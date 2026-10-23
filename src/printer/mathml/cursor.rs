//! Bounded bookmarks into borrowed immutable storage. Locations are captured
//! only from valid views/iterators, never from frontend-provided byte offsets.
use super::*;
use crate::atom::ListIterator;
use std::{collections::VecDeque, ops::Range, sync::Mutex};

const MAX_LOCATIONS: usize = 64;
const MAX_CURSORS: usize = 128;

#[derive(Clone)]
struct Cursor {
    node: Range<usize>,
    index: usize,
    tail: Range<usize>,
    remaining: usize,
}

#[derive(Default)]
struct Bookmarks {
    locations: VecDeque<(Vec<usize>, Range<usize>)>,
    cursors: VecDeque<Cursor>,
}

pub(crate) struct MathMlCursorCache<'a> {
    expression: AtomView<'a>,
    bookmarks: Mutex<Bookmarks>,
}

impl<'a> MathMlCursorCache<'a> {
    pub(crate) fn new(expression: AtomView<'a>) -> Self {
        Self {
            expression,
            bookmarks: Mutex::default(),
        }
    }

    fn data(&self) -> &[u8] {
        self.expression.get_data()
    }

    fn range(&self, data: &[u8]) -> Option<Range<usize>> {
        let root = self.data();
        let start = (data.as_ptr() as usize).checked_sub(root.as_ptr() as usize)?;
        let end = start.checked_add(data.len())?;
        (end <= root.len()).then_some(start..end)
    }

    pub(crate) fn locate(&self, path: &[usize]) -> Option<AtomView<'_>> {
        // Byte ranges lose the ownership context required by nested aliases.
        // Replay alias-containing paths from the original live view instead.
        let prefix = if self.expression.has_alias() {
            None
        } else {
            let mut bookmarks = self.bookmarks.lock().unwrap();
            let index = bookmarks
                .locations
                .iter()
                .enumerate()
                .filter(|(_, (prefix, _))| path.starts_with(prefix))
                .max_by_key(|(_, (prefix, _))| prefix.len())
                .map(|(index, _)| index);
            index.map(|index| {
                let entry = bookmarks.locations.remove(index).unwrap();
                let result = (entry.0.len(), entry.1.clone());
                bookmarks.locations.push_back(entry);
                result
            })
        };
        let (mut view, prefix_len) = match prefix {
            Some((len, range)) => (AtomView::from(&self.data()[range]), len),
            None => (self.expression, 0),
        };
        for (depth, index) in path.iter().enumerate().skip(prefix_len) {
            if view.mathml_child_count() == 0 {
                return None;
            }
            view = MathMlChildren::new(view, *index, Some(self)).next()?.1;
            let Some(range) = self.range(view.get_data()) else {
                // Alias bodies live in their own allocations. They remain
                // borrowed through the original expression, but are not offsets
                // into its contiguous root byte buffer.
                continue;
            };
            let prefix = path[..=depth].to_vec();
            let mut bookmarks = self.bookmarks.lock().unwrap();
            if let Some(index) = bookmarks.locations.iter().position(|(p, _)| *p == prefix) {
                bookmarks.locations.remove(index);
            }
            bookmarks.locations.push_back((prefix, range));
            while bookmarks.locations.len() > MAX_LOCATIONS {
                bookmarks.locations.pop_front();
            }
        }
        Some(view)
    }

    fn resume<'b>(
        &self,
        node: AtomView<'b>,
        iter: ListIterator<'b>,
        start: usize,
    ) -> (ListIterator<'b>, usize) {
        // An alias's single child is in a separate allocation, including its
        // exhausted iterator cursor. It cannot use root-buffer bookmarks.
        if matches!(node, AtomView::Alias(_)) {
            return (iter, 0);
        }
        let Some(node_range) = self.range(node.get_data()) else {
            return (iter, 0);
        };
        let cursor = {
            let mut bookmarks = self.bookmarks.lock().unwrap();
            let index = bookmarks
                .cursors
                .iter()
                .enumerate()
                .filter(|(_, c)| c.node == node_range && c.index <= start)
                .max_by_key(|(_, c)| c.index)
                .map(|(index, _)| index);
            index.map(|index| {
                let cursor = bookmarks.cursors.remove(index).unwrap();
                bookmarks.cursors.push_back(cursor.clone());
                cursor
            })
        };
        let Some(cursor) = cursor else {
            return (iter, 0);
        };
        let tail = &node.get_data()
            [cursor.tail.start - node_range.start..cursor.tail.end - node_range.start];
        // SAFETY: `remember` captured this exact iterator suffix and remaining
        // count for this node. The cache borrows its immutable backing Atom, so
        // its owner cannot mutate or release the bytes while they are borrowed.
        // Foreign views are rejected by `range`; no cache lock spans callbacks.
        (
            unsafe { iter.resume_at(tail, cursor.remaining) },
            cursor.index,
        )
    }

    fn remember(&self, node: AtomView<'_>, index: usize, iter: &ListIterator<'_>) {
        if matches!(node, AtomView::Alias(_)) {
            return;
        }
        let Some(node) = self.range(node.get_data()) else {
            return;
        };
        let (data, remaining) = iter.cursor();
        let tail = if remaining == 0 {
            node.end..node.end
        } else {
            let Some(range) = self.range(data) else {
                return;
            };
            range
        };
        let cursor = Cursor {
            node,
            index,
            tail,
            remaining,
        };
        let mut bookmarks = self.bookmarks.lock().unwrap();
        if let Some(index) = bookmarks
            .cursors
            .iter()
            .position(|c| c.node == cursor.node && c.index == index)
        {
            bookmarks.cursors.remove(index);
        }
        bookmarks.cursors.push_back(cursor);
        while bookmarks.cursors.len() > MAX_CURSORS {
            bookmarks.cursors.pop_front();
        }
    }
}

pub(super) struct MathMlChildren<'a> {
    node: AtomView<'a>,
    iter: ListIterator<'a>,
    index: usize,
    cache: Option<&'a MathMlCursorCache<'a>>,
}

impl<'a> MathMlChildren<'a> {
    pub(super) fn new(
        node: AtomView<'a>,
        start: usize,
        cache: Option<&'a MathMlCursorCache<'a>>,
    ) -> Self {
        let iter = node.mathml_children();
        let (mut iter, mut index) = match cache {
            Some(cache) => cache.resume(node, iter, start),
            None => (iter, 0),
        };
        let start = start.min(node.mathml_child_count());
        if start > index {
            iter.nth(start - index - 1);
            index = start;
        }
        let result = Self {
            node,
            iter,
            index,
            cache,
        };
        result.remember();
        result
    }

    fn remember(&self) {
        if let Some(cache) = self.cache {
            cache.remember(self.node, self.index, &self.iter);
        }
    }
}

impl<'a> Iterator for MathMlChildren<'a> {
    type Item = (usize, AtomView<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()?;
        let index = self.index;
        self.index += 1;
        Some((index, value))
    }
}

impl Drop for MathMlChildren<'_> {
    fn drop(&mut self) {
        self.remember();
    }
}
