//! Parallel forward solving with exactly the serial result.
//!
//! [`SparseRowReducer::add_csr_rows_parallel`] inserts a batch of rows and produces the same `U`,
//! pivots, `L` (pattern and values, with or without the rows of dependent inputs) and per-row
//! return values as calling [`SparseRowReducer::add_row`] on the rows in order, bit for bit, for
//! every window and block size and every number of threads.
//!
//! # Why it is exact
//!
//! Let `V` be the set of `U` rows present when row `r` is forward solved serially and `W` any
//! subset of `V`. Run the serial loop for `r` against `W` only. Pivots are never removed or
//! changed, so by induction over the visited columns the two runs agree as long as every visited
//! nonzero column has a pivot in `W`. Hence:
//!
//! * (a) if `r` reduces to zero against `W`, the serial run is identical (same subtractions in the
//!   same order, same `L`);
//! * (b) if `r` stops at column `c` (nonzero, no pivot in `W`), the serial run is identical up to
//!   `c`, and it differs only if `V` has a pivot at `c`, i.e. if a row EARLIER than `r` ends with
//!   pivot `c`. Resuming from the state at `c` continues exactly as the serial run.
//!
//! # Algorithm
//!
//! A sliding window of unresolved rows, in input order. Each round:
//!
//! * Parallel phase. The rows that need work are cut into blocks of consecutive rows; every block
//!   is reduced by one task, row after row, against the committed `U`, the window rows that are
//!   already final, and PROVISIONALLY the copy rows (accepted without reduction, if their column
//!   survives) earlier in the same block. A row records which provisional rows it used.
//! * Serial commit walk in input order, with `bound` = the smallest stop column of an earlier
//!   unresolved row (stop columns never decrease over the rounds, and a row's final pivot cannot
//!   lie left of its stop column):
//!   - a row that used a provisional row that was not accepted is invalidated: it goes back to its
//!     state before this round (validated in the previous walk) and its bound is that state's stop
//!     column (or its first column), so it never revisits columns left of an earlier bound;
//!   - otherwise its view was a subset of its serial view: a dependent row is final (a); a row
//!     stopped at an unpivoted `c < bound` is final and claims `c` (b: no earlier row can still end
//!     at `c`); a row whose `c` was claimed meanwhile resumes from its state next round; any other
//!     row waits.
//!   The first unresolved row always becomes final, so the loop terminates.
//! * Flush of the resolved prefix in input order: accepted rows receive their `U` row index,
//!   tagged references (pivots and `L` entries pointing at window rows) are rewritten, and the
//!   serial early exits (empty row, full rank) are reproduced.
//!
//! A final or provisional row that is later in the input than an unresolved row `e` claims a
//! column left of `e`'s bound; `e` only visits columns right of it (input rows and hence all `U`
//! rows have strictly increasing column indices, so a subtraction only touches columns right of
//! the visited one), so `e` never sees a pivot of a later row. Every accepted or dependent row
//! therefore saw a subset of its serial view, which is all the lemma needs.

use std::{cmp::Reverse, collections::VecDeque, sync::Mutex, time::Instant};

use rayon::prelude::*;

use super::{LuLMode, Scratch, SparseMatrix, SparseRowReducer};
use crate::domains::Field;

/// Tag bit of a pivot or `L` reference to a row of the current batch that is final but not yet
/// flushed into `U` (or provisional); the low bits are the row's position in the batch.
const TAG: u32 = 1 << 31;

/// No provisional pivot on a column.
const NONE: u32 = u32::MAX;

/// Which rows of a block serve as provisional pivots for the later rows of the block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Provisional {
    /// Copy rows only (accepted without reduction if their column survives): nearly always
    /// accepted in a batch sorted by decreasing first column.
    #[default]
    Copies,
    /// Every stopped row (more chains resolved per round, but rows that used a rejected
    /// provisional row are undone).
    All,
    /// None: every row waits for final pivots.
    None,
}

/// Tuning of [`SparseRowReducer::add_csr_rows_parallel`]. No setting changes the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelOptions {
    /// The largest number of unresolved rows in flight.
    pub window: usize,
    /// The number of consecutive working rows one task reduces in order.
    pub block: usize,
    /// The provisional pivots within a block.
    pub provisional: Provisional,
}

impl Default for ParallelOptions {
    fn default() -> Self {
        // measured on four-loop reproducers: a window of 4096 keeps the speculative states small
        // (2-5 % extra work) and still gives a modelled 14x on 16 cores
        Self {
            window: 4096,
            block: 16,
            provisional: Provisional::Copies,
        }
    }
}

/// Outcome of one input row of [`SparseRowReducer::add_csr_rows_parallel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowOutcome {
    /// The pivot column of the row in `U` if it was linearly independent (the return value of
    /// [`SparseRowReducer::add_row`]).
    pub pivot: Option<u32>,
    /// The index of the `L` row recorded for this input row, if any.
    pub l_row: Option<u32>,
}

/// Work counters of one call of [`SparseRowReducer::add_csr_rows_parallel`]. The counts depend on
/// the window and block sizes but not on the number of threads; the seconds are measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParallelStats {
    /// Input rows.
    pub rows: usize,
    /// Parallel phases (each followed by one serial commit walk).
    pub rounds: usize,
    /// Row reductions run (first reductions, resumptions and repeated undone reductions).
    pub reductions: usize,
    /// Resumptions of a row whose stop column was claimed by an earlier row.
    pub resumed: usize,
    /// Undone reductions of a row that used a provisional row that was not accepted.
    pub invalidated: usize,
    /// Rows accepted by copying (no reduction needed).
    pub copies: usize,
    /// Pivot-row entries subtracted (the work of the serial code, plus that of undone reductions).
    pub subtracted: u64,
    /// State entries scattered at resumptions and gathered at stops, and copied input entries.
    pub state_entries: u64,
    /// Per round: the total work and the work of the largest block (entries).
    pub round_work: Vec<(u64, u64)>,
    /// Rows visited by the serial commit walks.
    pub walk_visits: u64,
    /// Wall seconds of the parallel phases and of the serial walks and flushes.
    pub parallel_seconds: f64,
    pub serial_seconds: f64,
    /// The serial seconds split into admission, block cutting, commit walk and flush.
    pub serial_split: [f64; 4],
}

impl ParallelStats {
    /// Total work of the parallel phases (entries).
    pub fn work(&self) -> u64 {
        self.round_work.iter().map(|&(total, _)| total).sum()
    }

    /// Modelled speed-up of the parallel phases on `threads` dedicated cores: the total work over
    /// the sum, per round, of the larger of an even share and the largest block.
    pub fn modelled_speedup(&self, threads: usize) -> f64 {
        let t = threads.max(1) as f64;
        let span: f64 = self
            .round_work
            .iter()
            .map(|&(total, largest)| (total as f64 / t).max(largest as f64))
            .sum();
        if span == 0.0 {
            1.0
        } else {
            self.work() as f64 / span
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    /// Needs a (first, resumed or repeated) reduction in the next parallel phase.
    Work,
    /// Stopped at a nonzero column without pivot in the view of its last reduction; `prepared`
    /// holds the row as it would be stored in `U`.
    Stopped(u32),
    /// Reduced to zero (final once its provisional uses are validated).
    Dependent,
    /// Final: accepted with this pivot column; its row is in the window buffer.
    Accepted(u32),
    /// Final: empty input row (no `L` row, as in the serial code).
    Empty,
}

struct WindowRow<E> {
    /// Position in the batch.
    pos: u32,
    /// The first column of the input row.
    lead_col: u32,
    status: Status,
    /// No reduction done yet: the state is the input row.
    fresh: bool,
    /// Unnormalised state at the stop column (stop column first, then increasing columns,
    /// nonzero values only); empty while `fresh`.
    cols: Vec<u32>,
    vals: Vec<E>,
    /// The normalised values of a stopped row as they will be stored in `U` (its columns are
    /// those of the input row for a fresh copy row, else `cols`).
    prepared: Option<Prepared<E>>,
    /// Positions of the provisional rows used by the last reduction (to be validated).
    uses: Vec<u32>,
    /// The state before the last reduction, kept while `uses` is not validated.
    saved: Option<Saved<E>>,
    /// `L` entries recorded so far (`U` row indices or tagged batch positions) and, in
    /// [`LuLMode::Full`], their values.
    l_cols: Vec<u32>,
    l_vals: Vec<E>,
}

/// The state of a row before a reduction that used provisional rows.
struct Saved<E> {
    fresh: bool,
    cols: Vec<u32>,
    vals: Vec<E>,
    l_len: usize,
    l_vals_len: usize,
}

impl<E> WindowRow<E> {
    /// Undo the last reduction (it used a provisional row that was not accepted).
    fn restore(&mut self) {
        let saved = self
            .saved
            .take()
            .expect("a row with provisional uses keeps its state");
        self.status = Status::Work;
        self.fresh = saved.fresh;
        self.cols = saved.cols;
        self.vals = saved.vals;
        self.l_cols.truncate(saved.l_len);
        self.l_vals.truncate(saved.l_vals_len);
        self.prepared = None;
        self.uses.clear();
    }

    /// The column where the next reduction starts (a lower bound of the final pivot).
    fn start_col(&self) -> u32 {
        if self.fresh {
            self.lead_col
        } else {
            self.cols[0]
        }
    }
}

/// The input rows in compressed sparse row form.
struct Input<'a, E> {
    values: &'a [E],
    col_idcs: &'a [u32],
    row_ptrs: &'a [usize],
}

impl<E> Clone for Input<'_, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E> Copy for Input<'_, E> {}

impl<'a, E> Input<'a, E> {
    #[inline]
    fn row(&self, pos: u32) -> (&'a [E], &'a [u32]) {
        let start = self.row_ptrs[pos as usize];
        let end = self.row_ptrs[pos as usize + 1];
        (&self.values[start..end], &self.col_idcs[start..end])
    }
}

/// The normalised values and the leading coefficient of a stopped row.
struct Prepared<E> {
    vals: Vec<E>,
    /// The leading coefficient before normalisation (the diagonal of `L`).
    lead: E,
}

/// A final accepted row of the window, normalised as it is stored in `U`.
struct BufferRow<E> {
    cols: Vec<u32>,
    vals: Vec<E>,
    /// The leading coefficient before normalisation (the diagonal of `L`).
    lead: E,
}

/// Per-thread scratch: the dense row of the serial code and the provisional pivots of a block.
struct ThreadScratch<F: Field> {
    dense: Scratch<F>,
    /// Block index of the provisional row with this pivot column, or [`NONE`].
    local: Vec<u32>,
    /// Columns set in `local`.
    local_cols: Vec<u32>,
}

/// Read-only view of the committed and final pivots during a parallel phase.
struct View<'a, F: Field> {
    u: &'a SparseMatrix<F>,
    pivots: &'a [Option<u32>],
    buffer: &'a [Option<BufferRow<F::Element>>],
    slot_of_pos: &'a [u32],
}

impl<'a, F: Field> View<'a, F> {
    #[inline]
    fn row(&self, reference: u32) -> (&'a [F::Element], &'a [u32]) {
        if reference & TAG == 0 {
            let start = self.u.row_ptrs[reference as usize];
            let end = self.u.row_ptrs[reference as usize + 1];
            (&self.u.values[start..end], &self.u.col_idcs[start..end])
        } else {
            let slot = self.slot_of_pos[(reference & !TAG) as usize] as usize;
            let row = self.buffer[slot]
                .as_ref()
                .expect("a tagged pivot refers to a buffered row");
            (&row.vals, &row.cols)
        }
    }
}

/// Reserve `additional` elements and touch the whole spare capacity in parallel, so that later
/// serial appends find their pages mapped. Writes only uninitialised slots (no length change).
fn prefault<T: Clone + Send + Sync>(vec: &mut Vec<T>, additional: usize, fill: &T) {
    vec.reserve(additional);
    vec.spare_capacity_mut()
        .par_chunks_mut(1 << 15)
        .for_each(|chunk| {
            for slot in chunk {
                slot.write(fill.clone());
            }
        });
}

fn normalise<F: Field>(field: &F, vals: &[F::Element], copy: bool) -> Prepared<F::Element> {
    let lead = vals[0].clone();
    let inv = field.inv(&lead);
    // the copy shortcut multiplies `val * inv`, the gather `inv * val` (as in the serial code)
    let vals = if field.is_one(&inv) {
        vals.to_vec()
    } else if copy {
        vals.iter().map(|val| field.mul(val, &inv)).collect()
    } else {
        vals.iter().map(|val| field.mul(&inv, val)).collect()
    };
    Prepared { vals, lead }
}

impl<F: Field + Sync + Send> SparseRowReducer<F>
where
    F::Element: Sync + Send,
{
    /// Forward solve the rows of `mat` in parallel, with exactly the result of
    /// [`Self::add_matrix`] (see [`Self::add_csr_rows_parallel`]).
    pub fn add_matrix_parallel(
        &mut self,
        mat: &SparseMatrix<F>,
        options: ParallelOptions,
    ) -> (Vec<RowOutcome>, ParallelStats) {
        assert_eq!(mat.ncols(), self.u.ncols());
        self.add_csr_rows_parallel(&mat.values, &mat.col_idcs, &mat.row_ptrs, options)
    }

    /// Forward solve a batch of rows in compressed sparse row form in parallel (on the current
    /// rayon pool), with exactly the result of calling [`Self::add_row`] on every row in order:
    /// the same `U`, pivots, `L` (in every [`LuLMode`], with or without the `L` rows of dependent
    /// rows) and return values, for all `options` and every number of threads.
    ///
    /// Row `i` has the values `values[row_ptrs[i]..row_ptrs[i + 1]]` at the columns
    /// `col_idcs[row_ptrs[i]..row_ptrs[i + 1]]`, which must be strictly increasing (this is
    /// checked). Every row already in the reducer must also have strictly increasing columns,
    /// which holds when every row ever added had them.
    ///
    /// The [`ParallelOptions`] change the speed and the work counters, never the result.
    pub fn add_csr_rows_parallel(
        &mut self,
        values: &[F::Element],
        col_idcs: &[u32],
        row_ptrs: &[usize],
        options: ParallelOptions,
    ) -> (Vec<RowOutcome>, ParallelStats) {
        assert!(!row_ptrs.is_empty(), "row_ptrs holds at least the start");
        assert_eq!(values.len(), col_idcs.len());
        assert_eq!(*row_ptrs.last().unwrap(), values.len());
        let nrows = row_ptrs.len() - 1;
        assert!(
            nrows < TAG as usize,
            "batch too large for tagged references"
        );
        assert!(
            (self.u.nrows as usize) + nrows < TAG as usize,
            "U too large for tagged references"
        );
        for row in row_ptrs.windows(2) {
            let cols = &col_idcs[row[0]..row[1]];
            assert!(
                cols.windows(2).all(|pair| pair[0] < pair[1]),
                "parallel forward solving needs strictly increasing column indices"
            );
            assert!(cols.last().map_or(true, |&col| col < self.u.ncols));
        }

        // Pre-fault the expected growth of U and L in parallel. The flush appends to them on one
        // thread, and on a loaded host the page faults of fresh memory dominate that serial step.
        // Fill-in makes U grow by up to about twice the input entries.
        if !std::mem::needs_drop::<F::Element>() {
            let zero = self.u.field.zero();
            prefault(&mut self.u.values, 2 * values.len(), &zero);
            prefault(&mut self.u.col_idcs, 2 * values.len(), &0);
            if self.mode != LuLMode::None {
                prefault(&mut self.l.col_idcs, values.len(), &0);
                if self.mode == LuLMode::Full {
                    prefault(&mut self.l.values, values.len(), &zero);
                }
            }
        }

        let window = options.window.max(1);
        let block = options.block.max(1);
        let provisional = options.provisional;
        let ncols = self.u.ncols as usize;
        let record_l = self.mode != LuLMode::None;
        let full = self.mode == LuLMode::Full;
        let input = Input {
            values,
            col_idcs,
            row_ptrs,
        };

        let mut stats = ParallelStats {
            rows: nrows,
            ..ParallelStats::default()
        };
        let mut outcomes = vec![
            RowOutcome {
                pivot: None,
                l_row: None,
            };
            nrows
        ];
        // U row index of every flushed accepted row of the batch, by position.
        let mut real_of_pos = vec![u32::MAX; nrows];
        // Buffer slot of every final accepted row that is not flushed yet, by position.
        let mut slot_of_pos = vec![u32::MAX; nrows];
        let mut buffer: Vec<Option<BufferRow<F::Element>>> = Vec::new();
        let mut free_slots: Vec<u32> = Vec::new();

        let threads = rayon::current_num_threads().max(1);
        let scratches: Vec<Mutex<Option<ThreadScratch<F>>>> =
            (0..threads).map(|_| Mutex::new(None)).collect();

        let mut pending: VecDeque<WindowRow<F::Element>> = VecDeque::new();
        // positions of the unresolved (working or stopped) pending rows, increasing
        let mut open: Vec<u32> = Vec::new();
        let mut next: u32 = 0;

        loop {
            let serial_started = Instant::now();
            // admit new rows (all later than every pending row): at most `window` unresolved
            // rows, and at most `8 * window` rows waiting for the flush of their prefix
            while open.len() < window
                && pending.len() < window.saturating_mul(8)
                && (next as usize) < nrows
            {
                let (_, cols) = input.row(next);
                let empty = cols.is_empty();
                pending.push_back(WindowRow {
                    pos: next,
                    lead_col: cols.first().copied().unwrap_or(0),
                    status: if empty { Status::Empty } else { Status::Work },
                    fresh: true,
                    cols: Vec::new(),
                    vals: Vec::new(),
                    prepared: None,
                    uses: Vec::new(),
                    saved: None,
                    l_cols: Vec::new(),
                    l_vals: Vec::new(),
                });
                if !empty {
                    open.push(next);
                }
                next += 1;
            }
            if pending.is_empty() {
                stats.serial_seconds += serial_started.elapsed().as_secs_f64();
                break;
            }

            stats.serial_split[0] += serial_started.elapsed().as_secs_f64();
            let cut_started = Instant::now();
            // cut the working rows into blocks of `block` rows (deterministically): block k spans
            // the pending rows from its first working row to the next block's first working row
            let front = pending.front().map_or(0, |row| row.pos);
            let rows = pending.make_contiguous();
            let mut cuts = Vec::new();
            let mut in_block = 0;
            let mut working = 0;
            let mut resumed = 0;
            let mut last = 0;
            for &pos in &open {
                let i = (pos - front) as usize;
                if rows[i].status == Status::Work {
                    if in_block == 0 || in_block == block {
                        cuts.push(i);
                        in_block = 0;
                    }
                    in_block += 1;
                    working += 1;
                    resumed += (!rows[i].fresh) as usize;
                    last = i;
                }
            }
            cuts.push(last + 1);
            stats.serial_split[1] += cut_started.elapsed().as_secs_f64();
            stats.serial_seconds += serial_started.elapsed().as_secs_f64();

            // parallel phase: reduce the blocks against the frozen view
            if working > 0 {
                let parallel_started = Instant::now();
                stats.rounds += 1;
                stats.reductions += working;
                stats.resumed += resumed;
                let mut blocks: Vec<&mut [WindowRow<F::Element>]> = Vec::with_capacity(cuts.len());
                let mut rest = &mut rows[cuts[0]..];
                for pair in cuts.windows(2) {
                    let (head, tail) = rest.split_at_mut(pair[1] - pair[0]);
                    blocks.push(head);
                    rest = tail;
                }
                let view = View {
                    u: &self.u,
                    pivots: &self.pivots,
                    buffer: &buffer,
                    slot_of_pos: &slot_of_pos,
                };
                let field = &self.u.field;
                let scratches = &scratches;
                let work: Vec<(u64, u64)> = blocks
                    .into_par_iter()
                    .map(|block_rows| {
                        let index = rayon::current_thread_index().unwrap_or(0) % threads;
                        let mut guard = scratches[index].lock().unwrap();
                        let scratch = guard.get_or_insert_with(|| ThreadScratch {
                            dense: Scratch::new(ncols as u32, field),
                            local: vec![NONE; ncols],
                            local_cols: Vec::new(),
                        });
                        Self::reduce_block(
                            &view,
                            field,
                            block_rows,
                            input,
                            scratch,
                            provisional,
                            record_l,
                            full,
                        )
                    })
                    .collect();
                let (mut total, mut largest) = (0, 0);
                for &(subtracted, state) in &work {
                    stats.subtracted += subtracted;
                    stats.state_entries += state;
                    total += subtracted + state;
                    largest = largest.max(subtracted + state);
                }
                stats.round_work.push((total, largest));
                stats.parallel_seconds += parallel_started.elapsed().as_secs_f64();
            }

            // serial commit walk in input order over the unresolved rows (the rows reduced in this
            // round are among them)
            let serial_started = Instant::now();
            let mut bound = u32::MAX;
            for &pos in &open {
                let i = (pos - front) as usize;
                stats.walk_visits += 1;
                // provisional uses are earlier rows, already decided in this walk
                let valid = pending[i].uses.iter().all(|&pos| {
                    pos < front
                        || matches!(pending[(pos - front) as usize].status, Status::Accepted(_))
                });
                let row = &mut pending[i];
                if !valid {
                    row.restore();
                    stats.invalidated += 1;
                    bound = bound.min(row.start_col());
                    continue;
                }
                row.uses.clear();
                row.saved = None;
                match row.status {
                    Status::Stopped(col) => {
                        if self.pivots[col as usize].is_some() {
                            row.status = Status::Work;
                            row.prepared = None;
                            bound = bound.min(col);
                        } else if col < bound {
                            if row.fresh {
                                stats.copies += 1;
                            }
                            let prepared = row.prepared.take().expect("a stopped row is prepared");
                            let cols = if row.fresh {
                                input.row(row.pos).1.to_vec()
                            } else {
                                std::mem::take(&mut row.cols)
                            };
                            row.vals = Vec::new();
                            let buffered = BufferRow {
                                cols,
                                vals: prepared.vals,
                                lead: prepared.lead,
                            };
                            let slot = match free_slots.pop() {
                                Some(slot) => {
                                    buffer[slot as usize] = Some(buffered);
                                    slot
                                }
                                None => {
                                    buffer.push(Some(buffered));
                                    (buffer.len() - 1) as u32
                                }
                            };
                            slot_of_pos[row.pos as usize] = slot;
                            self.pivots[col as usize] = Some(TAG | row.pos);
                            row.status = Status::Accepted(col);
                        } else {
                            bound = bound.min(col);
                        }
                    }
                    Status::Dependent => {
                        // final: its L row is kept only when dependent L rows are retained
                        if !(self.retain_dependent_l && record_l) {
                            row.l_cols = Vec::new();
                            row.l_vals = Vec::new();
                        }
                        row.cols = Vec::new();
                        row.vals = Vec::new();
                    }
                    Status::Work => unreachable!("every working row was reduced"),
                    Status::Accepted(_) | Status::Empty => {}
                }
            }
            open.retain(|&pos| {
                matches!(
                    pending[(pos - front) as usize].status,
                    Status::Work | Status::Stopped(_)
                )
            });

            stats.serial_split[2] += serial_started.elapsed().as_secs_f64();
            let flush_started = Instant::now();
            // flush the resolved prefix in input order
            while let Some(row) = pending.front() {
                if matches!(row.status, Status::Work | Status::Stopped(_)) {
                    break;
                }
                let row = pending.pop_front().unwrap();
                self.flush_row(
                    row,
                    &mut outcomes,
                    &mut real_of_pos,
                    &slot_of_pos,
                    &mut buffer,
                    &mut free_slots,
                );
            }
            stats.serial_split[3] += flush_started.elapsed().as_secs_f64();
            stats.serial_seconds += serial_started.elapsed().as_secs_f64();
        }

        debug_assert!(buffer.iter().all(Option::is_none));
        (outcomes, stats)
    }

    /// Reduce the working rows of one block in order. A copy row becomes a provisional pivot for
    /// the later rows of the block.
    /// Returns the work: (pivot-row entries subtracted, state entries moved).
    fn reduce_block(
        view: &View<'_, F>,
        field: &F,
        rows: &mut [WindowRow<F::Element>],
        input: Input<'_, F::Element>,
        scratch: &mut ThreadScratch<F>,
        provisional: Provisional,
        record_l: bool,
        full: bool,
    ) -> (u64, u64) {
        let mut work = (0, 0);
        for i in 0..rows.len() {
            let (before, rest) = rows.split_at_mut(i);
            let row = &mut rest[0];
            match row.status {
                Status::Work => {
                    let (subtracted, state) = Self::reduce_speculatively(
                        view,
                        field,
                        row,
                        input,
                        &mut scratch.dense,
                        &scratch.local,
                        before,
                        record_l,
                        full,
                    );
                    work.0 += subtracted;
                    work.1 += state;
                }
                // waiting rows are not provisional pivots: they are blocked by an earlier row, so
                // rows using them would likely be undone
                Status::Stopped(_) | Status::Dependent | Status::Accepted(_) | Status::Empty => {
                    continue;
                }
            }
            // Only COPY rows (fresh rows whose first column is free) become provisional pivots. In
            // a batch sorted by decreasing first column, an earlier row stops at or right of its
            // own first column, so a copy row can only lose its column to an earlier row with the
            // same first column, which is nearly always in the same block: copies are almost
            // always accepted. A row that stopped after a reduction may be waiting for a pivot of
            // another block; rows using it would be undone, so later rows rather wait for it.
            let candidate = match provisional {
                Provisional::Copies => row.fresh,
                Provisional::All => true,
                Provisional::None => false,
            };
            if let (Status::Stopped(col), true) = (row.status, candidate) {
                let col = col as usize;
                if scratch.local[col] == NONE && view.pivots[col].is_none() {
                    scratch.local[col] = i as u32;
                    scratch.local_cols.push(col as u32);
                }
            }
        }
        for &col in &scratch.local_cols {
            scratch.local[col as usize] = NONE;
        }
        scratch.local_cols.clear();
        work
    }

    /// One (first, resumed or repeated) reduction of a window row: the serial loop of
    /// `forward_solve_row` against the view and the provisional rows `before` it in its block,
    /// stopping at the first nonzero column without pivot. Returns the work.
    fn reduce_speculatively(
        view: &View<'_, F>,
        field: &F,
        row: &mut WindowRow<F::Element>,
        input: Input<'_, F::Element>,
        scratch: &mut Scratch<F>,
        local: &[u32],
        before: &[WindowRow<F::Element>],
        record_l: bool,
        full: bool,
    ) -> (u64, u64) {
        debug_assert!(scratch.heap.is_empty());
        debug_assert!(row.uses.is_empty() && row.saved.is_none());
        let mut subtracted = 0u64;
        let mut state = 0u64;
        let was_fresh = row.fresh;
        let l_len = row.l_cols.len();
        let l_vals_len = row.l_vals.len();
        let mut old: (Vec<u32>, Vec<F::Element>) = (Vec::new(), Vec::new());
        if row.fresh {
            let (values, col_idcs) = input.row(row.pos);
            let first = col_idcs[0] as usize;
            // the serial shortcut: copy when the first column has no pivot
            if view.pivots[first].is_none() && local[first] == NONE {
                row.prepared = Some(normalise(field, values, true));
                row.status = Status::Stopped(col_idcs[0]);
                return (0, values.len() as u64);
            }
            row.fresh = false;
            for (val, &col) in values.iter().zip(col_idcs) {
                scratch.dense_row[col as usize] = val.clone();
                if !scratch.touched[col as usize] {
                    scratch.touched[col as usize] = true;
                    scratch.heap.push(Reverse(col));
                }
            }
        } else {
            old = (std::mem::take(&mut row.cols), std::mem::take(&mut row.vals));
            state += old.0.len() as u64;
            for (val, &col) in old.1.iter().zip(&old.0) {
                scratch.dense_row[col as usize] = val.clone();
                scratch.touched[col as usize] = true;
                scratch.heap.push(Reverse(col));
            }
        }
        // keep the state before this reduction until its provisional uses are validated
        let checkpoint = |row: &mut WindowRow<F::Element>, old: (Vec<u32>, Vec<F::Element>)| {
            if !row.uses.is_empty() {
                row.saved = Some(Saved {
                    fresh: was_fresh,
                    cols: old.0,
                    vals: old.1,
                    l_len,
                    l_vals_len,
                });
            }
        };

        while let Some(Reverse(col)) = scratch.heap.pop() {
            let col = col as usize;
            debug_assert!(scratch.touched[col]);
            if field.is_zero(&scratch.dense_row[col]) {
                scratch.touched[col] = false;
                continue;
            }
            let pivot = match view.pivots[col] {
                Some(reference) => {
                    let (vals, cols) = view.row(reference);
                    Some((reference, vals, cols))
                }
                None if local[col] != NONE => {
                    let provisional = &before[local[col] as usize];
                    row.uses.push(provisional.pos);
                    let prepared = provisional
                        .prepared
                        .as_ref()
                        .expect("a provisional pivot is prepared");
                    let cols = if provisional.fresh {
                        input.row(provisional.pos).1
                    } else {
                        &provisional.cols[..]
                    };
                    Some((TAG | provisional.pos, &prepared.vals[..], cols))
                }
                None => None,
            };
            if let Some((reference, pivot_values, pivot_cols)) = pivot {
                let pivot_val = scratch.dense_row[col].clone();
                Self::scatter_with_heap(
                    &mut scratch.dense_row,
                    &field.neg(pivot_val.clone()),
                    pivot_values,
                    pivot_cols,
                    field,
                    &mut scratch.touched,
                    &mut scratch.heap,
                );
                subtracted += pivot_cols.len() as u64;
                debug_assert!(field.is_zero(&scratch.dense_row[col]));
                scratch.touched[col] = false;
                if record_l {
                    row.l_cols.push(reference);
                    if full {
                        row.l_vals.push(pivot_val);
                    }
                }
            } else {
                // stop: save the state (this column first, then the rest in increasing order)
                scratch.list.clear();
                scratch.list.push(col as u32);
                scratch
                    .list
                    .extend(scratch.heap.drain().map(|Reverse(col)| col));
                scratch.list[1..].sort_unstable();
                row.cols.reserve(scratch.list.len());
                row.vals.reserve(scratch.list.len());
                for &i in &scratch.list {
                    let i = i as usize;
                    let val = std::mem::replace(&mut scratch.dense_row[i], field.zero());
                    if !field.is_zero(&val) {
                        row.cols.push(i as u32);
                        row.vals.push(val);
                    }
                    scratch.touched[i] = false;
                }
                scratch.list.clear();
                state += 2 * row.cols.len() as u64;
                row.prepared = Some(normalise(field, &row.vals, false));
                row.status = Status::Stopped(col as u32);
                checkpoint(row, old);
                return (subtracted, state);
            }
        }
        row.status = Status::Dependent;
        checkpoint(row, old);
        (subtracted, state)
    }

    /// Commit one final row in input order: the serial early exits, `U` and `L` rows, and the
    /// rewriting of tagged references.
    fn flush_row(
        &mut self,
        row: WindowRow<F::Element>,
        outcomes: &mut [RowOutcome],
        real_of_pos: &mut [u32],
        slot_of_pos: &[u32],
        buffer: &mut [Option<BufferRow<F::Element>>],
        free_slots: &mut Vec<u32>,
    ) {
        let pos = row.pos as usize;
        // the serial early exit: full rank reached, or an empty row
        if self.u.nrows == self.u.ncols || row.status == Status::Empty {
            // at full rank every column has a pivot, so no row can have been accepted
            assert!(
                !matches!(row.status, Status::Accepted(_)),
                "an accepted row at full rank"
            );
            return;
        }
        // L entries refer to U rows or to EARLIER rows of the batch, which are flushed already
        fn real(reference: u32, real_of_pos: &[u32]) -> u32 {
            if reference & TAG == 0 {
                reference
            } else {
                let real = real_of_pos[(reference & !TAG) as usize];
                debug_assert!(real != u32::MAX, "an L entry refers to an earlier row");
                real
            }
        }
        match row.status {
            Status::Dependent => {
                if self.mode != LuLMode::None && self.retain_dependent_l {
                    self.l
                        .col_idcs
                        .extend(row.l_cols.iter().map(|&r| real(r, real_of_pos)));
                    if self.mode == LuLMode::Full {
                        self.l.values.extend(row.l_vals);
                    }
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    outcomes[pos].l_row = Some(self.l.nrows);
                    self.l.nrows += 1;
                }
            }
            Status::Accepted(col) => {
                let slot = slot_of_pos[pos];
                let buffered = buffer[slot as usize]
                    .take()
                    .expect("an accepted row is buffered");
                free_slots.push(slot);
                let index = self.u.nrows;
                real_of_pos[pos] = index;
                self.pivots[col as usize] = Some(index);
                self.u.col_idcs.extend_from_slice(&buffered.cols);
                self.u.values.extend(buffered.vals);
                self.u.row_ptrs.push(self.u.values.len());
                self.u.nrows += 1;
                if self.mode != LuLMode::None {
                    self.l
                        .col_idcs
                        .extend(row.l_cols.iter().map(|&r| real(r, real_of_pos)));
                    self.l.col_idcs.push(index);
                    if self.mode == LuLMode::Full {
                        self.l.values.extend(row.l_vals);
                        self.l.values.push(buffered.lead);
                    }
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    outcomes[pos].l_row = Some(self.l.nrows);
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                outcomes[pos].pivot = Some(col);
            }
            Status::Work | Status::Stopped(_) | Status::Empty => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::{ParallelOptions, Provisional, RowOutcome};
    use crate::domains::{
        Ring, RingOps, Set,
        finite_field::{FiniteFieldCore, Zp64},
    };
    use crate::tensors::sparse::{LuLMode, SparseRowReducer};

    type El = <Zp64 as Set>::Element;
    type Row = (Vec<El>, Vec<u32>);

    /// Random rows with strictly increasing columns: fresh random rows, combinations of earlier
    /// rows (dependent or near-dependent), rows sharing leading columns, explicit zeros at
    /// non-leading positions and empty rows.
    fn random_rows(rng: &mut StdRng, field: &Zp64, nrows: usize, ncols: usize) -> Vec<Row> {
        let mut rows: Vec<Row> = vec![];
        for _ in 0..nrows {
            let mut dense = vec![field.zero(); ncols];
            let mut present = vec![false; ncols];
            let kind = rng.random_range(0..10);
            if kind == 0 {
                rows.push((vec![], vec![]));
                continue;
            }
            if !rows.is_empty() && kind <= 3 {
                for _ in 0..rng.random_range(1..=3) {
                    let other = &rows[rng.random_range(0..rows.len())];
                    let factor = field.to_element(rng.random_range(1..field.get_prime()));
                    for (v, &c) in other.0.iter().zip(&other.1) {
                        dense[c as usize] = field.add(&dense[c as usize], &field.mul(&factor, v));
                        present[c as usize] = true;
                    }
                }
            } else {
                // a leading column from a small set, so that leading columns collide
                let lead = rng.random_range(0..ncols.min(4).max(1)) * (ncols / 4).max(1);
                let lead = lead.min(ncols - 1);
                dense[lead] = field.to_element(rng.random_range(1..field.get_prime()));
                present[lead] = true;
                let density = rng.random_range(0.05..0.6);
                for c in lead + 1..ncols {
                    if rng.random_bool(density) {
                        dense[c] = field.to_element(rng.random_range(0..field.get_prime()));
                        present[c] = true;
                    }
                }
            }
            // keep explicit zeros only at non-leading positions (as a caller may pass them)
            let mut row: Row = (vec![], vec![]);
            for c in 0..ncols {
                let keep = if row.1.is_empty() {
                    !field.is_zero(&dense[c])
                } else {
                    present[c] && (!field.is_zero(&dense[c]) || rng.random_bool(0.2))
                };
                if keep {
                    row.0.push(dense[c].clone());
                    row.1.push(c as u32);
                }
            }
            rows.push(row);
        }
        rows
    }

    fn csr(rows: &[Row]) -> (Vec<El>, Vec<u32>, Vec<usize>) {
        let mut values = vec![];
        let mut cols = vec![];
        let mut ptrs = vec![0];
        for (v, c) in rows {
            values.extend_from_slice(v);
            cols.extend_from_slice(c);
            ptrs.push(values.len());
        }
        (values, cols, ptrs)
    }

    fn assert_same(serial: &SparseRowReducer<Zp64>, parallel: &SparseRowReducer<Zp64>) {
        assert_eq!(serial.u.nrows, parallel.u.nrows);
        assert_eq!(serial.u.ncols, parallel.u.ncols);
        assert_eq!(serial.u.row_ptrs, parallel.u.row_ptrs);
        assert_eq!(serial.u.col_idcs, parallel.u.col_idcs);
        assert_eq!(serial.u.values, parallel.u.values);
        assert_eq!(serial.pivots, parallel.pivots);
        assert_eq!(serial.l.nrows, parallel.l.nrows);
        assert_eq!(serial.l.ncols, parallel.l.ncols);
        assert_eq!(serial.l.row_ptrs, parallel.l.row_ptrs);
        assert_eq!(serial.l.col_idcs, parallel.l.col_idcs);
        assert_eq!(serial.l.values, parallel.l.values);
    }

    fn serial_batch(reducer: &mut SparseRowReducer<Zp64>, rows: &[Row]) -> Vec<RowOutcome> {
        rows.iter()
            .map(|(v, c)| {
                let before = reducer.l.nrows;
                let pivot = reducer.add_row(v, c);
                RowOutcome {
                    pivot,
                    l_row: (reducer.l.nrows > before).then_some(before),
                }
            })
            .collect()
    }

    /// The parallel forward solve equals the serial one field by field, for every L mode,
    /// with and without dependent L rows, for several batches with new columns in between,
    /// for many windows and thread counts.
    #[test]
    fn parallel_forward_solve_equals_serial() {
        let pools: Vec<rayon::ThreadPool> = [1, 2, 4, 8, 16]
            .iter()
            .map(|&n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap()
            })
            .collect();
        let mut rng = StdRng::seed_from_u64(0x9a7e_11e1);
        let mut cases = 0;
        let (mut resumed, mut waited, mut invalidated) = (0, 0, 0);
        for prime in [3, 5, 13, 2305843009213693951] {
            let field = Zp64::new(prime);
            for _ in 0..12 {
                let ncols = rng.random_range(1..48);
                let nbatches = rng.random_range(1..=3);
                let batches: Vec<(Vec<u32>, Vec<Row>)> = {
                    let mut width = ncols;
                    (0..nbatches)
                        .map(|b| {
                            // new columns before every batch but the first
                            let mut new_cols = vec![];
                            if b > 0 {
                                for _ in 0..rng.random_range(0..4) {
                                    new_cols.push(rng.random_range(0..=width as u32));
                                }
                                new_cols.sort_unstable();
                                width += new_cols.len();
                            }
                            let nrows = rng.random_range(0..80);
                            (new_cols, random_rows(&mut rng, &field, nrows, width))
                        })
                        .collect()
                };
                for mode in [LuLMode::Full, LuLMode::Pattern, LuLMode::None] {
                    for retain in [true, false] {
                        let mut serial = SparseRowReducer::new(ncols as u32, field.clone(), mode);
                        serial.set_retain_dependent_l(retain);
                        let mut expected = vec![];
                        let mut states = vec![];
                        for (new_cols, rows) in &batches {
                            if !new_cols.is_empty() {
                                serial.add_cols(new_cols);
                            }
                            states.push(serial.clone());
                            expected.push(serial_batch(&mut serial, rows));
                        }
                        for (window, block) in [
                            (1, 1),
                            (2, 1),
                            (3, 2),
                            (7, 1),
                            (7, 3),
                            (64, 1),
                            (64, 8),
                            (100_000, 1),
                            (100_000, 16),
                            (100_000, 100_000),
                        ] {
                            for provisional in
                                [Provisional::Copies, Provisional::All, Provisional::None]
                            {
                                for pool in &pools {
                                    let mut parallel =
                                        SparseRowReducer::new(ncols as u32, field.clone(), mode);
                                    parallel.set_retain_dependent_l(retain);
                                    for (b, (new_cols, rows)) in batches.iter().enumerate() {
                                        if !new_cols.is_empty() {
                                            parallel.add_cols(new_cols);
                                        }
                                        assert_same(&states[b], &parallel);
                                        let (values, cols, ptrs) = csr(rows);
                                        let (outcomes, stats) = pool.install(|| {
                                            parallel.add_csr_rows_parallel(
                                                &values,
                                                &cols,
                                                &ptrs,
                                                ParallelOptions {
                                                    window,
                                                    block,
                                                    provisional,
                                                },
                                            )
                                        });
                                        assert_eq!(outcomes, expected[b]);
                                        assert_eq!(stats.rows, rows.len());
                                        resumed += stats.resumed;
                                        waited += (stats.rounds > 1) as usize;
                                        invalidated += stats.invalidated;
                                        assert_eq!(
                                            stats.work(),
                                            stats.subtracted + stats.state_entries
                                        );
                                    }
                                    assert_same(&serial, &parallel);
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 1000);
        // the conflict machinery is exercised (resumptions, several rounds per batch)
        assert!(
            resumed > 1000 && waited > 1000,
            "resumed {resumed} waited {waited}"
        );
        // and so are undone reductions after a provisional row was not accepted
        assert!(invalidated > 100, "invalidated {invalidated}");
    }

    /// Work counters depend on the window only, never on the number of threads.
    #[test]
    fn parallel_stats_do_not_depend_on_threads() {
        let field = Zp64::new(13);
        let mut rng = StdRng::seed_from_u64(7);
        let rows = random_rows(&mut rng, &field, 400, 60);
        let (values, cols, ptrs) = csr(&rows);
        let counts = |s: &super::ParallelStats| {
            (
                s.rows,
                s.rounds,
                s.reductions,
                s.resumed,
                s.invalidated,
                s.copies,
                s.subtracted,
                s.state_entries,
                s.round_work.clone(),
                s.walk_visits,
            )
        };
        let mut reference = None;
        for threads in [1, 3, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut reducer = SparseRowReducer::new(60, field.clone(), LuLMode::Pattern);
            let (_, stats) = pool.install(|| {
                reducer.add_csr_rows_parallel(
                    &values,
                    &cols,
                    &ptrs,
                    ParallelOptions {
                        window: 32,
                        block: 4,
                        provisional: Provisional::All,
                    },
                )
            });
            match &reference {
                None => reference = Some(counts(&stats)),
                Some(reference) => assert_eq!(reference, &counts(&stats)),
            }
        }
    }

    /// Full rank reached inside a batch: later rows record no L row, as in the serial code.
    #[test]
    fn parallel_forward_solve_at_full_rank() {
        let field = Zp64::new(5);
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..50 {
            let rows = random_rows(&mut rng, &field, 40, 4);
            let mut serial = SparseRowReducer::new(4, field.clone(), LuLMode::Full);
            let expected = serial_batch(&mut serial, &rows);
            let (values, cols, ptrs) = csr(&rows);
            for (window, block) in [(1, 1), (5, 2), (50, 50)] {
                let mut parallel = SparseRowReducer::new(4, field.clone(), LuLMode::Full);
                let options = ParallelOptions {
                    window,
                    block,
                    provisional: Provisional::All,
                };
                let (outcomes, _) = parallel.add_csr_rows_parallel(&values, &cols, &ptrs, options);
                assert_eq!(outcomes, expected);
                assert_same(&serial, &parallel);
            }
        }
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn parallel_forward_solve_rejects_unsorted_rows() {
        let field = Zp64::new(5);
        let mut reducer = SparseRowReducer::new(4, field.clone(), LuLMode::Pattern);
        let one = field.one();
        reducer.add_csr_rows_parallel(
            &[one.clone(), one],
            &[2, 1],
            &[0, 2],
            ParallelOptions::default(),
        );
    }
}
