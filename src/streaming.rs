//! Methods for streaming large expressions to disk.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    ops::{Add, AddAssign},
    sync::{Arc, Mutex},
};

use brotli::{CompressorWriter, Decompressor};
use byteorder::{LittleEndian, WriteBytesExt};
use rayon::{ThreadPool, prelude::*};
use smartstring::{LazyCompact, SmartString};

use crate::{
    LicenseManager,
    atom::{Atom, AtomView},
    state::{RecycledAtom, State, Workspace},
};

static TEMP_FILES_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A stream that can be read from by using `name`.
pub trait ReadableNamedStream: Read + Send {
    fn open(name: &str) -> Self;
}

impl ReadableNamedStream for BufReader<File> {
    fn open(name: &str) -> Self {
        BufReader::new(File::open(name).unwrap())
    }
}

impl ReadableNamedStream for Decompressor<BufReader<File>> {
    fn open(name: &str) -> Self {
        brotli::Decompressor::new(BufReader::new(File::open(name).unwrap()), 4096)
    }
}

/// A stream that can be written to by using `name`.
pub trait WriteableNamedStream: Write + Send {
    type Reader: ReadableNamedStream;

    fn create(name: &str) -> Self;
}

impl WriteableNamedStream for BufWriter<File> {
    type Reader = BufReader<File>;

    fn create(name: &str) -> Self {
        BufWriter::new(File::create(name).unwrap())
    }
}

impl WriteableNamedStream for CompressorWriter<BufWriter<File>> {
    type Reader = Decompressor<BufReader<File>>;

    fn create(name: &str) -> Self {
        CompressorWriter::new(BufWriter::new(File::create(name).unwrap()), 4096, 6, 22)
    }
}

/// Setting for term streaming.
#[derive(Clone)]
pub struct TermStreamerConfig {
    /// The number of cores to use.
    pub n_cores: usize,
    /// The path where expressions are written.
    pub path: String,
    /// The maximum size of the memory buffer.
    pub max_mem_bytes: usize,
}

impl Default for TermStreamerConfig {
    fn default() -> Self {
        Self {
            n_cores: 4,
            path: ".".to_owned(),
            max_mem_bytes: 1073741824, // 1 GB
        }
    }
}

struct TermInputStream<'a, R: ReadableNamedStream> {
    mem_buf: &'a [Atom],
    file_buf: Vec<R>,
    pos: usize,
    mem_pos: usize,
}

impl<R: ReadableNamedStream> Iterator for TermInputStream<'_, R> {
    type Item = Atom;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == 0 {
            if self.mem_pos < self.mem_buf.len() {
                self.mem_pos += 1;
                return Some(self.mem_buf[self.mem_pos - 1].clone());
            }

            self.pos += 1;
        }

        while self.pos <= self.file_buf.len() {
            let mut a = Atom::new();
            if let Ok(()) = a.read(&mut self.file_buf[self.pos - 1]) {
                return Some(a);
            }

            self.pos += 1;
        }

        None
    }
}

/// A term streamer that has terms partly in memory and partly on another storage device.
///
/// It allows for operations on the terms in the expression only, via [map](TermStreamer::map).
/// Consider using a writer that compresses files, such as [CompressorWriter].
///
/// # Examples
///
/// ```
/// # use std::io::BufWriter;
/// # use std::fs::File;
/// use symbolica::{atom::AtomCore, parse};
/// use symbolica::streaming::TermStreamer;
/// let input = parse!("(x+1)*y + (y+1)^2*z");
///
/// let mut stream = TermStreamer::<BufWriter<File>>::new(Default::default());
/// stream.push(input);
///
/// // map every term in the expression
/// stream = stream.map(|x| x.expand());
///
/// let r = stream.to_expression();
///
/// let res = parse!("y+z+x*y+2*y*z+y^2*z");
/// assert_eq!(r, res);
/// ```
pub struct TermStreamer<W: WriteableNamedStream> {
    mem_buf: Vec<Atom>,
    mem_size: usize,
    num_terms: usize,
    total_size: usize,
    file_buf: Vec<W>,
    config: TermStreamerConfig,
    filename: String,
    thread_pool: Arc<rayon::ThreadPool>,
    generation: usize,
}

impl<W: WriteableNamedStream> Drop for TermStreamer<W> {
    /// Remove all temporary files created by the term streamer.
    fn drop(&mut self) {
        for x in &mut (0..self.file_buf.len()) {
            std::fs::remove_file(self.get_filename(x)).unwrap();
        }
    }
}

impl<W: WriteableNamedStream> Default for TermStreamer<W>
where
    for<'a> AtomView<'a>: Send,
    Atom: Send,
{
    fn default() -> Self {
        Self::new(TermStreamerConfig::default())
    }
}

impl<W: WriteableNamedStream> Add<&mut TermStreamer<W>> for &mut TermStreamer<W> {
    type Output = TermStreamer<W>;

    fn add(self, rhs: &mut TermStreamer<W>) -> Self::Output {
        let mut n = self.next_generation();
        let r1 = self.reader();
        let r2 = rhs.reader();

        for a in r1.chain(r2) {
            n.push(a);
        }

        n
    }
}

impl<W: WriteableNamedStream> AddAssign<&mut TermStreamer<W>> for TermStreamer<W> {
    fn add_assign(&mut self, rhs: &mut TermStreamer<W>) {
        for a in rhs.reader() {
            self.push(a);
        }
    }
}

impl<W: WriteableNamedStream> Add<Atom> for TermStreamer<W> {
    type Output = TermStreamer<W>;

    fn add(mut self, rhs: Atom) -> Self::Output {
        self.push(rhs);
        self
    }
}

impl<W: WriteableNamedStream> TermStreamer<W> {
    /// Create a new term streamer.
    pub fn new(config: TermStreamerConfig) -> Self {
        let pid = std::process::id();
        let uid = TEMP_FILES_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let filename = format!("{}/{}_{:x}", config.path, pid, uid);

        Self {
            mem_buf: vec![],
            mem_size: 0,
            num_terms: 0,
            total_size: 0,
            file_buf: vec![],
            filename,
            thread_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(if LicenseManager::is_licensed() {
                        config.n_cores
                    } else {
                        1
                    })
                    .build()
                    .unwrap(),
            ),
            config,
            generation: 0,
        }
    }

    fn next_generation(&self) -> Self {
        Self {
            mem_buf: vec![],
            mem_size: 0,
            num_terms: 0,
            total_size: 0,
            file_buf: vec![],
            filename: self.filename.clone(),
            config: self.config.clone(),
            thread_pool: self.thread_pool.clone(),
            generation: self.generation + 1,
        }
    }

    fn get_filename(&self, index: usize) -> String {
        format!("{}_{}_{}.tmp", self.filename, self.generation, index)
    }

    /// Returns true iff the stream fits in memory.
    pub fn fits_in_memory(&self) -> bool {
        self.file_buf.is_empty()
    }

    /// Get the number of terms in the stream.
    pub fn get_num_terms(&self) -> usize {
        self.num_terms
    }

    /// Add terms to the buffer.
    pub fn push(&mut self, a: Atom) {
        if let AtomView::Add(aa) = a.as_view() {
            for arg in aa.iter() {
                self.push_sorted_impl(arg.to_owned());
            }
        } else {
            self.push_sorted_impl(a);
        }
    }

    fn push_sorted_impl(&mut self, a: Atom) {
        let size = a.as_view().get_byte_size();
        self.mem_buf.push(a);
        self.mem_size += size;
        self.num_terms += 1;
        self.total_size += size;

        if self.mem_size >= self.config.max_mem_bytes {
            self.sort();

            if self.mem_size * 2 > self.config.max_mem_bytes {
                self.flush();
            }
        }
    }

    /// Write the memory buffer to a new file.
    fn flush(&mut self) {
        self.file_buf
            .push(W::create(&self.get_filename(self.file_buf.len())));

        let f = self.file_buf.last_mut().unwrap();
        for x in self.mem_buf.drain(..) {
            x.as_view().write(&mut *f).unwrap();
        }
        self.mem_size = 0;
    }

    /// Import terms and their state from a binary stream into the term streamer.
    /// The number of imported terms is returned.
    ///
    /// The state will be merged with the current one. If a symbol has conflicting attributes, the conflict
    /// can be resolved using the renaming function `conflict_fn`.
    ///
    /// A term stream can be exported using [TermStreamer::export].
    pub fn import<R: Read>(
        &mut self,
        source: &mut R,
        conflict_fn: Option<Box<dyn Fn(&str) -> SmartString<LazyCompact>>>,
    ) -> Result<u64, std::io::Error> {
        let state_map = State::import(source, conflict_fn)?;

        let mut n_terms_buf = [0; 8];
        source.read_exact(&mut n_terms_buf)?;
        let n_terms = u64::from_le_bytes(n_terms_buf);

        for _ in 0..n_terms {
            let mut tmp = Atom::new();
            tmp.read(&mut *source)?;
            self.push(tmp.as_view().rename(&state_map));
        }

        self.normalize();

        Ok(n_terms)
    }

    /// Export terms and their state to a binary stream.
    /// The resulting file can be read back using [TermStreamer::import] or
    /// by using [Atom::import]. In the latter case, the whole term stream will be read into memory
    /// as a single expression.
    pub fn export<S: std::io::Write>(&mut self, dest: &mut S) -> Result<(), std::io::Error> {
        self.normalize();

        State::export(dest)?;

        dest.write_u64::<LittleEndian>(self.num_terms as u64)?;

        for t in &self.mem_buf {
            t.as_view().write(&mut *dest)?;
        }

        for x in &mut self.file_buf {
            x.flush().unwrap();
        }

        let mut tmp = Atom::new();
        for i in 0..self.file_buf.len() {
            let mut r = W::Reader::open(&self.get_filename(i));

            // note that we cannot use read_to_end as the file may still be opened by a writer
            while let Ok(()) = tmp.read(&mut r) {
                tmp.as_view().write(&mut *dest)?;
            }
        }

        Ok(())
    }

    /// Sort all the terms in the memory buffer.
    fn sort(&mut self) {
        self.mem_buf
            .par_sort_by(|a, b| a.as_view().cmp_terms(&b.as_view()));

        let mut out = Vec::with_capacity(self.mem_buf.len());
        let old_size = self.mem_buf.len();
        let mut new_size = 0;

        if !self.mem_buf.is_empty() {
            let mut last_buf = self.mem_buf.remove(0);

            let mut handle: RecycledAtom = Atom::new().into();

            for cur_buf in self.mem_buf.drain(..) {
                if !last_buf.merge_terms(cur_buf.as_view(), &mut handle) {
                    // we are done merging
                    {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if !n.is_zero() {
                                new_size += v.get_byte_size();
                                out.push(last_buf);
                            }
                        } else {
                            new_size += v.get_byte_size();
                            out.push(last_buf);
                        }
                    }
                    last_buf = cur_buf;
                }
            }

            if let AtomView::Num(n) = last_buf.as_view() {
                if !n.is_zero() {
                    new_size += last_buf.as_view().get_byte_size();
                    out.push(last_buf);
                }
            } else {
                new_size += last_buf.as_view().get_byte_size();
                out.push(last_buf);
            }
        }

        self.mem_buf = out;
        self.num_terms += self.mem_buf.len();
        self.num_terms -= old_size;
        self.total_size += new_size;
        self.total_size -= self.mem_size;
        self.mem_size = new_size;
    }

    /// Fuse sorted streams into one sorted stream.
    pub fn normalize(&mut self) {
        self.sort();

        if self.file_buf.is_empty() {
            return;
        }

        self.mem_buf.reverse();

        let mut head = vec![self.mem_buf.pop()];

        let n_files = self.file_buf.len();

        for b in &mut self.file_buf {
            b.flush().unwrap();
        }

        let mut files: Vec<_> = (0..n_files)
            .map(|i| W::Reader::open(&self.get_filename(i)))
            .collect();

        for ff in &mut files {
            let mut a = Atom::new();
            if let Ok(()) = a.read(ff) {
                head.push(Some(a));
            } else {
                head.push(None);
            }
        }

        let mut new_stream = self.next_generation();

        let mut last = Atom::new();

        let mut smallest = (0..head.len()).collect::<Vec<_>>();
        let mut helper = Atom::new();
        loop {
            // find minimal element
            smallest.sort_unstable_by(|a, b| {
                if let Some(aa) = &head[*a] {
                    if let Some(bb) = &head[*b] {
                        aa.as_view().cmp_terms(&bb.as_view())
                    } else {
                        std::cmp::Ordering::Less
                    }
                } else {
                    std::cmp::Ordering::Greater
                }
            });

            let Some(c) = head[smallest[0]].take() else {
                // all None
                break;
            };

            // load the next element
            if smallest[0] == 0 {
                head[0] = self.mem_buf.pop();
            } else {
                let mut a = Atom::new();
                if let Ok(()) = a.read(&mut files[smallest[0] - 1]) {
                    head[smallest[0]] = Some(a);
                }
            }

            if !last.merge_terms(c.as_view(), &mut helper) {
                if let AtomView::Num(n) = last.as_view() {
                    if !n.is_zero() {
                        new_stream.push_sorted_impl(last.clone());
                    }
                } else {
                    new_stream.push_sorted_impl(last.clone());
                }

                last.set_from_view(&c.as_view());
            }
        }

        if let AtomView::Num(n) = last.as_view() {
            if !n.is_zero() {
                new_stream.push_sorted_impl(last.clone());
            }
        } else {
            new_stream.push_sorted_impl(last.clone());
        }

        if !new_stream.file_buf.is_empty() {
            // the memory buffer has the larger elements than the
            // file streams, so write the elements to a file
            new_stream.flush();
        }

        *self = new_stream;
    }

    /// Convert the term stream into an expression. This may exceed the available memory.
    pub fn to_expression(&mut self) -> Atom {
        self.normalize();

        let mut a = Atom::new();
        let add = a.to_add();

        for x in self.reader() {
            add.extend(x.as_view());
        }

        if add.get_nargs() == 1 {
            let mut b = Atom::new();
            b.set_from_view(&add.to_add_view().iter().next().unwrap());
            return b;
        }

        add.set_normalized(true);
        a
    }

    fn reader(&mut self) -> TermInputStream<'_, W::Reader> {
        let num_files = self.file_buf.len();

        for x in &mut self.file_buf {
            x.flush().unwrap();
        }

        TermInputStream {
            mem_buf: &self.mem_buf,
            file_buf: (0..num_files)
                .map(|i| W::Reader::open(&self.get_filename(i)))
                .collect(),
            pos: 0,
            mem_pos: 0,
        }
    }

    /// Map every term in the stream using the function `f`. The resulting terms
    /// are a stream as well, which is returned by this function.
    pub fn map(&mut self, f: impl Fn(Atom) -> Atom + Send + Sync) -> Self {
        if self.thread_pool.current_num_threads() == 1 {
            return self.map_single_thread(f);
        }

        let t = self.thread_pool.clone();

        let new_out = self.next_generation();

        let reader = self.reader();

        let out_wrap = Mutex::new(new_out);

        t.install(
            #[inline(always)]
            || {
                reader.par_bridge().for_each(|x| {
                    let r = f(x);
                    out_wrap.lock().unwrap().push(r);
                });
            },
        );

        out_wrap.into_inner().unwrap()
    }

    /// Map every term in the stream using the function `f` using a single thread. The resulting terms
    /// are a stream as well, which is returned by this function.
    pub fn map_single_thread(&mut self, f: impl Fn(Atom) -> Atom) -> Self {
        let mut new_out = self.next_generation();

        let reader = self.reader();

        for x in reader {
            new_out.push(f(x));
        }

        new_out
    }

    /// Check if two term streams are equal. This will normalize both streams.
    pub fn eq(&mut self, other: &mut Self) -> bool {
        self.normalize();
        other.normalize();

        self.reader().eq(other.reader())
    }

    /// Get the total size of the term stream in bytes.
    pub fn get_byte_size(&self) -> usize {
        self.total_size
    }

    /// Clear all terms from the term streamer.
    pub fn clear(&mut self) {
        self.mem_buf.clear();
        self.mem_size = 0;
        self.num_terms = 0;
        self.total_size = 0;

        for x in &mut (0..self.file_buf.len()) {
            std::fs::remove_file(self.get_filename(x)).unwrap();
        }

        self.file_buf.clear();
    }
}

impl AtomView<'_> {
    /// Map the function `f` over all terms.
    pub(crate) fn map_terms_single_core(&self, f: impl Fn(AtomView) -> Atom) -> Atom {
        if let AtomView::Add(aa) = self {
            Workspace::get_local().with(|ws| {
                let mut r = ws.new_atom();
                let rr = r.to_add();
                for arg in aa {
                    rr.extend(f(arg).as_view());
                }
                let mut out = Atom::new();
                r.as_view().normalize(ws, &mut out);
                out
            })
        } else {
            f(*self)
        }
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    pub(crate) fn map_terms(
        &self,
        f: impl Fn(AtomView) -> Atom + Send + Sync,
        n_cores: usize,
    ) -> Atom {
        if n_cores < 2 || !LicenseManager::is_licensed() {
            return self.map_terms_single_core(f);
        }

        if let AtomView::Add(_) = self {
            let t = rayon::ThreadPoolBuilder::new()
                .num_threads(n_cores)
                .build()
                .unwrap();

            self.map_terms_with_pool(f, &t)
        } else {
            f(*self)
        }
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    pub(crate) fn map_terms_with_pool(
        &self,
        f: impl Fn(AtomView) -> Atom + Send + Sync,
        p: &ThreadPool,
    ) -> Atom {
        if !LicenseManager::is_licensed() {
            return self.map_terms_single_core(f);
        }

        if let AtomView::Add(aa) = self {
            let out_wrap = Mutex::new(vec![]);
            let args = aa.iter().collect::<Vec<_>>();

            p.install(
                #[inline(always)]
                || {
                    args.par_iter().for_each(|x| {
                        let r = f(*x);
                        out_wrap.lock().unwrap().push(r);
                    });
                },
            );

            let res = out_wrap.into_inner().unwrap();

            Workspace::get_local().with(|ws| {
                let mut r = ws.new_atom();
                let rr = r.to_add();
                for arg in res {
                    rr.extend(arg.as_view());
                }
                let mut out = Atom::new();
                r.as_view().normalize(ws, &mut out);
                out
            })
        } else {
            f(*self)
        }
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::BufWriter};

    use brotli::CompressorWriter;

    use crate::{
        atom::{Atom, AtomCore, AtomType},
        function,
        id::WildcardRestriction,
        parse,
        streaming::{TermStreamer, TermStreamerConfig},
        symbol,
    };

    #[test]
    fn import_export() {
        let mut streamer =
            TermStreamer::<CompressorWriter<BufWriter<File>>>::new(TermStreamerConfig {
                n_cores: 1,
                path: ".".to_owned(),
                max_mem_bytes: 180,
            });

        for i in (0..100).rev() {
            streamer.push(function!(symbol!("f1"), i));
        }

        let mut r = vec![];
        streamer.export(&mut r).unwrap();

        let b = Atom::import(&mut r.as_slice(), None).unwrap();

        streamer.clear();

        streamer.import(&mut r.as_slice(), None).unwrap();
        let c = streamer.to_expression();
        assert_eq!(b, c);
    }

    #[test]
    fn file_stream() {
        let mut streamer =
            TermStreamer::<CompressorWriter<BufWriter<File>>>::new(TermStreamerConfig {
                n_cores: 4,
                path: ".".to_owned(),
                max_mem_bytes: 20,
            });

        let input = parse!("v1 + f1(v1) + 2*f1(v2) + 7*f1(v3) + v2 + v3 + v4");
        streamer.push(input);

        let _ = streamer.reader();

        streamer = streamer + parse!("f1(v1)");

        streamer = streamer.map(|f| f);

        let pattern = parse!("f1(x_)").to_pattern();
        let rhs = parse!("f1(v1) + v1").to_pattern();

        streamer = streamer.map(|x| x.replace(&pattern).with(&rhs).expand());

        streamer.normalize();

        let r = streamer.to_expression();

        let res = parse!("12*v1+v2+v3+v4+11*f1(v1)");
        assert_eq!(r, res);
    }

    #[test]
    fn file_stream_with_rationals() {
        let mut streamer =
            TermStreamer::<CompressorWriter<BufWriter<File>>>::new(TermStreamerConfig {
                n_cores: 4,
                path: ".".to_owned(),
                max_mem_bytes: 20,
            });

        let input = parse!("v1*coeff(v2/v3+1)+v2*coeff(v3+1)+v3*coeff(1/v2)");
        streamer.push(input);

        let pattern = parse!("v1_").to_pattern();
        let rhs = parse!("v1").to_pattern();

        streamer = streamer.map(|x| {
            x.replace(&pattern)
                .when(symbol!("v1_").restrict(WildcardRestriction::IsAtomType(AtomType::Var)))
                .with(&rhs)
                .expand()
        });

        streamer.normalize();

        let r = streamer.to_expression();

        let res = parse!("coeff((v3+2*v2*v3+v2*v3^2+v2^2)/(v2*v3))*v1");
        assert_eq!(r - res, Atom::Zero);
    }

    #[test]
    fn memory_stream() {
        let input = parse!("v1 + f1(v1) + 2*f1(v2) + 7*f1(v3)");
        let pattern = parse!("f1(x_)").to_pattern();
        let rhs = parse!("f1(v1) + v1").to_pattern();

        let mut stream = TermStreamer::<BufWriter<File>>::new(TermStreamerConfig::default());
        stream.push(input);

        // map every term in the expression
        stream = stream.map(|x| x.replace(&pattern).with(&rhs).expand());

        let r = stream.to_expression();

        let res = parse!("11*v1+10*f1(v1)");
        assert_eq!(r, res);
    }

    #[test]
    fn term_map() {
        let input = parse!("v1 + v2 + v3 + v4");

        let r = input
            .as_view()
            .map_terms(|x| Atom::num(1) + &x.to_owned(), 4);

        let r2 = input
            .as_view()
            .map_terms(|x| Atom::num(1) + &x.to_owned(), 1);
        assert_eq!(r, r2);

        let res = parse!("v1 + v2 + v3 + v4 + 4");
        assert_eq!(r, res);
    }
}
