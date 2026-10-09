//! PFH0 pack files (Empire / Napoleon: Total War).
//!
//! Layout: b"PFH0", u32 type, u32 replaced_pack_count, u32 replaced_index_size, u32 file_count,
//! u32 file_index_size, replaced pack names (NUL-terminated), file index (u32 size + NUL-terminated
//! latin1 name), then the file data in index order. Member names use backslashes and are looked up
//! case-insensitively (`/` is accepted as a separator).

use anyhow::{Context, Result, anyhow, bail};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const TYPE_BOOT: u32 = 0;
pub const TYPE_RELEASE: u32 = 1;
pub const TYPE_PATCH: u32 = 2;
pub const TYPE_MOD: u32 = 3;
pub const TYPE_MOVIE: u32 = 4;

const HEADER: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Backslash-separated, exactly as stored.
    pub name: String,
    pub offset: u64,
    pub size: u32,
}

#[derive(Clone, Debug)]
pub struct Pack {
    pub path: PathBuf,
    pub type_: u32,
    pub replaces: Vec<String>,
    pub entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

/// Lookup key of a member name: backslash separators, lower case.
pub fn key(name: &str) -> String {
    name.replace('/', "\\").to_lowercase()
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

impl Pack {
    pub fn open(path: &Path) -> Result<Pack> {
        Self::open_inner(path).with_context(|| format!("pack {}", path.display()))
    }

    fn open_inner(path: &Path) -> Result<Pack> {
        let mut f = File::open(path)?;
        let len = f.metadata()?.len();
        if len < HEADER as u64 {
            bail!("file has {len} bytes, shorter than the {HEADER} byte PFH0 header");
        }
        let mut head = [0u8; HEADER];
        f.read_exact(&mut head)?;
        if &head[0..4] != b"PFH0" {
            bail!("unsupported pack magic {:?}", &head[0..4]);
        }
        let type_ = le32(&head, 4);
        let nrep = le32(&head, 8) as u64;
        let repsz = le32(&head, 12) as u64;
        let nfiles = le32(&head, 16) as u64;
        let idxsz = le32(&head, 20) as u64;
        let data_start = HEADER as u64 + repsz + idxsz;
        if data_start > len {
            bail!("header declares {repsz} + {idxsz} index bytes but the file ends at {len}");
        }
        // Each index record needs at least a size word and a terminator.
        if nfiles > idxsz / 5 {
            bail!("header declares {nfiles} files but only {idxsz} index bytes");
        }
        if nrep > repsz {
            bail!("header declares {nrep} replaced packs in {repsz} bytes");
        }
        let mut rep = vec![0u8; repsz as usize];
        f.read_exact(&mut rep)?;
        let mut idx = vec![0u8; idxsz as usize];
        f.read_exact(&mut idx)?;

        let mut replaces = Vec::with_capacity(nrep as usize);
        let mut p = 0usize;
        for i in 0..nrep {
            let end = rep[p..]
                .iter()
                .position(|&c| c == 0)
                .ok_or_else(|| anyhow!("replaced pack name {i} is not terminated"))?
                + p;
            replaces.push(rep[p..end].iter().map(|&c| c as char).collect());
            p = end + 1;
        }

        let mut entries = Vec::with_capacity(nfiles as usize);
        let mut by_name = HashMap::with_capacity(nfiles as usize);
        let mut offset = data_start;
        let mut i = 0usize;
        for n in 0..nfiles {
            if i + 5 > idx.len() {
                bail!("file index truncated before entry {n}");
            }
            let size = le32(&idx, i);
            let end = idx[i + 4..]
                .iter()
                .position(|&c| c == 0)
                .ok_or_else(|| anyhow!("name of entry {n} is not terminated"))?
                + i
                + 4;
            let name: String = idx[i + 4..end].iter().map(|&c| c as char).collect();
            if name.is_empty() {
                bail!("entry {n} has an empty name");
            }
            by_name.insert(key(&name), entries.len());
            entries.push(Entry { name, offset, size });
            offset += size as u64;
            i = end + 1;
        }
        if i > idx.len() {
            bail!("file index overruns its declared size");
        }
        if offset > len {
            bail!(
                "data for {nfiles} files needs {} bytes but the file has {len}",
                offset
            );
        }
        Ok(Pack {
            path: path.to_owned(),
            type_,
            replaces,
            entries,
            by_name,
        })
    }

    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.by_name.get(&key(name)).map(|&i| &self.entries[i])
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let e = self
            .find(name)
            .ok_or_else(|| anyhow!("{} has no member {name}", self.path.display()))?;
        self.read_entry(e)
    }

    pub fn read_entry(&self, e: &Entry) -> Result<Vec<u8>> {
        let mut out = vec![0u8; e.size as usize];
        self.reader(e)?
            .read_exact(&mut out)
            .with_context(|| format!("read {} from {}", e.name, self.path.display()))?;
        Ok(out)
    }

    /// Streaming reader positioned on one member (reads at most `e.size` bytes).
    pub fn reader(&self, e: &Entry) -> Result<std::io::Take<File>> {
        let mut f =
            File::open(&self.path).with_context(|| format!("open {}", self.path.display()))?;
        f.seek(SeekFrom::Start(e.offset))?;
        Ok(f.take(e.size as u64))
    }
}

/// Validates and normalises a member name for writing.
fn member_name(raw: &str) -> Result<String> {
    let name = raw.replace('/', "\\");
    if name.is_empty() {
        bail!("empty pack member name");
    }
    for part in name.split('\\') {
        if part.is_empty() || part == "." || part == ".." {
            bail!("unsafe pack member name {raw:?}");
        }
    }
    for c in name.chars() {
        if (c as u32) > 0xFF || c == '\0' || c == ':' {
            bail!("pack member name {raw:?} contains {c:?}, which is not storable");
        }
    }
    Ok(name)
}

/// Writes a PFH0 pack. `files` maps backslash or slash separated member names to contents. The pack is
/// written to a temporary sibling and moved over `path` only when complete.
pub fn write_pack(path: &Path, files: &BTreeMap<String, Vec<u8>>, type_: u32) -> Result<()> {
    let mut members: Vec<(String, &Vec<u8>)> = Vec::with_capacity(files.len());
    for (raw, data) in files {
        members.push((member_name(raw)?, data));
    }
    members.sort_by_key(|(n, _)| n.to_lowercase());
    for pair in members.windows(2) {
        if pair[0].0.to_lowercase() == pair[1].0.to_lowercase() {
            bail!("pack members {:?} and {:?} collide", pair[0].0, pair[1].0);
        }
    }
    let mut index = Vec::new();
    for (name, data) in &members {
        let size = u32::try_from(data.len()).map_err(|_| anyhow!("member {name} exceeds 4 GiB"))?;
        index.extend_from_slice(&size.to_le_bytes());
        index.extend(name.chars().map(|c| c as u8));
        index.push(0);
    }
    let index_size = u32::try_from(index.len()).map_err(|_| anyhow!("pack index exceeds 4 GiB"))?;
    let count = u32::try_from(members.len()).map_err(|_| anyhow!("too many pack members"))?;

    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let mut tmp_name = path
        .file_name()
        .ok_or_else(|| anyhow!("pack path {} has no file name", path.display()))?
        .to_owned();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    let write = || -> Result<()> {
        let mut out = BufWriter::new(
            File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?,
        );
        out.write_all(b"PFH0")?;
        for word in [type_, 0, 0, count, index_size] {
            out.write_all(&word.to_le_bytes())?;
        }
        out.write_all(&index)?;
        for (_, data) in &members {
            out.write_all(data)?;
        }
        out.flush()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(anyhow!(e).context(format!("move {} over {}", tmp.display(), path.display())));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Scratch directory removed on drop (read-only snapshot files are made writable first).
    pub struct TempDir(pub PathBuf);

    impl TempDir {
        pub fn new(tag: &str) -> TempDir {
            static N: AtomicU32 = AtomicU32::new(0);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "kgc_{tag}_{}_{}_{stamp}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            fn unlock(dir: &Path) {
                if let Ok(rd) = std::fs::read_dir(dir) {
                    for e in rd.flatten() {
                        let p = e.path();
                        if p.is_dir() {
                            unlock(&p);
                        } else if let Ok(meta) = std::fs::metadata(&p) {
                            let mut perm = meta.permissions();
                            perm.set_readonly(false);
                            let _ = std::fs::set_permissions(&p, perm);
                        }
                    }
                }
            }
            unlock(&self.0);
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::TempDir;
    use super::*;

    fn files(list: &[(&str, &[u8])]) -> BTreeMap<String, Vec<u8>> {
        list.iter()
            .map(|(n, d)| (n.to_string(), d.to_vec()))
            .collect()
    }

    #[test]
    fn write_then_open_round_trips_names_sizes_and_bytes() {
        let dir = TempDir::new("pack_rt");
        let path = dir.path().join("nested").join("x.pack");
        write_pack(
            &path,
            &files(&[
                ("db/Zeta/nap_world", b"zz"),
                ("text/localisation.loc", b""),
                ("db\\alpha\\row", &[0, 1, 255]),
            ]),
            TYPE_MOD,
        )
        .unwrap();
        let pack = Pack::open(&path).unwrap();
        assert_eq!(pack.type_, TYPE_MOD);
        let names: Vec<&str> = pack.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "db\\alpha\\row",
                "db\\Zeta\\nap_world",
                "text\\localisation.loc"
            ]
        );
        assert_eq!(pack.read("DB/ZETA/NAP_WORLD").unwrap(), b"zz");
        assert_eq!(pack.read("db\\alpha\\row").unwrap(), [0, 1, 255]);
        assert_eq!(pack.read("text/localisation.loc").unwrap(), b"");
        assert!(pack.read("db\\missing").is_err());
        let offsets: Vec<u64> = pack.entries.iter().map(|e| e.offset).collect();
        assert_eq!(offsets[1], offsets[0] + 3);
        assert!(!path.with_file_name("x.pack.tmp").exists());
    }

    #[test]
    fn rewriting_replaces_the_previous_pack() {
        let dir = TempDir::new("pack_replace");
        let path = dir.path().join("x.pack");
        write_pack(&path, &files(&[("a", b"old")]), TYPE_MOD).unwrap();
        write_pack(&path, &files(&[("a", b"new!")]), TYPE_MOD).unwrap();
        assert_eq!(Pack::open(&path).unwrap().read("a").unwrap(), b"new!");
    }

    #[test]
    fn writer_rejects_unsafe_or_colliding_names() {
        let dir = TempDir::new("pack_names");
        let path = dir.path().join("x.pack");
        for bad in [
            "../evil",
            "a/../b",
            "/abs",
            "a//b",
            "",
            "c:\\x",
            "snow\u{2603}",
            "nul\0name",
        ] {
            assert!(
                write_pack(&path, &files(&[(bad, b"x")]), TYPE_MOD).is_err(),
                "{bad:?} must be rejected"
            );
        }
        assert!(write_pack(&path, &files(&[("A/b", b"1"), ("a\\B", b"2")]), TYPE_MOD).is_err());
        assert!(!path.exists());
    }

    fn sample_bytes() -> Vec<u8> {
        let dir = TempDir::new("pack_sample");
        let path = dir.path().join("s.pack");
        write_pack(
            &path,
            &files(&[("one", b"11111"), ("two\\three", b"222")]),
            TYPE_PATCH,
        )
        .unwrap();
        fs::read(path).unwrap()
    }

    fn open_bytes(bytes: &[u8]) -> Result<Pack> {
        let dir = TempDir::new("pack_open");
        let path = dir.path().join("t.pack");
        fs::write(&path, bytes).unwrap();
        Pack::open(&path)
    }

    #[test]
    fn every_truncated_pack_is_refused() {
        let bytes = sample_bytes();
        assert!(open_bytes(&bytes).is_ok());
        // Data is the tail of the file, so every shorter prefix must fail the bounds checks.
        for cut in 0..bytes.len() {
            assert!(
                open_bytes(&bytes[..cut]).is_err(),
                "prefix of {cut} bytes must be rejected"
            );
        }
    }

    #[test]
    fn rejects_bad_magic_and_inconsistent_header_counts() {
        let bytes = sample_bytes();
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(open_bytes(&magic).is_err());
        for (at, value) in [
            (16usize, 1_000_000u32),
            (20, u32::MAX),
            (12, u32::MAX),
            (8, 5),
        ] {
            let mut bad = bytes.clone();
            bad[at..at + 4].copy_from_slice(&value.to_le_bytes());
            assert!(open_bytes(&bad).is_err(), "header word at {at} = {value}");
        }
    }

    #[test]
    fn rejects_unterminated_names_and_oversized_members() {
        let bytes = sample_bytes();
        let idx_size = le32(&bytes, 20) as usize;
        let mut unterminated = bytes.clone();
        unterminated[HEADER + idx_size - 1] = b'x'; // last index terminator becomes part of the name
        assert!(open_bytes(&unterminated).is_err());
        let mut huge = bytes.clone();
        huge[HEADER..HEADER + 4].copy_from_slice(&u32::MAX.to_le_bytes()); // first member claims 4 GiB
        assert!(open_bytes(&huge).is_err());
    }

    #[test]
    fn reads_are_bounded_to_the_member() {
        let dir = TempDir::new("pack_reader");
        let path = dir.path().join("x.pack");
        write_pack(&path, &files(&[("a", b"AAAA"), ("b", b"BB")]), TYPE_MOD).unwrap();
        let pack = Pack::open(&path).unwrap();
        let mut got = Vec::new();
        pack.reader(pack.find("a").unwrap())
            .unwrap()
            .read_to_end(&mut got)
            .unwrap();
        assert_eq!(got, b"AAAA");
    }
}
