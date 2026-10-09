use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufWriter, Read, Write},
    path::Path,
};
use walkdir::WalkDir;
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

pub(crate) const ZIP_NAME: &str = "GrandCampaign-NTW.zip";

/// Game payloads stay under data/. Compatibility metadata has a separate ZIP root
/// so dragging only data/ into the game never installs project or provenance files.
pub(crate) fn package(
    data: &Path,
    destination: &Path,
    metadata: &BTreeMap<String, Vec<u8>>,
) -> Result<usize> {
    for name in metadata.keys() {
        ensure!(
            !name.is_empty() && !name.contains(['/', '\\', ':']) && name != "." && name != "..",
            "Invalid export metadata filename {name}"
        );
    }
    let mut files = Vec::new();
    for entry in WalkDir::new(data) {
        let entry = entry?;
        if entry.file_type().is_symlink() {
            bail!("Refusing a linked export path: {}", entry.path().display());
        }
        if entry.file_type().is_file() {
            let name = format!(
                "data/{}",
                entry
                    .path()
                    .strip_prefix(data)?
                    .to_string_lossy()
                    .replace('\\', "/")
            );
            files.push((name, entry.into_path()));
        }
    }
    files.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    ensure!(!files.is_empty(), "Cannot export an empty campaign");
    let mut zip = ZipWriter::new(BufWriter::new(File::create(destination)?));
    for (name, path) in &files {
        // Supertexture tiles already contain zlib streams; recompressing them wastes CPU.
        let method = if path.extension().is_some_and(|ext| ext == "stpd") {
            CompressionMethod::Stored
        } else {
            CompressionMethod::Deflated
        };
        let options = SimpleFileOptions::default()
            .compression_method(method)
            .unix_permissions(0o644);
        zip.start_file(name, options)?;
        io::copy(&mut File::open(path)?, &mut zip).with_context(|| format!("Archive {name}"))?;
    }
    for (name, bytes) in metadata {
        zip.start_file(
            format!("kgc-export/{name}"),
            SimpleFileOptions::default()
                .compression_method(CompressionMethod::Deflated)
                .unix_permissions(0o644),
        )?;
        zip.write_all(bytes)?;
    }
    let mut writer = zip.finish()?;
    writer.flush()?;
    drop(writer);

    // Read every entry back: checks folder names, file sizes, decompression, and ZIP CRCs.
    let mut archive = ZipArchive::new(File::open(destination)?)?;
    ensure!(
        archive.len() == files.len() + metadata.len(),
        "ZIP entry count differs from the planned game and metadata files"
    );
    for (name, path) in &files {
        let mut entry = archive.by_name(name)?;
        ensure!(entry.enclosed_name().is_some(), "Unsafe ZIP member {name}");
        let size = io::copy(&mut entry, &mut io::sink())
            .with_context(|| format!("Verify ZIP member {name}"))?;
        ensure!(
            size == fs::metadata(path)?.len(),
            "ZIP size mismatch for {name}"
        );
    }
    for (name, expected) in metadata {
        let mut entry = archive.by_name(&format!("kgc-export/{name}"))?;
        let mut actual = Vec::with_capacity(expected.len());
        entry.read_to_end(&mut actual)?;
        ensure!(&actual == expected, "ZIP metadata mismatch for {name}");
    }
    Ok(files.len())
}
