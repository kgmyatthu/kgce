use anyhow::{Context, Result, bail};
use std::{
    fs,
    path::{Component, Path},
};

/// Guard all user-selected write destinations, including paths redirected through directory links.
/// Conversion never needs a writable game installation: the final deliverable is a ZIP.
pub(crate) fn ensure_writable_path(path: &Path) -> Result<()> {
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        bail!(
            "Write destinations must not contain '..': {}",
            path.display()
        );
    }
    let absolute = std::path::absolute(path)?;
    let existing = absolute
        .ancestors()
        .find(|p| fs::symlink_metadata(p).is_ok())
        .context("Write destination has no existing ancestor")?;
    let resolved = fs::canonicalize(existing)
        .with_context(|| format!("Resolve write destination {}", path.display()))?;
    for ancestor in resolved.ancestors() {
        let executable =
            ancestor.join("Napoleon.exe").is_file() || ancestor.join("Empire.exe").is_file();
        let data = ancestor.join("boot.pack").is_file()
            && (ancestor.join("data.pack").is_file() || ancestor.join("main.pack").is_file());
        let installation = ancestor.join("data/boot.pack").is_file()
            && (ancestor.join("data/data.pack").is_file()
                || ancestor.join("data/main.pack").is_file());
        if executable || data || installation {
            bail!(
                "Game installations are read-only. Refusing to write {}. Choose an editor workspace outside the game; export a ZIP and install it manually.",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::Project, pack::test_util::TempDir};

    #[test]
    fn project_save_cannot_replace_a_game_file_or_create_files_under_an_install() {
        let dir = TempDir::new("game_write_guard");
        let game = dir.path().join("game");
        fs::create_dir_all(game.join("data")).unwrap();
        fs::write(game.join("Napoleon.exe"), b"game marker").unwrap();
        let target = game.join("data/project.json");
        fs::write(&target, b"original game bytes").unwrap();
        assert!(Project::default().save(&target).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"original game bytes");
        let new = game.join("new/export/project.json");
        assert!(Project::default().save(&new).is_err());
        assert!(!game.join("new").exists());
        let outside = dir.path().join("editor/project.json");
        Project::default().save(&outside).unwrap();
        assert_eq!(Project::load(&outside).unwrap().name, "Grand Campaign");
    }
}
