use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn git(directory: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(directory)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// Whether `directory` is the top level of its own git checkout. A vendored
/// copy inside another repository is not: its version and rebuild triggers
/// must not follow the enclosing repository's history.
fn is_own_checkout(directory: &Path) -> bool {
    let Some(top) = git(directory, &["rev-parse", "--show-toplevel"]) else {
        return false;
    };
    match (Path::new(&top).canonicalize(), directory.canonicalize()) {
        (Ok(top), Ok(directory)) => top == directory,
        _ => false,
    }
}

/// Files whose change can change `git describe`: `HEAD`, the checked-out
/// branch ref (or `packed-refs` when that ref is packed) and the tags. Not the
/// whole `refs` directory: another branch or worktree moving must not
/// rebuild this crate.
pub(crate) fn git_metadata_paths(directory: &Path) -> Vec<PathBuf> {
    let git_path = |entry: &str| {
        git(
            directory,
            &["rev-parse", "--path-format=absolute", "--git-path", entry],
        )
        .map(PathBuf::from)
    };
    let mut paths = vec![];
    if let Some(head) = git_path("HEAD") {
        paths.push(head);
    }
    if let Some(branch) = git(directory, &["symbolic-ref", "-q", "HEAD"]) {
        match git_path(&branch) {
            Some(file) if file.exists() => paths.push(file),
            _ => paths.extend(git_path("packed-refs")),
        }
    }
    paths.extend(git_path("refs/tags"));
    paths.into_iter().filter(|path| path.exists()).collect()
}

fn get_git_version(directory: &Path) -> Option<String> {
    git(directory, &["describe", "--tags", "--always"])
}

fn main() {
    let cargo_version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let directory =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_else(|| ".".into()));
    let own = is_own_checkout(&directory);

    let version = own
        .then(|| get_git_version(&directory))
        .flatten()
        .unwrap_or(cargo_version);

    println!("cargo:rustc-env=SYMBOLICA_VERSION=symbolica-{}", version);
    if own {
        for path in git_metadata_paths(&directory) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
