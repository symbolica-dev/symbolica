use std::{
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) fn git_metadata_paths(directory: &Path) -> Vec<PathBuf> {
    ["HEAD", "refs", "packed-refs"]
        .into_iter()
        .filter_map(|entry| {
            let output = Command::new("git")
                .current_dir(directory)
                .args(["rev-parse", "--path-format=absolute", "--git-path", entry])
                .output()
                .ok()?;
            if !output.status.success() {
                return None;
            }
            let path = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
            path.exists().then_some(path)
        })
        .collect()
}

fn get_git_version() -> Option<String> {
    let output = Command::new("git")
        .args(["describe", "--tags", "--always"])
        .output()
        .ok()?;

    if output.status.success() {
        let git_desc = String::from_utf8(output.stdout).ok()?;
        Some(git_desc.trim().to_string())
    } else {
        None
    }
}

fn main() {
    let cargo_version = std::env::var("CARGO_PKG_VERSION").unwrap();

    let version = match get_git_version() {
        Some(v) => v,
        None => cargo_version,
    };

    println!("cargo:rustc-env=SYMBOLICA_VERSION=symbolica-{}", version);
    for path in git_metadata_paths(Path::new(".")) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}
