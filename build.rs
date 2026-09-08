use std::process::Command;

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
    // In a linked worktree .git is a file. Watching the nonexistent .git/HEAD
    // makes Cargo rebuild the entire library on every invocation.
    for name in ["HEAD", "refs", "packed-refs"] {
        if let Ok(output) = Command::new("git")
            .args(["rev-parse", "--git-path", name])
            .output()
            && output.status.success()
            && let Ok(path) = String::from_utf8(output.stdout)
            && std::path::Path::new(path.trim()).exists()
        {
            println!("cargo:rerun-if-changed={}", path.trim());
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
