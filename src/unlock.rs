use std::{
    collections::{HashMap, HashSet},
    fs::{DirBuilder, File, OpenOptions, TryLockError},
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicU32, Ordering::Relaxed},
    },
    thread::ThreadId,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
#[cfg(not(target_arch = "wasm32"))]
use directories::ProjectDirs;
use ed25519_compact::{PublicKey, Signature};
use tinyjson::JsonValue;

const LIBRARY_UNLOCK_PUBLIC_KEY: &str = "jmTdC49SNxaJA7kzYC-1tICg26hZMH7M9QesZ8W2lCU";

#[cfg(any(test, debug_assertions))]
const TEST_UNLOCK_PUBLIC_KEY: &str = "FTlMMgb7IKbxHiS-E7bp_W2ZqhPeqXpZ_Or40Jpjvns";
#[cfg(any(test, debug_assertions))]
const TEST_UNLOCK_LICENSE: &str = "SYMBOLICA_UNLOCK_TEST";

#[cfg(any(test, debug_assertions))]
pub(crate) const TEST_UNLOCK_TOKEN: &str = concat!(
    "eyJsaWNlbnNlIjoiU1lNQk9MSUNBX1VOTE9DS19URVNUIiwicGFja2FnZSI6InB5c2VjZGVjIiwidmVyc2lvbiI6MX0",
    ".",
    "VKUj1gBSnqHBETCWB7UV2ySynCfmPlTZ8RvEx4HR0Nr4w1n-SDHkneMOKFAuKfDmvY14YmW9WwDx7JDCHKKOBQ"
);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UnlockClaims {
    pub package: String,
    license: String,
    token_id: String,
}

struct ActiveUnlockGuards {
    pid: u32,
    owner_threads: HashMap<ThreadId, usize>,
}

static ACTIVE_UNLOCK_GUARDS: LazyLock<Mutex<HashMap<String, ActiveUnlockGuards>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CHECKED_UNLOCK_LICENSES: LazyLock<Mutex<HashSet<(u32, String)>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn verify_signature(
    public_key: &str,
    payload: &[u8],
    signature: &Signature,
) -> Result<bool, String> {
    let public_key_bytes: [u8; 32] = URL_SAFE_NO_PAD
        .decode(public_key)
        .map_err(|_| "Invalid compiled Symbolica library unlock public key".to_owned())?
        .try_into()
        .map_err(|_| "Invalid compiled Symbolica library unlock public key length".to_owned())?;

    Ok(PublicKey::new(public_key_bytes)
        .verify(payload, signature)
        .is_ok())
}

fn get_usize_claim(claims: &HashMap<String, JsonValue>, name: &str) -> Result<usize, String> {
    let value = claims
        .get(name)
        .and_then(JsonValue::get::<f64>)
        .copied()
        .ok_or_else(|| format!("Library unlock token is missing integer claim '{name}'"))?;

    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(format!(
            "Library unlock token has invalid integer claim '{name}'"
        ));
    }

    Ok(value as usize)
}

pub(crate) fn verify_token(token: &str) -> Result<UnlockClaims, String> {
    let (payload, signature_text) = token
        .split_once('.')
        .ok_or_else(|| "Invalid Symbolica library unlock token format".to_owned())?;

    if signature_text.contains('.') {
        return Err("Invalid Symbolica library unlock token format".to_owned());
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|_| "Invalid Symbolica library unlock token payload encoding".to_owned())?;
    let signature_bytes: [u8; 64] = URL_SAFE_NO_PAD
        .decode(signature_text)
        .map_err(|_| "Invalid Symbolica library unlock token signature encoding".to_owned())?
        .try_into()
        .map_err(|_| "Invalid Symbolica library unlock token signature length".to_owned())?;
    let signature = Signature::new(signature_bytes);
    let signature_is_valid =
        verify_signature(LIBRARY_UNLOCK_PUBLIC_KEY, &payload_bytes, &signature)?;
    #[cfg(any(test, debug_assertions))]
    let signature_is_valid =
        signature_is_valid || verify_signature(TEST_UNLOCK_PUBLIC_KEY, &payload_bytes, &signature)?;
    if !signature_is_valid {
        return Err("Invalid Symbolica library unlock token signature".to_owned());
    }

    let payload = std::str::from_utf8(&payload_bytes)
        .map_err(|_| "Symbolica library unlock token payload is not UTF-8".to_owned())?;
    let value: JsonValue = payload
        .parse()
        .map_err(|_| "Invalid Symbolica library unlock token JSON".to_owned())?;
    let claims = value
        .get::<HashMap<String, JsonValue>>()
        .ok_or_else(|| "Symbolica library unlock token payload is not an object".to_owned())?;

    let version = get_usize_claim(claims, "version")?;
    if version != 1 {
        return Err(format!(
            "Unsupported Symbolica library unlock token version {version}"
        ));
    }

    let package = claims
        .get("package")
        .and_then(JsonValue::get::<String>)
        .filter(|package| {
            !package.is_empty()
                && package
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_'))
        })
        .cloned()
        .ok_or_else(|| "Symbolica library unlock token has invalid package claim".to_owned())?;
    let license = claims
        .get("license")
        .and_then(JsonValue::get::<String>)
        .filter(|license| license.starts_with("SYMBOLICA_UNLOCK_") && license.len() > 17)
        .cloned()
        .ok_or_else(|| "Symbolica library unlock token has invalid license claim".to_owned())?;

    Ok(UnlockClaims {
        package,
        license,
        token_id: signature_text.to_owned(),
    })
}

pub(crate) fn start_license_check(claims: &UnlockClaims) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(any(test, debug_assertions))]
        if claims.license == TEST_UNLOCK_LICENSE
            && claims.token_id == TEST_UNLOCK_TOKEN.split_once('.').unwrap().1
        {
            return;
        }

        let pid = std::process::id();
        let mut checked = CHECKED_UNLOCK_LICENSES.lock().unwrap();
        if checked.insert((pid, claims.license.clone())) {
            crate::LicenseManager::check_library_unlock_registration(claims.license.clone());
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn lock_directory() -> io::Result<(PathBuf, bool)> {
    if let Some(path) = std::env::var_os("SYMBOLICA_LOCK_DIR") {
        return Ok((PathBuf::from(path), false));
    }

    let project_dirs = ProjectDirs::from("io", "symbolica", "symbolica").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Could not determine the per-user Symbolica lock directory; set SYMBOLICA_LOCK_DIR",
        )
    })?;
    let base = project_dirs
        .runtime_dir()
        .unwrap_or_else(|| project_dirs.cache_dir());
    Ok((base.join("locks"), true))
}

#[cfg(not(target_arch = "wasm32"))]
fn prepare_lock_directory(path: &Path, repair_permissions: bool) -> io::Result<()> {
    let mut builder = DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)?;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Symbolica lock path '{}' is not a directory",
                path.display()
            ),
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        if metadata.permissions().mode() & 0o077 != 0 {
            if repair_permissions {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "SYMBOLICA_LOCK_DIR '{}' must only be accessible by its owner",
                        path.display()
                    ),
                ));
            }
        }
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn try_acquire_lock_in(directory: &Path, name: &str) -> io::Result<Option<File>> {
    let path = directory.join(name);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Symbolica lock path '{}' is not a regular file",
                    path.display()
                ),
            ));
        }
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }

    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let file = options.open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(error)) => Err(error),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn try_acquire_lock(name: &str) -> io::Result<Option<File>> {
    let (directory, repair_permissions) = lock_directory()?;
    prepare_lock_directory(&directory, repair_permissions)?;
    try_acquire_lock_in(&directory, name)
}

/// A verified unlock for a Rust library.
///
/// Construct this through [`crate::register_library_unlock!`], then call [`Self::enter`] around every
/// synchronous library operation. Worker closures must enter their own guard.
#[derive(Clone, Debug)]
pub struct LibraryUnlock {
    claims: UnlockClaims,
    checked_pid: Arc<AtomicU32>,
}

impl LibraryUnlock {
    /// Verify a token and bind it to the crate name supplied by [`crate::register_library_unlock!`].
    #[doc(hidden)]
    pub fn for_crate(token: &str, crate_name: &str) -> Result<Self, String> {
        let claims = verify_token(token)?;
        if claims.package != crate_name {
            return Err(format!(
                "Library unlock token for package '{}' cannot be registered by crate '{}'",
                claims.package, crate_name
            ));
        }
        start_license_check(&claims);
        Ok(Self {
            claims,
            checked_pid: Arc::new(AtomicU32::new(std::process::id())),
        })
    }

    /// Authorize Symbolica calls on the current thread until the returned guard is dropped.
    pub fn enter(&self) -> LibraryUnlockGuard {
        let pid = std::process::id();
        if self.checked_pid.swap(pid, Relaxed) != pid {
            start_license_check(&self.claims);
        }
        LibraryUnlockGuard::activate(&self.claims)
    }
}

/// A thread-bound Rust library unlock guard.
///
/// Same-thread callbacks inherit the authorization. The guard is deliberately not `Send`; worker
/// threads must enter their own guard from a cloned [`LibraryUnlock`].
pub struct LibraryUnlockGuard {
    token_id: String,
    pid: u32,
    owner_thread: ThreadId,
    _not_send: PhantomData<Rc<()>>,
}

impl LibraryUnlockGuard {
    fn activate(claims: &UnlockClaims) -> Self {
        let token_id = claims.token_id.clone();
        let pid = std::process::id();
        let owner_thread = std::thread::current().id();
        let mut guards = ACTIVE_UNLOCK_GUARDS.lock().unwrap();

        if guards.get(&token_id).is_some_and(|guard| guard.pid != pid) {
            guards.remove(&token_id);
        }

        if let Some(active) = guards.get_mut(&token_id) {
            *active.owner_threads.entry(owner_thread).or_default() += 1;
        } else {
            let mut owner_threads = HashMap::new();
            owner_threads.insert(owner_thread, 1);
            guards.insert(token_id.clone(), ActiveUnlockGuards { pid, owner_threads });
        }

        Self {
            token_id,
            pid,
            owner_thread,
            _not_send: PhantomData,
        }
    }
}

impl Drop for LibraryUnlockGuard {
    fn drop(&mut self) {
        if self.pid != std::process::id() {
            return;
        }

        let mut guards = ACTIVE_UNLOCK_GUARDS.lock().unwrap();
        let remove = if let Some(active) = guards.get_mut(&self.token_id) {
            if let Some(count) = active.owner_threads.get_mut(&self.owner_thread) {
                *count -= 1;
                if *count == 0 {
                    active.owner_threads.remove(&self.owner_thread);
                }
            }
            active.owner_threads.is_empty()
        } else {
            false
        };

        if remove {
            guards.remove(&self.token_id);
        }
    }
}

/// Return whether the current Rust thread owns an active library unlock guard.
pub(crate) fn current_thread_has_guard() -> bool {
    let pid = std::process::id();
    let thread_id = std::thread::current().id();
    ACTIVE_UNLOCK_GUARDS
        .lock()
        .unwrap()
        .values()
        .any(|active| active.pid == pid && active.owner_threads.contains_key(&thread_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    fn test_lock_directory(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("symbolica-lock-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn verifies_development_unlock_token() {
        assert_eq!(
            verify_token(TEST_UNLOCK_TOKEN).unwrap(),
            UnlockClaims {
                package: "pysecdec".to_owned(),
                license: TEST_UNLOCK_LICENSE.to_owned(),
                token_id: TEST_UNLOCK_TOKEN.split_once('.').unwrap().1.to_owned(),
            }
        );
    }

    #[test]
    fn rejects_tampered_unlock_token() {
        let mut token = TEST_UNLOCK_TOKEN.to_owned();
        token.replace_range(..1, "f");
        assert_eq!(
            verify_token(&token).unwrap_err(),
            "Invalid Symbolica library unlock token signature"
        );
    }

    #[test]
    fn rust_guard_is_thread_bound() {
        let registration = LibraryUnlock::for_crate(TEST_UNLOCK_TOKEN, "pysecdec").unwrap();
        assert!(!current_thread_has_guard());
        {
            let _guard = registration.enter();
            assert!(current_thread_has_guard());
            assert!(!std::thread::spawn(current_thread_has_guard).join().unwrap());
        }
        assert!(!current_thread_has_guard());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn private_per_user_lock_excludes_other_handles() {
        let directory = test_lock_directory("exclusive");
        let _ = std::fs::remove_dir_all(&directory);
        prepare_lock_directory(&directory, true).unwrap();

        let first = try_acquire_lock_in(&directory, "test.lock")
            .unwrap()
            .unwrap();
        assert!(
            try_acquire_lock_in(&directory, "test.lock")
                .unwrap()
                .is_none()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            assert_eq!(
                std::fs::metadata(&directory).unwrap().permissions().mode() & 0o077,
                0
            );
            assert_eq!(
                std::fs::metadata(directory.join("test.lock"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o077,
                0
            );
        }

        drop(first);
        assert!(
            try_acquire_lock_in(&directory, "test.lock")
                .unwrap()
                .is_some()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn explicit_lock_directory_must_be_private() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_lock_directory("permissions");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o755)).unwrap();

        let error = prepare_lock_directory(&directory, false).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        std::fs::remove_dir(directory).unwrap();
    }
}
