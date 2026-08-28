use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions, TryLockError},
    io,
    path::PathBuf,
    sync::{LazyLock, Mutex},
    thread::ThreadId,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::{Signature, VerifyingKey};
use tinyjson::JsonValue;

const TEST_OEM_PUBLIC_KEY: &str = "zZ0TCRh69ahYSVwbj4I-n_2gB7xPF49l_19DV13OuyA";

#[cfg(test)]
pub(crate) const TEST_OEM_TOKEN: &str = concat!(
    "eyJ2ZXJzaW9uIjoxLCJwYWNrYWdlIjoicHlzZWNkZWMiLCJtYXhfcHJvY2Vzc2VzIjo0LCJtYXhfdGhyZWFkc19wZXJfcHJvY2VzcyI6OH0K",
    ".",
    "ce4TV4ICSXSNBESisJSbAheXXL5T5Fts6FPdMlSlHVCA0bJFIlLRj3s2iyobmIZk7yOR0Ix_vjVnufkULUjwCw"
);

const OEM_CONCURRENCY_WARNING: &str =
    "┌─────────────────────────────────────────────────────────────────────┐
│ Symbolica OEM concurrency exceeds the library's declared allowance. │
└─────────────────────────────────────────────────────────────────────┘";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OemClaims {
    pub package: String,
    pub max_processes: usize,
    pub max_threads_per_process: usize,
    token_id: String,
}

struct ActiveOemLease {
    claims: OemClaims,
    pid: u32,
    active_scopes: usize,
    _process_slot: File,
    threads: HashSet<ThreadId>,
}

static OEM_LEASES: LazyLock<Mutex<HashMap<String, ActiveOemLease>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct OemThreadRegistrations(RefCell<HashSet<String>>);

impl Drop for OemThreadRegistrations {
    fn drop(&mut self) {
        let thread_id = std::thread::current().id();
        let mut leases = OEM_LEASES.lock().unwrap();
        for token_id in self.0.get_mut().drain() {
            if let Some(lease) = leases.get_mut(&token_id) {
                lease.threads.remove(&thread_id);
            }
        }
    }
}

std::thread_local! {
    static OEM_THREAD_REGISTRATIONS: OemThreadRegistrations =
        OemThreadRegistrations(RefCell::new(HashSet::new()));
}

fn configured_public_key() -> Option<&'static str> {
    if let Some(key) = option_env!("SYMBOLICA_PYTHON_OEM_PUBLIC_KEY") {
        return Some(key);
    }

    #[cfg(debug_assertions)]
    {
        Some(TEST_OEM_PUBLIC_KEY)
    }

    #[cfg(not(debug_assertions))]
    {
        None
    }
}

fn get_usize_claim(claims: &HashMap<String, JsonValue>, name: &str) -> Result<usize, String> {
    let value = claims
        .get(name)
        .and_then(JsonValue::get::<f64>)
        .copied()
        .ok_or_else(|| format!("OEM token is missing integer claim '{name}'"))?;

    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(format!("OEM token has invalid integer claim '{name}'"));
    }

    Ok(value as usize)
}

pub(crate) fn verify_token(token: &str) -> Result<OemClaims, String> {
    let (payload, signature) = token
        .split_once('.')
        .ok_or_else(|| "Invalid Symbolica OEM token format".to_owned())?;

    if signature.contains('.') {
        return Err("Invalid Symbolica OEM token format".to_owned());
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|_| "Invalid Symbolica OEM token payload encoding".to_owned())?;
    let signature_bytes: [u8; 64] = URL_SAFE_NO_PAD
        .decode(signature)
        .map_err(|_| "Invalid Symbolica OEM token signature encoding".to_owned())?
        .try_into()
        .map_err(|_| "Invalid Symbolica OEM token signature length".to_owned())?;
    let public_key_bytes: [u8; 32] =
        URL_SAFE_NO_PAD
            .decode(configured_public_key().ok_or_else(|| {
                "This Symbolica build has no Python OEM verification key".to_owned()
            })?)
            .map_err(|_| "Invalid compiled Symbolica OEM public key".to_owned())?
            .try_into()
            .map_err(|_| "Invalid compiled Symbolica OEM public key length".to_owned())?;

    let public_key = VerifyingKey::from_bytes(&public_key_bytes)
        .map_err(|_| "Invalid compiled Symbolica OEM public key".to_owned())?;
    public_key
        .verify_strict(&payload_bytes, &Signature::from_bytes(&signature_bytes))
        .map_err(|_| "Invalid Symbolica OEM token signature".to_owned())?;

    let payload = std::str::from_utf8(&payload_bytes)
        .map_err(|_| "Symbolica OEM token payload is not UTF-8".to_owned())?;
    let value: JsonValue = payload
        .parse()
        .map_err(|_| "Invalid Symbolica OEM token JSON".to_owned())?;
    let claims = value
        .get::<HashMap<String, JsonValue>>()
        .ok_or_else(|| "Symbolica OEM token payload is not an object".to_owned())?;

    let version = get_usize_claim(claims, "version")?;
    if version != 1 {
        return Err(format!("Unsupported Symbolica OEM token version {version}"));
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
        .ok_or_else(|| "Symbolica OEM token has invalid package claim".to_owned())?;

    Ok(OemClaims {
        package,
        max_processes: get_usize_claim(claims, "max_processes")?,
        max_threads_per_process: get_usize_claim(claims, "max_threads_per_process")?,
        token_id: signature.to_owned(),
    })
}

fn lock_path(name: &str) -> PathBuf {
    let mut path = std::env::var_os("SYMBOLICA_LOCK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    path.push(name);
    path
}

pub(crate) fn try_acquire_lock(name: &str) -> io::Result<Option<File>> {
    let path = lock_path(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o666);
    }

    let file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o666));
    }

    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(error)) => Err(error),
    }
}

fn acquire_process_slot(claims: &OemClaims) -> Result<File, String> {
    for slot in 0..claims.max_processes {
        let name = format!("symbolica-oem-{}-{slot}.lock", claims.token_id);
        match try_acquire_lock(&name) {
            Ok(Some(file)) => return Ok(file),
            Ok(None) => {}
            Err(error) => return Err(format!("Could not acquire Symbolica OEM lock: {error}")),
        }
    }

    println!("{OEM_CONCURRENCY_WARNING}");
    std::process::abort();
}

pub(crate) struct OemScopeGuard {
    token_id: String,
    pid: u32,
}

impl OemScopeGuard {
    pub(crate) fn activate(claims: OemClaims) -> Result<Self, String> {
        let token_id = claims.token_id.clone();
        let pid = std::process::id();
        let mut leases = OEM_LEASES.lock().unwrap();

        if leases.get(&token_id).is_some_and(|lease| lease.pid != pid) {
            leases.remove(&token_id);
        }

        if let Some(lease) = leases.get_mut(&token_id) {
            if lease.claims != claims {
                return Err("Conflicting Symbolica OEM claims for the same token".to_owned());
            }
            lease.active_scopes += 1;
        } else {
            let process_slot = acquire_process_slot(&claims)?;
            leases.insert(
                token_id.clone(),
                ActiveOemLease {
                    claims,
                    pid,
                    active_scopes: 1,
                    _process_slot: process_slot,
                    threads: HashSet::new(),
                },
            );
        }

        Ok(Self { token_id, pid })
    }
}

impl Drop for OemScopeGuard {
    fn drop(&mut self) {
        if self.pid != std::process::id() {
            return;
        }

        let mut leases = OEM_LEASES.lock().unwrap();
        let remove = if let Some(lease) = leases.get_mut(&self.token_id) {
            lease.active_scopes -= 1;
            lease.active_scopes == 0
        } else {
            false
        };

        if remove {
            leases.remove(&self.token_id);
        }
    }
}

/// Register the current thread with an active OEM lease and return its limit and occupied seats.
pub(crate) fn active_thread_allowance() -> Option<(usize, usize)> {
    let pid = std::process::id();
    let thread_id = std::thread::current().id();
    let mut leases = OEM_LEASES.lock().unwrap();
    if !leases.values().any(|lease| lease.pid == pid) {
        return None;
    }

    let token_id = leases
        .iter()
        .find_map(|(token_id, lease)| {
            (lease.pid == pid && lease.threads.contains(&thread_id)).then(|| token_id.clone())
        })
        .or_else(|| {
            leases.iter().find_map(|(token_id, lease)| {
                (lease.pid == pid && lease.threads.len() < lease.claims.max_threads_per_process)
                    .then(|| token_id.clone())
            })
        });

    let Some(token_id) = token_id else {
        println!("{OEM_CONCURRENCY_WARNING}");
        std::process::abort();
    };

    let lease = leases.get_mut(&token_id).unwrap();
    lease.threads.insert(thread_id);
    let limit = lease.claims.max_threads_per_process;
    let occupied = lease.threads.len();
    drop(leases);

    OEM_THREAD_REGISTRATIONS.with(|registrations| {
        registrations.0.borrow_mut().insert(token_id);
    });

    Some((limit, occupied))
}

pub(crate) fn active_thread_limit() -> Option<usize> {
    active_thread_allowance().map(|(limit, _)| limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_development_oem_token() {
        assert_eq!(
            verify_token(TEST_OEM_TOKEN).unwrap(),
            OemClaims {
                package: "pysecdec".to_owned(),
                max_processes: 4,
                max_threads_per_process: 8,
                token_id: TEST_OEM_TOKEN.split_once('.').unwrap().1.to_owned(),
            }
        );
    }

    #[test]
    fn rejects_tampered_oem_token() {
        let mut token = TEST_OEM_TOKEN.to_owned();
        token.replace_range(..1, "f");
        assert_eq!(
            verify_token(&token).unwrap_err(),
            "Invalid Symbolica OEM token signature"
        );
    }
}
