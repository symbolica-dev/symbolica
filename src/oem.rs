use std::{
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
    "eyJ2ZXJzaW9uIjoxLCJwYWNrYWdlIjoicHlzZWNkZWMiLCJtYXhfcHJvY2Vzc2VzIjo0fQo",
    ".",
    "BlpNuJqgYtiWDgokAZO2Q8udQUB9WTIJn6EueXmm35G85NSMCngcpMGkxdzFNRXGVP3m6nR-KyhOK64Y7sZHAw"
);

const OEM_CONCURRENCY_WARNING: &str =
    "┌─────────────────────────────────────────────────────────────────────┐
│ Symbolica OEM concurrency exceeds the library's declared allowance. │
└─────────────────────────────────────────────────────────────────────┘";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OemClaims {
    pub package: String,
    pub max_processes: usize,
    token_id: String,
}

struct ActiveOemLease {
    claims: OemClaims,
    pid: u32,
    active_scopes: usize,
    reserved_new_threads: usize,
    _process_slot: File,
    owner_threads: HashMap<ThreadId, usize>,
    additional_threads: HashSet<ThreadId>,
}

static OEM_LEASES: LazyLock<Mutex<HashMap<String, ActiveOemLease>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

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
    owner_thread: ThreadId,
    reserved_new_threads: usize,
}

impl OemScopeGuard {
    pub(crate) fn activate(claims: OemClaims, reserved_new_threads: usize) -> Result<Self, String> {
        let token_id = claims.token_id.clone();
        let pid = std::process::id();
        let owner_thread = std::thread::current().id();
        let mut leases = OEM_LEASES.lock().unwrap();

        if leases.get(&token_id).is_some_and(|lease| lease.pid != pid) {
            leases.remove(&token_id);
        }

        if let Some(lease) = leases.get_mut(&token_id) {
            if lease.claims != claims {
                return Err("Conflicting Symbolica OEM claims for the same token".to_owned());
            }
            let new_reservation = lease
                .reserved_new_threads
                .checked_add(reserved_new_threads)
                .ok_or_else(|| "OEM thread reservation overflow".to_owned())?;
            lease.active_scopes += 1;
            lease.reserved_new_threads = new_reservation;
            *lease.owner_threads.entry(owner_thread).or_default() += 1;
        } else {
            let process_slot = acquire_process_slot(&claims)?;
            let mut owner_threads = HashMap::new();
            owner_threads.insert(owner_thread, 1);
            leases.insert(
                token_id.clone(),
                ActiveOemLease {
                    claims,
                    pid,
                    active_scopes: 1,
                    reserved_new_threads,
                    _process_slot: process_slot,
                    owner_threads,
                    additional_threads: HashSet::new(),
                },
            );
        }

        Ok(Self {
            token_id,
            pid,
            owner_thread,
            reserved_new_threads,
        })
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
            lease.reserved_new_threads = lease
                .reserved_new_threads
                .saturating_sub(self.reserved_new_threads);
            if let Some(count) = lease.owner_threads.get_mut(&self.owner_thread) {
                *count -= 1;
                if *count == 0 {
                    lease.owner_threads.remove(&self.owner_thread);
                }
            }
            lease.active_scopes == 0
        } else {
            false
        };

        if remove {
            leases.remove(&self.token_id);
        }
    }
}

/// Register the current thread with an active OEM lease.
///
/// A scope's entering thread is covered automatically. Every other distinct thread identity
/// consumes one of the runtime reservations until the outermost scope for the token closes.
pub(crate) fn register_current_thread() -> bool {
    let pid = std::process::id();
    let thread_id = std::thread::current().id();
    let mut leases = OEM_LEASES.lock().unwrap();
    if !leases.values().any(|lease| lease.pid == pid) {
        return false;
    }

    if leases.values().any(|lease| {
        lease.pid == pid
            && (lease.owner_threads.contains_key(&thread_id)
                || lease.additional_threads.contains(&thread_id))
    }) {
        return true;
    }

    let lease = leases.values_mut().find(|lease| {
        lease.pid == pid && lease.additional_threads.len() < lease.reserved_new_threads
    });

    let Some(lease) = lease else {
        println!("{OEM_CONCURRENCY_WARNING}");
        std::process::abort();
    };

    lease.additional_threads.insert(thread_id);
    true
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
