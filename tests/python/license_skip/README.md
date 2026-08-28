# Python OEM scope and concurrency test

This fixture models a pure-Python library named `pysecdec` that embeds a signed OEM token. The token
declares up to four concurrent processes and eight Symbolica-using threads per process. The library
opens an OEM scope around its work, including user callbacks, while the separate `user.py` remains
ordinary unlicensed code outside that dynamic scope.

The test starts an ordinary unlicensed Symbolica process that owns a lockfile and verifies that:

- Library calls, callbacks, and eight library threads use the OEM allowance.
- A ninth thread or fifth OEM process aborts with the concurrency warning.
- Direct user code before or after the library scope collides with the ordinary lock.
- Copying the token into `user.py` fails because the signed package claim is `pysecdec`.

Development builds accept the fixture's dedicated test public key. Release builds require
`SYMBOLICA_PYTHON_OEM_PUBLIC_KEY` to be set at compile time and do not include that test key.

Run it against an installed development build of Symbolica with:

```shell
python tests/python/license_skip/test_oem_scope.py -v
```
