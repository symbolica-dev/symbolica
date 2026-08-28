# Python OEM scope and concurrency test

This fixture models a pure-Python library named `pysecdec` that embeds a signed OEM token. The token
declares up to four concurrent processes but does not limit the library's threads. Each OEM scope
declares how many additional thread identities that operation expects to introduce. The library
reserves eight for its threaded operation and opens scopes around user callbacks, while the
separate `user.py` remains ordinary unlicensed code outside those dynamic scopes.

The test starts an ordinary unlicensed Symbolica process that owns a lockfile and verifies that:

- Library calls and callbacks use the automatically covered calling thread.
- Eight library threads fit the operation's runtime reservation.
- A ninth thread or fifth OEM process aborts with the concurrency warning.
- Direct user code before or after the library scope collides with the ordinary lock.
- Copying the token into `user.py` fails because the signed package claim is `pysecdec`.

Development builds accept the fixture's dedicated test public key. Release builds require
`SYMBOLICA_PYTHON_OEM_PUBLIC_KEY` to be set at compile time and do not include that test key.

Run it against an installed development build of Symbolica with:

```shell
python tests/python/license_skip/test_oem_scope.py -v
```
