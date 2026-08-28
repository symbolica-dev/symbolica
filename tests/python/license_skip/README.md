# Python file-local license skip test

This fixture models a library named `pysecdec` that declares `SKIP_LICENSE = True`. Symbolica calls
whose active Python frame belongs to `pysecdec.py` skip the license manager. The separate `user.py`
does not opt out, so Symbolica calls made directly there or from one of its callbacks use normal
license handling.

`pysecdec.py` defines `__all__` so that `from pysecdec import *` cannot accidentally copy the
`SKIP_LICENSE` marker into user globals. A real multi-file package must place the marker in every
module whose own Symbolica operations should skip checks.

The library deliberately does not wrap callback invocation in `with skip_license():`. That guard
is thread-scoped and would also exempt Symbolica calls made synchronously by the user callback.

The test starts an unlicensed Symbolica process that owns a fresh local port, then runs four child
processes to verify both successful library paths and expected user-code port collisions.

Run it against an installed development build of Symbolica with:

```shell
python tests/python/license_skip/test_port_isolation.py -v
```
