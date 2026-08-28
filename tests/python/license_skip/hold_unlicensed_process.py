"""Own the ordinary unlicensed Symbolica lock until the parent test closes stdin."""

import sys
from pathlib import Path

from symbolica import E


E("port_holder")
Path(sys.argv[1]).write_text("ready", encoding="utf-8")
sys.stdin.read()
