"""Own an unlicensed Symbolica port until the parent test closes stdin."""

import sys
import time
from pathlib import Path

from symbolica import E


E("port_holder")

# LicenseManager starts its port-owning thread before E returns. Give that thread time to enter its
# bind/sleep loop before telling the parent to launch collision scenarios.
time.sleep(0.2)
Path(sys.argv[1]).write_text("ready", encoding="utf-8")
sys.stdin.read()
