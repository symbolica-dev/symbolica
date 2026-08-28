"""Hold one pysecdec OEM process slot until the parent closes stdin."""

import sys

import pysecdec


pysecdec.hold_oem_process(sys.argv[1])
