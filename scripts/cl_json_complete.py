#!/usr/bin/env python
"""Exit 0 iff <rollout json> already holds <trials> scored results.

One definition of "this job is finished", shared by every dispatcher, the GT sweep and the
watchdog. rollout.py writes its JSON incrementally so a killed run leaves a short file; an
existence test would skip it forever, which is how a campaign silently ends up with n<20
cells.
"""
import json, sys

try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError, IndexError):
    sys.exit(1)
sys.exit(0 if len(d.get("results", [])) >= int(sys.argv[2]) else 1)
