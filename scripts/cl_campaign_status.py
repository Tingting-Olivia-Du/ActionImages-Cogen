#!/usr/bin/env python
"""How much of a closed-loop campaign queue is actually finished on disk?

    python scripts/cl_campaign_status.py <queue.txt> [trials]   -> "<done> <total>"

A queue line is either `<model> <task>` (the RLBench-only phase-1 queue) or
`<model> <backend> <task>` (the unified phase-2 queue); the backend picks the report
directory and the tag prefix. A job counts as done only when its JSON carries `trials`
scored results -- the same test the dispatchers use, so the watchdog and the workers can
never disagree about what is left.
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RLB = os.path.join(REPO, "reports", "closedloop_unseen14")
MS = os.path.join(REPO, "reports", "closedloop_maniskill")


def json_for(line):
    f = line.split()
    if len(f) == 2:
        model, task = f
        return os.path.join(RLB, f"rollout_u14_{model}_{task}.json")
    model, backend, task = f
    if backend == "rlb":
        return os.path.join(RLB, f"rollout_u14_{model}_{task}.json")
    pfx = "ms" if backend == "ms_heldout" else "msv1"
    return os.path.join(MS, f"rollout_{pfx}_{model}_{task}.json")


def complete(path, trials):
    try:
        return len(json.load(open(path)).get("results", [])) >= trials
    except (OSError, ValueError):
        return False


def main():
    queue, trials = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
    lines = [l.strip() for l in open(queue) if l.strip()]
    done = sum(1 for l in lines if complete(json_for(l), trials))
    print(f"{done} {len(lines)}")


if __name__ == "__main__":
    main()
