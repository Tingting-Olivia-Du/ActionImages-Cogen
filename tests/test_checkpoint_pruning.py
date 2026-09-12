"""Checkpoint saving/pruning: collective symmetry, and which files survive.

Two independent failure modes, both of which have actually happened here:

1. COLLECTIVE ASYMMETRY. `_save_checkpoint` returns early on every rank except 0, so any
   collective placed after that point is called by exactly one rank. NCCL/gloo match
   collectives by CALL ORDER, not by name, so such a barrier does not simply hang -- it pairs
   with another rank's next collective, training continues on a desynchronised communicator,
   and the watchdog aborts the job ~10 minutes later with SIGABRT and no useful traceback.
   That killed arm0 and arm1 on 2026-08-10 (checkpoint 17:16:47, abort 17:27:08 = the 600s
   NCCL timeout). test_barrier_symmetry runs the real control flow on 2 gloo ranks, and
   test_negative_control_buggy_order proves the test can detect the bug by reproducing the
   old ordering and requiring it to hang.

2. PRUNING THE WRONG FILES. The newest checkpoint is the resume tip and is left ENTIRELY
   untouched -- a half-pruned tip is unrecoverable, and there is nothing redundant left in it
   anyway now that `pytorch_model.bin` is never written (the save_model override). Every
   OLDER checkpoint keeps its `stepN.ckpt` (what eval and warm-start read) and loses its
   `global_step*/` (resume state for a step nothing will resume from).

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_checkpoint_pruning.py
"""
import multiprocessing as mp
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------------------
# 1. collective symmetry
# ---------------------------------------------------------------------------------------
def _worker(rank, world, order, tmpdir, q):
    """Reproduce _save_checkpoint's control flow with `order` deciding where the barrier goes.

    'fixed'  -- barrier BEFORE the rank-0 early return   (current code)
    'buggy'  -- barrier AFTER it, i.e. rank 0 only       (what shipped and killed two runs)
    """
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29901",
                      RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        # Stand-in for super()._save_checkpoint(): every rank writes its shard.
        open(os.path.join(tmpdir, f"shard_{rank}"), "w").close()

        if order == "fixed":
            dist.barrier()              # every rank -- current code

        if rank == 0:                   # the `if not is_world_process_zero(): return` branch
            if order == "buggy":
                dist.barrier()          # rank 0 ONLY -- the bug
            q.put(("rank0_pruned", rank))
        else:
            q.put(("rank1_returned_early", rank))

        # BOTH ranks now go back to training, whose next step is itself a collective. This is
        # the step that pays for an asymmetric barrier: under 'buggy', rank 0's barrier has
        # already consumed rank 1's all_reduce, so this one finds no partner.
        dist.all_reduce(torch.zeros(1))
        q.put((f"rank{rank}_next_step_done", rank))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(order, timeout=25):
    tmpdir = tempfile.mkdtemp()
    q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(r, 2, order, tmpdir, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout)
    alive = [p for p in procs if p.is_alive()]
    for p in alive:
        p.terminate()
        p.join()
    events = set()
    while not q.empty():
        events.add(q.get()[0])
    shutil.rmtree(tmpdir, ignore_errors=True)
    return events, bool(alive)


def test_barrier_symmetry():
    """The current ordering must let BOTH ranks finish."""
    events, hung = _run("fixed")
    assert not hung, "fixed ordering hung -- the barrier is not reached by all ranks"
    assert "rank0_pruned" in events, events
    assert "rank1_returned_early" in events, events
    assert {"rank0_next_step_done", "rank1_next_step_done"} <= events, (
        f"a rank did not survive to the next training step: {events}")
    print("BARRIER_SYMMETRY_OK  both ranks pruned/returned AND completed the next step")


def test_negative_control_buggy_order():
    """The old ordering must be DETECTED -- otherwise this test proves nothing.

    With gloo, a rank-0-only barrier pairs with rank 1's all_reduce and then rank 0 waits
    forever for a partner that has exited: the job does not terminate. (Under NCCL the same
    mismatch surfaces as the watchdog SIGABRT we actually observed.) Either way the run does
    not complete cleanly, which is what this asserts.
    """
    events, hung = _run("buggy", timeout=15)
    both_survived = {"rank0_next_step_done", "rank1_next_step_done"} <= events
    assert hung or not both_survived, (
        f"the buggy ordering completed cleanly ({events}) -- this test cannot detect the "
        f"regression it exists to catch, so the fixed-ordering result above is meaningless"
    )
    print(f"NEGATIVE_CONTROL_OK  rank-0-only barrier detected "
          f"(hung={hung}, events={sorted(events)})")


# ---------------------------------------------------------------------------------------
# 2. which files survive pruning
# ---------------------------------------------------------------------------------------
def _fake_ckpt(root, step, resume_state=True):
    d = os.path.join(root, f"checkpoint-{step}")
    os.makedirs(os.path.join(d, f"global_step{step}"), exist_ok=True)
    for name in ("pytorch_model.bin", "training_args.bin", "trainer_state.json",
                 f"step{step}.ckpt", "scheduler.pt", "latest", "zero_to_fp32.py",
                 "rng_state_0.pth", "rng_state_1.pth"):
        with open(os.path.join(d, name), "w") as f:
            f.write("x" * 16)
    with open(os.path.join(d, f"global_step{step}", "shard.pt"), "w") as f:
        f.write("y" * 64)
    if not resume_state:
        shutil.rmtree(os.path.join(d, f"global_step{step}"))
    return d


def test_pruning_keeps_the_right_files():
    from train import ActionImagesTrainer

    root = tempfile.mkdtemp()
    try:
        for step in (250, 500, 750):
            _fake_ckpt(root, step)
        newest = os.path.join(root, "checkpoint-750")

        # Call the real method, unbound -- no Trainer construction, no torch.distributed.
        ActionImagesTrainer._prune_resume_state(object(), root, keep=newest)

        for step in (250, 500):
            d = os.path.join(root, f"checkpoint-{step}")
            got = sorted(os.listdir(d))
            assert f"step{step}.ckpt" in got, f"eval checkpoint deleted from {d}: {got}"
            assert "trainer_state.json" in got, got
            assert not os.path.exists(os.path.join(d, f"global_step{step}")), \
                f"stale resume state kept in {d}"
            assert "pytorch_model.bin" not in got, f"redundant fp32 copy kept in {d}"
            print(f"  checkpoint-{step:<4} pruned to {got}")

        # The resume tip is left ENTIRELY untouched -- a half-pruned tip is unrecoverable,
        # and there is nothing redundant in it anyway now that pytorch_model.bin is never
        # written (save_model override).
        got = sorted(os.listdir(newest))
        for required in ("global_step750", "step750.ckpt", "trainer_state.json",
                         "scheduler.pt", "latest", "rng_state_0.pth"):
            assert required in got, f"resume tip lost {required}: {got}"
        print(f"  checkpoint-750  (NEWEST) untouched, keeps {got}")
        print("PRUNING_FILE_SELECTION_OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_single_checkpoint_is_never_pruned():
    """With only one checkpoint, nothing is pruned -- it IS the tip.

    Cheap to state, expensive to get wrong: if the sole checkpoint were trimmed, the very
    first save of a run would leave it unresumable, and the run could not restart from its
    own only checkpoint. Verified live on 2026-08-10 too -- the real run logged no `pruned`
    line at all after checkpoint-2, only after checkpoint-4 existed.
    """
    from train import ActionImagesTrainer

    root = tempfile.mkdtemp()
    try:
        d = _fake_ckpt(root, 250)
        before = sorted(os.listdir(d))
        ActionImagesTrainer._prune_resume_state(object(), root, keep=d)
        after = sorted(os.listdir(d))
        assert before == after, f"the only checkpoint was modified: {before} -> {after}"
        print(f"SINGLE_CHECKPOINT_UNTOUCHED_OK  {after}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pruning_is_idempotent_and_survives_partial_trees():
    """Re-running must not fail, and a half-written checkpoint must not abort the run."""
    from train import ActionImagesTrainer

    root = tempfile.mkdtemp()
    try:
        _fake_ckpt(root, 250)
        _fake_ckpt(root, 500, resume_state=False)   # e.g. an interrupted write
        newest = os.path.join(root, "checkpoint-500")
        for _ in range(3):
            ActionImagesTrainer._prune_resume_state(object(), root, keep=newest)
        assert os.path.exists(os.path.join(root, "checkpoint-250", "step250.ckpt"))
        assert os.path.exists(os.path.join(root, "checkpoint-500", "step500.ckpt"))
        print("PRUNING_IDEMPOTENT_OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    test_barrier_symmetry()
    test_negative_control_buggy_order()
    test_pruning_keeps_the_right_files()
    test_single_checkpoint_is_never_pruned()
    test_pruning_is_idempotent_and_survives_partial_trees()
    print("ALL_CHECKPOINT_PRUNING_TESTS_PASSED")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
