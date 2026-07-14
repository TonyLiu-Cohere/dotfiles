---
name: submit-run
description: >-
  Submit a fax/tax training run to a Kubernetes GPU cluster with the `sweep`
  tool. Use when the user wants to launch, submit, or kick off a training job
  (pretraining, finetune, RLVR) from a fax/tax monorepo checkout or worktree,
  monitor it, and pull step times. Covers the launcher pattern, environment
  gotchas at older commits, and how to read logs when `kjobs sweeplogs` misbehaves.
---
# Submit a fax training run with `sweep`

A run is defined by a small Python **launcher** that builds a `sweep.SweepConfig`,
then submitted with `uv run python <launcher>.py --submit start`. `sweep` builds a
Docker image from the repo checkout and submits a Kubernetes job. The image's
dependency stack (jax, rocm/cuda plugins, etc.) comes from the **repo's lockfiles
at the checked-out commit**, not from your local `.venv` — so the commit/worktree
you launch from determines what actually runs.

## 1. Confirm what you're submitting
State back, before touching anything:
- **Which code**: commit / branch / worktree (this fixes the jax + kernel stack).
- **Workload**: cluster, queue, GPU count, mesh, batch size, sequence length, ckpt.
- **Command kind**: `FaxRunCommand.FINETUNE` / pretrain / RLVR.
- **Run config source**: a resolved `run_config` JSON is the robust path (see §5).

## 2. Write (or reuse) the launcher
A launcher is a `sweep.SweepConfig` subclass. Minimum shape:

```python
import sweep
from workspace.code.templates.utils import get_repo_root

class MyRun(sweep.SweepConfig):
    settings: sweep.SweepSettings = sweep.SweepSettings(
        experiment_name="my_experiment",
        cluster=sweep.Cluster.oke_us_phx_1_prod,
    )
    posttrain: sweep.FaxConfig = sweep.FaxConfig(
        repo=sweep.RepoConfig(directory=get_repo_root(), kind=sweep.RepoKinds.fax),
        partition="gpu_64",
        queue=sweep.Queue.pre_training_queue,
        priority_class=sweep.PriorityClass.dev_medium,
        run_config="/abs/path/to/base_run_config.json",
        ckpt_path="s3://.../ckpt-480",
        fax_run_command=sweep.FaxRunCommand.FINETUNE,
        patch_run_config=dict(
            total_train_steps=20,
            first_step_to_profile=-1,   # -1 = NO profiling; profiling adds ~7s/step overhead
            log_loss_and_steptime_every_steps=1,
            log_status_every_steps=100, # heavy status/MoE logging: keep sparse
            ckpt={"force_at_init": False},
        ),
    )
    def get_search_space(self): return [{}]

if __name__ == "__main__":
    sweep.cli.run_sweep_with_flags(MyRun(), debugging_artefacts=[__file__])
```

Notes:
- `repo.directory=get_repo_root()` → the image is built from **this** checkout/worktree.
- Use an env var (e.g. `os.environ["MY_LABEL"]`) in `experiment_name` so the same
  launcher can produce distinctly-named sweeps; each submission needs a unique name.
- **For honest step times, disable profiling** (`first_step_to_profile=-1`,
  `profile_on_resume=False`) and keep status logging sparse. Profiling artifacts
  have masqueraded as real regressions.

## 3. Prefer worktrees over `git checkout` for A/B code comparisons
To compare two commits, add a worktree per commit so each keeps its own `.venv`:

```bash
git -C ~/repos/tax worktree add --detach ~/repos/exp-A <commitA>
git -C ~/repos/tax worktree add --detach ~/repos/exp-B <commitB>
# copy the launcher + run_config into each worktree's workspace/, then submit from each
```

The launcher/config live as **untracked** files, so they survive `git checkout` and
don't collide with tracked paths. Keep the launcher + `base_run_config.json`
byte-identical across arms (verify with `md5sum`) so the only variable is the commit.

## 4. Submit
From the workspace dir of the checkout/worktree:

```bash
cd <repo>/workspace
MY_LABEL=before uv run python perf_probe/my_run.py --submit start
```

`uv run` re-syncs `.venv` to the checked-out lockfile first (this is what swaps the
jax stack), then builds the image and submits.

### Gotcha: `sweep`/`co` version skew at older commits
At an older commit the pinned `sweep` may be incompatible with the `co` it resolves
(e.g. `ImportError: cannot import name 'schemas' from 'co.trainjob.client'`). Fix on
the **submit side only** (does not change the built image, which uses the repo lock):

```bash
uv pip install "sweep==1.10.0"                 # a version compatible with the new co
MY_LABEL=after uv run --no-sync python perf_probe/my_run.py --submit start
```

`--no-sync` stops `uv run` from reverting your pinned `sweep`. `kjobs` is a shell
function, so it can't be wrapped with `timeout`/`uv run` (→ exit 127); call it directly.

## 5. Bypass `command_data` with a resolved run_config
`command_data` needs its own venv + a reachable Redis and often blocks submission.
Instead extract a fully-resolved `run_config` from an existing W&B run and point
`run_config=` at that JSON. If an older commit's Pydantic schema (`extra="forbid"`)
rejects newer fields, strip exactly the offending keys until it validates.

## 6. Monitor and pull step times
```bash
# status
kjobs --context oke-us-phx-1-prod --namespace default list sweeps <sweep_name>

# find the driver pod (the exec/rank-0 pod prints "Training at step")
kubectl --context oke-us-phx-1-prod -n default get pods | grep <sweep_name>
kubectl --context oke-us-phx-1-prod -n default \
  logs <sweep_name>...-exec-0-0-xxxxx --tail=6000 | grep "Training at step"
```

Prefer `kubectl logs` on the `-exec-0-0-*` pod over `kjobs sweeplogs`, which streams
via fsnotify and frequently dies with "too many open files". Ignore step 0 (compile)
and occasional S3/logging outliers; average the steady-state steps.

## 7. Checkpoint
After submitting, record: sweep name, W&B link, commit, and expected vs observed
step time. Watch that exactly one sweep was created (retries have spawned duplicates;
cancel extras with `kjobs cancel`). Don't claim a result until steady-state step
times are observed in the logs.
