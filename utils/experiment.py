"""Run T1-T12 in order under each memory condition; one JSON file per (condition, run).

From the repo root:
    python -m utils.experiment --out runs/memory_main --runs 1-5
    python -m utils.experiment --out runs/memory_smoke --runs 1 --conditions episodic,structured --tasks T1,T2,T3,T4

Finished (condition, run) files are skipped, so an interrupted experiment resumes where it stopped.
Memory is created fresh for every run. Sequences are queued run by run, so the conditions are
interleaved in time.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import base_instructions, run_task, task_cost
from .memory import make_memory

REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "data" / "tasks.json"
CONDITIONS = ("none", "episodic", "structured")
DEFAULT_MODEL = "gpt-5.6-luna"


def load_spec():
    return json.loads(SPEC_PATH.read_text())


def load_env(start=None):
    """Read the first .env in this folder or a parent folder. Variables already set win."""
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        path = folder / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, value = line.split("=", 1)
                    os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
            return path
    return None


def make_client():
    from openai import OpenAI
    # Anaconda's old brotli breaks httpx's decoder; gzip/deflate avoids it (as in the course notebooks).
    return OpenAI(default_headers={"Accept-Encoding": "gzip, deflate"})


def sequence_path(out_dir, condition, run):
    return Path(out_dir) / f"{condition}_run{run:02d}.json"


def trace_cost(trace):
    return task_cost(trace) + trace.get("memory_update", {}).get("memory_cost", 0.0)


def run_sequence(client, model, spec, condition, run, task_ids=None, verbose=False, log=print, traces=None):
    """One run: fresh memory, then the tasks in order. Appends each trace to `traces` and returns it.

    A trace is appended as soon as its task is answered, so if the sequence fails part-way the
    caller still has every answered task and its cost.
    """
    instructions = base_instructions(spec)
    memory = make_memory(condition, client, model, instructions)
    traces = [] if traces is None else traces
    for task in spec["tasks"]:
        if task_ids and task["id"] not in task_ids:
            continue
        memory_text = memory.context_for(task)
        retrieved = list(getattr(memory, "last_retrieved", []))
        trace = run_task(client, model, task, instructions, memory_text, verbose=verbose)
        trace.update(run=run, condition=condition, memory_shown=memory_text, retrieved=retrieved)
        traces.append(trace)
        trace.update(memory_update=memory.update(task, trace, trace["correct"]), memory_after=memory.snapshot())
        log(f"{condition:>10} run {run:>2} {task['id']:>3}: {'correct' if trace['correct'] else 'WRONG  '}"
            f"  tools {trace['tool_calls']:>2}  {trace['label'] or ''}")
        if verbose and condition != "none":
            print(f"\n--- memory after {task['id']} ---\n{json.dumps(trace['memory_after'], indent=1)[:3000]}\n")
    return traces


def save_sequence(out_dir, condition, run, model, traces, started):
    path = sequence_path(out_dir, condition, run)
    payload = {"condition": condition, "run": run, "model": model, "started_at": started,
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "tasks": [t["task"] for t in traces],
               "traces": traces}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str, ensure_ascii=False))
    tmp.replace(path)
    return path


def save_failed(out_dir, condition, run, model, traces, started, error):
    """Keep the answered tasks of a failed sequence in failed/: counted by spent_usd, not by load_traces."""
    folder = Path(out_dir) / "failed"
    folder.mkdir(exist_ok=True)
    path = folder / f"{condition}_run{run:02d}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {"condition": condition, "run": run, "model": model, "started_at": started, "error": error,
               "traces": traces}
    path.write_text(json.dumps(payload, indent=1, default=str, ensure_ascii=False))
    return path


def load_traces(*dirs):
    """All traces of finished sequences in the given run folders, in (folder, condition, run) file order."""
    traces = []
    for d in dirs:
        for path in sorted(Path(d).glob("*_run*.json")):
            traces += json.loads(path.read_text())["traces"]
    return traces


def spent_usd(*dirs):
    """USD over the finished and the failed sequences in the given run folders."""
    total = 0.0
    for d in map(Path, dirs):
        if d.exists():
            total += sum(trace_cost(t) for t in load_traces(d))
            for path in sorted((d / "failed").glob("*.json")):
                total += sum(trace_cost(t) for t in json.loads(path.read_text())["traces"])
    return total


def run_all(client, model, spec, out_dir, runs, conditions=CONDITIONS, task_ids=None, workers=3,
            budget_usd=3.0, budget_dirs=(), log=print):
    """Run every missing (condition, run) sequence.

    The budget covers budget_dirs plus out_dir. It is checked when each sequence starts, so with
    several workers the spend can pass budget_usd by up to `workers` sequences (about $0.04 each).
    A saved sequence counts as finished only if it holds exactly the requested tasks.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [t["id"] for t in spec["tasks"] if not task_ids or t["id"] in task_ids]
    jobs = []
    for run in runs:
        for c in conditions:
            path = sequence_path(out_dir, c, run)
            if not path.exists():
                jobs.append((run, c))
                continue
            saved = json.loads(path.read_text())
            tasks = saved.get("tasks") or [t["task"] for t in saved["traces"]]
            if tasks != wanted:
                raise ValueError(f"{path} holds tasks {tasks}, not the requested {wanted}; use another folder")
    log(f"{len(jobs)} sequences to run in {out_dir} ({len(runs) * len(conditions) - len(jobs)} already saved)")
    budget_dirs = tuple(dict.fromkeys(Path(d).resolve() for d in (*budget_dirs, out_dir)))

    def job(run, condition):
        spent = spent_usd(*budget_dirs)
        if spent > budget_usd:
            raise RuntimeError(f"Budget reached: ${spent:.2f} > ${budget_usd:.2f}")
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        traces = []
        try:
            run_sequence(client, model, spec, condition, run, task_ids=task_ids, log=log, traces=traces)
        except Exception as exc:
            if traces:
                save_failed(out_dir, condition, run, model, traces, started, f"{type(exc).__name__}: {exc}")
            raise
        save_sequence(out_dir, condition, run, model, traces, started)
        acc = sum(t["correct"] for t in traces)
        log(f"saved {condition} run {run}: {acc}/{len(traces)} correct, ${sum(map(trace_cost, traces)):.4f}")

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, run, c): (run, c) for run, c in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # keep the other sequences going; rerun resumes the failed one
                errors.append((futures[future], exc))
                log(f"FAILED {futures[future]}: {type(exc).__name__}: {exc}")
    log(f"done; spent so far ${spent_usd(*budget_dirs):.4f}" + (f"; {len(errors)} failed" if errors else ""))
    return errors


def _parse_runs(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", default="1-5", help='e.g. "1-5" or "6,7"')
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--tasks", default=None, help="comma-separated task ids (default: all 12)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--budget", type=float, default=3.0, help="USD over --out and every folder in --budget-dirs")
    parser.add_argument("--budget-dirs", default="runs/memory_smoke,runs/memory_main")
    args = parser.parse_args()

    load_env(REPO)
    model = os.getenv("CS839_MODEL", DEFAULT_MODEL)
    client = make_client()
    print(f"Model: {model}", flush=True)
    errors = run_all(client, model, load_spec(), REPO / args.out, _parse_runs(args.runs),
                     conditions=tuple(args.conditions.split(",")),
                     task_ids=args.tasks.split(",") if args.tasks else None,
                     workers=args.workers, budget_usd=args.budget,
                     budget_dirs=[REPO / d for d in args.budget_dirs.split(",")],
                     log=lambda s: print(s, flush=True))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
