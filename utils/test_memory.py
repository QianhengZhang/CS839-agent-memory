"""Offline checks for the memory conditions (no API calls).

Run from the repo root: python utils/test_memory.py
Fake traces are built by replaying each task's reference_calls through the real tools. That is
test-only: reference_calls are never shown to the agent or the memory writer.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.agent import base_instructions, compact_trace, describe_names, execute_tool, run_task  # noqa: E402
from utils.memory import (EpisodicMemory, NoMemory, StructuredMemory, memory_words,  # noqa: E402
                          truncate_to_budget)

SPEC = json.loads((Path(__file__).resolve().parent.parent / "data" / "tasks.json").read_text())
TASKS = SPEC["tasks"]
T = {t["id"]: t for t in TASKS}
NON_STATES = {"District of Columbia", "Puerto Rico", "Guam", "United States Virgin Islands",
              "American Samoa", "Commonwealth of the Northern Mariana Islands"}
S50 = [r["name"] for r in execute_tool("list_regions", {})["result"] if r["name"] not in NON_STATES]
POISON = "POISON"


# ---------- stub OpenAI client ----------

class _Usage:
    def __init__(self, i=100, o=20):
        self.input_tokens, self.output_tokens = i, o


class _Response:
    def __init__(self, text):
        self.output_text, self.output, self.usage = text, [], _Usage()


class StubClient:
    """client.responses.create() returns the queued replies in order and records every request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)


# ---------- fake traces ----------

def fake_trace(task):
    calls = []
    for i, (name, args) in enumerate(task["reference_calls"], 1):
        args = dict(args)
        for k in ("names", "to_names"):
            if args.get(k) == "<50 states>":
                args[k] = S50
            elif args.get(k) == "<50 states + DC>":
                args[k] = S50 + ["District of Columbia"]
        calls.append({"step": i, "name": name, "arguments": args,
                      "result": json.dumps(execute_tool(name, args))[:8000]})
    return {"task": task["id"], "question": task["question"], "calls": calls,
            "final_answer": json.dumps(task["answer"]), "label": f"{POISON}-label", "parsed": POISON}


def poisoned(task):
    t = dict(task)
    t.update(answer=POISON, grading=POISON, rules=[POISON], reference_calls=[[POISON, {}]],
             failure_conditions=[{"answer": POISON, "cause": POISON}])
    return t


def record(i, words=5):
    return {"id": f"m{i}", "type": "rule", "when": "area requests", "do": " ".join(["word"] * words),
            "evidence": [f"T{i}:right"]}


def writer_reply(records, changes=("added m1",)):
    return json.dumps({"memory": records, "changes": list(changes)})


# ---------- checks ----------

def check(name, ok, detail=""):
    ok = bool(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail and not ok else ''}")
    return ok


def test_compact_trace():
    ok = check("names: 'all' kept as is", describe_names("all") == '"all"')
    d = describe_names(S50)
    ok &= check("names: 50 states listed by what was left out",
                d.startswith("<50 names: the 56 dataset rows except ") and "District of Columbia" in d, d)
    ok &= check("names: short lists written out", describe_names(["Wisconsin"]) == '["Wisconsin"]')
    d = describe_names(S50 + ["Texas"])
    ok &= check("names: repeated names reported", d.startswith("<51 names") and "repeated Texas" in d, d)
    d = describe_names(S50[:49] + ["Atlantis"])
    ok &= check("names: unknown names reported", "unknown names ['Atlantis']" in d and S50[49] in d, d)
    malformed = [{"step": 1, "name": "get_area", "arguments": {"names": [{"name": "Ohio", "code": "OH"}] * 13},
                  "result": '{"ok": false}'},
                 {"step": 2, "name": "get_area", "arguments": ["not", "an", "object"], "result": '{"ok": false}'},
                 {"step": 3, "name": "get_area", "arguments": {"names": [1] * 13}, "result": '{"ok": false}'}]
    try:
        compact_trace("q", malformed, None)
        ok &= check("compact trace survives malformed tool arguments", True)
    except Exception as exc:
        ok &= check("compact trace survives malformed tool arguments", False, f"{type(exc).__name__}: {exc}")
    text = compact_trace(T["T6"]["question"], fake_trace(T["T6"])["calls"], '{"states": []}')
    ok &= check("compact trace shows the projection argument", 'projection="EPSG:5070"' in text, text)
    ok &= check("compact trace: no step limit message when answered", "step limit" not in text)
    return ok


def test_no_memory_prompt_unchanged():
    expected = (SPEC["scenario"] + " " + SPEC["system_prompt_definitions"] + " "
                "You cannot run code. Use the calculator tool for arithmetic and the rank tool for sorting.")
    client = StubClient(['{"state": "California", "population": 39355309}'])
    trace = run_task(client, "stub", T["T1"], base_instructions(SPEC), NoMemory().context_for(T["T1"]))
    req = client.requests[0]
    ok = check("no memory: instructions identical to hw1_no_memory.ipynb", req["instructions"] == expected)
    ok &= check("no memory: user prompt is question + answer format",
                req["input"][0]["content"] == (f"{T['T1']['question']}\n\nWhen you are done, reply with only a "
                                               f"JSON object in this format:\n{T['T1']['answer_format']}"))
    ok &= check("stub task graded correct", trace["correct"])
    client = StubClient(['{"state": "California", "population": 39355309}'])
    run_task(client, "stub", T["T1"], base_instructions(SPEC), "MEMORY: x")
    ok &= check("memory text appended after the instructions",
                client.requests[0]["instructions"] == expected + "\n\nMEMORY: x")
    return ok


def test_episodic():
    mem, shown = EpisodicMemory(k=3), {}
    for task in TASKS:
        context = mem.context_for(poisoned(task))
        shown[task["id"]] = list(mem.last_retrieved)
        assert POISON not in context
        mem.update(poisoned(task), fake_trace(task), correct=True)
    print("     retrieved episodes:", ", ".join(f"{k}<-{'/'.join(v) or '-'}" for k, v in shown.items()))
    ok = check("episodic: nothing to retrieve for T1", shown["T1"] == [])
    ok &= check("episodic: at most k episodes, only earlier tasks",
                all(len(v) <= 3 and all(int(e[1:]) < int(k[1:]) for e in v) for k, v in shown.items()))
    ok &= check("episodic: T7 (most dense) retrieves T5 (least dense)", "T5" in shown["T7"])
    ok &= check("episodic: T6 (smallest area) retrieves T2 (area)", "T2" in shown["T6"])
    ok &= check("episodic: T12 (density) retrieves a density task", {"T5", "T7"} & set(shown["T12"]))
    ok &= check("episodic: T10 (fewest people) retrieves a population task",
                {"T1", "T3", "T4", "T8"} & set(shown["T10"]))
    ok &= check("episodic: episode text has the verdict", "judged CORRECT" in mem.snapshot()[0]["text"])
    ok &= check("episodic: no poisoned field stored", POISON not in json.dumps(mem.snapshot()))
    return ok


def test_structured():
    inst = base_instructions(SPEC)
    task, trace = poisoned(T["T2"]), fake_trace(T["T2"])

    client = StubClient([writer_reply([record(1)])])
    mem = StructuredMemory(client, "stub", inst)
    info = mem.update(task, trace, correct=False)
    request = client.requests[0]["input"][0]["content"]
    ok = check("structured: writer sees the verdict", "REVIEWER VERDICT: INCORRECT" in request)
    ok &= check("structured: writer request has no poisoned field", POISON not in json.dumps(client.requests))
    ok &= check("structured: memory updated", mem.snapshot() == [record(1)], json.dumps(mem.snapshot()))
    ok &= check("structured: usage counted", info["memory_calls"] == 1 and info["memory_input_tokens"] == 100)
    context = mem.context_for(task)
    ok &= check("structured: context renders records", context.startswith("MEMORY:") and "[m1 | rule]" in context)
    ok &= check("structured: context has no poisoned field", POISON not in context)
    ok &= check("structured: json_schema requested", client.requests[0]["text"]["format"]["type"] == "json_schema")

    big = [record(i) for i in range(1, 11)]
    client = StubClient([writer_reply(big), writer_reply(big)])
    mem = StructuredMemory(client, "stub", inst)
    info = mem.update(task, trace, correct=True)
    ok &= check("structured: over budget re-prompts once", len(client.requests) == 2)
    ok &= check("structured: then truncates to 8 records", len(mem.snapshot()) == 8 and info["budget_truncated"])

    client = StubClient([writer_reply(big), "not json"])
    mem = StructuredMemory(client, "stub", inst)
    info = mem.update(task, trace, correct=True)
    ok &= check("structured: over budget then unparseable keeps the first reply, truncated",
                len(mem.snapshot()) == 8 and info["budget_truncated"])

    client = StubClient(["not json", "still not json"])
    mem = StructuredMemory(client, "stub", inst)
    mem.records = [record(1)]
    info = mem.update(task, trace, correct=True)
    ok &= check("structured: unparseable replies keep the old memory",
                mem.snapshot() == [record(1)] and "kept the previous memory" in info["problems"])

    import httpx2
    import openai
    unsupported = openai.BadRequestError(
        "Unsupported parameter: 'text.format'", body=None,
        response=httpx2.Response(400, request=httpx2.Request("POST", "https://api.openai.com/v1/responses")))
    client = StubClient([unsupported, writer_reply([record(2)])])
    mem = StructuredMemory(client, "stub", inst)
    mem.update(task, trace, correct=True)
    ok &= check("structured: falls back to plain JSON without structured outputs",
                not mem.use_schema and "text" not in client.requests[1] and mem.snapshot() == [record(2)])
    mem = StructuredMemory(StubClient([ValueError("bad format")]), "stub", inst)
    try:
        mem.update(task, trace, correct=True)
        ok &= check("structured: other errors mentioning 'format' are raised, not a fallback", False)
    except ValueError:
        ok &= check("structured: other errors mentioning 'format' are raised, not a fallback", mem.use_schema)

    old = [record(i) for i in range(1, 7)]
    client = StubClient([writer_reply(old + [record(i) for i in range(7, 11)])] * 2)
    mem = StructuredMemory(client, "stub", inst)
    mem.records = [dict(r) for r in old]
    info = mem.update(task, trace, correct=False)
    ids = [r["id"] for r in mem.snapshot()]
    ok &= check("structured: truncation keeps the records added in this update",
                ids == ["m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"], str(ids))
    ok &= check("structured: truncation is written to the change log",
                "dropped m1 (over budget)" in info["changes"] and "dropped m2 (over budget)" in info["changes"])

    client = StubClient([writer_reply([dict(record(1), evidence="T2:wrong")])])
    mem = StructuredMemory(client, "stub", inst)
    mem.update(task, trace, correct=False)
    ok &= check("structured: evidence given as a string is kept whole", mem.snapshot()[0]["evidence"] == ["T2:wrong"])

    ok &= check("budget: word cap drops records until within budget",
                memory_words(truncate_to_budget([record(i, 40) for i in range(5)], 8, 150)) <= 150)
    supported = dict(record(1), evidence=["T1:right", "T2:right"])
    kept = truncate_to_budget([supported] + [record(i) for i in range(2, 11)], 8, 150)
    ok &= check("budget: drops the least supported, earliest records first",
                kept[0] == supported and [r["id"] for r in kept[1:]] == [f"m{i}" for i in range(4, 11)],
                str([r["id"] for r in kept]))
    return ok


def test_failed_sequence():
    import tempfile

    from utils.experiment import load_traces, run_all, spent_usd
    client = StubClient(['{"state": "California", "population": 39355309}', RuntimeError("API down")])
    with tempfile.TemporaryDirectory() as tmp:
        errors = run_all(client, "stub", SPEC, tmp, runs=[1], conditions=("none",), task_ids=["T1", "T2"],
                         workers=1, log=lambda s: None)
        failed = list((Path(tmp) / "failed").glob("*.json"))
        ok = check("failed sequence: reported, and not loaded as a finished run",
                   len(errors) == 1 and load_traces(tmp) == [])
        ok &= check("failed sequence: answered tasks kept in failed/ and counted in the budget",
                    len(failed) == 1 and spent_usd(tmp) > 0)

    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
        client = StubClient(['{"state": "California", "population": 39355309}'])
        run_all(client, "stub", SPEC, tmp, runs=[1], conditions=("none",), task_ids=["T1"], workers=1,
                log=lambda s: None)
        try:
            run_all(StubClient([]), "stub", SPEC, tmp, runs=[1], conditions=("none",), workers=1, log=lambda s: None)
            ok &= check("resume: a T1-only file is not taken as a finished full run", False)
        except ValueError:
            ok &= check("resume: a T1-only file is not taken as a finished full run", True)
        errors = run_all(StubClient([]), "stub", SPEC, tmp, runs=[2], conditions=("none",), task_ids=["T1"],
                         workers=1, budget_usd=0.0, budget_dirs=(other,), log=lambda s: None)
        ok &= check("budget: the output folder is always counted",
                    len(errors) == 1 and "Budget reached" in str(errors[0][1]))
    return ok


if __name__ == "__main__":
    results = [test_compact_trace(), test_no_memory_prompt_unchanged(), test_episodic(), test_structured(),
               test_failed_sequence()]
    print("\nALL PASS" if all(results) else "\nFAILURES ABOVE")
