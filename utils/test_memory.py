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

    client = StubClient(["not json", "still not json"])
    mem = StructuredMemory(client, "stub", inst)
    mem.records = [record(1)]
    info = mem.update(task, trace, correct=True)
    ok &= check("structured: unparseable replies keep the old memory",
                mem.snapshot() == [record(1)] and "kept the previous memory" in info["problems"])

    client = StubClient([ValueError("Unsupported parameter: text.format"), writer_reply([record(2)])])
    mem = StructuredMemory(client, "stub", inst)
    mem.update(task, trace, correct=True)
    ok &= check("structured: falls back to plain JSON without structured outputs",
                not mem.use_schema and "text" not in client.requests[1] and mem.snapshot() == [record(2)])

    ok &= check("budget: word cap drops trailing records",
                memory_words(truncate_to_budget([record(i, 40) for i in range(5)], 8, 150)) <= 150)
    return ok


if __name__ == "__main__":
    results = [test_compact_trace(), test_no_memory_prompt_unchanged(), test_episodic(), test_structured()]
    print("\nALL PASS" if all(results) else "\nFAILURES ABOVE")
