"""The three memory conditions: none, episodic and structured.

Every memory has the same interface:
    context_for(task)            -> text appended to the task agent's instructions ("" for none)
    update(task, trace, correct) -> dict describing what changed (and writer token usage)
    snapshot()                   -> JSON-serializable state after the last update

Feedback is binary: update() receives only whether the answer was judged correct. It reads the
request, the tool calls and the final answer from the trace, and the task id; never the reference
answer, failure_conditions, grader labels or reference_calls.
"""

import copy
import json
import math
import re
from collections import Counter

import openai

from .agent import PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT, compact_trace, create_response, format_args
from .tools import TOOL_SPECS


def _verdict(correct):
    return "CORRECT" if correct else "INCORRECT"


class NoMemory:
    name = "none"

    def context_for(self, task):
        return ""

    def update(self, task, trace, correct):
        return {}

    def snapshot(self):
        return None


# ---------- episodic memory ----------

EPISODIC_HEADER = (
    "PAST EPISODES: your own earlier attempts at related requests, each with the reviewer's verdict "
    "(CORRECT or INCORRECT; the reviewer gave no other feedback). They may or may not apply to the "
    "current request."
)

_STOPWORDS = set(
    "a an and are as at be been by did do does each for from had has have how in is it its of on or "
    "our that the their them these they this to was were what when which who will with".split()
)


def _features(text):
    """Character 4-grams of each non-stopword (so 'densely' and 'density' share 'dens')."""
    feats = Counter()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _STOPWORDS:
            continue
        padded = f"#{word}#"
        for i in range(max(1, len(padded) - 3)):
            feats[padded[i:i + 4]] += 1
    return feats


def _tfidf(docs):
    """TF-IDF vectors (dicts) for a list of feature Counters, with smoothed IDF over these docs."""
    n = len(docs)
    df = Counter(f for d in docs for f in d)
    vecs = []
    for d in docs:
        v = {f: (1 + math.log(c)) * (math.log((n + 1) / (df[f] + 1)) + 1) for f, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({f: x / norm for f, x in v.items()})
    return vecs


def _cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(f, 0.0) for f, x in a.items())


class EpisodicMemory:
    """Stores every finished task as an episode; shows the k most similar past episodes."""

    name = "episodic"

    def __init__(self, k=3):
        self.k = k
        self.episodes = []        # {"task", "correct", "text", "index_text"}
        self.last_retrieved = []  # task ids shown for the latest context_for() call
        self.last_scores = {}

    def retrieve(self, question):
        if not self.episodes:
            return []
        vecs = _tfidf([_features(question)] + [_features(e["index_text"]) for e in self.episodes])
        query, docs = vecs[0], vecs[1:]
        scores = [(_cosine(query, d), i) for i, d in enumerate(docs)]
        scores.sort(key=lambda s: (-round(s[0], 6), -s[1]))  # ties: most recent first
        self.last_scores = {self.episodes[i]["task"]: round(s, 3) for s, i in scores}
        return [self.episodes[i] for _, i in scores[:self.k]]

    def context_for(self, task):
        hits = self.retrieve(task["question"])
        self.last_retrieved = [e["task"] for e in hits]
        if not hits:
            return ""
        return EPISODIC_HEADER + "\n\n" + "\n\n".join(e["text"] for e in hits)

    def update(self, task, trace, correct):
        body = compact_trace(trace["question"], trace["calls"], trace["final_answer"])
        text = f"[Episode {task['id']}, judged {_verdict(correct)}]\n{body}"
        calls = " ".join(f"{c['name']} {format_args(c['arguments'])}" for c in trace["calls"])
        index_text = " ".join([trace["question"], calls, trace["final_answer"] or ""])
        self.episodes.append({"task": task["id"], "correct": bool(correct), "text": text,
                              "index_text": index_text})
        return {"stored": task["id"], "episodes": len(self.episodes)}

    def snapshot(self):
        return [{"task": e["task"], "correct": e["correct"], "text": e["text"]} for e in self.episodes]


# ---------- structured memory ----------

STRUCTURED_HEADER = (
    "MEMORY: lessons you recorded after earlier requests, based on whether a reviewer judged each "
    "answer CORRECT or INCORRECT. Apply a lesson only when its 'when' condition matches the current "
    "request."
)

RECORD_TYPES = ("rule", "procedure", "warning", "exception")

WRITER_INSTRUCTIONS = """You maintain the long-term memory of an analyst agent. The agent answers a sequence of short, different data requests using a fixed set of tools. After each request you see what the agent did and whether a reviewer judged the final answer CORRECT or INCORRECT. The reviewer gives no other feedback and never reveals the right answer.

Update the memory so that the agent does better on future, different requests.

Each memory record has:
- id: a short unique string such as "m1"
- type: "rule", "procedure", "warning" or "exception"
- when: the kind of request or tool use the record applies to, stated generally
- do: what to do or avoid, concretely; name tools and argument values when they matter
- evidence: outcomes that support the record, as "<task id>:right" or "<task id>:wrong"

Policy:
- Record reusable lessons about the data and tools, not the answer to this request.
- If the answer was INCORRECT, diagnose the most likely cause from the trace, the tool descriptions and the wording of the request, and write or revise a record so the agent avoids it. Say "likely" when the diagnosis is uncertain. If the agent followed a memory record and was still wrong, revise or narrow that record instead of repeating it.
- If the answer was CORRECT, keep the records the agent followed and add the evidence. Add a new record only for a choice in the trace that was not obvious and probably mattered.
- If new evidence conflicts with an existing record, do not silently delete the record: narrow its "when" and add an "exception" record for the specific case.
- Delete records that the evidence has shown to be wrong, and merge duplicates.
- Keep at most {max_records} records and at most {max_words} words in total across all "when" and "do" fields. Keep evidence lists short.

Return only a JSON object: {{"memory": [<records>], "changes": [<one short line per record added, updated or deleted, e.g. "added m3: ...">]}}"""

WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": list(RECORD_TYPES)},
                    "when": {"type": "string"},
                    "do": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "type", "when", "do", "evidence"],
                "additionalProperties": False,
            },
        },
        "changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["memory", "changes"],
    "additionalProperties": False,
}


def memory_words(records):
    return sum(len(r["when"].split()) + len(r["do"].split()) for r in records)


def render_records(records):
    lines = []
    for r in records:
        evidence = f" (evidence: {', '.join(r['evidence'])})" if r["evidence"] else ""
        lines.append(f"- [{r['id']} | {r['type']}] When: {r['when']} Do: {r['do']}{evidence}")
    return "\n".join(lines)


def parse_writer_reply(text):
    """Pull {"memory", "changes"} out of the writer's reply and validate the records."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the reply")
    data = json.loads(cleaned[start:end + 1])
    records = data.get("memory")
    if not isinstance(records, list):
        raise ValueError('"memory" must be a list')
    clean, seen = [], set()
    for i, r in enumerate(records):
        if not isinstance(r, dict) or not all(k in r for k in ("type", "when", "do")):
            raise ValueError(f"record {i} needs type, when and do")
        rid = str(r.get("id") or f"m{i + 1}")
        while rid in seen:
            rid += "b"
        seen.add(rid)
        rtype = r["type"] if r["type"] in RECORD_TYPES else "rule"
        evidence = r.get("evidence") or []
        evidence = [str(e) for e in ([evidence] if isinstance(evidence, str) else evidence)][-6:]
        clean.append({"id": rid, "type": rtype, "when": str(r["when"]).strip(), "do": str(r["do"]).strip(),
                      "evidence": evidence})
    changes = [str(c) for c in data.get("changes") or []]
    return clean, changes


def truncate_to_budget(records, max_records, max_words, protect=()):
    """Drop records until both limits hold: records whose id is not in `protect` first, then the least
    supported (fewest evidence entries; ties: the earliest).

    The writer is not asked to rank its records, so the order alone says nothing about importance.
    """
    kept = list(records)
    while kept and (len(kept) > max_records or memory_words(kept) > max_words):
        kept.pop(min(range(len(kept)), key=lambda i: (kept[i]["id"] in protect, len(kept[i]["evidence"]), i)))
    return kept


class StructuredMemory:
    """A compact list of records maintained by an LLM writer after every task."""

    name = "structured"

    def __init__(self, client, model, agent_instructions, max_records=8, max_words=150, use_schema=True):
        self.client = client
        self.model = model
        self.agent_instructions = agent_instructions
        self.max_records = max_records
        self.max_words = max_words
        self.use_schema = use_schema
        self.records = []
        self.tools_text = json.dumps(TOOL_SPECS, ensure_ascii=False)

    def context_for(self, task):
        if not self.records:
            return ""
        return STRUCTURED_HEADER + "\n" + render_records(self.records)

    def writer_request(self, task_id, question, calls, final_answer, correct):
        current = json.dumps(self.records, indent=1, ensure_ascii=False) if self.records else "(empty)"
        return (
            f"AGENT INSTRUCTIONS\n{self.agent_instructions}\n\n"
            f"TOOLS AVAILABLE TO THE AGENT\n{self.tools_text}\n\n"
            f"CURRENT MEMORY (this is what the agent saw for the latest request)\n{current}\n\n"
            f"LATEST REQUEST: {task_id}\n{compact_trace(question, calls, final_answer)}\n\n"
            f"REVIEWER VERDICT: {_verdict(correct)}\n\n"
            "Return the updated memory as a JSON object."
        )

    def _call_writer(self, messages, usage):
        kwargs = {"model": self.model,
                  "instructions": WRITER_INSTRUCTIONS.format(max_records=self.max_records,
                                                             max_words=self.max_words),
                  "input": messages}
        if self.use_schema:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "memory_update",
                                         "schema": WRITER_SCHEMA, "strict": True}}
        try:
            response = create_response(self.client, **kwargs)
        except openai.BadRequestError as exc:  # e.g. a proxy without structured outputs: fall back to plain JSON
            if not self.use_schema or not any(s in str(exc) for s in ("text.format", "json_schema", "response_format")):
                raise
            self.use_schema = False
            kwargs.pop("text")
            response = create_response(self.client, **kwargs)
        usage["calls"] += 1
        if response.usage:
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
        return response.output_text

    def update(self, task, trace, correct):
        request = self.writer_request(task["id"], trace["question"], trace["calls"], trace["final_answer"],
                                      correct)
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        messages = [{"role": "user", "content": request}]
        info = {"request": request, "changes": [], "problems": [], "budget_truncated": False}
        records = None  # the latest reply that parsed; an over-budget one is truncated below
        for attempt in range(2):
            reply = self._call_writer(messages, usage)
            try:
                parsed, changes = parse_writer_reply(reply)
            except (ValueError, json.JSONDecodeError) as exc:
                info["problems"].append(f"unparseable reply: {exc}")
                messages += [{"role": "assistant", "content": reply or ""},
                             {"role": "user", "content": f"That reply could not be parsed ({exc}). "
                                                         "Return only the JSON object."}]
                continue
            records = parsed
            info["changes"] = changes
            n, words = len(records), memory_words(records)
            if n <= self.max_records and words <= self.max_words:
                break
            info["problems"].append(f"over budget: {n} records, {words} words")
            if attempt == 0:
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": (
                                 f"The memory has {n} records and {words} words; the limit is "
                                 f"{self.max_records} records and {self.max_words} words. Merge or drop the "
                                 "least useful records and return the full JSON object again.")}]
        if records is None:
            info["problems"].append("kept the previous memory")
        else:
            before = {r["id"]: r for r in self.records}
            fresh = {r["id"] for r in records if before.get(r["id"]) != r}  # added or changed in this update
            trimmed = truncate_to_budget(records, self.max_records, self.max_words, protect=fresh)
            dropped = [r["id"] for r in records if r not in trimmed]
            info["budget_truncated"] = bool(dropped)
            info["changes"] += [f"dropped {rid} (over budget)" for rid in dropped]
            self.records = trimmed
        info.update(memory_calls=usage["calls"], memory_input_tokens=usage["input_tokens"],
                    memory_output_tokens=usage["output_tokens"],
                    memory_cost=(usage["input_tokens"] * PRICE_PER_M_INPUT
                                 + usage["output_tokens"] * PRICE_PER_M_OUTPUT) / 1e6)
        return info

    def snapshot(self):
        return copy.deepcopy(self.records)


def make_memory(condition, client=None, model=None, agent_instructions=None):
    if condition == "none":
        return NoMemory()
    if condition == "episodic":
        return EpisodicMemory(k=3)
    if condition == "structured":
        return StructuredMemory(client, model, agent_instructions)
    raise ValueError(f"unknown condition {condition!r}")
