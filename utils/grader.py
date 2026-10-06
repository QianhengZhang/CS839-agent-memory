"""Grade an agent's final reply against tasks.json.

    from grader import grade
    result = grade(task, reply_text)   # task = one entry of tasks.json["tasks"]
    result["correct"], result["label"]

Rules (same as tasks.json "grading"): populations, counts and state sets exact;
areas, densities and distances within 2%. When the answer is wrong, `label` names the
matching known mistake from task["failure_conditions"], or "other".
"""

import json
import re

TOL = 0.02


# ---------- parsing ----------

def parse_reply(text):
    """Pull the first JSON object out of the model's reply (code fences allowed)."""
    if text is None:
        raise ValueError("no reply")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(cleaned[start:end + 1])


def _num(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return float(x)


def _name(x):
    return str(x).strip().lower()


def to_canonical(tid, a):
    """Map the reply format (task["answer_format"]) to the shape of task["answer"]."""
    if tid == "T1":
        return [a["state"], a["population"]]
    if tid == "T2":
        return {"same_state": a["same_state"], "areas_km2": a["areas_km2"]}
    if tid in ("T3", "T4"):
        return a["count"]
    if tid in ("T5", "T6", "T7", "T10"):
        return a["states"]
    if tid == "T8":
        return a["population"]
    if tid == "T9":
        return a["distance_km"]
    if tid == "T11":
        return {"states": a["states"], "total_population": a["total_population"]}
    if tid == "T12":
        return {"count": a["count"], "overall_density": a["overall_density"]}
    raise KeyError(tid)


# ---------- comparisons ----------

def _close(a, b):
    return abs(_num(a) - _num(b)) <= TOL * abs(_num(b))


def _exact(a, b):
    return round(_num(a)) == round(_num(b))


def _ranked(ans, ref, exact_values):
    """Same names in the same order, values exact or within 2%."""
    if len(ans) != len(ref):
        return False
    cmp = _exact if exact_values else _close
    return all(_name(a[0]) == _name(r[0]) and cmp(a[1], r[1]) for a, r in zip(ans, ref))


def _areas(ans, ref):
    got = {_name(k): v for k, v in ans.items()}
    return all(_name(k) in got and _close(got[_name(k)], v) for k, v in ref.items())


def _matches(tid, ans, ref, full=True):
    """full=True: the task's grading rule. full=False: the looser test used to label a known mistake."""
    if tid == "T1":
        return _name(ans[0]) == _name(ref[0]) and _exact(ans[1], ref[1])
    if tid == "T2":
        same = bool(ans["same_state"]) == bool(ref["same_state"]) if full else True
        return same and _areas(ans["areas_km2"], ref["areas_km2"])
    if tid in ("T3", "T4", "T8"):
        return _exact(ans, ref)
    if tid in ("T5", "T6", "T7"):
        return _ranked(ans, ref, exact_values=False)
    if tid == "T10":
        return _ranked(ans, ref, exact_values=True) if full else [_name(a[0]) for a in ans] == [_name(r[0]) for r in ref]
    if tid == "T9":
        return _close(ans, ref)
    if tid == "T11":
        same_set = {_name(s) for s in ans["states"]} == {_name(s) for s in ref["states"]}
        return same_set and _exact(ans["total_population"], ref["total_population"]) if full else same_set
    if tid == "T12":
        return _exact(ans["count"], ref["count"]) and (_close(ans["overall_density"], ref["overall_density"]) if full else True)
    raise KeyError(tid)


# ---------- public API ----------

def grade_canonical(task, ans):
    tid = task["id"]
    try:
        if _matches(tid, ans, task["answer"], full=True):
            return {"correct": True, "label": None}
        for fc in task["failure_conditions"]:
            if _matches(tid, ans, fc["answer"], full=False):
                return {"correct": False, "label": fc["cause"]}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {"correct": False, "label": f"wrong answer shape: {exc}"}
    return {"correct": False, "label": "other"}


def grade(task, reply_text):
    """Grade the raw reply text. Returns {"correct", "label", "parsed"}."""
    try:
        parsed = parse_reply(reply_text)
        ans = to_canonical(task["id"], parsed)
    except Exception as exc:  # unparseable or missing keys
        return {"correct": False, "label": f"bad format: {exc}", "parsed": None}
    out = grade_canonical(task, ans)
    out["parsed"] = parsed
    return out
