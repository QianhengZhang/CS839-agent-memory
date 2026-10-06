"""Replay every reference solution and every known mistake through the real tools and the grader.

Run from the repo root: python utils/test_tasks.py
Passes when each reference solution is graded correct and each mistake is graded wrong
with its own label. No API calls.
"""

import json

from grader import grade
from tools import TOOLS

from pathlib import Path

SPEC = json.load(open(Path(__file__).resolve().parent.parent / "data" / "tasks.json"))
T = {t["id"]: t for t in SPEC["tasks"]}
NON_STATES = {"District of Columbia", "Puerto Rico", "Guam", "United States Virgin Islands",
              "American Samoa", "Commonwealth of the Northern Mariana Islands"}
S50 = [r["name"] for r in TOOLS["list_regions"]() if r["name"] not in NON_STATES]
WI = 5972787


def call(name, args):
    args = dict(args)
    for k in ("names", "to_names"):
        if args.get(k) == "<50 states>":
            args[k] = S50
        elif args.get(k) == "<50 states + DC>":
            args[k] = S50 + ["District of Columbia"]
    return TOOLS[name](**args)


def rows(res, key):
    return [[r["name"], r[key]] for r in res["rows"]]


def reply(tid, results):
    """Turn the last tool results into the JSON reply the task asks for."""
    r = results
    if tid == "T1":
        top = r[0]["rows"][0]
        return {"state": top["name"], "population": top["population"]}
    if tid == "T2":
        a = {x["name"]: x["area_km2"] for x in r[2]["rows"]}
        return {"same_state": r[0]["rows"][0]["name"] == r[1]["rows"][0]["name"], "areas_km2": a}
    if tid in ("T3", "T4"):
        return {"count": r[-1]["count"]}
    if tid == "T6":
        return {"states": rows(r[0], "area_km2")}
    if tid in ("T5", "T7"):
        return {"states": rows(r[0], "people_per_km2")}
    if tid == "T8":
        return {"population": r[0]["total"]}
    if tid == "T9":
        return {"distance_km": r[0]["rows"][0]["distance_km"]}
    if tid == "T10":
        return {"states": rows(r[0], "population")}
    if tid == "T11":
        return {"states": [x["name"] for x in r[0]["rows"]], "total_population": r[1]["total"]}
    if tid == "T12":
        return {"count": r[3]["count"], "overall_density": r[2] / 2}
    raise KeyError(tid)


def run(tid, calls):
    results = [call(name, args) for name, args in calls]
    return grade(T[tid], json.dumps(reply(tid, results)))


# Known mistakes, written as the tool calls an agent would make.
DEFAULT, ALL = {}, "all"
MISTAKES = {
    "T2": [[["get_population", {"names": "<50 states>", "order": "desc", "limit": 1}],
            ["get_area", {"names": "<50 states>", "order": "desc", "limit": 1}],
            ["get_area", {"names": ["California", "Alaska"]}]]],
    "T4": [[["get_population", {"names": ["Wisconsin"]}],
            ["get_population", {"names": ALL, "less_than": WI}]]],
    "T6": [[["get_area", {"names": ALL, "projection": "EPSG:5070", "order": "asc", "limit": 5}]],
           [["get_area", {"names": "<50 states>", "order": "asc", "limit": 5}]]],
    "T7": [[["get_density", {"names": ALL, "projection": "EPSG:5070", "order": "desc", "limit": 5}]],
           [["get_density", {"names": "<50 states>", "order": "desc", "limit": 5}]]],
    "T5": [[["get_density", {"names": "<50 states>", "order": "asc", "limit": 5}]]],
    "T8": [[["get_population", {"names": "<50 states>"}]],
           [["get_population", {"names": ALL}]]],
    "T9": [[["distance", {"from_name": "Alaska", "to_names": ["Maine"], "projection": "EPSG:5070"}]],
           [["distance", {"from_name": "Alaska", "to_names": ["Maine"]}]]],
    "T10": [[["get_population", {"names": ALL, "order": "asc", "limit": 5}]]],
    "T11": [[["distance", {"from_name": "Wisconsin", "to_names": "<50 states>", "less_than": 800}],
             ["get_population", {"names": ["Illinois", "Iowa", "Michigan", "Minnesota", "Wisconsin"]}]]],
}


def t12_mistake(names, projection):
    p = call("get_population", {"names": names})["total"]
    a = call("get_area", {"names": names, "projection": projection})["total"]
    d = call("get_density", {"names": names, "projection": projection, "greater_than": 2 * p / a})
    return grade(T["T12"], json.dumps({"count": d["count"], "overall_density": p / a}))


if __name__ == "__main__":
    ok = True
    for tid, task in T.items():
        g = run(tid, task["reference_calls"])
        print(f"{tid:>3} reference ({len(task['reference_calls'])} calls): {'correct' if g['correct'] else 'WRONG ' + str(g['label'])}")
        ok &= g["correct"]
    print()
    labels = {fc["cause"] for t in T.values() for fc in t["failure_conditions"]}
    for tid, seqs in MISTAKES.items():
        for seq in seqs:
            g = run(tid, seq)
            print(f"{tid:>3} mistake -> {'CORRECT?!' if g['correct'] else g['label']}")
            ok &= (not g["correct"]) and g["label"] in labels
    for names, proj in (("<50 states>", "EPSG:3857"), (ALL, "EPSG:5070")):
        g = t12_mistake(names, proj)
        print(f"T12 mistake -> {'CORRECT?!' if g['correct'] else g['label']}")
        ok &= (not g["correct"]) and g["label"] in labels
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
