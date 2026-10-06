"""Compute reference answers and known wrong answers for T1-T12, then write tasks.json.

Uses only the agent's own tools (tools.py), so the reference answer is what a correct
sequence of tool calls returns. Run from the repo root:  python utils/build_tasks.py
"""

import json
from pathlib import Path

from tools import TOOL_SPECS, TOOLS, distance, get_area, get_density, get_population, list_regions, rank


def vals(res, key):
    """{name: value} from a tool result {count, total, rows}."""
    return {r["name"]: r[key] for r in res["rows"]}

NON_STATES = ["District of Columbia", "Puerto Rico", "Guam", "United States Virgin Islands",
              "American Samoa", "Commonwealth of the Northern Mariana Islands"]
ALL = [r["name"] for r in list_regions()]
STATES = [n for n in ALL if n not in NON_STATES]
assert len(ALL) == 56 and len(STATES) == 50

POP = vals(get_population("all"), "population")  # None for the 4 island territories
HAS_POP = [n for n in ALL if POP[n] is not None]  # 50 states + DC + PR
A_EA = vals(get_area("all", "EPSG:5070"), "area_km2")
A_WM = vals(get_area("all", "EPSG:3857"), "area_km2")
D_EA = vals(get_density(HAS_POP, "EPSG:5070"), "people_per_km2")
D_WM = vals(get_density(HAS_POP, "EPSG:3857"), "people_per_km2")
WI = POP["Wisconsin"]
TOL = 0.02


def sub(d, names):
    return {n: d[n] for n in names if n in d}


def fmt(pairs, nd=1):
    return [[n, round(v, nd)] for n, v in pairs]


tasks = []
add = tasks.append

add({"id": "T1", "type": "warm-up", "rules": [],
     "question": "Our annual report opens with the most populous state. Which state is it, and what was its population in 2025?",
     "tools_needed": ["list_regions", "get_population", "rank"],
     "answer": list(rank(sub(POP, STATES), "desc", 1)[0]),
     "grading": "State name and population exact.",
     "failure_conditions": []})

top = rank(sub(POP, STATES), "desc", 1)[0][0]
big = rank(sub(A_EA, STATES), "desc", 1)[0][0]
add({"id": "T2", "type": "learn", "rules": ["B"],
     "question": "A draft slide says the most populous state is also the largest state by area. Is that right? Give the area in km² of both states.",
     "tools_needed": ["list_regions", "get_population", "rank", "get_area"],
     "answer": {"same_state": top == big, "areas_km2": {top: A_EA[top], big: A_EA[big]}},
     "grading": f"Yes/no exact; each area within {TOL:.0%}.",
     "failure_conditions": [
         {"answer": {"same_state": False, "areas_km2": {top: A_WM[top], big: A_WM[big]}},
          "cause": "Default projection EPSG:3857 (rule B). The yes/no part is still right."}]})

add({"id": "T3", "type": "warm-up", "rules": [],
     "question": "For the Wisconsin state brief: how many states had more people than Wisconsin in 2025?",
     "tools_needed": ["list_regions", "get_population"],
     "answer": sum(POP[n] > WI for n in STATES),
     "grading": "Exact.",
     "failure_conditions": []})

add({"id": "T4", "type": "learn", "rules": ["A"],
     "question": "A federal pilot program is open only to states with fewer people than Wisconsin in 2025. How many states are eligible?",
     "tools_needed": ["list_regions", "get_population"],
     "answer": sum(POP[n] < WI for n in STATES),
     "grading": "Exact.",
     "failure_conditions": [
         {"answer": sum(POP[n] < WI for n in HAS_POP),
          "cause": "Counted every row with a population, including DC and Puerto Rico (rule A)."}]})

add({"id": "T6", "type": "reuse", "rules": ["A", "B"],
     "question": "A small-states grant covers the 5 states with the smallest area. List them with their areas in km².",
     "tools_needed": ["list_regions", "get_area", "rank"],
     "answer": fmt(rank(sub(A_EA, STATES), "asc", 5)),
     "grading": f"Same 5 states in the same order; each area within {TOL:.0%}.",
     "failure_conditions": [
         {"answer": fmt(rank(A_EA, "asc", 5)), "cause": "Included DC and territories (rule A)."},
         {"answer": fmt(rank(sub(A_WM, STATES), "asc", 5)), "cause": "Default projection EPSG:3857 (rule B)."},
         {"answer": [[n, A_WM[n]] for n, _ in rank(sub(A_EA, STATES), "asc", 5)],
          "cause": "Right states, but areas from the default projection EPSG:3857 (rule B)."}]})

add({"id": "T7", "type": "reuse", "rules": ["A", "B"],
     "question": "A transit funding memo needs the 5 most densely populated states. List them with their densities in people per km².",
     "tools_needed": ["list_regions", "get_area", "get_population", "calculator", "rank"],
     "answer": fmt(rank(sub(D_EA, STATES), "desc", 5)),
     "grading": f"Same 5 states in the same order; each density within {TOL:.0%}.",
     "failure_conditions": [
         {"answer": fmt(rank(D_EA, "desc", 5)), "cause": "Included DC and Puerto Rico (rule A)."},
         {"answer": fmt(rank(sub(D_WM, STATES), "desc", 5)), "cause": "Default projection EPSG:3857 (rule B)."}]})

add({"id": "T5", "type": "reuse", "rules": ["B"],
     "question": "A rural health program targets the 5 least densely populated states. List them with their densities in people per km².",
     "tools_needed": ["list_regions", "get_area", "get_population", "calculator", "rank"],
     "answer": fmt(rank(sub(D_EA, STATES), "asc", 5), 2),
     "grading": f"Same 5 states in the same order; each density within {TOL:.0%}.",
     "failure_conditions": [
         {"answer": fmt(rank(sub(D_WM, STATES), "asc", 5), 2), "cause": "Default projection EPSG:3857 (rule B)."}]})

s50 = sum(POP[n] for n in STATES)
dc, pr = POP["District of Columbia"], POP["Puerto Rico"]
add({"id": "T8", "type": "exception", "rules": ["A (exception)"],
     "question": "The report's headline figure is the population of the United States in the Census Bureau's 2025 estimates. What is it?",
     "tools_needed": ["list_regions", "get_population", "calculator"],
     "answer": s50 + dc,
     "grading": "Exact. The Census Bureau's United States total = 50 states + DC (no Puerto Rico).",
     "failure_conditions": [
         {"answer": s50, "cause": "Excluded DC by applying rule A where it does not hold."},
         {"answer": s50 + dc + pr, "cause": "Summed every row with a population, including Puerto Rico."}]})

add({"id": "T9", "type": "exception", "rules": ["B (exception)"],
     "question": "A map caption needs the distance in km between the centroids of Alaska and Maine. What is it?",
     "tools_needed": ["distance"],
     "answer": vals(distance("Alaska", ["Maine"], "geodesic"), "distance_km")["Maine"],
     "grading": f"Within {TOL:.0%}. Long distances are measured on the earth's surface (geodesic).",
     "failure_conditions": [
         {"answer": vals(distance("Alaska", ["Maine"], "planar", "EPSG:5070"), "distance_km")["Maine"],
          "cause": "Planar distance in EPSG:5070: applied rule B (equal-area for area) to distance."},
         {"answer": vals(distance("Alaska", ["Maine"], "planar", "EPSG:3857"), "distance_km")["Maine"],
          "cause": "Default planar distance in EPSG:3857."}]})

add({"id": "T10", "type": "return", "rules": ["A"],
     "question": "A memo on small-population states needs the 5 states with the fewest people in 2025. List them with their populations.",
     "tools_needed": ["list_regions", "get_population", "rank"],
     "answer": [list(p) for p in rank(sub(POP, STATES), "asc", 5)],
     "grading": "Same 5 states in the same order; populations exact.",
     "failure_conditions": [
         {"answer": [list(p) for p in rank(sub(POP, STATES + ["District of Columbia"]), "asc", 5)],
          "cause": "Included DC: rule A not applied, or the T8 exception carried over to a states-only question."}]})

others = [n for n in ALL if n != "Wisconsin"]
def within(method, proj="EPSG:3857"):
    d = vals(distance("Wisconsin", others, method, proj), "distance_km")
    return sorted(n for n in STATES if n != "Wisconsin" and d[n] <= 800)
w_geo, w_ea, w_wm = within("geodesic"), within("planar", "EPSG:5070"), within("planar", "EPSG:3857")
tot = WI + sum(POP[n] for n in w_geo)
add({"id": "T11", "type": "combine", "rules": ["B (exception)"],
     "question": ("A new regional office in Wisconsin will serve every state whose centroid is within 800 km of Wisconsin's centroid. "
                  "Which states does it serve, and what is the 2025 population of Wisconsin plus those states?"),
     "tools_needed": ["list_regions", "distance", "get_population", "calculator"],
     "answer": {"states": w_geo, "total_population": tot},
     "grading": "State set exact; total exact.",
     "planar_5070_same_set": w_ea == w_geo,
     "failure_conditions": [
         {"answer": {"states": w_wm, "total_population": WI + sum(POP[n] for n in w_wm)},
          "cause": "Default planar distance in EPSG:3857: distances inflated, states missed."}]
     + ([] if w_ea == w_geo else [
         {"answer": {"states": w_ea, "total_population": WI + sum(POP[n] for n in w_ea)},
          "cause": "Planar distance in EPSG:5070 (rule B applied to distance)."}])})

def above(dens, area, names):
    overall = sum(POP[n] for n in names) / sum(area[n] for n in names)
    return round(overall, 2), sorted(n for n in names if dens[n] > 2 * overall)
ov, ab = above(D_EA, A_EA, STATES)
ov_wm, ab_wm = above(D_WM, A_WM, STATES)
ov_all, ab_all = above(D_EA, A_EA, HAS_POP)
add({"id": "T12", "type": "return", "rules": ["A", "B"],
     "question": ("A funding formula gives extra weight to states whose population density is more than twice the overall density of the 50 states "
                  "(their total population divided by their total area). How many states get the extra weight?"),
     "tools_needed": ["list_regions", "get_area", "get_population", "calculator"],
     "answer": {"overall_density": ov, "count": len(ab), "states": ab},
     "grading": f"Count exact; overall density within {TOL:.0%}.",
     "failure_conditions": [
         {"answer": {"overall_density": ov_wm, "count": len(ab_wm)},
          "cause": "Default projection EPSG:3857 for area (rule B not reapplied after T9)."},
         {"answer": {"overall_density": ov_all, "count": len(ab_all)},
          "cause": "Used every row with a population, including DC and Puerto Rico (rule A)."}]})

# Reference solution per task: the tool calls a correct agent would make.
# "<50 states>" stands for the list of the 50 state names; "<50 states + DC>" adds District of Columbia.
# For documentation and for test_tasks.py only. Never show these to the agent.
S50, S50_DC = "<50 states>", "<50 states + DC>"
_pop50 = sum(POP[n] for n in STATES)
_area50 = round(sum(A_EA[n] for n in STATES), 1)
REFERENCE_CALLS = {
    "T1": [["get_population", {"names": S50, "order": "desc", "limit": 1}]],
    "T2": [["get_population", {"names": S50, "order": "desc", "limit": 1}],
           ["get_area", {"names": S50, "projection": "EPSG:5070", "order": "desc", "limit": 1}],
           ["get_area", {"names": [top, big], "projection": "EPSG:5070"}]],
    "T3": [["get_population", {"names": ["Wisconsin"]}],
           ["get_population", {"names": S50, "greater_than": WI}]],
    "T4": [["get_population", {"names": ["Wisconsin"]}],
           ["get_population", {"names": S50, "less_than": WI}]],
    "T6": [["get_area", {"names": S50, "projection": "EPSG:5070", "order": "asc", "limit": 5}]],
    "T7": [["get_density", {"names": S50, "projection": "EPSG:5070", "order": "desc", "limit": 5}]],
    "T5": [["get_density", {"names": S50, "projection": "EPSG:5070", "order": "asc", "limit": 5}]],
    "T8": [["get_population", {"names": S50_DC}]],
    "T9": [["distance", {"from_name": "Alaska", "to_names": ["Maine"], "method": "geodesic"}]],
    "T10": [["get_population", {"names": S50, "order": "asc", "limit": 5}]],
    "T11": [["distance", {"from_name": "Wisconsin", "to_names": S50, "method": "geodesic", "less_than": 800}],
            ["get_population", {"names": w_geo + ["Wisconsin"]}]],
    "T12": [["get_population", {"names": S50}],
            ["get_area", {"names": S50, "projection": "EPSG:5070"}],
            ["calculator", {"expression": f"2 * {_pop50} / {_area50}"}],
            ["get_density", {"names": S50, "projection": "EPSG:5070", "greater_than": round(2 * _pop50 / _area50, 2)}]],
}
for t in tasks:
    t["reference_calls"] = REFERENCE_CALLS[t["id"]]

# The JSON shape the agent must reply with. The grader (grader.py) reads exactly these keys.
ANSWER_FORMAT = {
    "T1": '{"state": "<name>", "population": <integer>}',
    "T2": '{"same_state": <true or false>, "areas_km2": {"<state>": <number>, "<state>": <number>}}',
    "T3": '{"count": <integer>}',
    "T4": '{"count": <integer>}',
    "T6": '{"states": [["<state>", <area_km2>], ...]}  (5 entries, smallest first)',
    "T7": '{"states": [["<state>", <people_per_km2>], ...]}  (5 entries, highest first)',
    "T5": '{"states": [["<state>", <people_per_km2>], ...]}  (5 entries, lowest first)',
    "T8": '{"population": <integer>}',
    "T9": '{"distance_km": <number>}',
    "T10": '{"states": [["<state>", <population>], ...]}  (5 entries, fewest first)',
    "T11": '{"states": ["<state>", ...], "total_population": <integer>}  (states without Wisconsin)',
    "T12": '{"count": <integer>, "overall_density": <number>}',
}
for t in tasks:
    t["answer_format"] = ANSWER_FORMAT[t["id"]]

# No-memory pilot: 4 runs (runs/no_memory_20261006_113712.json and runs/no_memory_20261006_114512.json),
# done before the reorder; IDs here are the new positions.
PILOT_NO_MEMORY = {"T1": "4/4", "T2": "0/4", "T3": "4/4", "T4": "4/4", "T5": "3/4", "T6": "0/4",
                   "T7": "0/4", "T8": "4/4", "T9": "3/4", "T10": "3/4", "T11": "3/4", "T12": "3/4"}
for t in tasks:
    t["pilot_no_memory"] = PILOT_NO_MEMORY[t["id"]]

tasks.sort(key=lambda t: int(t["id"][1:]))  # sequence order = numeric order

spec = {
    "data": ("us_states_popest2025.geojson: TIGER/Line 2025 state boundaries (56 rows) joined with "
             "Census Bureau NST-EST2025-POP (Vintage 2025 estimates, released Jan 2026). "
             "Population = July 1, 2025 estimate; null for Guam, USVI, American Samoa, CNMI."),
    "scenario": ("You are a GIS analyst at a national nonprofit. Colleagues send short data requests "
                 "for reports, memos and grant work. Answer each one with the dataset and tools."),
    "system_prompt_definitions": (
        "Use only the provided data and tools. Area means the area of a region's boundary in the dataset, "
        "in km². Population means the Census Bureau's estimate for July 1, 2025. Population density means "
        "population divided by area, in people per km²."),
    "rules": {
        "A": "'States' means the 50 states. The dataset also has DC, Puerto Rico and 4 island territories.",
        "B": "get_area defaults to EPSG:3857, which inflates area. Use EPSG:5070 for area. "
             "Exception: long distances should be geodesic.",
    },
    "task_groups": {"no_trap": ["T1", "T3"], "learn": ["T2", "T4"], "reuse": ["T5", "T6", "T7"],
                    "exception": ["T8", "T9"], "return_or_combine": ["T10", "T11", "T12"]},
    "tools": TOOL_SPECS,
    "tasks": tasks,
}
TASKS_PATH = Path(__file__).resolve().parent.parent / "data" / "tasks.json"
with open(TASKS_PATH, "w") as f:
    json.dump(spec, f, indent=2, ensure_ascii=False)

for t in tasks:
    print(t["id"], "|", json.dumps(t["answer"])[:150])
    for fc in t["failure_conditions"]:
        print("      wrong:", json.dumps(fc["answer"])[:110], "<-", fc["cause"][:45])
print("T11 planar 5070 gives same set:", w_ea == w_geo)
