# CS 839 HW1: Continual learning through agent memory

An agent answers 12 GIS tasks in order. We compare no memory, episodic memory and structured memory.

```
hw1_no_memory.ipynb  the no-memory condition on its own
hw1_memory.ipynb     the full experiment: no memory, episodic and structured memory, results
data/
  tasks.json         12 tasks: question, answer format, answer, known mistakes, pilot results
  us_states_popest2025.geojson
utils/
  tools.py           the 8 tools the agent calls, and their descriptions
  grader.py          grade(task, reply_text)
  build_tasks.py     regenerates data/tasks.json (only if a task changes)
  test_tasks.py      checks every task without API calls
  agent.py           the agent loop of hw1_no_memory.ipynb, plus a memory hook
  memory.py          the three memory conditions
  experiment.py      runs the sequences and saves one JSON file per (condition, run)
  test_memory.py     checks the memory code with a stub client, without API calls
runs/                saved traces (memory_smoke/, memory_main/)
results/, figures/   tables and plots made by hw1_memory.ipynb
```

## Setup

From this folder:

```
pip install -r requirements.txt matplotlib
python utils/test_tasks.py     # should end with ALL PASS
python utils/test_memory.py    # should end with ALL PASS
```

Then open `hw1_memory.ipynb` (or `hw1_no_memory.ipynb`). It reads `OPENAI_API_KEY` from a `.env` file in this
folder or a parent folder, or asks for the team API key, and uses `gpt-5.6-luna`.

## Data

TIGER/Line 2025 state boundaries joined with the Census Bureau's Vintage 2025 population estimates
(July 1, 2025). 56 rows: 50 states, DC, Puerto Rico and 4 island territories (no population).
Boundaries include water inside the state line.

## Two rules the agent can learn

- **Rule A:** "states" means the 50 states. Exception in T8: the U.S. total includes DC.
- **Rule B:** use `projection="EPSG:5070"` for area. Exception in T9: long distances are geodesic.

## Running the memory conditions

- One run = T1 to T12 in order. Clear memory at the start of each run.
- 5 runs per condition, conditions interleaved. Re-run no memory too.
- Never show `reference_calls` from `tasks.json` to the agent.
- Feedback is binary: after each task the memory learns only whether the answer was correct.

`hw1_memory.ipynb` runs everything; the same runs from the command line:

```
python -m utils.experiment --out runs/memory_main --runs 1-5
```

Finished (condition, run) files are skipped, so an interrupted experiment resumes.

## Results (Oct 11, 10 runs per condition, $0.86 in total)

Correct out of 120: no memory 71, episodic 62, structured 74. Per task: `results/summary_by_task.csv` and
`figures/success_by_task.png`; cost per condition: `results/cost_by_condition.csv`. Memory snapshots, the final
structured memories, the two trajectories and the findings are in `hw1_memory.ipynb` (sections 10–14).
