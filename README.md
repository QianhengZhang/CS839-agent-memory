# CS 839 HW1: Continual learning through agent memory

An agent answers 12 GIS tasks in order. We compare no memory, episodic memory and structured memory.
Design doc (agent and memory): [proposal](https://claude.ai/code/artifact/52cd4132-37d1-4ce1-9f8b-b7d7270594f1).

```
hw1.ipynb            the experiment (currently the no-memory condition)
data/
  tasks.json         12 tasks: question, answer format, answer, known mistakes, pilot results
  us_states_popest2025.geojson
utils/
  tools.py           the 8 tools the agent calls, and their descriptions
  grader.py          grade(task, reply_text)
  build_tasks.py     regenerates data/tasks.json (only if a task changes)
  test_tasks.py      checks every task without API calls
```

## Setup

From this folder:

```
pip install -r requirements.txt
python utils/test_tasks.py     # should end with ALL PASS
```

Then open `hw1.ipynb`. It asks for the team API key and uses `gpt-5.6-luna`.

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
