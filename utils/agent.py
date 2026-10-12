"""The task agent shared by the three memory conditions.

run_task() is the loop from hw1_no_memory.ipynb (user request -> tool calls -> local results -> answer).
The only addition is `memory_text`, appended to the instructions. With memory_text="" the request
is the same as in hw1_no_memory.ipynb, so the no-memory condition is unchanged.

compact_trace() turns a finished task into the short text that both memory systems store.
It uses only the request, the tool calls and results, and the final answer; never the reference
answer, the grader label or reference_calls from tasks.json.
"""

import json
import threading
import time

import openai

from .grader import grade
from .tools import TOOL_SPECS, TOOLS

MAX_STEPS = 15            # model calls per task; reaching it counts as a failure
MAX_TOOL_OUTPUT = 8000    # characters; list_regions alone is about 2,800
PRICE_PER_M_INPUT = 0.2   # USD per 1M input tokens for gpt-5.6-luna
PRICE_PER_M_OUTPUT = 1.2  # USD per 1M output tokens

ALL_ROWS = [r["name"] for r in TOOLS["list_regions"]()]


def base_instructions(spec):
    """The instructions of hw1_no_memory.ipynb, identical for every task and condition."""
    return (spec["scenario"] + " " + spec["system_prompt_definitions"] + " "
            "You cannot run code. Use the calculator tool for arithmetic and the rank tool for sorting.")


def strip_defaults(schema):
    if isinstance(schema, dict):
        return {k: strip_defaults(v) for k, v in schema.items() if k != "default"}
    if isinstance(schema, list):
        return [strip_defaults(v) for v in schema]
    return schema


TOOL_DEFS = [
    {
        "type": "function",
        "name": spec["name"],
        "description": spec["description"],
        "parameters": strip_defaults(spec["parameters"]),
        "strict": False,
    }
    for spec in TOOL_SPECS
]

# Conditions run in parallel threads; tools.py shares GeoPandas/pyproj objects, so calls are serialized.
_TOOL_LOCK = threading.Lock()


def execute_tool(name, arguments):
    if name not in TOOLS:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        with _TOOL_LOCK:
            return {"ok": True, "result": TOOLS[name](**arguments)}
    except Exception as exc:  # bad arguments come back to the model as an error
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
              openai.InternalServerError)


def create_response(client, retries=5, **kwargs):
    """client.responses.create, retried with backoff on rate limits and transient server errors."""
    for attempt in range(retries):
        try:
            return client.responses.create(**kwargs)
        except _RETRYABLE:
            if attempt == retries - 1:
                raise
            time.sleep(3 * 2 ** attempt)


def run_task(client, model, task, instructions, memory_text="", verbose=False):
    """Answer one task in a fresh conversation. Returns the trace, graded."""
    prompt = (f"{task['question']}\n\n"
              f"When you are done, reply with only a JSON object in this format:\n{task['answer_format']}")
    full_instructions = instructions + ("\n\n" + memory_text if memory_text else "")
    input_items = [{"role": "user", "content": prompt}]
    trace = {"task": task["id"], "question": task["question"], "calls": [], "final_answer": None,
             "model_calls": 0, "input_tokens": 0, "output_tokens": 0}
    if verbose:
        print(f"[{task['id']}] USER: {task['question']}\n")

    for step in range(1, MAX_STEPS + 1):
        response = create_response(
            client,
            model=model,
            instructions=full_instructions,
            input=input_items,
            tools=TOOL_DEFS,
            tool_choice="auto",
        )
        trace["model_calls"] += 1
        if response.usage:
            trace["input_tokens"] += response.usage.input_tokens
            trace["output_tokens"] += response.usage.output_tokens

        input_items += response.output
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            trace["final_answer"] = response.output_text
            if verbose:
                print(f"ASSISTANT: {trace['final_answer']}")
            break

        for call in function_calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError as exc:
                arguments, result = {}, {"ok": False, "error": f"Invalid JSON arguments: {exc}"}
            else:
                result = execute_tool(call.name, arguments)
            output = json.dumps(result)[:MAX_TOOL_OUTPUT]
            trace["calls"].append({"step": step, "name": call.name, "arguments": arguments, "result": output})
            if verbose:
                print(f"TOOL REQUEST: {call.name}({json.dumps(arguments)[:200]})")
                print(f"TOOL RESULT:  {output[:200]}{'...' if len(output) > 200 else ''}\n")
            input_items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
    else:
        trace["final_answer"] = None  # step limit reached

    trace["tool_calls"] = len(trace["calls"])
    trace.update(grade(task, trace["final_answer"]))
    if verbose:
        verdict = "CORRECT" if trace["correct"] else f"WRONG ({trace['label']})"
        print(f"\n{verdict} | tool calls {trace['tool_calls']} | tokens in {trace['input_tokens']}, "
              f"out {trace['output_tokens']}")
    return trace


def task_cost(trace):
    """USD for the task agent's calls in one trace (memory writer calls are counted separately)."""
    return (trace["input_tokens"] * PRICE_PER_M_INPUT + trace["output_tokens"] * PRICE_PER_M_OUTPUT) / 1e6


# ---------- compact trace: what the memory systems store ----------

def _raw(value, limit=200):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def describe_names(names):
    """Short, neutral description of a names / to_names argument (lists of 50+ names are common)."""
    if isinstance(names, list) and not all(isinstance(n, str) for n in names):
        return _raw(names)  # malformed: the tool call itself fails
    if not isinstance(names, list) or len(names) <= 12:
        return json.dumps(names, ensure_ascii=False)
    chosen = set(names)
    left_out = [n for n in ALL_ROWS if n not in chosen]
    repeated = sorted({n for n in names if names.count(n) > 1})
    unknown = sorted({n for n in names if n not in ALL_ROWS})
    text = f"<{len(names)} names: the {len(ALL_ROWS)} dataset rows"
    text += f" except {', '.join(left_out)}" if left_out else ""
    text += f"; repeated {', '.join(repeated)}" if repeated else ""
    text += f"; unknown names {unknown}" if unknown else ""
    return text + ">"


def format_args(arguments):
    if not isinstance(arguments, dict):  # the model can send any JSON; the tool call itself fails
        return _raw(arguments)
    parts = []
    for key, value in arguments.items():
        if key in ("names", "to_names"):
            parts.append(f"{key}={describe_names(value)}")
        elif key == "values" and isinstance(value, dict) and len(value) > 12:
            parts.append(f"values=<dict over {describe_names(list(value)).strip('<>')}>")
        else:
            parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
    return ", ".join(parts)


def compact_trace(question, calls, final_answer, result_chars=300):
    """The request, each tool call with its (truncated) result, and the final answer."""
    lines = [f"Request: {question}"]
    if calls:
        lines.append("Tool calls:")
        for i, call in enumerate(calls, 1):
            result = call["result"]
            if len(result) > result_chars:
                result = result[:result_chars] + "..."
            lines.append(f"  {i}. {call['name']}({format_args(call['arguments'])}) -> {result}")
    else:
        lines.append("Tool calls: none")
    answer = final_answer.strip() if final_answer else "(none: step limit reached)"
    lines.append(f"Final answer: {answer}")
    return "\n".join(lines)
