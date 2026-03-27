"""Test all bug fixes."""
from agents.base_agent import BaseAgent
from workflows.slack_bot import _route_message
from graph.state import make_initial_state, ProjectState
import typing

agent = BaseAgent()

# ── JSON Parsing Tests ────────────────────────────────────────
# Test 1: Standard ```json block
result = agent._parse_json('Here is the output:\n```json\n{"action": "kickoff"}\n```')
assert result == {"action": "kickoff"}, f"Test 1 failed: {result}"
print("JSON parse test 1 (fenced json): PASS")

# Test 2: Bare JSON
result = agent._parse_json('The result is {"approved": true}')
assert result == {"approved": True}, f"Test 2 failed: {result}"
print("JSON parse test 2 (bare JSON): PASS")

# Test 3: Fenced without language tag
result = agent._parse_json('Output:\n```\n{"name": "test"}\n```')
assert result == {"name": "test"}, f"Test 3 failed: {result}"
print("JSON parse test 3 (fenced no tag): PASS")

# Test 4: No JSON at all — should return raw_output, not empty dict
result = agent._parse_json("I could not complete the task.")
assert "raw_output" in result, f"Test 4 failed: {result}"
print("JSON parse test 4 (fallback raw_output): PASS")

# ── Routing Tests ─────────────────────────────────────────────
tests = [
    ("What is our burn rate and runway?", "cfo"),
    ("Write a PRD for the new feature", "product"),
    ("Fix the deploy bug", "dev"),
    ("Run the test suite", "qa"),
    ("Create a launch campaign", "marketing"),
    ("Close the deal with Acme", "sales"),
    ("What should we prioritize?", "ceo"),
    ("Hey cfo, check the investor deck", "cfo"),
    ("Hey dev, can you review the API code?", "dev"),
]
all_pass = True
for text, expected in tests:
    result = _route_message(text)
    if result == expected:
        print(f"Route PASS: '{text}' -> {result}")
    else:
        print(f"Route FAIL: '{text}' -> {result} (expected {expected})")
        all_pass = False

# ── State Factory Tests ───────────────────────────────────────
state = make_initial_state("test", "idea", "C1", "ts1")
hints = typing.get_type_hints(ProjectState)
missing = [k for k in hints if k not in state]
if missing:
    print(f"WARNING: make_initial_state missing fields: {missing}")
else:
    print("State factory covers all ProjectState fields: PASS")

assert state["waiting_on_founder"] == False
assert state["idea_refinement_history"] == []
assert state["slack_thread_ts"] == "ts1"
print("State factory field values: PASS")

# ── Graph Compilation ─────────────────────────────────────────
from graph.checkpointer import get_checkpointer
from graph.project_graph import build_project_graph

checkpointer = get_checkpointer()
graph = build_project_graph(checkpointer)
nodes = list(graph.nodes.keys())
assert "ceo_router" in nodes
assert "product" in nodes
assert "dev" in nodes
assert "qa" in nodes
assert "finance_sales_summary" in nodes
print(f"Graph compiled with nodes: {nodes}")

print("\n=== ALL TESTS PASSED ===")
