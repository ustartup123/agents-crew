"""Test the LangGraph topology by patching agent nodes with mock functions.

This verifies:
- CEO → Product → [PRD Review] → Dev → QA → [Code Review] → Marketing → [Marketing Review] → Finance+Sales → END
- Dev → Product clarification loop works
- QA → Dev iteration loop works
- PRD rejection loop works
- Marketing rejection loop works
- Business node collects messages correctly
- Founder command parsing
"""

import unittest
from unittest.mock import patch, MagicMock
from graph.state import make_initial_state
from graph.checkpointer import get_checkpointer
from graph.project_graph import build_project_graph


# ── Mock node returns ────────────────────────────────────────────────────────

def _mock_cfo(state):
    return {
        "notion_financial_url": "https://notion.so/cfo",
        "current_node": "cfo",
        "status_summary": "Financial plan complete.",
        "messages": [{
            "from_agent": "cfo", "to_agent": "team",
            "content": "Financial plan done.", "message_type": "update",
            "timestamp": "2026-01-01T00:04:00Z",
        }],
    }


def _mock_sales(state):
    return {
        "notion_sales_url": "https://notion.so/sales",
        "current_node": "sales",
        "status_summary": "Sales done.",
        "messages": [{
            "from_agent": "sales", "to_agent": "team",
            "content": "Sales playbook done.", "message_type": "update",
            "timestamp": "2026-01-01T00:06:00Z",
        }],
    }


def _mock_ceo_summary(state):
    return {
        "phase": "done",
        "current_node": "done",
        "status_summary": "Project complete.",
        "messages": [{
            "from_agent": "ceo", "to_agent": "founder",
            "content": "Project complete.", "message_type": "update",
            "timestamp": "2026-01-01T00:07:00Z",
        }],
    }


class TestFullHappyPath(unittest.TestCase):
    """Full path: CEO → Product → [PRD approve] → Dev → QA → [Code approve] → Marketing → [Mkt approve] → Finance+Sales → END."""

    @patch("graph.project_graph.ceo_summary_node", _mock_ceo_summary)
    @patch("graph.project_graph.sales_node", _mock_sales)
    @patch("graph.project_graph.cfo_node", _mock_cfo)
    @patch("graph.nodes.MarketingAgent")
    @patch("graph.nodes.QAAgent")
    @patch("graph.nodes.DevAgent")
    @patch("graph.nodes.ProductAgent")
    @patch("graph.nodes.CEOAgent")
    @patch("graph.nodes.slack_reply_thread")
    @patch("graph.nodes.slack_send_message")
    @patch("graph.nodes.close_sandbox")
    @patch("graph.nodes.update_project")
    def test_happy_path_all_gates(self, mock_registry, mock_close,
                                   mock_slack_send, mock_slack_reply,
                                   MockCEO, MockProduct, MockDev, MockQA, MockMarketing):
        # CEO
        mock_ceo = MagicMock()
        mock_ceo.role = "CEO"
        mock_ceo.execute_kickoff.return_value = {
            "action": "kickoff",
            "project_name": "test-project",
            "agents_needed": ["product", "dev", "qa", "cfo", "marketing", "sales"],
            "kickoff_summary": "Let's build this!",
        }
        MockCEO.return_value = mock_ceo

        # Product
        mock_product = MagicMock()
        mock_product.role = "VP of Product"
        mock_product.execute.return_value = {
            "notion_prd_url": "https://notion.so/prd",
            "prd_content": "PRD text",
        }
        MockProduct.return_value = mock_product

        # Dev
        mock_dev = MagicMock()
        mock_dev.role = "Lead Developer"
        mock_dev.execute.return_value = {"needs_clarification": False, "summary": "Code committed."}
        MockDev.return_value = mock_dev

        # QA
        mock_qa = MagicMock()
        mock_qa.role = "QA Lead"
        mock_qa.execute.return_value = {"qa_approved": True, "test_summary": "5/5 tests pass."}
        MockQA.return_value = mock_qa

        # Marketing
        mock_mkt = MagicMock()
        mock_mkt.role = "Head of Marketing"
        mock_mkt.execute.return_value = {
            "notion_gtm_url": "https://notion.so/gtm",
            "gtm_content": "GTM strategy text",
            "summary": "GTM done.",
        }
        MockMarketing.return_value = mock_mkt

        checkpointer = get_checkpointer()
        graph = build_project_graph(checkpointer)
        config = {"configurable": {"thread_id": "test-full-happy"}}
        initial_state = make_initial_state(
            project_id="test-full-happy",
            idea="A task tracking SaaS for freelancers",
            slack_channel="C_TEST",
        )

        # Phase 1: Run until PRD review gate
        graph.invoke(initial_state, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_prd_review", snapshot.next)

        # Phase 2: Approve PRD → Dev → QA → code review gate
        graph.update_state(config, {
            "pending_review": {"gate": "prd_review", "approved": True},
        })
        graph.invoke(None, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_code_review", snapshot.next)

        # Phase 3: Approve code → Marketing → marketing review gate
        graph.update_state(config, {
            "pending_review": {"gate": "code_review", "approved": True},
        })
        graph.invoke(None, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_marketing_review", snapshot.next)

        # Phase 4: Approve marketing → Finance+Sales+Summary → END
        graph.update_state(config, {
            "pending_review": {"gate": "marketing_review", "approved": True},
        })
        result = graph.invoke(None, config=config)

        # Verify final state
        self.assertEqual(result["phase"], "done")
        self.assertEqual(result["project_name"], "test-project")
        self.assertTrue(result["qa_approved"])
        self.assertEqual(result.get("notion_gtm_url"), "https://notion.so/gtm")
        self.assertEqual(result.get("notion_financial_url"), "https://notion.so/cfo")
        self.assertEqual(result.get("notion_sales_url"), "https://notion.so/sales")

        messages = result.get("messages", [])
        self.assertGreaterEqual(len(messages), 5)

        print(f"Full happy path with all 3 gates: PASS ({len(messages)} messages)")


class TestPRDRejection(unittest.TestCase):
    """Test that rejecting the PRD loops back to Product."""

    @patch("graph.nodes.ProductAgent")
    @patch("graph.nodes.CEOAgent")
    @patch("graph.nodes.slack_reply_thread")
    @patch("graph.nodes.slack_send_message")
    @patch("graph.nodes.close_sandbox")
    @patch("graph.nodes.update_project")
    def test_prd_reject_loops(self, mock_registry, mock_close,
                               mock_slack_send, mock_slack_reply,
                               MockCEO, MockProduct):
        mock_ceo = MagicMock()
        mock_ceo.role = "CEO"
        mock_ceo.execute_kickoff.return_value = {
            "action": "kickoff", "project_name": "reject-test",
            "agents_needed": ["product", "dev", "qa"],
            "kickoff_summary": "Go!",
        }
        MockCEO.return_value = mock_ceo

        product_calls = {"n": 0}
        mock_product = MagicMock()
        mock_product.role = "VP of Product"
        def product_execute(state_arg):
            product_calls["n"] += 1
            return {"notion_prd_url": "https://notion.so/prd", "prd_content": f"PRD v{product_calls['n']}"}
        mock_product.execute.side_effect = product_execute
        MockProduct.return_value = mock_product

        checkpointer = get_checkpointer()
        graph = build_project_graph(checkpointer)
        config = {"configurable": {"thread_id": "test-prd-reject-v2"}}
        initial_state = make_initial_state(
            project_id="test-prd-reject-v2", idea="An app", slack_channel="C_TEST",
        )

        # Run to PRD review
        graph.invoke(initial_state, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_prd_review", snapshot.next)

        # Reject PRD
        graph.update_state(config, {
            "pending_review": {"gate": "prd_review", "approved": False},
            "review_rejection_reason": "Needs more user stories",
        })
        graph.invoke(None, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_prd_review", snapshot.next)
        self.assertEqual(product_calls["n"], 2)

        print(f"PRD rejection loop: PASS (Product called {product_calls['n']} times)")


class TestMarketingRejection(unittest.TestCase):
    """Test that rejecting marketing plan loops back to Marketing."""

    @patch("graph.project_graph.ceo_summary_node", _mock_ceo_summary)
    @patch("graph.project_graph.sales_node", _mock_sales)
    @patch("graph.project_graph.cfo_node", _mock_cfo)
    @patch("graph.nodes.MarketingAgent")
    @patch("graph.nodes.QAAgent")
    @patch("graph.nodes.DevAgent")
    @patch("graph.nodes.ProductAgent")
    @patch("graph.nodes.CEOAgent")
    @patch("graph.nodes.slack_reply_thread")
    @patch("graph.nodes.slack_send_message")
    @patch("graph.nodes.close_sandbox")
    @patch("graph.nodes.update_project")
    def test_marketing_reject_loops(self, mock_registry, mock_close,
                                     mock_slack_send, mock_slack_reply,
                                     MockCEO, MockProduct, MockDev, MockQA, MockMarketing):
        mock_ceo = MagicMock()
        mock_ceo.role = "CEO"
        mock_ceo.execute_kickoff.return_value = {
            "action": "kickoff", "project_name": "mkt-reject",
            "agents_needed": ["product", "dev", "qa", "cfo", "marketing", "sales"],
            "kickoff_summary": "Go!",
        }
        MockCEO.return_value = mock_ceo

        mock_product = MagicMock()
        mock_product.role = "VP of Product"
        mock_product.execute.return_value = {"prd_content": "PRD"}
        MockProduct.return_value = mock_product

        mock_dev = MagicMock()
        mock_dev.role = "Lead Developer"
        mock_dev.execute.return_value = {"summary": "Code done."}
        MockDev.return_value = mock_dev

        mock_qa = MagicMock()
        mock_qa.role = "QA Lead"
        mock_qa.execute.return_value = {"qa_approved": True, "test_summary": "Pass"}
        MockQA.return_value = mock_qa

        mkt_calls = {"n": 0}
        mock_mkt = MagicMock()
        mock_mkt.role = "Head of Marketing"
        def mkt_execute(state_arg):
            mkt_calls["n"] += 1
            return {"notion_gtm_url": "https://notion.so/gtm", "summary": f"GTM v{mkt_calls['n']}"}
        mock_mkt.execute.side_effect = mkt_execute
        MockMarketing.return_value = mock_mkt

        checkpointer = get_checkpointer()
        graph = build_project_graph(checkpointer)
        config = {"configurable": {"thread_id": "test-mkt-reject"}}
        initial_state = make_initial_state(
            project_id="test-mkt-reject", idea="An app", slack_channel="C_TEST",
        )

        # Run to PRD review → approve
        graph.invoke(initial_state, config=config)
        graph.update_state(config, {"pending_review": {"gate": "prd_review", "approved": True}})

        # Run to code review → approve
        graph.invoke(None, config=config)
        graph.update_state(config, {"pending_review": {"gate": "code_review", "approved": True}})

        # Run to marketing review
        graph.invoke(None, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_marketing_review", snapshot.next)
        self.assertEqual(mkt_calls["n"], 1)

        # Reject marketing
        graph.update_state(config, {
            "pending_review": {"gate": "marketing_review", "approved": False},
            "review_rejection_reason": "Need more focus on SEO",
        })
        graph.invoke(None, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_marketing_review", snapshot.next)
        self.assertEqual(mkt_calls["n"], 2)

        # Approve marketing → finish
        graph.update_state(config, {
            "pending_review": {"gate": "marketing_review", "approved": True},
        })
        result = graph.invoke(None, config=config)
        self.assertEqual(result["phase"], "done")

        print(f"Marketing rejection loop: PASS (Marketing called {mkt_calls['n']} times)")


class TestQAIterationLoop(unittest.TestCase):
    """Test that QA failures loop back to Dev."""

    @patch("graph.project_graph.ceo_summary_node", _mock_ceo_summary)
    @patch("graph.project_graph.sales_node", _mock_sales)
    @patch("graph.project_graph.cfo_node", _mock_cfo)
    @patch("graph.nodes.MarketingAgent")
    @patch("graph.nodes.QAAgent")
    @patch("graph.nodes.DevAgent")
    @patch("graph.nodes.ProductAgent")
    @patch("graph.nodes.CEOAgent")
    @patch("graph.nodes.slack_reply_thread")
    @patch("graph.nodes.slack_send_message")
    @patch("graph.nodes.close_sandbox")
    @patch("graph.nodes.update_project")
    def test_qa_fails_then_passes(self, mock_registry, mock_close,
                                   mock_slack_send, mock_slack_reply,
                                   MockCEO, MockProduct, MockDev, MockQA, MockMarketing):
        mock_ceo = MagicMock()
        mock_ceo.role = "CEO"
        mock_ceo.execute_kickoff.return_value = {
            "action": "kickoff", "project_name": "iter-test",
            "agents_needed": ["product", "dev", "qa"],
            "kickoff_summary": "Go!",
        }
        MockCEO.return_value = mock_ceo

        mock_product = MagicMock()
        mock_product.role = "VP of Product"
        mock_product.execute.return_value = {"prd_content": "PRD"}
        MockProduct.return_value = mock_product

        mock_dev = MagicMock()
        mock_dev.role = "Lead Developer"
        mock_dev.execute.return_value = {"summary": "Fixed bugs."}
        MockDev.return_value = mock_dev

        mock_mkt = MagicMock()
        mock_mkt.role = "Head of Marketing"
        mock_mkt.execute.return_value = {"notion_gtm_url": "", "summary": "GTM"}
        MockMarketing.return_value = mock_mkt

        qa_call_count = {"n": 0}
        def qa_side_effect(state_arg=None):
            qa_call_count["n"] += 1
            mock_qa_inst = MagicMock()
            mock_qa_inst.role = "QA Lead"
            if qa_call_count["n"] <= 1:
                mock_qa_inst.execute.return_value = {
                    "qa_approved": False, "qa_feedback": "Bug found.",
                    "test_summary": "3/5 failed",
                }
            else:
                mock_qa_inst.execute.return_value = {
                    "qa_approved": True, "test_summary": "5/5 pass",
                }
            return mock_qa_inst
        MockQA.side_effect = qa_side_effect

        checkpointer = get_checkpointer()
        graph = build_project_graph(checkpointer)
        config = {"configurable": {"thread_id": "test-qa-loop-v3"}}
        initial_state = make_initial_state(
            project_id="test-qa-loop-v3", idea="An app", slack_channel="C_TEST",
        )

        # Run to PRD review → approve
        graph.invoke(initial_state, config=config)
        graph.update_state(config, {"pending_review": {"gate": "prd_review", "approved": True}})

        # Run: Dev → QA (fail) → Dev → QA (pass) → code review gate
        graph.invoke(None, config=config)
        snapshot = graph.get_state(config)
        self.assertIn("wait_for_code_review", snapshot.next)
        self.assertTrue(snapshot.values["qa_approved"])
        self.assertEqual(snapshot.values["code_iterations"], 2)

        print(f"QA iteration loop: PASS (iterations={snapshot.values['code_iterations']})")


class TestFinanceSalesMessageCollection(unittest.TestCase):
    """Test that finance_sales_summary node collects messages from all sub-agents."""

    def test_messages_not_lost(self):
        from graph.project_graph import _finance_sales_summary_node

        with patch("graph.project_graph.cfo_node", _mock_cfo), \
             patch("graph.project_graph.sales_node", _mock_sales), \
             patch("graph.project_graph.ceo_summary_node", _mock_ceo_summary):

            state = make_initial_state("msg-test", "test", "C1")
            state["agents_needed"] = ["cfo", "sales"]

            result = _finance_sales_summary_node(state)

        messages = result.get("messages", [])
        agents_in_messages = {m["from_agent"] for m in messages}
        self.assertIn("cfo", agents_in_messages, "CFO messages missing!")
        self.assertIn("sales", agents_in_messages, "Sales messages missing!")
        self.assertIn("ceo", agents_in_messages, "CEO summary messages missing!")
        self.assertEqual(len(messages), 3)
        print(f"Finance+Sales message collection: PASS ({len(messages)} messages)")


class TestFounderCommandParsing(unittest.TestCase):
    """Test that the command parser correctly identifies all founder commands."""

    def test_commands_parsed(self):
        from workflows.slack_bot import _parse_founder_command

        # status
        cmd = _parse_founder_command("status")
        self.assertEqual(cmd[0], "status")
        self.assertIsNone(cmd[1][0])

        cmd = _parse_founder_command("status my-project")
        self.assertEqual(cmd[0], "status")
        self.assertEqual(cmd[1][0], "my-project")

        # products
        cmd = _parse_founder_command("products")
        self.assertEqual(cmd[0], "products")

        cmd = _parse_founder_command("product list")
        self.assertEqual(cmd[0], "products")

        # demo
        cmd = _parse_founder_command("demo")
        self.assertEqual(cmd[0], "demo")

        cmd = _parse_founder_command("demo my-project")
        self.assertEqual(cmd[0], "demo")
        self.assertEqual(cmd[1][0], "my-project")

        # approve / reject
        cmd = _parse_founder_command("approve")
        self.assertEqual(cmd[0], "approve")

        cmd = _parse_founder_command("reject needs more tests")
        self.assertEqual(cmd[0], "reject")
        self.assertEqual(cmd[1][0], "needs more tests")

        # feedback
        cmd = _parse_founder_command("feedback dev: use FastAPI not Flask")
        self.assertEqual(cmd[0], "feedback")
        self.assertEqual(cmd[1][0], "dev")
        self.assertEqual(cmd[1][1], "use FastAPI not Flask")

        # pause / resume
        cmd = _parse_founder_command("pause my-project")
        self.assertEqual(cmd[0], "pause")

        cmd = _parse_founder_command("resume my-project")
        self.assertEqual(cmd[0], "resume")

        # Not a command
        result = _parse_founder_command("build me a SaaS app")
        self.assertIsNone(result)

        result = _parse_founder_command("what's our burn rate?")
        self.assertIsNone(result)

        print("Founder command parsing: PASS (all commands including products, demo)")


class TestJSONParsing(unittest.TestCase):
    """Test the improved JSON parsing in BaseAgent."""

    def test_parse_strategies(self):
        from agents.base_agent import BaseAgent
        agent = BaseAgent()

        # Strategy 0: Direct JSON
        result = agent._parse_json('{"key": "value"}')
        self.assertEqual(result["key"], "value")

        # Strategy 1: Fenced json block
        result = agent._parse_json('Here is the result:\n```json\n{"key": "fenced"}\n```')
        self.assertEqual(result["key"], "fenced")

        # Strategy 2: Fenced block without language
        result = agent._parse_json('Result:\n```\n{"key": "nolang"}\n```')
        self.assertEqual(result["key"], "nolang")

        # Strategy 3: Nested JSON in text
        result = agent._parse_json('Some text {"outer": {"inner": "val"}} more text')
        self.assertEqual(result["outer"]["inner"], "val")

        # Fallback: raw output
        result = agent._parse_json("no json here at all")
        self.assertIn("raw_output", result)

        print("JSON parsing: PASS (all strategies)")


class TestRetryDecorator(unittest.TestCase):
    """Test the retry decorator."""

    def test_retry_succeeds_on_second_attempt(self):
        from tools.retry import retry

        call_count = {"n": 0}

        @retry(max_retries=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        def flaky_fn():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("transient error")
            return "success"

        result = flaky_fn()
        self.assertEqual(result, "success")
        self.assertEqual(call_count["n"], 2)

    def test_retry_exhausted(self):
        from tools.retry import retry

        @retry(max_retries=2, base_delay=0.01, retryable_exceptions=(ValueError,))
        def always_fail():
            raise ValueError("permanent error")

        with self.assertRaises(ValueError):
            always_fail()

        print("Retry decorator: PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
