import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from conversation_memory import HistorySummarySubagent


class FakeResponse:
    content = "已合并的历史摘要"


class FakeChatModel:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = []

    def bind(self, **_kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.should_fail:
            raise RuntimeError("simulated API failure")
        return FakeResponse()


def make_history(turn_count):
    messages = [SystemMessage(content="base system prompt")]
    for index in range(turn_count):
        messages.extend(
            [
                HumanMessage(content=f"question {index}"),
                AIMessage(content=f"answer {index}"),
            ]
        )
    return messages


class HistorySummarySubagentTests(unittest.TestCase):
    def test_nine_turn_fallback_archives_oldest_three(self):
        model = FakeChatModel()
        subagent = HistorySummarySubagent(model, context_window_tokens=1_000_000)

        result = subagent.compact_if_needed(
            summary="",
            messages=make_history(9),
            upcoming_query="next question",
            router_instructions="router prompt",
            tool_schemas=[],
        )

        self.assertEqual(result.archived_turns, 3)
        self.assertEqual(sum(isinstance(m, HumanMessage) for m in result.messages), 6)
        self.assertEqual(result.messages[0].content, "base system prompt")
        self.assertEqual(result.summary, "已合并的历史摘要")
        self.assertEqual(len(model.calls), 1)

    def test_token_threshold_can_trigger_before_nine_turns(self):
        model = FakeChatModel()
        subagent = HistorySummarySubagent(model, context_window_tokens=50)

        result = subagent.compact_if_needed(
            summary="",
            messages=make_history(7),
            upcoming_query="next question",
            router_instructions="router prompt",
            tool_schemas=[],
        )

        self.assertEqual(result.archived_turns, 1)
        self.assertEqual(sum(isinstance(m, HumanMessage) for m in result.messages), 6)

    def test_summary_failure_keeps_original_history(self):
        model = FakeChatModel(should_fail=True)
        subagent = HistorySummarySubagent(model, context_window_tokens=1_000_000)
        messages = make_history(9)

        result = subagent.compact_if_needed(
            summary="previous summary",
            messages=messages,
            upcoming_query="next question",
            router_instructions="router prompt",
            tool_schemas=[],
        )

        self.assertEqual(result.messages, messages)
        self.assertEqual(result.summary, "previous summary")
        self.assertEqual(result.archived_turns, 0)
        self.assertIn("原对话已保留", result.warning)


if __name__ == "__main__":
    unittest.main()
