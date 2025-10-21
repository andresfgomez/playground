from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
import time
import uuid

# ---- Conversation item model ----
@dataclass
class ConvItem:
    role: str                     # "user" | "assistant" | "tool"
    content: Any                  # str for text, or dict for tool call/result
    type: str = "text"            # "text" | "tool_call" | "tool_result"
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

# ---- In-memory store with token-windowing ----
class ConversationStore:
    """
    Keeps a running list of items (user/assistant messages and tool calls/results).
    Provides a window of items that fit within an approximate token budget.
    """
    def __init__(self, approx_token_budget: int = 6000):
        self.items: Deque[ConvItem] = deque()
        self.approx_token_budget = approx_token_budget
        self.conversation_id: Optional[str] = None   # server-side Conversations API id, once created

    def add_user(self, text: str):
        self.items.append(ConvItem(role="user", content=text, type="text"))

    def add_assistant(self, text: str):
        self.items.append(ConvItem(role="assistant", content=text, type="text"))

    def add_tool_call(self, name: str, arguments: Dict[str, Any]):
        self.items.append(ConvItem(
            role="assistant",
            type="tool_call",
            content={"tool_name": name, "arguments": arguments}
        ))

    def add_tool_result(self, name: str, result: Any, call_id: Optional[str] = None):
        self.items.append(ConvItem(
            role="tool",
            type="tool_result",
            content={"tool_name": name, "result": result, "call_id": call_id}
        ))

    # very rough token approximation: ~4 chars/token
    @staticmethod
    def _approx_tokens_of_item(it: ConvItem) -> int:
        if it.type == "text":
            s = str(it.content)
        else:
            s = str(it.content)
        return max(1, len(s) // 4)

    def window(self) -> List[ConvItem]:
        """Return the most recent items that fit in the budget."""
        budget = self.approx_token_budget
        out: List[ConvItem] = []
        for it in reversed(self.items):
            tks = self._approx_tokens_of_item(it)
            if tks > budget and not out:
                # always include at least the most recent item even if it “blows” the budget
                out.append(it)
                break
            if tks <= budget:
                out.append(it)
                budget -= tks
            else:
                break
        return list(reversed(out))

    # Convert to Responses API "input" items
    def to_openai_input(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for it in self.window():
            if it.type == "text":
                result.append({"role": it.role, "content": it.content})
            elif it.type == "tool_call":
                # The model produced a tool call in a prior turn (we store it),
                # but when we continue the conversation we normally DON'T resend tool_call items
                # unless you’re replaying the entire chain. We keep them in memory for auditing.
                continue
            elif it.type == "tool_result":
                # Tool results must be returned to the model as a tool message:
                result.append({
                    "role": "tool",
                    "content": str(it.content["result"]),
                    "name": it.content["tool_name"]
                })
        return result