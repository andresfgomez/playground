from openai import OpenAI

client = OpenAI()  # expects OPENAI_API_KEY in env

MODEL = "gpt-5-mini"  # pick your model
INSTRUCTIONS = "You are a helpful assistant. When tools are useful, call them."

def run_turn(store: ConversationStore, user_text: Optional[str] = None):
    # 1) Add latest user input
    if user_text:
        store.add_user(user_text)

    # 2) Create (or continue) a conversation
    input_items = store.to_openai_input()

    # Option A: Persist with Conversations API by attaching store.conversation_id
    # Option B: Use previous_response_id chaining for better efficiency
    kwargs = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "input": input_items,
        "tools": TOOLS_SPEC,
        "tool_choice": "auto",
    }
    if store.conversation_id:
        kwargs["conversation"] = store.conversation_id

    # 3) Ask the model
    resp = client.responses.create(**kwargs)

    # If this is the very first call and the server created a conversation, capture it:
    if getattr(resp, "conversation", None) and not store.conversation_id:
        store.conversation_id = resp.conversation.id  # provided by Conversations API

    # 4) Process output items
    # The Responses API returns a list of output "items" (text, tool calls, etc.).
    # We'll scan for tool calls; if none, we just record the assistant text.
    # Pseudocode below assumes `resp.output` is iterable of items with .type
    tool_called = False
    assistant_text_chunks = []

    for item in getattr(resp, "output", []):
        if item.type == "message" and item.role == "assistant":
            # may contain text segments; concatenate
            for part in item.content:
                if part.type == "output_text":
                    assistant_text_chunks.append(part.text)

        elif item.type == "tool_call":
            tool_called = True
            name = item.name
            arguments = item.arguments  # dict of args parsed by the model
            store.add_tool_call(name, arguments)

            # 5) Execute the tool
            fn = TOOL_EXEC.get(name)
            result = None
            error = None
            try:
                if fn is None:
                    raise ValueError(f"Unknown tool: {name}")
                result = fn(**arguments)
            except Exception as e:
                error = str(e)
                result = f"Tool '{name}' failed: {error}"

            # 6) Return the tool result to the model (continuation):
            store.add_tool_result(name, result, call_id=item.id)

            # Continue the response using previous_response_id to keep the model’s chain
            cont = client.responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                tools=TOOLS_SPEC,
                tool_choice="auto",
                # pass only new tool message or let store.window() include it:
                input=store.to_openai_input(),
                previous_response_id=resp.id
            )

            # collect any final assistant text from continuation
            for citem in getattr(cont, "output", []):
                if citem.type == "message" and citem.role == "assistant":
                    for part in citem.content:
                        if part.type == "output_text":
                            assistant_text_chunks.append(part.text)

            # (Optional) update conversation id if provided on continuation
            if getattr(cont, "conversation", None) and not store.conversation_id:
                store.conversation_id = cont.conversation.id

    final_text = "\n".join(assistant_text_chunks).strip()
    if final_text:
        store.add_assistant(final_text)

    return final_text