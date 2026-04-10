import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ── Define Tools ──────────────────────────────────────────
@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a given city."""
    return f"The weather in {city} is sunny and 28°C."

tools = [add_numbers, get_weather]

# ── LLM with tools bound ──────────────────────────────────
llm = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])
llm_with_tools = llm.bind_tools(tools)

# ── Nodes ─────────────────────────────────────────────────
def llm_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are a helpful assistant with tools.")
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

# ToolNode handles running the actual tool automatically
tool_node = ToolNode(tools)

# ── Router — does LLM want a tool or is it done? ──────────
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:       # LLM wants to call a tool
        return "tool_node"
    return END                # LLM has final answer

# ── Build Graph ───────────────────────────────────────────
builder = StateGraph(MessagesState)

builder.add_node("llm_node",  llm_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_node")
builder.add_conditional_edges("llm_node", should_continue)
builder.add_edge("tool_node", "llm_node")  # after tool runs → back to LLM

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ── Run ───────────────────────────────────────────────────
config = {"configurable": {"thread_id": "tool_test"}}

result = graph.invoke(
    {"messages": [HumanMessage(content="What is 42 + 58?")]},
    config=config
)
print(result["messages"][-1].content)
# → "42 + 58 = 100"  ✅  (used add_numbers tool)

result = graph.invoke(
    {"messages": [HumanMessage(content="What's the weather in Karachi?")]},
    config=config
)
print(result["messages"][-1].content)
# → "The weather in Karachi is sunny and 28°C."  ✅
