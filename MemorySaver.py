import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ── LLM ──────────────────────────────────────────────────
llm = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])

# ── Node ──────────────────────────────────────────────────
def chat_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are a friendly assistant.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

# ── Build Graph ───────────────────────────────────────────
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# ── Add Checkpointer ──────────────────────────────────────
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)  # ← only change needed!

# ── Run multi-turn conversation ───────────────────────────
config = {"configurable": {"thread_id": "ali_chat"}}

# Turn 1
result = graph.invoke(
    {"messages": [HumanMessage(content="Hi! My name is Ali.")]},
    config=config      # ← pass config every time
)
print(result["messages"][-1].content)
# → "Hi Ali! Nice to meet you!"

# Turn 2 — graph REMEMBERS Ali from turn 1
result = graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config=config      # ← same thread_id = same conversation
)
print(result["messages"][-1].content)
# → "Your name is Ali!"  ✅
