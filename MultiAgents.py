import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── State ─────────────────────────────────────────────────
class AgentState(MessagesState):
    next_agent: str   # "researcher" | "writer" | "coder" | "FINISH"

# ── LLM ──────────────────────────────────────────────────
llm = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])

# ── Supervisor Node ───────────────────────────────────────
# Reads the user request and decides which worker to call
def supervisor_node(state: AgentState) -> dict:
    user_message = state["messages"][-1].content

    prompt = f"""You are a supervisor. Route this task to the right worker.
Workers available:
- researcher: for finding facts, answering questions
- writer: for writing essays, emails, or content
- coder: for writing or explaining code

Task: "{user_message}"

Reply with ONLY one word: researcher, writer, or coder."""

    result = llm.invoke([HumanMessage(content=prompt)])
    next_agent = result.content.strip().lower()

    if next_agent not in ["researcher", "writer", "coder"]:
        next_agent = "researcher"

    print(f"[supervisor] Routing to: {next_agent}")
    return {"next_agent": next_agent}

# ── Worker Nodes ──────────────────────────────────────────
def researcher_node(state: AgentState) -> dict:
    system = SystemMessage(content=(
        "You are a research specialist. "
        "Answer questions with clear, factual information."
    ))
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response], "next_agent": "FINISH"}

def writer_node(state: AgentState) -> dict:
    system = SystemMessage(content=(
        "You are a writing specialist. "
        "Write clear, engaging, well-structured content."
    ))
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response], "next_agent": "FINISH"}

def coder_node(state: AgentState) -> dict:
    system = SystemMessage(content=(
        "You are a coding specialist. "
        "Write clean, well-commented code with explanations."
    ))
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response], "next_agent": "FINISH"}

# ── Router — supervisor decides next step ─────────────────
def route_to_agent(state: AgentState) -> str:
    return state["next_agent"]   # "researcher", "writer", or "coder"

# ── Build Graph ───────────────────────────────────────────
builder = StateGraph(AgentState)

# Register nodes
builder.add_node("supervisor",  supervisor_node)
builder.add_node("researcher",  researcher_node)
builder.add_node("writer",      writer_node)
builder.add_node("coder",       coder_node)

# Flow
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_to_agent)

# All workers end the graph when done
builder.add_edge("researcher", END)
builder.add_edge("writer",     END)
builder.add_edge("coder",      END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ── Test ──────────────────────────────────────────────────
config = {"configurable": {"thread_id": "multi_agent_test"}}

# Test 1 — should route to researcher
result = graph.invoke(
    {"messages": [HumanMessage(content="What is quantum computing?")]},
    config=config
)
print(result["messages"][-1].content)
# [supervisor] Routing to: researcher
# → "Quantum computing is..."

# Test 2 — should route to coder
result = graph.invoke(
    {"messages": [HumanMessage(content="Write a Python function to reverse a string")]},
    config={"configurable": {"thread_id": "code_test"}}
)
print(result["messages"][-1].content)
# [supervisor] Routing to: coder
# → "def reverse_string(s): ..."
