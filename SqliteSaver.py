import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver       # ← different import
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])

def chat_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are a friendly assistant.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# ── SqliteSaver — saves to a file called memory.db ────────
with SqliteSaver.from_conn_string("memory.db") as memory:
    graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "ali_chat"}}

    # Turn 1
    graph.invoke(
        {"messages": [HumanMessage(content="My name is Ali.")]},
        config=config
    )

    # Turn 2 — even if you restart the program, Ali is remembered
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config
    )
    print(result["messages"][-1].content)
    # → "Your name is Ali!"  ✅
