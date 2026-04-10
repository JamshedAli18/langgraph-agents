from langgraph.checkpoint.postgres import PostgresSaver

DB_URL = "postgresql://user:password@localhost:5432/mydb"

with PostgresSaver.from_conn_string(DB_URL) as memory:
    # One-time setup — creates the tables LangGraph needs
    memory.setup()

    graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "ali_chat"}}

    graph.invoke(
        {"messages": [HumanMessage(content="My name is Ali")]},
        config=config
    )
