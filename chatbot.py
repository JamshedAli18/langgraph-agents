from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="groq/compound-mini")

from langgraph.graph.message import add_messages
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: State) -> State:

    messages = state["messages"]

    response = model.invoke(messages)

    return {"messages": [response]}


from langgraph.graph import StateGraph, START, END

checkpointer = InMemorySaver()
graph = StateGraph(State)

graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")

graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer)

thread_id = '1'

while True:

    user_input = input("You: ")

    print("You: ", user_input)

    if user_input.strip().lower() in ["exit", "quit"]:
        break

    config = {"configurable": {"thread_id": thread_id}}

    response = chatbot.invoke({'messages': [HumanMessage(content = user_input)]},config=config)

    print("Bot: ", response['messages'][-1].content)
