from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="groq/compound-mini")

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

from langgraph.graph import END,START
from langgraph.graph import StateGraph

def gen_joke(state:JokeState):

    prompt = f"generate a joke about given topic{state['topic']}"

    response = model.invoke(prompt).content

    return {"joke":response}

def explain_joke(state:JokeState):

    prompt = f"explain the joke in simple words and short {state['joke']}"

    response = model.invoke(prompt).content

    return {"explanation": response}


checkpointer = InMemorySaver()

graph = StateGraph(JokeState)

graph.add_node('gen_joke',gen_joke)
graph.add_node('explain_joke',explain_joke)


graph.add_edge(START,'gen_joke')
graph.add_edge('gen_joke','explain_joke')
graph.add_edge('explain_joke',END)

workflow = graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id":"1"}}

initial_state = {
    "topic": "AI" 
}

print(workflow.invoke(initial_state,config=config1))

list(workflow.get_state_history(config=config1))

config2 = {"configurable": {"thread_id": "2" }}
print(workflow.invoke({"topic":"pastsa"},config=config2))
