import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from typing import TypedDict,Annotated

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_model = ChatGroq(model="llama-3.1-8b-instant")

class PromptChainState(TypedDict):
    topic : str
    outline: str
    content: str
    rating : float

def generate_outline(state: PromptChainState) -> PromptChainState:
    topic = state["topic"]
    outline = groq_model.invoke(f"Generate a single outline from the topic: {topic}").content
    
    return state | {"outline": outline}

def generate_content(state: PromptChainState) -> PromptChainState:
    topic = state["topic"]
    outline = state["outline"]
    content = groq_model.invoke(f"Generate content for blog on the topic: {topic} with outline: {outline}").content
    
    return state | {"content": content}

def generate_rating(state: PromptChainState) -> PromptChainState:
    #genrating rating from 1 to 10 based on writing

    rating = groq_model.invoke(f"Generate rating for blog on the topic: {state['topic']} with outline: {state['outline']} and content: {state['content']}").content
    
    return state | {"rating": rating}


graph = StateGraph(PromptChainState)

graph.add_node("generate_outline",generate_outline)
graph.add_node("generate_content",generate_content)
graph.add_node("generate_rating",generate_rating)

graph.add_edge(START,"generate_outline")
graph.add_edge("generate_outline","generate_content")
graph.add_edge("generate_content","generate_rating")
graph.add_edge("generate_rating",END)

graph = graph.compile()

initial_state = {"topic": "AI"}

for s in graph.stream(initial_state):
    print(s)
