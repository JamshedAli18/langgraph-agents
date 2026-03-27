import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from typing import TypedDict,Annotated

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_model = ChatGroq(model="llama-3.1-8b-instant")

class AnalysisState(TypedDict):
    topic : str
    pros : str
    cons : str
    risks : str
    final_report : str

def analyze_pros(state:AnalysisState):
    topic = state["topic"]
    prompt = f"Analyze the pros of {topic} and generate only 2 pros"
    response = groq_model.invoke(prompt)
    return {"pros":response.content}

def analyze_cons(state:AnalysisState):
    topic = state["topic"]
    prompt = f"Analyze the cons of {topic} and generate only 2 cons"
    response = groq_model.invoke(prompt)
    return {"cons":response.content}

def analyze_risks(state:AnalysisState):
    topic = state["topic"]
    prompt = f"Analyze the risks of {topic} and generate only 2 risks"
    response = groq_model.invoke(prompt)
    return {"risks":response.content}

def generate_final_report(state:AnalysisState):
    topic = state["topic"]
    pros = state["pros"]
    cons = state["cons"]
    risks = state["risks"]
    prompt = f"Generate a final report in bullet points on {topic} with pros: {pros}, cons: {cons}, risks: {risks}"
    response = groq_model.invoke(prompt)
    return {"final_report":response.content}



graph = StateGraph(AnalysisState)

graph.add_node("pros",analyze_pros)
graph.add_node("cons",analyze_cons)
graph.add_node("risks",analyze_risks)
graph.add_node("final_report",generate_final_report)

graph.add_edge(START,"pros")
graph.add_edge(START,"cons")
graph.add_edge(START,"risks")
graph.add_edge("pros",END)
graph.add_edge("cons",END)
graph.add_edge("risks",END)

workflow = graph.compile()

initial_state = {"topic": "Langchain"}

for s in workflow.stream(initial_state):
    print(s)

