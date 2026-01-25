import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_model = ChatGroq(model="llama-3.1-8b-instant")

class LLMWorkflow(TypedDict):
    question: str
    answer: str


def llm_workflow(state: LLMWorkflow) -> LLMWorkflow:

    question = state["question"]

    prompt = f"Answer the following question: {question}"

    answer = groq_model.invoke(prompt).content

    state["answer"] = answer

    return state



graph = StateGraph(state_schema=LLMWorkflow)

graph.add_node("llm_workflow", llm_workflow)


graph.add_edge(START, "llm_workflow")
graph.add_edge("llm_workflow", END)

workflow = graph.compile()




initail_state = {"question": "What is the capital of France? and tell me why it is the capital"}

final_state = workflow.invoke(initail_state)

print(final_state['answer'])
