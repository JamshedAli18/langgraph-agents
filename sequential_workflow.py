from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class BMI_calculator(TypedDict):
    weight: float
    height: float
    bmi : float
    label: str


def bmi_calculator(state: BMI_calculator) -> BMI_calculator:

    weight_kg = state['weight']
    height_m = state['height']

    bmi = weight_kg / (height_m ** 2)
    
    state['bmi'] = round(bmi, 2)

    return state


def bmi_label(state: BMI_calculator) -> BMI_calculator:
    bmi = state["bmi"]

    if bmi < 18.5:
        state["label"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["label"] = "Fit"
    elif 25 <= bmi < 30:
        state["label"] = "Overweight"
    else:
        state["label"] = "Obese"

    return state


graph = StateGraph(state_schema=BMI_calculator)

graph.add_node('bmi_calculator', bmi_calculator)
graph.add_node('bmi_label', bmi_label)

graph.add_edge(START, 'bmi_calculator')
graph.add_edge('bmi_calculator', 'bmi_label')
graph.add_edge('bmi_label', END)

workflow = graph.compile()

initail_state = {"weight": 100.3, "height": 1}

final_state = workflow.invoke(initail_state)

