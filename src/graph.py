from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from preprocessing.restructure_prompt import rewrite_prompt
from preprocessing.web_scraping import get_result, download_images
from typing import Literal
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    origin_query: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    restructure_query: Annotated[list[AnyMessage], operator.add]
    image_list: Annotated[list[AnyMessage], operator.add]

def rewrite_prompt_node(state: dict):
    return rewrite_prompt(state.origin_query[-1])

def get_http_result_node(state: dict):
    return get_result(state.restructure_query[-1])

def download_images_node(state: dict):
    return download_images(state.image_list[-1])

def preprocess_image_node(state: dict):
    return 

def select_best_candidate_node(state: dict):
    return

def generate_image_node(state: dict):
    return

def run_trellis_node(state: dict):
    return 

def evaluate_images():
    return 

# Conditional logic
def should_continue(state: GraphState) -> Literal[""]:
    return

def main():
    # Build workflow
    agent_builder = StateGraph(GraphState)

    # Add nodes
    agent_builder.add_node("rewrite_prompt_node", rewrite_prompt_node) 
    agent_builder.add_node("get_http_result_node", get_http_result_node)
    agent_builder.add_node("download_images_node", download_images_node)
    agent_builder.add_node("preprocesss_image_node", preprocess_image_node)
    agent_builder.add_node("select_best_candidate_node", select_best_candidate_node)
    agent_builder.add_node("generate_image_node", generate_image_node)
    agent_builder.add_node("run_trellis_node", run_trellis_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "rewrite_prompt_node")
    agent_builder.add_edge("rewrite_prompt_node", "get_http_result_node")
    agent_builder.add_edge("get_http_result_node", "download_images_node")
    agent_builder.add_conditional_edges("download_images_node", should_continue, ["select_best_candidate_node", "generate_image_node"])
    agent_builder.add_edge("select_best_candidate_node", "preprocesss_image_node")
    agent_builder.add_edge("generate_image_node", "preprocesss_image_nodes")
    agent_builder.add_edge("preprocess_image_nodes", "run_trellis_node")
    agent_builder.add_edge("run_trellis_node", END)

    # Compile the agent
    agent = agent_builder.compile()

    # Show the agent
    from IPython.display import Image, display
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    # Invoke
    from langchain.messages import HumanMessage
    messages = [HumanMessage(content="Add 3 and 4.")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()-