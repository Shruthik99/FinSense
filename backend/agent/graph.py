from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    node_analyze_spending,
    node_fetch_live_data,
    node_retrieve_knowledge,
    node_generate_roast,
    node_generate_coach_plan
)


def create_finsense_graph():
    """
    Creates the LangGraph agent reasoning graph.
    Defines the flow: analyze → fetch → retrieve → roast → coach
    """
    # Initialize the graph with our state schema
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("analyze_spending", node_analyze_spending)
    graph.add_node("fetch_live_data", node_fetch_live_data)
    graph.add_node("retrieve_knowledge", node_retrieve_knowledge)
    graph.add_node("generate_roast", node_generate_roast)
    graph.add_node("generate_coach_plan", node_generate_coach_plan)

    # Define the flow — each node leads to the next
    graph.set_entry_point("analyze_spending")
    graph.add_edge("analyze_spending", "fetch_live_data")
    graph.add_edge("fetch_live_data", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "generate_roast")
    graph.add_edge("generate_roast", "generate_coach_plan")
    graph.add_edge("generate_coach_plan", END)

    # Compile the graph
    return graph.compile()


# The main agent instance
finsense_agent = create_finsense_graph()


def run_agent(budget_input: dict) -> AgentState:
    """
    Runs the full agent pipeline on a budget input.
    Returns the complete state with all results.
    """
    from agent.state import create_initial_state
    initial_state = create_initial_state(budget_input)
    final_state = finsense_agent.invoke(initial_state)
    return final_state