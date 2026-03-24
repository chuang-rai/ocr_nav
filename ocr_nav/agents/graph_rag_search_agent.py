from pathlib import Path

from pydantic_ai import Agent

from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.skills.graph_rag_search_skills import _parse_skills_md
from ocr_nav.tools.graph_rag_tools import (
    GraphRAGDeps,
    execute_cypher_query,
    execute_python_code,
    semantic_search_in_graph,
    visualize_nodes_edges,
)

# ------------------------------------------------------------------ #
# Agent construction
# ------------------------------------------------------------------ #

_SKILLS_MD_PATH = Path(__file__).parent.parent.parent / "ocr_nav" / "skills" / "graph_rag_search_skills.md"


def build_agent(
    graph_rag: BaseGraphRAG, model_name: str = "google-gla:gemini-3.1-flash-lite-preview"
) -> Agent[GraphRAGDeps, int]:
    """Build a PydanticAI agent with the same GraphRAG tools as the Gemini version."""

    tool_list = [execute_cypher_query, execute_python_code, semantic_search_in_graph, visualize_nodes_edges]

    # Reuse the system prompt authored in the companion markdown file
    system_prompt, _ = _parse_skills_md(_SKILLS_MD_PATH)
    existing_node_type_str = "Existing node types:" + ",".join(graph_rag.get_existing_node_types())
    existing_edge_type_str = "Existing edge types:" + ",".join(graph_rag.get_existing_rel_types())
    system_prompt += "\n" + existing_node_type_str + "\n" + existing_edge_type_str

    agent: Agent[GraphRAGDeps, int] = Agent(
        model_name, system_prompt=system_prompt, deps_type=GraphRAGDeps, retries=3, output_type=int
    )

    # -- Tool definitions ------------------------------------------------ #
    for tool in tool_list:
        agent.tool(tool)

    return agent
