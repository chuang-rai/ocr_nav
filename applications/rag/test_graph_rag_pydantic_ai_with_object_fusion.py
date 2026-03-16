"""
PydanticAI-based agentic system for Graph RAG querying with object fusion.

This is a PydanticAI reimplementation of the query logic in
test_graph_rag_llm_query_with_object_fusion.py.  PydanticAI manages the
tool-calling loop automatically, so we only need to declare tools and deps.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf
from pydantic_ai import Agent, RunContext
from termcolor import cprint

from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.skills.graph_rag_search_skills import _parse_skills_md, _SKILLS_MD_PATH
from ocr_nav.utils.rag_utils import visualize_nodes_edges as _visualize_nodes_edges


# ------------------------------------------------------------------ #
# Dependencies – injected into every tool call via RunContext
# ------------------------------------------------------------------ #


@dataclass
class GraphRAGDeps:
    """Runtime dependencies available to every tool invocation."""

    graph_rag: BaseGraphRAG
    output_dir: Path
    query_text: str = ""


# ------------------------------------------------------------------ #
# Agent construction
# ------------------------------------------------------------------ #


def build_agent(model_name: str = "google-gla:gemini-3.1-flash-lite-preview") -> Agent[GraphRAGDeps, str]:
    """Build a PydanticAI agent with the same GraphRAG tools as the Gemini version."""

    # Reuse the system prompt authored in the companion markdown file
    system_prompt, _ = _parse_skills_md(_SKILLS_MD_PATH)

    agent: Agent[GraphRAGDeps, str] = Agent(
        model_name,
        system_prompt=system_prompt,
        deps_type=GraphRAGDeps,
        retries=3,
    )

    # -- Tool definitions ------------------------------------------------ #

    @agent.tool
    def execute_cypher_query(ctx: RunContext[GraphRAGDeps], query: str) -> str:
        """Execute a Cypher query against the Kuzu graph database and return the results.
        Use this to query the graph for nodes, relationships, and properties."""
        cprint(f"Executing Cypher query: {query}", "cyan")
        result = ctx.deps.graph_rag.execute_cypher_query(query)
        if "Error" in str(result):
            return f"Error: {result}"
        return str(result)

    @agent.tool
    def execute_python_code(ctx: RunContext[GraphRAGDeps], code: str) -> str:
        """Execute Python code that interacts with the `graph_rag` object (a BaseGraphRAG instance).
        The code has access to a `graph_rag` variable.
        Store the result in a variable named `retrieval_result`."""
        cprint(f"Executing Python code:\n{code}", "cyan")
        local_vars = {"graph_rag": ctx.deps.graph_rag}
        try:
            exec(code, {}, local_vars)
            result = local_vars.get("retrieval_result", None)
            return str(result)
        except Exception as e:
            cprint(f"Error executing Python code: {e}", "red")
            return f"Error: {e}"

    @agent.tool
    def semantic_search_in_graph(ctx: RunContext[GraphRAGDeps], query: str, top_k: int) -> str:
        """Perform a semantic search in the graph using the provided query text.
        Use the graph_rag's built-in retrieval methods to find relevant nodes."""
        cprint(f"Performing semantic search with query: {query}", "cyan")
        try:
            obj_score_tuples = ctx.deps.graph_rag.retrieve_node_and_score_by_query(
                "Object", query, "embedding", top_k=top_k
            )
            return str([(x[0]["id"], x[0]["labels"], x[1]) for x in obj_score_tuples])
        except Exception as e:
            cprint(f"Error performing semantic search: {e}", "red")
            return f"Error: {e}"

    @agent.tool
    def visualize_nodes_edges(ctx: RunContext[GraphRAGDeps], node_list: str, edge_list: str) -> str:
        """Visualize the results of a graph RAG query.
        node_list: A Python literal for a list of tuples, e.g. '[("Object", 1), ("Frame", 2)]'.
            Each tuple is (node_type: str, node_id: int).
        edge_list: A Python literal for a list of tuples, e.g. '[("Frame", 0, "CONTAINS", "Object", 1)]'.
            Each tuple is (src_type, src_id, rel_type, tgt_type, tgt_id).
        When the visualized node type is Object, remember to also add its related nodes."""
        try:
            nodes = eval(node_list)
            edges = eval(edge_list)
            cprint(f"Visualizing nodes: {nodes} and edges: {edges}", "cyan")
            _visualize_nodes_edges(ctx.deps.graph_rag, ctx.deps.output_dir, ctx.deps.query_text, nodes, edges)
            return "Visualization generated successfully."
        except Exception as e:
            cprint(f"Error visualizing nodes/edges: {e}", "red")
            return f"Error: {e}"

    return agent


# ------------------------------------------------------------------ #
# Main interactive loop
# ------------------------------------------------------------------ #


def main():
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_path = config_dir / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())
    bag_path = Path(args.bag_path)

    # Initialise GraphRAG
    graph_rag_path = bag_path.parent / "graph_rag"
    graph_rag = BaseGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")

    # Read the Gemini model name from the existing config and map to PydanticAI format
    gemini_config_path = config_dir / "llm" / "gemini_plus.yaml"
    with open(gemini_config_path, "r") as f:
        gemini_config = yaml.safe_load(f)
    pydantic_ai_model = f"google-gla:{gemini_config['params']['model_name']}"

    agent = build_agent(pydantic_ai_model)

    # Pre-build the vector index once
    graph_rag.build_node_index("Object", "embedding", metric="cosine", mu=16, efc=200)

    # Interactive query loop (mirrors the original while-loop)
    while True:
        query_text = input("Enter your query (or 'q' to quit): ")
        if query_text.strip().lower() == "q":
            break

        deps = GraphRAGDeps(
            graph_rag=graph_rag,
            output_dir=bag_path.parent,
            query_text=query_text,
        )

        result = agent.run_sync(query_text, deps=deps)
        cprint(f"Model: Final answer: {result.data}", "green")


if __name__ == "__main__":
    main()
