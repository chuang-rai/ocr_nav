from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import RunContext
from termcolor import cprint

from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.skills.graph_rag_search_skills import _SKILLS_MD_PATH, _parse_skills_md
from ocr_nav.utils.rag_utils import visualize_nodes_edges as _visualize_nodes_edges


@dataclass
class GraphRAGDeps:
    """Runtime dependencies available to every tool invocation."""

    graph_rag: BaseGraphRAG
    output_dir: Path
    query_text: str = ""


def execute_cypher_query(ctx: RunContext[GraphRAGDeps], query: str) -> str:
    """Execute a Cypher query against the Kuzu graph database and return the results.
    Use this to query the graph for nodes, relationships, and properties."""
    cprint(f"Executing Cypher query: {query}", "cyan")
    result = ctx.deps.graph_rag.execute_cypher_query(query)
    if "Error" in str(result):
        return f"Error: {result}"
    return str(result)


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


def semantic_search_in_graph(ctx: RunContext[GraphRAGDeps], query: str, top_k: int) -> str:
    """Perform a semantic search in the graph using the provided query text.
    Use the graph_rag's built-in retrieval methods to find relevant nodes.
    top_k is clamped to a minimum of 5 to ensure sufficient results."""
    top_k = max(top_k, 5)  # always retrieve at least 5 nodes
    cprint(f"Performing semantic search with query: {query}, top_k: {top_k}", "cyan")
    try:
        obj_score_tuples = ctx.deps.graph_rag.retrieve_node_and_score_by_query(
            "Object", query, "embedding", top_k=top_k
        )
        return str([(x[0]["id"], x[0]["labels"], x[1]) for x in obj_score_tuples])
    except Exception as e:
        cprint(f"Error performing semantic search: {e}", "red")
        return f"Error: {e}"


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
