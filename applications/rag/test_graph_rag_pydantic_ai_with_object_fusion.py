"""
PydanticAI-based agentic system for Graph RAG querying with object fusion.

This is a PydanticAI reimplementation of the query logic in
test_graph_rag_llm_query_with_object_fusion.py.  PydanticAI manages the
tool-calling loop automatically, so we only need to declare tools and deps.
"""

from pathlib import Path

import yaml
from omegaconf import OmegaConf
from termcolor import cprint

from ocr_nav.agents.graph_rag_search_agent import GraphRAGDeps, build_agent
from ocr_nav.rag.graph_rag import BaseGraphRAG


def main():
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_path = config_dir / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())
    bag_path = Path(args.bag_path)

    # Initialise GraphRAG
    graph_rag_path = bag_path.parent / "graph_rag_new"
    graph_rag = BaseGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")

    # Read the Gemini model name from the existing config and map to PydanticAI format
    gemini_config_path = config_dir / "vlm" / "gemini_plus.yaml"
    with open(gemini_config_path, "r") as f:
        gemini_config = yaml.safe_load(f)
    pydantic_ai_model = f"google-gla:{gemini_config['params']['model_name']}"

    agent = build_agent(graph_rag, pydantic_ai_model)

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
        cprint(f"Model: Final answer: {result.output}", "green")


if __name__ == "__main__":
    main()
