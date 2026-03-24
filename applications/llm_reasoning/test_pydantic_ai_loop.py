from pathlib import Path
from unittest import result

import numpy as np
from omegaconf import OmegaConf
from pydantic_ai import Agent, RunContext
from termcolor import cprint
from tqdm import tqdm

from ocr_nav.agents.environment_inspection_agent import EnvInspectionDeps, build_agent
from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.scene_graph.floor_graph import FloorGraph
from ocr_nav.utils.pyvista_vis_utils import create_plotter, draw_line, draw_sphere


def main():
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_path = config_dir / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())
    bag_path = Path(args.bag_path)
    plotter = create_plotter()

    # Initialise GraphRAG
    graph_rag_path = bag_path.parent / "graph_rag"
    graph_rag = BaseGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")
    floor_graph = FloorGraph(args.voxel_size)
    floor_graph.load_floor_graph(Path(args.output_dir) / f"{args.output_graph_name}.json")
    # print("Plotting voronoi graphs...")
    # for edge in tqdm(floor_graph.floor_graph.edges()):
    #     src, tar = edge
    #     src_pos = floor_graph.floor_graph.nodes[src]["pos"]
    #     tar_pos = floor_graph.floor_graph.nodes[tar]["pos"]
    #     line = draw_line(
    #         np.array([src_pos[0], src_pos[1], src_pos[2]]),
    #         np.array([tar_pos[0], tar_pos[1], tar_pos[2]]),
    #     )
    #     plotter.add_mesh(line, line_width=4, color="green", render_lines_as_tubes=True)
    # for node in tqdm(floor_graph.floor_graph.nodes()):
    #     node_pos = floor_graph.floor_graph.nodes[node]["pos"]
    #     sphere = draw_sphere(
    #         np.array([node_pos[0], node_pos[1], node_pos[2]]),
    #         radius=0.1,
    #     )
    #     plotter.add_mesh(sphere, color="green")

    # plotter.show()

    # Read the Gemini model name from the existing config and map to PydanticAI format
    gemini_config_path = config_dir / "vlm" / "gemini_plus.yaml"
    gemini_config = OmegaConf.load(gemini_config_path.as_posix())
    pydantic_ai_model = f"google-gla:{gemini_config['params']['model_name']}"

    env_inspection_agent = build_agent(pydantic_ai_model)

    # randomly select a start node for testing
    np.random.seed(1)
    nodes = list(floor_graph.floor_graph.nodes(data=True))
    start_node = nodes[np.random.randint(0, len(nodes))][0]
    start_pos = np.array(floor_graph.floor_graph.nodes[start_node]["pos"])

    env_inspection_deps = EnvInspectionDeps(
        graph_rag=graph_rag,
        floor_graph=floor_graph,
        output_dir=bag_path.parent,
        query_text="",
        current_position=start_pos,
    )

    while True:
        query_text = input("Enter your query (or 'q' to quit): ")
        if query_text.strip().lower() == "q":
            break
        env_inspection_deps.query_text = query_text
        result = env_inspection_agent.run_sync(query_text, deps=env_inspection_deps)
        cprint(f"Model: Final answer: {result.output}", "green")

    # graph_rag.build_node_index("Object", "embedding", metric="cosine", mu=16, efc=200)
    # agent.run_sync()


if __name__ == "__main__":
    main()
