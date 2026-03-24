import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib import colors
from pydantic_ai import Agent, RunContext
from termcolor import cprint
from tqdm import tqdm

from ocr_nav.agents.graph_rag_search_agent import GraphRAGDeps
from ocr_nav.agents.graph_rag_search_agent import build_agent as build_graph_rag_search_agent
from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.scene_graph.floor_graph import FloorGraph
from ocr_nav.skills.graph_rag_search_skills import _parse_skills_md
from ocr_nav.utils.pyvista_vis_utils import (
    create_plotter,
    draw_cube,
    draw_line,
    draw_point_cloud,
)

_SKILLS_MD_PATH = Path(__file__).parent.parent.parent / "ocr_nav" / "skills" / "environment_inspection_skills.md"


@dataclass
class EnvInspectionDeps:
    """Runtime dependencies available to every tool invocation."""

    graph_rag: BaseGraphRAG
    output_dir: Path
    floor_graph: FloorGraph
    query_text: str
    current_position: tuple[float, float, float] | np.ndarray


def build_agent(model_name: str = "google-gla:gemini-3.1-flash-lite-preview") -> Agent[EnvInspectionDeps, str]:
    """Build a PydanticAI agent with the same GraphRAG tools as the Gemini version."""

    # Reuse the system prompt authored in the companion markdown file
    system_prompt, _ = _parse_skills_md(_SKILLS_MD_PATH)

    agent: Agent[EnvInspectionDeps, str] = Agent(
        model_name,
        system_prompt=system_prompt,
        deps_type=EnvInspectionDeps,
        retries=3,
    )

    @agent.tool
    async def search_object_pos(ctx: RunContext[EnvInspectionDeps], query: str) -> str:
        """Search for an object in the graph and return its position as a potential goal for nav_to function."""
        graph_rag_agent = build_graph_rag_search_agent(ctx.deps.graph_rag, model_name)

        # append to the system prompt of the agent
        @graph_rag_agent.system_prompt
        def add_final_output_type() -> None:
            return "\n" + "The output should be the id of the most relevant object node in the graph as an integer."

        # define the output type
        graph_rag_deps = GraphRAGDeps(
            graph_rag=ctx.deps.graph_rag,
            output_dir=ctx.deps.output_dir,
            query_text=query,
        )
        result = await graph_rag_agent.run(query, deps=graph_rag_deps)
        if "Error" in str(result):
            return f"Error: {result}"

        obj_id = result.output
        if not isinstance(obj_id, int) and not (isinstance(obj_id, str) and obj_id.isdigit()):
            cprint(f"Error: Expected an integer object ID, but got: {obj_id}", "red")
            return f"Error: Expected an integer object ID, but got: {obj_id}"
        obj_id = int(obj_id)
        cprint(f"Most relevant object node ID: {obj_id}", "green")
        obj_pc_path = ctx.deps.graph_rag.kuzu_db_dir / "object_point_clouds" / f"object_{obj_id:05d}.ply"
        pc = o3d.io.read_point_cloud(obj_pc_path.as_posix())
        central_pos = np.mean(np.array(pc.points), axis=0)
        cprint(f"Object {obj_id} central position: {central_pos}", "green")
        return str(central_pos.tolist())

    # -- Tool definitions ------------------------------------------------ #

    @agent.tool
    def nav_to(ctx: RunContext[EnvInspectionDeps], goal_pos: tuple[float, float, float], obj_id: int) -> str:
        """Navigate to a goal position in the environment."""
        cprint(f"Going to goal position: {goal_pos}", "cyan")
        floor_graph = ctx.deps.floor_graph
        start_pos = np.array(ctx.deps.current_position)
        goal_pos = np.array(goal_pos)
        path_pos = floor_graph.plan_global_path(start_pos, goal_pos)
        if "Error" in str(path_pos):
            return f"Error: {path_pos}"

        # visualize the planned path
        # plot the starting node and the goal node
        print("Plotting voronoi graphs...")
        plotter = create_plotter()
        obj_pc_path = ctx.deps.graph_rag.kuzu_db_dir / "object_point_clouds" / f"object_{obj_id:05d}.ply"
        pc = o3d.io.read_point_cloud(obj_pc_path.as_posix())
        plotter = draw_point_cloud(plotter, np.asarray(pc.points), color="orange", point_size=5.0)
        for edge in tqdm(floor_graph.floor_graph.edges()):
            src, tar = edge
            src_pos = floor_graph.floor_graph.nodes[src]["pos"]
            tar_pos = floor_graph.floor_graph.nodes[tar]["pos"]
            line = draw_line(
                np.array([src_pos[0], src_pos[1], src_pos[2]]),
                np.array([tar_pos[0], tar_pos[1], tar_pos[2]]),
            )
            plotter.add_mesh(line, line_width=4, color="green", render_lines_as_tubes=True)

        plotter.show(interactive_update=True)
        start_cube = draw_cube(start_pos, size=0.5, color="blue")
        plotter.add_mesh(start_cube, color="blue", name="start_cube")
        goal_cube = draw_cube(goal_pos, size=0.5, color="red")
        plotter.add_mesh(goal_cube, color="red", name="goal_cube")

        # plot the path
        pbar = tqdm(range(len(path_pos) - 1))
        for i in pbar:
            line = draw_line(
                np.array([path_pos[i][0], path_pos[i][1], path_pos[i][2]]),
                np.array([path_pos[i + 1][0], path_pos[i + 1][1], path_pos[i + 1][2]]),
            )
            plotter.add_mesh(line, line_width=10, color="green", render_lines_as_tubes=True)
            plotter.update()
        plotter.show()

        return str(path_pos)

    return agent
