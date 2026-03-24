"""Graph RAG search skills for Claude tool use.

Skill definitions (system prompt + tool schemas) live in the companion markdown file
``graph_rag_search_skills.md``. This module parses that file at init time and
provides execution handlers for each declared tool.

Usage:
    from ocr_nav.skills.graph_rag_search_skills import GraphRAGSearchSkills

    skills = GraphRAGSearchSkills(graph_rag, output_dir=bag_path.parent)
    tools = skills.get_tools()
    system_prompt = skills.get_system_prompt()

    # In the agent loop:
    fn_response = skills.execute(fn_name, fn_args)
"""

import json
import re
from pathlib import Path

from termcolor import cprint

from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.utils.rag_utils import visualize_nodes_edges

_SKILLS_MD_PATH = Path(__file__).with_suffix(".md")


def _resolve_file_links(text: str, md_dir: Path) -> str:
    """Resolve markdown file links and inline their contents.

    Finds patterns like ``[description](relative/path.py)`` where the path
    points to an existing file relative to ``md_dir``, and replaces the link
    with the file's contents wrapped in a fenced code block.
    """

    def _replace_link(match: re.Match) -> str:
        description = match.group(1)
        rel_path = match.group(2)
        resolved = (md_dir / rel_path).resolve()
        if resolved.is_file():
            content = resolved.read_text()
            suffix = resolved.suffix.lstrip(".")
            return f"{description}\n```{suffix}\n{content}\n```"
        # Not a file link — leave as-is
        return match.group(0)

    return re.sub(r"\[([^\]]+)\]\(([^)]+\.[a-zA-Z0-9]+)\)", _replace_link, text)


def _parse_skills_md(md_path: Path) -> tuple[str, list[dict]]:
    """Parse a skills markdown file into a system prompt and tool definitions.

    Returns:
        (system_prompt, tool_defs) where each tool_def is a dict with keys:
        name, description, input_schema (a JSON Schema object).
    """
    text = md_path.read_text()

    # Resolve file links (e.g. [desc](../rag/graph_rag.py) → inlined source)
    text = _resolve_file_links(text, md_path.parent)

    # --- Extract system prompt: everything before "## Tools" ---
    tools_heading_match = re.search(r"^## Tools\s*$", text, re.MULTILINE)
    if tools_heading_match:
        system_prompt_section = text[: tools_heading_match.start()].strip()
    else:
        system_prompt_section = text.strip()

    # Strip the top-level heading (e.g. "# Graph RAG Search Skills")
    system_prompt_section = re.sub(r"^# .+\n+", "", system_prompt_section).strip()

    # --- Extract tool definitions: each ### heading under ## Tools ---
    tool_defs = []
    if tools_heading_match:
        tools_section = text[tools_heading_match.end() :]
        # Split by ### headings
        tool_blocks = re.split(r"^### +", tools_section, flags=re.MULTILINE)
        for block in tool_blocks:
            block = block.strip()
            if not block:
                continue
            # First line is the tool name
            lines = block.split("\n", 1)
            tool_name = lines[0].strip()
            tool_body = lines[1].strip() if len(lines) > 1 else ""

            # Extract JSON Schema from fenced ```json code block
            schema_match = re.search(r"```json\s*\n(.*?)\n```", tool_body, re.DOTALL)
            if schema_match:
                description = tool_body[: schema_match.start()].strip()
                # Remove trailing **Input Schema:** label if present
                description = re.sub(r"\*\*Input Schema:\*\*\s*$", "", description).strip()
                input_schema = json.loads(schema_match.group(1))
            else:
                description = tool_body.strip()
                input_schema = {"type": "object", "properties": {}, "required": []}

            tool_defs.append(
                {
                    "name": tool_name,
                    "description": description,
                    "input_schema": input_schema,
                }
            )

    return system_prompt_section, tool_defs


class GraphRAGSearchSkills:
    """Encapsulates Claude tool-use definitions for querying a GraphRAG.

    Skill definitions are loaded from ``graph_rag_search_skills.md`` next to this file.
    The markdown contains the system prompt text and a ``## Tools`` section with one
    ``### tool_name`` subsection per tool, each with a JSON Schema block.

    Args:
        graph_rag: An initialized ``BaseGraphRAG`` instance.
        output_dir: Optional directory for visualization outputs.
        skills_md_path: Override path to the skills markdown file.
    """

    def __init__(
        self,
        graph_rag: BaseGraphRAG,
        output_dir: str | Path | None = None,
        skills_md_path: str | Path | None = None,
    ):
        self.graph_rag = graph_rag
        self.output_dir = Path(output_dir) if output_dir is not None else None

        # Load and parse the markdown skill definitions
        md_path = Path(skills_md_path) if skills_md_path else _SKILLS_MD_PATH
        self._system_prompt_template, self._tool_defs = _parse_skills_md(md_path)

        # Registry mapping function name → handler method
        self._handlers: dict[str, callable] = {
            "execute_cypher_query": self._handle_execute_cypher_query,
            "execute_python_code": self._handle_execute_python_code,
            "semantic_search_in_graph": self._handle_semantic_search_in_graph,
            "visualize_nodes_edges": self._handle_visualize_nodes_edges,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_system_prompt(self) -> str:
        """Return the system prompt (file links already resolved at parse time)."""
        return self._system_prompt_template

    def get_tools(self) -> list[dict]:
        """Return tool definitions compatible with Claude's tool use API."""
        return self._tool_defs

    def execute(self, fn_name: str, fn_args: dict, query_text: str = "") -> dict:
        """Dispatch a function call to the appropriate handler.

        Args:
            fn_name: The function name from the ``tool_use`` content block ``name``.
            fn_args: The arguments dict from the ``tool_use`` content block ``input``.
            query_text: The original user query (used by visualization).

        Returns:
            A dict suitable for a Claude ``tool_result`` content block.
        """
        handler = self._handlers.get(fn_name)
        if handler is None:
            cprint(f"Unknown function call: {fn_name}", "red")
            return {"error": f"Unknown function: {fn_name}"}
        return handler(fn_args, query_text=query_text)

    # ------------------------------------------------------------------ #
    # Execution handlers
    # ------------------------------------------------------------------ #

    def _handle_execute_cypher_query(self, args: dict, **kwargs) -> dict:
        cypher_query = args["query"]
        cprint(f"Executing Cypher query: {cypher_query}", "cyan")
        retrieval_result = self.graph_rag.execute_cypher_query(cypher_query)
        if "Error" in str(retrieval_result):
            return {"error": str(retrieval_result)}
        return {"result": str(retrieval_result)}

    def _handle_execute_python_code(self, args: dict, **kwargs) -> dict:
        python_code = args["code"]
        cprint(f"Executing Python code:\n{python_code}", "cyan")
        local_vars = {"graph_rag": self.graph_rag}
        try:
            exec(python_code, {}, local_vars)
            result = local_vars.get("retrieval_result", None)
            return {"result": str(result)}
        except Exception as e:
            cprint(f"Error executing Python code: {e}", "red")
            return {"error": str(e)}

    def _handle_semantic_search_in_graph(self, args: dict, **kwargs) -> dict:
        search_query = args["query"]
        top_k = max(args["top_k"], 5)  # always retrieve at least 5 nodes
        cprint(f"Performing semantic search with query: {search_query}, top_k: {top_k}", "cyan")
        try:
            obj_score_tuples = self.graph_rag.retrieve_node_and_score_by_query(
                "Object", search_query, "embedding", top_k=top_k
            )
            return {"result": str([(x[0]["id"], x[0]["labels"], x[1]) for x in obj_score_tuples])}
        except Exception as e:
            cprint(f"Error performing semantic search: {e}", "red")
            return {"error": str(e)}

    def _handle_visualize_nodes_edges(self, args: dict, query_text: str = "", **kwargs) -> dict:
        if self.output_dir is None:
            return {"error": "output_dir not set — cannot visualize"}
        try:
            node_list = eval(args["node_list"])
            edge_list = eval(args["edge_list"])
            cprint(f"Visualizing nodes: {node_list} and edges: {edge_list}", "cyan")
            visualize_nodes_edges(self.graph_rag, self.output_dir, query_text, node_list, edge_list)
            return {"result": "Visualization generated successfully."}
        except Exception as e:
            cprint(f"Error visualizing nodes/edges: {e}", "red")
            raise e
            return {"error": str(e)}
