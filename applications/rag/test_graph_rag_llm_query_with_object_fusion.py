import yaml
from pathlib import Path
from omegaconf import OmegaConf
from termcolor import cprint
from google.genai import types
from rai_ai_core_library.utils import dynamic_model
from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.skills.graph_rag_search_skills import GraphRAGSearchSkills


def main():
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_path = config_dir / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())
    bag_path = Path(args.bag_path)
    annotation_dir = bag_path.parent / "qwen3vl_annotations_fast_8b"
    rgb_dir = bag_path.parent / "rgb"
    rgb_paths = sorted(list(rgb_dir.iterdir()))

    graph_rag_path = bag_path.parent / "graph_rag"
    graph_rag = BaseGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")
    # visualize_graphrag(graph_rag, graph_rag_path.parent)

    gemini_config_path = config_dir / "llm" / "gemini_plus.yaml"
    with open(gemini_config_path, "r") as file:
        gemini_config = yaml.safe_load(file)
    device = "cuda"
    gemini = dynamic_model(gemini_config, device=device)

    # Initialize skills
    skills = GraphRAGSearchSkills(graph_rag, output_dir=bag_path.parent)
    system_prompt = skills.get_system_prompt()
    tools = skills.get_tools()

    graph_rag.build_node_index("Object", "embedding", metric="cosine", mu=16, efc=200)
    k = 1
    while k != ord("q"):
        query_text = input("Enter your query (or 'q' to quit): ")
        if query_text == "q":
            break

        history = []
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=query_text)]))
        for _ in range(20):  # limit the number of retrieval and reasoning iterations to prevent infinite loop

            response = gemini.client.models.generate_content(
                model=gemini.model_name,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=tools,
                ),
            )

            # Check if the model returned a function call
            function_call_part = None
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_call_part = part
                    break

            if function_call_part is None:
                # No function call — this is the final text answer
                cprint(f"Model: Final answer: {response.text}", "green")
                break

            # Append the model's full content to history (preserves thought signatures automatically)
            history.append(response.candidates[0].content)

            fn_name = function_call_part.function_call.name
            fn_args = dict(function_call_part.function_call.args)
            fn_response = skills.execute(fn_name, fn_args, query_text=query_text)
            cprint(f"Executed function '{fn_name}' with args {fn_args}, got response: \n{fn_response}", "yellow")

            # Append function response to history
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fn_name, response=fn_response)],
                )
            )


if __name__ == "__main__":
    main()
