from __future__ import annotations
from typing import TYPE_CHECKING
import os

import numpy as np
from pathlib import Path
import open3d as o3d
from pyvis.network import Network
import cv2

if TYPE_CHECKING:
    from ocr_nav.rag.graph_rag import BaseGraphRAG


def convert_type_to_kuzu_type(value):
    if value in [int, np.int8, np.int16, np.int32, np.int64]:
        return "INT"
    elif value in [float, np.float32, np.float64]:
        return "DOUBLE"
    elif value is str:
        return "STRING"
    elif value is bool:
        return "BOOL"
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Only support fixed-length tuple of size 2")
        assert isinstance(value[1], int), "The second element of the tuple must be an integer representing the length"
        elem_type = convert_type_to_kuzu_type(value[0])
        return f"{elem_type}[{value[1]}]"
    elif isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Cannot determine the type of an empty list")
        elem_type = convert_type_to_kuzu_type(value[0])
        return f"{elem_type}[]"
    else:
        raise ValueError(f"Unsupported type: {type(value)}")


def visualize_graphrag(graph_rag: BaseGraphRAG, root_path: str | Path, show_img: bool = False):
    root_path = Path(root_path)
    rgb_dir = root_path / "rgb"
    rgb_paths = sorted(list(rgb_dir.iterdir()))
    # traverse through all the node types
    node_types = graph_rag.get_existing_node_types()
    edge_types = graph_rag.get_existing_rel_types()
    net = Network(height="1080px", width="100%", bgcolor="#222222", font_color="white")
    added_node_ids_set = set()
    for node_type in node_types:
        nodes = graph_rag.get_all_nodes_of_type(node_type)
        print(f"Node type: {node_type}, Number of nodes: {len(nodes)}")
        for node in nodes:
            if node_type == "Object":
                net.add_node(node_type + "_" + str(node["id"]), label=node["label"], title=node["attributes"])
            elif node_type == "Frame":
                net.add_node(
                    node_type + "_" + str(node["id"]),
                    label=f"Frame {node['id']}",
                    image=rgb_paths[node["id"]].as_posix() if show_img else None,
                    shape="image" if show_img else "ellipse",
                    title=f"Frame {node['id']}",
                )
            elif node_type == "Bbox":
                net.add_node(
                    node_type + "_" + str(node["id"]),
                    label=f"Bbox {node['id']}",
                    title=str(node["bbox"]),
                )

    for edge_type in edge_types:
        relationships = graph_rag.get_all_rels_of_type(edge_type)
        print(f"Edge type: {edge_type}, Number of edges: {len(relationships)}")
        for rel in relationships:
            src_node, rel, tar_node = rel
            print(rel)
            src_node_type = src_node["_label"]
            src_node_id = src_node["id"]
            tar_node_type = tar_node["_label"]
            tar_node_id = tar_node["id"]
            net.add_edge(
                src_node_type + "_" + str(src_node_id),
                tar_node_type + "_" + str(tar_node_id),
            )
    net.show_buttons(filter_=["nodes", "physics", "selection"])

    # 4. Fine-tune the "Obsidian" interaction
    net.set_options(
        """
    var options = {
    "nodes": {
        "borderWidth": 2,
        "color": {
        "highlight": { "background": "#ff0000", "border": "#ffffff" }
        }
    },
    "interaction": {
        "hover": true,
        "navigationButtons": true,
        "multiselect": true
    }
    }
    """
    )
    net.show((root_path / "graph_rag_visualization.html").as_posix(), notebook=False)


def visualize_graphrag_query_result(
    graph_rag: BaseGraphRAG,
    root_path: str | Path,
    query_text: str,
    query_result_obj_nodes: list[dict],
    show_img: bool = True,
):
    root_path = Path(root_path)
    rgb_dir = root_path / "rgb"
    rgb_paths = sorted(list(rgb_dir.iterdir()))
    query_text_str = query_text.replace(" ", "_").replace("/", "_")
    query_result_dir = root_path / "query_results" / query_text_str
    os.makedirs(query_result_dir, exist_ok=True)
    # traverse through all the node types
    node_types = graph_rag.get_existing_node_types()
    edge_types = graph_rag.get_existing_rel_types()
    net = Network(height="1080px", width="100%", bgcolor="#222222", font_color="white")
    added_node_ids_set = set()
    same_node_edge_map = {}
    for obj_node in query_result_obj_nodes:
        net.add_node(
            "Object_" + str(obj_node["id"]),
            label=",".join(obj_node["labels"]),
            title=f"Object_{obj_node['id']}: {obj_node['attributes']}",
        )
        obj_rel_tuples = graph_rag.retrieve_related_nodes_with_tar_node("Object", obj_node["id"])
        print(len(obj_rel_tuples))
        for obj_rel_tuple in obj_rel_tuples:
            if obj_rel_tuple[1]["_label"] == "IsSame":
                print(
                    f"Found IsSame relationship between Object_{obj_rel_tuple[0]['id']} and Object_{obj_rel_tuple[2]['id']}"
                )
                same_node_edge_map[obj_rel_tuple[0]["id"]] = obj_rel_tuple[2]["id"]
                continue

            box_node = obj_rel_tuple[0]
            box_id = box_node["id"]
            bbox = box_node["bbox"]
            frame_rel_bbox_tuple = graph_rag.retrieve_related_nodes_with_tar_node("Bbox", box_id)[0]
            frame_node = frame_rel_bbox_tuple[0]
            frame_id = frame_node["id"]
            bgr = cv2.imread(rgb_paths[frame_id].as_posix())
            cropped_bgr = bgr[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            box_path = query_result_dir / f"bbox_{box_id}.jpg"
            cv2.imwrite(box_path.as_posix(), cropped_bgr)

            net.add_node(
                "Bbox_" + str(box_id),
                label=f"Bbox {box_id}",
                image=box_path.as_posix() if show_img else None,
                shape="image" if show_img else "ellipse",
                title=str(box_node["bbox"]),
            )
            net.add_node(
                "Frame_" + str(frame_id),
                label=f"Frame {frame_id}",
                image=rgb_paths[frame_id].as_posix() if show_img else None,
                shape="image" if show_img else "ellipse",
                title=f"Frame {frame_id}",
            )
            net.add_edge("Object_" + str(obj_node["id"]), "Bbox_" + str(box_id))
            net.add_edge("Bbox_" + str(box_id), "Frame_" + str(frame_id))
    for src_id, tar_id in same_node_edge_map.items():
        nodes = net.get_nodes()
        if "Object_" + str(src_id) not in nodes:
            print("src node is not added")
            obj_node = graph_rag.retrieve_node_by_id("Object", src_id)
            net.add_node(
                "Object_" + str(obj_node["id"]),
                label=",".join(obj_node["labels"]),
                title=f"Object_{obj_node['id']}: {obj_node['attributes']}",
            )
        if "Object_" + str(tar_id) not in nodes:
            print("tar node is not added")
            obj_node = graph_rag.retrieve_node_by_id("Object", tar_id)
            net.add_node(
                "Object_" + str(obj_node["id"]),
                label=",".join(obj_node["labels"]),
                title=f"Object_{obj_node['id']}: {obj_node['attributes']}",
            )
        try:
            net.add_edge("Object_" + str(src_id), "Object_" + str(tar_id), color="red", title="IsSame")
        except:
            print(
                f"Failed to add IsSame edge between Object_{src_id} and Object_{tar_id}. They might already be connected with another IsSame edge."
            )
            pass

    net.show_buttons(filter_=["physics"])
    net.show((query_result_dir / f"graph_rag_query_result.html").as_posix(), notebook=False)


def visualize_nodes_edges(
    graph_rag: BaseGraphRAG,
    root_path: str | Path,
    query_text: str,
    node_list: list[tuple[str, int]],
    edge_list: list[tuple[str, int, str, int]],
) -> None:
    root_path = Path(root_path)
    rgb_dir = root_path / "rgb"
    pc_path = root_path / "glim_map.ply"
    map_pc = o3d.io.read_point_cloud(pc_path.as_posix())
    map_pc.paint_uniform_color([0.5, 0.5, 0.5])
    pc_dir = graph_rag.kuzu_db_dir / "object_point_clouds"
    rgb_paths = sorted(list(rgb_dir.iterdir()))
    query_text_str = query_text.replace(" ", "_").replace("/", "_").replace("?", "")
    query_result_dir = graph_rag.kuzu_db_dir / "query_results" / query_text_str
    os.makedirs(query_result_dir, exist_ok=True)
    net = Network(height="1080px", width="100%", bgcolor="#222222", font_color="white")
    obj_pc_list = []
    for node_type, node_id in node_list:
        node = graph_rag.retrieve_node_by_id(node_type, node_id)
        if node_type == "Object":
            obj_pc = o3d.io.read_point_cloud((pc_dir / f"object_{node_id:05d}.ply").as_posix())
            obj_pc.paint_uniform_color(np.random.rand(3))
            obj_pc_list.append(obj_pc)
            object_node = graph_rag.retrieve_node_by_id(node_type, node_id)
            net.add_node(
                node_type + "_" + str(node["id"]),
                label=",".join(node["labels"]),
                title=f"Object_{node['id']}: {node['attributes']}",
            )
            related_bbox_nodes = graph_rag.retrieve_related_nodes_with_tar_node("Object", node_id)
            for bbox_node, rel, tar_node in related_bbox_nodes:
                bbox = bbox_node["bbox"]
                if bbox is None:
                    continue
                frame_rel_bbox_tuple = graph_rag.retrieve_related_nodes_with_tar_node("Bbox", bbox_node["id"])[0]
                frame_node = frame_rel_bbox_tuple[0]
                frame_id = frame_node["id"]
                bgr = cv2.imread(rgb_paths[frame_id].as_posix())
                cropped_bgr = bgr[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                box_path = query_result_dir / f"bbox_{bbox_node['id']}.jpg"
                cv2.imwrite(box_path.as_posix(), cropped_bgr)
                net.add_node(
                    "Bbox_" + str(bbox_node["id"]),
                    label=f"Bbox {bbox_node['id']}",
                    image=box_path.as_posix(),
                    shape="image",
                    title=str(bbox_node["bbox"]),
                )
                net.add_node(
                    "Frame_" + str(frame_id),
                    label=f"Frame {frame_id}",
                    image=rgb_paths[frame_id].as_posix(),
                    shape="image",
                    title=f"Frame {frame_id}",
                )
                net.add_edge(node_type + "_" + str(node["id"]), "Bbox_" + str(bbox_node["id"]))
                net.add_edge("Bbox_" + str(bbox_node["id"]), "Frame_" + str(frame_id))
        elif node_type == "Frame":
            net.add_node(
                node_type + "_" + str(node["id"]),
                label=f"Frame {node['id']}",
                image=rgb_paths[node["id"]].as_posix(),
                shape="image",
                title=f"Frame {node['id']}",
            )
        elif node_type == "Bbox":
            bbox = node["bbox"]
            frame_rel_bbox_tuple = graph_rag.retrieve_related_nodes_with_tar_node("Bbox", node_id)[0]
            frame_node = frame_rel_bbox_tuple[0]
            frame_id = frame_node["id"]
            bgr = cv2.imread(rgb_paths[frame_id].as_posix())
            cropped_bgr = bgr[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            box_path = query_result_dir / f"bbox_{node['id']}.jpg"
            cv2.imwrite(box_path.as_posix(), cropped_bgr)
            net.add_node(
                node_type + "_" + str(node["id"]),
                label=f"Bbox {node['id']}",
                title=str(node["bbox"]),
                image=box_path.as_posix(),
                shape="image",
            )

    for src_type, src_id, rel_type, tar_type, tar_id in edge_list:
        color = "red" if rel_type == "IsSame" else "blue"
        net.add_edge(src_type + "_" + str(src_id), tar_type + "_" + str(tar_id), title=rel_type, color=color)
    net.show_buttons(filter_=["physics"])
    net.show((query_result_dir / f"graph_rag_query_result.html").as_posix(), notebook=False)
    o3d.visualization.draw_geometries(obj_pc_list + [map_pc])
