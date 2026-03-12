import os
import copy
from time import time
from collections import defaultdict
import shutil
import json
import numpy as np
import open3d as o3d
import cv2
import rclpy
from omegaconf import OmegaConf
from pathlib import Path
from rai_ai_core_library.utils import dynamic_model
from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.matcher.xfeat_matcher import XFeatMatcher
from ocr_nav.matcher.pc_associator import GlobalFragmentAssociator
from ocr_nav.utils.io_utils import BagIO
from ocr_nav.utils.visualization_utils import visualize_masks_on_image
from ocr_nav.utils.mapping_utils import (
    select_points_in_masks_batch,
    project_points,
    transform_point_cloud,
    to_numpy_pc,
    to_o3d_pc,
)


def init_sam(sam_config_path: str, device: str):
    sam_config = OmegaConf.load(sam_config_path)
    sam_model = dynamic_model(sam_config, device=device)
    print("Successfully loaded SAM model.")
    return sam_model


def compute_object_embedding_with_annotation(annotation: dict, embedding_model) -> np.ndarray:
    obj = annotation["label"]
    attrs = annotation.get("attributes", {})
    color = attrs.get("color", "")
    material = attrs.get("material", "")
    material_str = f"made of {material}" if material else ""
    function = attrs.get("function", "")
    function_str = f" for {function}" if function else ""
    narrative = f"A {color} {obj} {material_str}{function_str}."
    embedding = embedding_model.encode(narrative)
    return embedding


def scale_bbox(bbox_raw, w, h, normalize_max=1000):
    """Scale a normalized bounding box to pixel coordinates."""
    bbox = np.array(bbox_raw)
    bbox[0] = int(bbox[0] * w / normalize_max)
    bbox[1] = int(bbox[1] * h / normalize_max)
    bbox[2] = int(bbox[2] * w / normalize_max)
    bbox[3] = int(bbox[3] * h / normalize_max)
    return bbox


def add_bbox_and_link_to_frame(graph_rag: BaseGraphRAG, frame_id: int, global_bbox_id: int, bbox: list[int]):
    """Add a Bbox node and link it to a Frame (shared by both branches)."""
    graph_rag.add_node("Bbox", {"id": global_bbox_id, "bbox": bbox})
    graph_rag.add_relationship(
        "FrameContainsBbox",
        "Frame",
        frame_id,
        "Bbox",
        global_bbox_id,
        {"from_id": frame_id, "to_id": global_bbox_id},
    )


def update_object_with_new_annotation(
    graph_rag: BaseGraphRAG,
    object_id: int,
    new_object_anno: dict,
):
    obj_node = graph_rag.retrieve_node_by_id("Object", object_id)
    new_properties = {}
    label_set = set(obj_node["labels"])
    if new_object_anno["label"] not in label_set:
        label_set.add(new_object_anno["label"])
    new_properties["labels"] = list(label_set)

    updated_anno = {"label": "|".join(new_properties["labels"]), "attributes": {}}
    existing_attrs = obj_node["attributes"]
    new_attrs = []
    for attr in existing_attrs:
        key = attr.split(":")[0]
        old_value = attr.split(":")[1].strip()
        updated_attr = attr
        if key in new_object_anno["attributes"]:
            new_value = new_object_anno["attributes"][key]
            old_value_set = set([v.strip() for v in old_value.split("|")])
            if new_value not in old_value_set:
                updated_attr = f"{key}: {old_value} | {new_value}"
                updated_anno["attributes"][key] = f"{old_value} | {new_value}"
        new_attrs.append(updated_attr)

    new_properties["attributes"] = new_attrs

    embedding = compute_object_embedding_with_annotation(updated_anno, graph_rag.embedding_model)
    new_properties["embedding"] = embedding.tolist()

    graph_rag.update_node("Object", object_id, new_properties)


def link_bbox_to_object(
    graph_rag: BaseGraphRAG,
    global_bbox_id: int,
    object_id: int,
):
    """Add a BboxAssociatedWithObject relationship."""
    # update the object node's labels and attributes if needed

    graph_rag.add_relationship(
        "BboxAssociatedWithObject",
        "Bbox",
        global_bbox_id,
        "Object",
        object_id,
        {"from_id": global_bbox_id, "to_id": object_id},
    )


def merge_pc_and_resample(
    pc1: o3d.geometry.PointCloud, pc2: o3d.geometry.PointCloud, voxel_size: float | None = None
) -> o3d.geometry.PointCloud:
    """Merge two point clouds and resample to a fixed number of points."""
    merged_pc = pc1 + pc2
    if voxel_size is not None:
        merged_pc = merged_pc.voxel_down_sample(voxel_size)
    return merged_pc


def get_masked_pc_list_in_cam_frame(
    bagio: BagIO, masks: np.ndarray, lidar_list: list[np.ndarray], use_lidar_id: int
) -> list[o3d.geometry.PointCloud]:
    lidar2camera_tfs = bagio.get_lidar2camera_tfs()
    # tf_camera2anchor_lidar = np.linalg.inv(lidar2camera_tfs[bagio.anchor_lidar_id])
    tf_lidar2camera = lidar2camera_tfs[use_lidar_id]
    tf_camera2lidar = np.linalg.inv(lidar2camera_tfs[use_lidar_id])
    intr_mat = bagio.get_intrinsics()
    h, w = bagio.get_image_size()

    lidar_pc = lidar_list[use_lidar_id]
    lidar_pc_in_cam_frame = transform_point_cloud(lidar_pc, tf_lidar2camera)
    pts2d, _, pts3d = project_points(lidar_pc_in_cam_frame, intr_mat, w, h)
    masked_pc_list = select_points_in_masks_batch(masks, pts2d, pts3d)
    return masked_pc_list


def create_new_object(graph_rag, global_obj_id, obj_ann, embedding_list, attrs_list):
    """Create a new Object node."""
    graph_rag.add_node(
        "Object",
        {
            "id": global_obj_id,
            "labels": [obj_ann["label"]],
            "embedding": embedding_list,
            "attributes": attrs_list,
        },
    )


def add_object_pc(object_pc_dict: dict, object_id: int, pc: o3d.geometry.PointCloud, pose: np.ndarray = np.eye(4)):
    """Add a point cloud to the global object point cloud dictionary."""
    transformed_pc = copy.deepcopy(pc).transform(pose)
    if object_id in object_pc_dict:
        existing_pc = object_pc_dict[object_id]
        merged_pc = merge_pc_and_resample(existing_pc, transformed_pc)
        object_pc_dict[object_id] = merged_pc
    else:
        object_pc_dict[object_id] = transformed_pc


def add_fragment_pc(
    global_fragment_associator: GlobalFragmentAssociator,
    fragment_id: int,
    pc: o3d.geometry.PointCloud,
    pose: np.ndarray = np.eye(4),
):
    """Add a point cloud to the global fragment point cloud dictionary."""
    transformed_pc = copy.deepcopy(pc).transform(pose)
    if fragment_id in global_fragment_associator.global_fragment_pc_history:
        voxel_grids = global_fragment_associator.convert_pc_to_hierarchical_voxels(transformed_pc)
        global_fragment_associator.update_fragment(fragment_id, transformed_pc, voxel_grids)
    else:
        global_fragment_associator.add_new_fragment(transformed_pc, fragment_id)


def associate_fragments(
    to_be_associated_ids_list: list[int],
    global_fragment_associator: GlobalFragmentAssociator,
    early_stop_iou_threshold: float = 0.1,
):
    id_map = {}
    print(
        f"Trying to associate fragments {to_be_associated_ids_list} with historical fragments in the global fragment associator..."
    )
    for i in range(len(to_be_associated_ids_list)):
        to_be_associated_id = to_be_associated_ids_list[i]
        # to_be_associated_pc = copy.deepcopy(global_fragment_associator.global_fragment_pc_history[to_be_associated_id])
        associated_id = global_fragment_associator.associate_fragment_with_id(
            to_be_associated_id, early_stop_threshold=early_stop_iou_threshold
        )
        if associated_id is not None:
            id_map[to_be_associated_id] = associated_id

            # visualize associated fragments in open3d
            global_fragment_associator.global_fragment_pc_history[associated_id].paint_uniform_color(np.random.rand(3))
            print(f"Fragment {to_be_associated_id} is associated with fragment {associated_id}")
            # to_be_associated_pc.paint_uniform_color(np.random.rand(3))
            # o3d.visualization.draw_geometries(
            #     [
            #         global_fragment_associator.global_fragment_pc_history[associated_id],
            #         to_be_associated_pc,
            #     ]
            # )
    return global_fragment_associator, id_map


def try_associate_with_past(
    graph_rag: BaseGraphRAG, sim_mat: np.ndarray, bbox_id: int, last_global_bbox_ids: list[int], threshold: float = 0.8
):
    """
    Try to find a matching past object via similarity matrix.
    Returns associated_object_id and best_score if matched, else (None, best_score).
    """
    sim_scores = sim_mat[:, bbox_id]
    best_match_idx = np.argmax(sim_scores)
    best_score = sim_scores[best_match_idx]

    if best_score <= threshold:
        return None, best_score

    try:
        associated_bbox_id = last_global_bbox_ids[best_match_idx]
    except IndexError:
        print(
            f"IndexError: best_match_idx {best_match_idx} out of range for "
            f"last_global_bbox_ids with length {len(last_global_bbox_ids)}"
        )
        return None, best_score

    related_src_rel_tar_tuples = graph_rag.retrieve_related_nodes_with_src_node("Bbox", associated_bbox_id)
    if len(related_src_rel_tar_tuples) == 0:
        return None, best_score
    associated_object_id = related_src_rel_tar_tuples[0][2]["id"]
    return associated_object_id, best_score


def augment_bbox_list_with_unmatched(
    matcher: XFeatMatcher,
    sim_mat: np.ndarray,
    kpts_mask1_list: list[np.ndarray],
    last_bbox_list: list[np.ndarray],
    threshold: float = 0.2,
):
    """Propagate unmatched bboxes from the previous frame into the current bbox list."""
    score_last = np.sum(sim_mat, axis=1)
    not_matched_mask_ids = np.where(score_last < threshold)[0]
    not_in_anno_bbox_list_unnormalized = []
    not_in_anno_bbox_ids = []
    for not_matched_mask_id in not_matched_mask_ids:
        kpts_mask = kpts_mask1_list[not_matched_mask_id]
        if np.sum(kpts_mask) < 10:
            continue
        kps_in_curr_img = matcher.kps2[kpts_mask]
        x_min, y_min = np.min(kps_in_curr_img, axis=0)
        x_max, y_max = np.max(kps_in_curr_img, axis=0)
        bbox_curr_img = np.round([x_min, y_min, x_max, y_max]).astype(int).tolist()
        not_in_anno_bbox_list_unnormalized.append(bbox_curr_img)
        not_in_anno_bbox_ids.append(not_matched_mask_id)
        sim_score = np.zeros((len(last_bbox_list), 1), dtype=np.float32)
        sim_score[not_matched_mask_id, 0] = 1.0
        sim_mat = np.hstack((sim_mat, sim_score))
    return sim_mat, not_in_anno_bbox_list_unnormalized, not_in_anno_bbox_ids


def setup_graph_rag(bag_path, graph_rag_name, cache_segmentation=True):
    """Initialize graph RAG with schema definitions."""
    graph_rag_path = bag_path.parent / graph_rag_name

    graph_rag_pc_dir = graph_rag_path / "object_point_clouds"
    if graph_rag_pc_dir.exists():
        shutil.rmtree(graph_rag_pc_dir)
    os.makedirs(graph_rag_pc_dir, exist_ok=True)

    if cache_segmentation:
        bag_name = bag_path.stem
        cache_dir = graph_rag_path / (bag_name + "_semantic_seg")
        os.makedirs(cache_dir, exist_ok=True)
    graph_rag = BaseGraphRAG(graph_rag_path.as_posix(), overwrite=True, embedding_model_name="BAAI/bge-m3")
    graph_rag.define_node_type("Object", {"id": int, "labels": [str], "embedding": (float, 1024), "attributes": [str]})
    graph_rag.define_node_type("Bbox", {"id": int, "bbox": (int, 4)})
    graph_rag.define_node_type("Frame", {"id": int, "timestamp": str, "caption": str, "pose": (float, 16)})
    graph_rag.define_relationship_type("FrameContainsBbox", "Frame", "Bbox", {"from_id": int, "to_id": int})
    graph_rag.define_relationship_type("BboxAssociatedWithObject", "Bbox", "Object", {"from_id": int, "to_id": int})
    graph_rag.define_relationship_type("IsSame", "Object", "Object", {"from_id": int, "to_id": int})
    return graph_rag, graph_rag_pc_dir, cache_dir if cache_segmentation else None


def setup_bagio(bag_path, args):
    """Initialize the BagIO reader."""
    bagio = BagIO(
        bag_path.as_posix(),
        rgb_topic=args.bagio.rgb_topic,
        camera_info_topic=args.bagio.camera_info_topic,
        camera_frame_id=args.bagio.camera_frame_id,
        anchor_lidar_id=args.bagio.anchor_lidar_id,
        lidar_topic_list=args.bagio.lidar_topic_list,
        lidar_frame_ids=args.bagio.lidar_frame_ids,
        world_frame_id=args.bagio.world_frame_id,
        max_bag_total_time=args.bagio.max_bag_total_time_sec,
        sample_every=args.bagio.sample_every,
    )
    bagio.init_reader()
    return bagio


def main():
    config_path = Path(__file__).parent.parent.parent / "config" / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())

    sam_config_path = Path(__file__).parent.parent.parent / "config" / "segmentation" / "segment_anything.yaml"
    sam = init_sam(sam_config_path, device="cuda")

    bag_path = Path(args.bag_path)
    annotation_dir = bag_path.parent / "qwen3vl_annotations_fast_8b"
    anno_paths = sorted(list(annotation_dir.iterdir()))
    timelist = [int(anno_path.stem.split("_")[1]) for anno_path in anno_paths]

    anchor_lidar_id = args.bagio.anchor_lidar_id
    os.makedirs(annotation_dir, exist_ok=True)
    rclpy.init()

    bagio = setup_bagio(bag_path, args)
    graph_rag, graph_rag_pc_dir, cache_dir = setup_graph_rag(
        bag_path, graph_rag_name="graph_rag", cache_segmentation=args.cache_segmentation
    )
    xfeat_match_dir = graph_rag.kuzu_db_dir / "xfeat_matcher"
    os.makedirs(xfeat_match_dir, exist_ok=True)
    matcher = XFeatMatcher(top_k=4096)

    global_bbox_id = 0
    global_obj_id = 0
    last_img_np = None
    last_bbox_list = None
    last_global_bbox_ids = None
    global_obj_pc_dict = {}  # global_object_id -> point cloud
    global_obj_tracks_dict = defaultdict(list)  # global_object_id -> list of (frame_id, bbox)
    global_id2associator_id_map = {}
    global_fragment_associator = GlobalFragmentAssociator(threshold=0.3, voxel_size_list=[0.1, 0.2, 0.5, 1.0])

    while bagio.has_next():
        data = bagio.get_next_sync_data()
        try:
            lidar_list, img_np, anchor_lidar_pose, t_nanosec = data
        except TypeError:
            print("Data format error, skipping this frame.")
            continue
        w, h = img_np.shape[1], img_np.shape[0]
        tf_anchor_lidar2camera = bagio.get_lidar2camera_tfs()[args.bagio.anchor_lidar_id]
        tf_camera2anchor_lidar = np.linalg.inv(tf_anchor_lidar2camera)
        if t_nanosec not in timelist:
            continue

        frame_id = timelist.index(t_nanosec)
        with open(anno_paths[frame_id], "r") as f:
            annotation = json.load(f)

        # Add Frame node
        graph_rag.add_node(
            "Frame",
            {
                "id": frame_id,
                "timestamp": str(t_nanosec),
                "caption": annotation.get("description", ""),
                "pose": anchor_lidar_pose.flatten().tolist(),
            },
        )

        bbox_list = [np.array(obj_ann.get("bounding_box", [])) for obj_ann in annotation["objects"]]
        bbox_list = [scale_bbox(bbox, w, h, normalize_max=1000).tolist() for bbox in bbox_list]
        bbox_anno_list = [obj_ann["label"] for obj_ann in annotation["objects"]]
        sim_mat = None

        # Compute similarity with previous frame
        if last_bbox_list is not None:
            sim_mat, kpts_mask1_list = matcher.compute_bbox_similarities(
                last_img_np,
                img_np,
                last_bbox_list,
                bbox_list,
                w,
                h,
                normalize_max=None,
                vis=False,
                box_label_list_1=last_bbox_anno_list,
                save_vis_path=(xfeat_match_dir / f"xfeat_{frame_id}.jpg").as_posix(),
            )
            sim_mat, not_in_anno_bbox_list_unnormalized, not_in_anno_bbox_ids = augment_bbox_list_with_unmatched(
                matcher, sim_mat, kpts_mask1_list, last_bbox_list, threshold=0.2
            )
            bbox_list.extend(not_in_anno_bbox_list_unnormalized)
            bbox_anno_list.extend([last_bbox_anno_list[i] for i in not_in_anno_bbox_ids])
        if args.cache_segmentation:
            mask_path = cache_dir / f"mask_{t_nanosec}.npy"
            if not mask_path.exists():
                masks = sam.predict(bbox_list, img_np)
                np.save(mask_path, masks)
            else:
                masks = np.load(mask_path)
        else:
            masks = sam.predict(bbox_list, img_np)

        # masked_img = visualize_masks_on_image(img_np, masks)
        # cv2.imshow("Masked Image", masked_img)
        # cv2.waitKey(1)
        use_lidar_id = 1
        masked_pc_list = get_masked_pc_list_in_cam_frame(bagio, masks, lidar_list, use_lidar_id)
        # Process each detected object
        global_bbox_ids = []
        global_association_time_list = []
        local_association_time_list = []
        for bbox_id, bbox in enumerate(bbox_list):
            add_bbox_and_link_to_frame(graph_rag, frame_id, global_bbox_id, bbox)
            obj_ann = {}
            if bbox_id < len(annotation["objects"]):
                obj_ann = annotation["objects"][bbox_id]

            if sim_mat is not None:
                assert sim_mat.shape[0] == len(last_bbox_list) and sim_mat.shape[1] == len(bbox_list), (
                    f"Similarity matrix shape {sim_mat.shape} does not match bbox list sizes "
                    f"{len(last_bbox_list)} and {len(bbox_list)}"
                )
                associated_object_id, best_score = try_associate_with_past(
                    graph_rag, sim_mat, bbox_id, last_global_bbox_ids, threshold=0.8
                )
                if associated_object_id is not None:
                    # Matched: reuse existing object
                    if obj_ann:
                        update_object_with_new_annotation(graph_rag, associated_object_id, obj_ann)
                    link_bbox_to_object(graph_rag, global_bbox_id, associated_object_id)
                    add_object_pc(
                        global_obj_pc_dict,
                        associated_object_id,
                        masked_pc_list[bbox_id],
                        pose=anchor_lidar_pose @ tf_camera2anchor_lidar,
                    )
                    st = time()
                    add_fragment_pc(
                        global_fragment_associator,
                        associated_object_id,
                        masked_pc_list[bbox_id],
                        pose=anchor_lidar_pose @ tf_camera2anchor_lidar,
                    )
                    et = time()
                    # print(f"Time taken to add fragment pc for object {associated_object_id}: {et - st:.2f} seconds")
                    local_association_time_list.append(et - st)
                    global_obj_tracks_dict[associated_object_id].append((frame_id, bbox))
                    global_bbox_ids.append(global_bbox_id)
                    global_bbox_id += 1
                    continue

            # associate with historical fragments when tracks are lost
            # print(f"Time taken to associate with past fragments for bbox {bbox_id}: {et - st:.2f} seconds")

            # add new object if no match found
            embedding = compute_object_embedding_with_annotation(obj_ann, graph_rag.embedding_model)
            embedding_list = embedding.tolist()
            attrs_list = [f"{k}: {v}" for k, v in obj_ann["attributes"].items()]
            create_new_object(graph_rag, global_obj_id, obj_ann, embedding_list, attrs_list)
            link_bbox_to_object(graph_rag, global_bbox_id, global_obj_id)
            add_object_pc(
                global_obj_pc_dict,
                global_obj_id,
                masked_pc_list[bbox_id],
                pose=anchor_lidar_pose @ tf_camera2anchor_lidar,
            )
            print(f"adding fragment {global_obj_id}")
            st = time()
            add_fragment_pc(
                global_fragment_associator,
                global_obj_id,
                masked_pc_list[bbox_id],
                pose=anchor_lidar_pose @ tf_camera2anchor_lidar,
            )
            et = time()

            global_obj_tracks_dict[global_obj_id].append((frame_id, bbox))
            global_bbox_ids.append(global_bbox_id)
            global_bbox_id += 1
            global_obj_id += 1

        st = time()
        lost_track_ids = [
            obj_id
            for obj_id, tracks in global_obj_tracks_dict.items()
            if len(tracks) > 0 and tracks[-1][0] == frame_id - 1
        ]
        global_fragment_associator, id_map = associate_fragments(
            lost_track_ids, global_fragment_associator, early_stop_iou_threshold=0.1
        )
        for src_id, associated_id in id_map.items():
            src_node = graph_rag.retrieve_node_by_id("Object", src_id)
            associated_node = graph_rag.retrieve_node_by_id(
                "Object", associated_id
            )  # check if the associated object still exists in the graph

            print(
                f"Trying to associate lost track object {src_node['labels']} {src_id} with fragment, got associated fragment id {associated_node['labels']} {associated_id}"
            )
            graph_rag.add_relationship(
                "IsSame",
                "Object",
                src_id,
                "Object",
                associated_id,
                {"from_id": src_id, "to_id": associated_id},
            )
        et = time()
        global_association_time_list.append(et - st)

        last_img_np = img_np
        last_bbox_list = bbox_list
        last_bbox_anno_list = bbox_anno_list
        last_global_bbox_ids = global_bbox_ids

        # global_fragment_associator, id_map = associate_fragments(
        #     global_obj_pc_dict, global_fragment_associator, early_stop_iou_threshold=0.1
        # )

    for obj_id, pc in global_obj_pc_dict.items():
        pc_path = graph_rag_pc_dir / f"object_{obj_id:05d}.ply"
        o3d.io.write_point_cloud(pc_path.as_posix(), pc)
        print(f"Saved point cloud for object {obj_id} with {len(pc.points)} points to {pc_path}")


if __name__ == "__main__":
    main()
