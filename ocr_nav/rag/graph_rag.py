from pathlib import Path
import kuzu
from sentence_transformers import SentenceTransformer
from ocr_nav.utils.rag_utils import convert_type_to_kuzu_type


class BaseGraphRAG:
    def __init__(self, kuzu_db_dir: str, overwrite: bool = False, embedding_model_name: str = "BAAI/bge-m3"):
        kuzu_db_dir = Path(kuzu_db_dir)
        kuzu_db_path = kuzu_db_dir / (kuzu_db_dir.name + ".db")
        if overwrite and kuzu_db_path.exists():
            kuzu_db_path.unlink()
        kuzu_db_dir.mkdir(parents=True, exist_ok=True)
        self.kuzu_db = kuzu.Database(kuzu_db_path.as_posix())
        self.connection = kuzu.Connection(self.kuzu_db)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Successfully loaded embedding model: {embedding_model_name}")
        self.node_types = self._check_existing_node_types()
        self.rel_types = self._check_existing_rel_types()

    def _check_existing_node_types(self):
        result = self.connection.execute("CALL show_tables() WHERE type = 'NODE' RETURN name;")
        existing_node_types = set()
        while result.has_next():
            row = result.get_next()
            print("Existing node type:", row[0])
            existing_node_types.add(row[0])
        return existing_node_types

    def _check_existing_rel_types(self):
        result = self.connection.execute("CALL show_tables() WHERE type = 'REL' RETURN name;")
        existing_rel_types = set()
        while result.has_next():
            row = result.get_next()
            print("Existing relationship type:", row[0])
            existing_rel_types.add(row[0])
        return existing_rel_types

    def define_node_type(self, node_name: str, properties: dict):
        props_str = ", ".join([f"{k} {convert_type_to_kuzu_type(v)}" for k, v in properties.items()])
        id_definition = "id INT64"
        if "id" not in properties:
            props_str = id_definition + ", " + props_str
        query = f"CREATE NODE TABLE {node_name}({props_str}, PRIMARY KEY (id))"
        print(query)
        try:
            self.connection.execute(query)
            print(f"Successfully created node table {node_name}.")
        except RuntimeError as e:
            if "already exists" in str(e):
                print(f"Node table {node_name} already exists, skipping creation.")
            else:
                raise e  # Re-raise if it's a different error
        self.node_types.add(node_name)

    def define_relationship_type(self, rel_name: str, from_node_type: str, to_node_type: str, properties: dict):
        props_str = ", ".join([f"{k} {convert_type_to_kuzu_type(v)}" for k, v in properties.items()])
        query = f"CREATE REL TABLE {rel_name}(FROM {from_node_type} TO {to_node_type}, {props_str})"
        try:
            self.connection.execute(query)
            print(f"Successfully created relationship table {rel_name}.")
        except RuntimeError as e:
            if "already exists" in str(e):
                print(f"Relationship table {rel_name} already exists, skipping creation.")
            else:
                raise e  # Re-raise if it's a different error
        self.rel_types.add(rel_name)

    def add_node(self, node_type: str, properties: dict):
        if node_type not in self.node_types:
            raise ValueError(f"Node type {node_type} is not defined in the graph.")
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"CREATE (n:{node_type} {{{props_str}}})"
        self.connection.execute(query, properties)

    def add_relationship(
        self,
        rel_type: str,
        from_node_type: str,
        from_node_id: int,
        to_node_type: str,
        to_node_id: int,
        properties: dict,
    ):
        if rel_type not in self.rel_types:
            raise ValueError(f"Relationship type {rel_type} is not defined in the graph.")
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"""
        MATCH (a:{from_node_type} {{id: $from_id}}), (b:{to_node_type} {{id: $to_id}})
        CREATE (a)-[r:{rel_type} {{{props_str}}}]->(b)
        """
        params = {"from_id": from_node_id, "to_id": to_node_id}
        params.update(properties)
        self.connection.execute(query, params)

    def extend_node_type(self, node_name: str, new_properties: dict):
        for prop_name, prop_type in new_properties.items():
            query = f"ALTER TABLE {node_name} ADD COLUMN {prop_name} {convert_type_to_kuzu_type(prop_type)}"
            try:
                self.connection.execute(query)
                print(f"Successfully added property {prop_name} to node {node_name}.")
            except RuntimeError as e:
                if "already exists" in str(e):
                    print(f"Property {prop_name} already exists in node {node_name}, skipping...")
                else:
                    raise e  # Re-raise if it's a different error

    def build_node_index(
        self, node_type: str, property_name: str, metric: str = "cosine", mu: int = 16, efc: int = 200
    ):
        index_name = f"{node_type}_{property_name}_idx"
        query = (
            f"CALL CREATE_VECTOR_INDEX({node_type}, {index_name}, "
            + f"{property_name}, metric:='{metric}', mu:={mu}, efc:={efc})"
        )
        try:
            self.connection.execute("LOAD EXTENSION vector")
            self.connection.execute(query)
            print(f"Successfully created index {index_name} on {node_type}({property_name}).")
        except RuntimeError as e:
            if "already exists" in str(e):
                print(f"Index {index_name} already exists, skipping...")
            else:
                raise e  # Re-raise if it's a different error


class EmbodiedGraphRAG(BaseGraphRAG):
    def __init__(self, kuzu_db_dir: str, overwrite: bool = False, embedding_model_name: str = "BAAI/bge-m3"):
        super().__init__(kuzu_db_dir, overwrite, embedding_model_name)
        self.obj_id = 0

    def _create_schema(self):
        tables = [
            "CREATE NODE TABLE Frame(id INT64, timestamp STRING, caption STRING, PRIMARY KEY (id))",  # SigLIP2 image embedding size is 1152
            """
            CREATE NODE TABLE Object(
                id INT64,
                label STRING, 
                attributes STRING[], 
                embedding FLOAT[1024], 
                bbox INT64[4],
                PRIMARY KEY (id)
            )
            """,  # BGE-M3 text embedding size is 1024
            "CREATE REL TABLE CONTAINS(FROM Frame TO Object, bbox INT64[4])",
        ]
        for table in tables:
            try:
                self.connection.execute(table)
                print(f"Successfully created table {table.split()[3].split('(')[0]}.")
            except RuntimeError as e:
                if "already exists" in str(e):
                    print("Table already exists, skipping creation.")
                else:
                    raise e  # Re-raise if it's a different error

    def build_index(self):
        try:
            self.connection.execute("LOAD EXTENSION vector")
            self.connection.execute(
                "CALL CREATE_VECTOR_INDEX('Object', 'object_embeddings_idx', 'embedding', metric:='cosine', mu:=16, efc:=200)"
            )
            print("Index created successfully.")
        except Exception as e:
            if "already exists" in str(e):
                print("Index already exists, skipping...")
            else:
                raise e

    def ingest_json_frame(self, frame_id: int, timestamp: str, frame_caption: str, objects_json: list[dict]):
        # Create the Frame
        self.connection.execute(
            "MERGE (f:Frame {id: $id})" " ON CREATE SET f.timestamp = $ts, f.caption = $caption",
            {"id": frame_id, "ts": timestamp, "caption": frame_caption},
        )

        for obj in objects_json:
            attrs = obj["attributes"]
            attrs_list = [f"{k}: {v}" for k, v in attrs.items()]
            # Convert JSON to a semantic narrative for the vector
            color = attrs.get("color", "")
            material = attrs.get("material", "")
            material_str = f"made of {material}" if material else ""
            function = attrs.get("function", "")
            function_str = f" for {function}" if function else ""
            narrative = f"A {color} {obj['label']} {material_str}{function_str}."
            embedding = self.embedding_model.encode(narrative).tolist()

            # Merge/Create Object and Link to Frame
            self.connection.execute(
                """
                MERGE (o:Object {id: $id})
                ON CREATE SET o.label = $label, o.embedding = $vec, o.attributes = $attrs, o.bbox = $bbox
                WITH o
                MATCH (f:Frame {id: $frame_id})
                CREATE (f)-[:CONTAINS {bbox: $bbox}]->(o)
            """,
                {
                    "id": self.obj_id,
                    "label": obj["label"],
                    "vec": embedding,
                    "attrs": attrs_list,
                    "frame_id": frame_id,
                    "bbox": obj["bounding_box"],
                },
            )
            self.obj_id += 1

    def find_images_by_concept(self, query_text: str, top_k: int = 3):
        self.build_index()
        query_vec = self.embedding_model.encode(query_text).tolist()

        # 1. Vector search for relevant objects
        # 2. Graph hop to the Frame (image) containing them
        result = self.connection.execute(
            """
            CALL QUERY_VECTOR_INDEX('Object', 'object_embeddings_idx', $vec, $top_k)
            YIELD node AS o, distance
            MATCH (f:Frame)-[r:FrameContainsObject]->(o)
            RETURN f.id, f.timestamp, o.label, r.bbox, distance
            """,
            {"vec": query_vec, "top_k": top_k},
        )
        img_ids = []
        box_info = []
        while result.has_next():
            row = result.get_next()
            print(f"Found {row[1]} in {row[0]} (Distance: {row[4]:.4f}) at BBox {row[3]}")
            img_ids.append(row[0])
            box_info.append({"label": row[2], "bounding_box": row[3], "distance": row[4]})
        return img_ids, box_info
