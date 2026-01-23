import kuzu
from sentence_transformers import SentenceTransformer


class EmbodiedGraphRAG:
    def __init__(self, kuzu_db_path: str, embedding_model_name: str = "BAAI/bge-m3"):
        self.kuzu_db = kuzu.Database(kuzu_db_path)
        self.connection = kuzu.Connection(self.kuzu_db)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Successfully loaded embedding model: {embedding_model_name}")
        self._create_schema()

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
            MATCH (f:Frame)-[r:CONTAINS]->(o)
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
