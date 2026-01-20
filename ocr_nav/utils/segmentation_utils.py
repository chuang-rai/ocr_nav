import yaml
import numpy as np
from rai_ai_core_library.detection.grounding_dino import GroundingDino
from rai_ai_core_library.segmentation.sam import Sam
from rai_ai_core_library.utils import dynamic_model
from rai_ai_core_library.base_models import DetectionResult


class GroundingDinoSamSegmenter:
    def __init__(self, grounding_dino_config_path: str, sam_config_path: str, device: str):
        with open(grounding_dino_config_path, "r") as file:
            det_config = yaml.safe_load(file)
        with open(sam_config_path, "r") as file:
            sam_config = yaml.safe_load(file)

        self.grounding_dino_model = dynamic_model(det_config, device=device)
        print("Successfully loaded Grounding DINO model.")
        self.sam_model = dynamic_model(sam_config, device=device)
        print("Successfully loaded SAM model.")

    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        # Use Grounding DINO to get bounding boxes for the text prompt
        detection_results: DetectionResult = self.grounding_dino_model.predict(image, [text_prompt])

        # Use SAM to get segmentation masks for the bounding boxes
        masks = []
        for detection_result in detection_results:
            box = detection_result.bbox  # (x_min, y_min, x_max, y_max) in np.ndarray
            box = box.tolist()
            mask = self.sam_model.predict([box], image)  # np.ndarray of shape (1, H, W)
            masks.append(mask)

        # Combine all masks into a single mask
        if masks:
            combined_mask = np.any(masks, axis=0).astype(np.uint8).squeeze()
        else:
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return combined_mask
