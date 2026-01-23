from PIL import Image
import requests
import torch
from transformers import AutoProcessor, AutoModel

# 1. Load the model and processor
model_id = "google/siglip2-so400m-patch14-384"  # Example variant
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Prepare inputs (Image + Text)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of two cats", "a remote controller", "a photo of a dog", "a red sofa"]

# 3. Extract Image Features
image_inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)
    # L2 Normalization (recommended for similarity tasks)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

# 4. Extract Text Features
# Note: SigLIP2 often uses a fixed max_length (e.g., 64) for best results
text_inputs = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    # L2 Normalization
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

print(f"Image Feature Shape: {image_features.shape}")  # [1, embedding_dim]
print(f"Text Feature Shape: {text_features.shape}")  # [num_texts, embedding_dim]

# 5. Compute similarities
similarity = torch.matmul(image_features, text_features.T)
print("Similarity Scores:", similarity)
