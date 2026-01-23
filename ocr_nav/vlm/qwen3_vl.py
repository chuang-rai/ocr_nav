from transformers import AutoModelForImageTextToText, AutoProcessor
import torch


class QWen3VLQueryInterface:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype="auto", attn_implementation="flash_attention_2", device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def query(self, prompt: str, image_bytes: bytes) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_bytes}",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        # Post-processing
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
