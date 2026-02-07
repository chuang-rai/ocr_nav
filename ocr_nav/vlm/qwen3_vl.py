import os
from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch


class QWen3VLQueryInterface:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     model_name, dtype="auto", attn_implementation="flash_attention_2", device_map=device
        # )
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype="auto", device_map=device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def query(self, prompt: str, image_bytes: bytes = None) -> str:
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
        if image_bytes is None:
            messages[0]["content"] = [{"type": "text", "text": prompt}]

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


class QWen3VLvLLMQueryInterface:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["OMP_NUM_THREADS"] = "1"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            mm_encoder_tp_mode="data",
            enable_expert_parallel=False,
            max_model_len=4096,
            tensor_parallel_size=torch.cuda.device_count(),
            mm_processor_kwargs={
                "limit_mm_per_prompt": {"image": 1, "video": 0},
            },
            seed=0,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            top_k=-1,
            stop_token_ids=[],
        )

    def prepare_inputs_for_vllm(self, messages, processor):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # qwen_vl_utils 0.0.14+ reqired
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        print(f"video_kwargs: {video_kwargs}")

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}

    def query(self, prompt: str, image_bytes: bytes = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_bytes}",
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            }
        ]

        inputs = [self.prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text

    def batch_query(self, prompts: list[str], image_bytes_list: list[bytes]) -> list[str]:
        batch_size = len(prompts)
        messages_list = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{image_bytes_list[i]}",
                        },
                        {"type": "text", "text": f"{prompts[i]}"},
                    ],
                }
            ]
            messages_list.append(messages)

        inputs = [self.prepare_inputs_for_vllm(message, self.processor) for message in messages_list]
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        generated_texts = [outputs[i].outputs[0].text for i in range(batch_size)]
        return generated_texts
