import os
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from qwen_vl_utils import process_vision_info
import torch


class DeepSeekOCRvLLMQueryInterface:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["OMP_NUM_THREADS"] = "1"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
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

    def query(self, prompt: str, image: Image.Image) -> str:
        model_input = [{"prompt": prompt, "multi_modal_data": {"image": image}}]

        model_outputs = self.llm.generate(model_input, self.sampling_params)
        generated_text = model_outputs[0].outputs[0].text
        return generated_text

    def batch_query(self, prompts: list[str], image_list: list[Image.Image]) -> list[str]:
        batch_size = len(prompts)
        messages_list = []
        for i in range(batch_size):
            messages = {"prompt": prompts[i], "multi_modal_data": {"image": image_list[i]}}
            messages_list.append(messages)

        outputs = self.llm.generate(messages_list, sampling_params=self.sampling_params)
        generated_texts = [outputs[i].outputs[0].text for i in range(batch_size)]
        return generated_texts
