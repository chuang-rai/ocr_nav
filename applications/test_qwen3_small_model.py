# -*- coding: utf-8 -*-
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def prepare_inputs_for_vllm(messages, processor):
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


if __name__ == "__main__":

    # TODO: change to your own checkpoint path
    checkpoint_path = "Qwen/Qwen3-1.7B"
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.75,
        seed=0,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        max_tokens=1024,
    )

    messages = [
        {
            "role": "system",
            # "content": "You are a data extraction engine. Respond ONLY with valid Cypher. Do not include any introductory text, explanations, or Markdown code blocks unless specifically requested. If you cannot fulfill the request, return an empty string.",
            "content": "Some text is provided below. Given the text, extract up to 10 knowledge triples in the form of `subject,predicate,object` on each line. Do not include any introductory text, explanations, or Markdown code blocks unless specifically requested. No think.\n",
        }
    ]
    while True:
        user_input = input("User: ")
        if user_input.lower() == "q":
            break

        messages.append(
            {
                "role": "user",
                # "content": f"{user_input}",
                "content": f"Open-vocabulary scene understanding is crucial for robotic applications, enabling robots to comprehend complex 3D environmental contexts and supporting various downstream tasks such as navigation and manipulation. However, achieving accurate scene understanding in real-world scenarios remains challenging.",
            }
        )
        outputs = llm.chat(messages, sampling_params, use_tqdm=False)
        print("LLM: ", outputs[0].outputs[0].text)
        messages.append(
            {
                "role": "assistant",
                "content": outputs[0].outputs[0].text,
            }
        )
