import asyncio
import re
import os
import pathlib
import time

import argparse
import torch

if torch.version.cuda == "11.8":  # type: ignore
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from ocr_nav.thirdparty.deepseek_ocr.deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm
from ocr_nav.thirdparty.deepseek_ocr.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from ocr_nav.thirdparty.deepseek_ocr.process.image_process import DeepseekOCRProcessor
from ocr_nav.thirdparty.deepseek_ocr.config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE
from ocr_nav.ocr_nav.utils.io_utils import load_image
from ocr_nav.ocr_nav.utils.visualization_utils import draw_bounding_boxes


ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def process_image_with_refs(image, ref_texts, output_dir):
    result_image = draw_bounding_boxes(image, ref_texts, output_dir)
    return result_image


async def stream_generate(image=None, prompt=""):

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logits_processors = [
        NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})
    ]  # whitelist: <td>, </td>

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        # ignore_eos=False,
    )

    request_id = f"request-{int(time.time())}"

    printed_length = 0

    if image and "<image>" in prompt:
        request = {"prompt": prompt, "multi_modal_data": {"image": image}}
    elif prompt:
        request = {"prompt": prompt}
    else:
        assert False, f"prompt is none!!!"
    async for request_output in engine.generate(request, sampling_params, request_id):  # type: ignore
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end="", flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print("\n")

    return final_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/test_video_sbb/not_blurry_images",
        help="Input image directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/test_video_sbb/not_blurry_images_ocr",
        help="OCR output directory",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/images", exist_ok=True)

    input_dir = pathlib.Path(args.input_dir)

    total_time = 0.0

    for image_path in tqdm(list(input_dir.glob("*")), desc="images"):
        start_time = time.time()
        image_name = image_path.stem

        image = load_image(image_path)
        assert image is not None
        image = image.convert("RGB")

        if "<image>" in PROMPT:

            image_features = DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=CROP_MODE
            )
        else:
            image_features = ""

        prompt = PROMPT

        result_out = asyncio.run(stream_generate(image_features, prompt))

        save_results = 1

        if save_results and "<image>" in prompt:
            print("=" * 15 + "save results:" + "=" * 15)

            image_draw = image.copy()

            outputs = result_out

            with open(f"{args.output_dir}/{image_name}_result_ori.mmd", "w", encoding="utf-8") as afile:
                afile.write(outputs)

            matches_ref, matches_images, mathes_other = re_match(outputs)
            # print(matches_ref)
            result = process_image_with_refs(image_draw, matches_ref, args.output_dir)

            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, f"![](images/" + str(idx) + ".jpg)\n")

            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, "").replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

            # if 'structural formula' in conversation[0]['content']:
            #     outputs = '<smiles>' + outputs + '</smiles>'
            with open(f"{args.output_dir}/{image_name}_result.mmd", "w", encoding="utf-8") as afile:
                afile.write(outputs)

            if "line_type" in outputs:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Circle

                lines = eval(outputs)["Line"]["line"]

                line_type = eval(outputs)["Line"]["line_type"]
                # print(lines)

                endpoints = eval(outputs)["Line"]["line_endpoint"]

                fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(" -- ")[0])
                        p1 = eval(line.split(" -- ")[-1])

                        if line_type[idx] == "--":
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k")
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k")

                        ax.scatter(p0[0], p0[1], s=5, color="k")
                        ax.scatter(p1[0], p1[1], s=5, color="k")
                    except:
                        pass

                for endpoint in endpoints:

                    label = endpoint.split(": ")[0]
                    (x, y) = eval(endpoint.split(": ")[1])
                    ax.annotate(
                        label, (x, y), xytext=(1, 1), textcoords="offset points", fontsize=5, fontweight="light"
                    )

                try:
                    if "Circle" in eval(outputs).keys():
                        circle_centers = eval(outputs)["Circle"]["circle_center"]
                        radius = eval(outputs)["Circle"]["radius"]

                        for center, r in zip(circle_centers, radius):
                            center = eval(center.split(": ")[1])
                            circle = Circle(center, radius=r, fill=False, edgecolor="black", linewidth=0.8)
                            ax.add_patch(circle)
                except:
                    pass

                plt.savefig(f"{args.output_dir}/{image_name}_geo.jpg")
                plt.close()

            result.save(f"{args.output_dir}/{image_name}_result_with_boxes.jpg")
        end_time = time.time()
        total_time += end_time - start_time
        print(f"Processing time for {image_name}: {end_time - start_time:.2f} seconds")
    print(f"Average processing time per image: {total_time / len(list(input_dir.glob('*'))):.2f} seconds")
