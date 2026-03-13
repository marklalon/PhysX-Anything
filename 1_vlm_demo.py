from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import base64
import os
import ipdb
import numpy as np
from PIL import Image
import trimesh
from rembg import remove
import argparse
def voxel_encode(voxels: np.ndarray, size: int = 32) -> np.ndarray:

    voxels = np.asarray(voxels, dtype=np.int64)
    assert voxels.ndim == 2 and voxels.shape[1] == 3, "voxels shape should be (N,3)"
    assert size == 32, "size=32（2^5）。"
    if (voxels < 0).any() or (voxels >= size).any():
        raise ValueError("xyz should be within [0, 32).")

    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    return (x << 10) | (y << 5) | z


def voxel_decode(indices: np.ndarray, size: int = 32) -> np.ndarray:

    indices = np.asarray(indices, dtype=np.int64).ravel()
    assert size == 32, "size=32（2^5）。"
    if (indices < 0).any() or (indices >= size**3).any():

        indices=indices.clip(0,size**3-1)
        print("index should be within [0, 32768).")


    x = (indices >> 10) & 31
    y = (indices >> 5)  & 31
    z = indices & 31
    return np.stack([x, y, z], axis=1)



def ints_to_space_separated_str(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int64).ravel()
    return " ".join(map(str, arr))



def merge_adjacent_to_dash(s: str) -> str:

    if not s.strip():
        return ""

    nums = list(map(int, s.split()))

    nums = sorted(set(nums))

    result = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            result.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n
    result.append(f"{start}-{prev}" if start != prev else f"{start}")
    return " ".join(result)



def dash_str_to_ints(s: str) -> np.ndarray:

    if not s.strip():
        return np.array([], dtype=np.int64)

    out = []
    for token in s.split():
        if "-" in token:
            a, b = map(int, token.split("-"))
            if a > b:
                a, b = b, a 
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))
    return np.array(sorted(set(out)), dtype=np.int64)


def addmessage(message,before,after):
    answer={}
    answer['role']='assistant'
    answer['content']=[{"type": "text", "text": before}]
    question={}
    question['role']='user'
    question['content']=[{"type": "text", "text": after}]
    newmessage=message.copy()
    newmessage.append(answer)
    newmessage.append(question)
    return newmessage



def generate_save(model,messages,save_dir,save_name='test',save=True):


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)


    generated_ids = model.generate(**inputs, do_sample=False, max_length=32768)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if save:
        with open(os.path.join(save_dir,save_name+'.txt'),'w') as file:
            file.write( output_text[0])
    return output_text[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="输入图片路径（如 ./demo/foo.png）")
    parser.add_argument("--save_part_ply", action='store_true', default=True,
                        help="保存部件 PLY 文件")
    parser.add_argument("--remove_bg", action='store_true', default=False,
                        help="移除背景")
    parser.add_argument("--ckpt", type=str, default='./pretrain/vlm')
    parser.add_argument("--load_in_8bit", action='store_true', default=True,
                        help="使用 8-bit 量化（默认开启）")
    parser.add_argument("--full_precision", action='store_true', default=False,
                        help="禁用 8-bit 量化，使用 BF16 全精度")
    args = parser.parse_args()

    image_path = args.image
    name = os.path.basename(image_path)
    
    print(f"[INFO] 输入图片: {image_path}")
    print(f"[INFO] 输出目录: output/{os.path.splitext(name)[0]}")

    use_8bit = args.load_in_8bit and not args.full_precision
    
    if use_8bit:
        print("[INFO] 使用 8-bit 量化模式加载模型...")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.ckpt,
                    quantization_config=quant_cfg,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
        print("[INFO] 模型加载完成 (8-bit 量化)")
    else:
        print("[INFO] 使用 BF16 模式加载模型...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.ckpt,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
        print("[INFO] 模型加载完成 (BF16)")
    min_pixels = 65536
    max_pixels = 262144

    processor = AutoProcessor.from_pretrained(args.ckpt, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
    if not processor.chat_template and hasattr(processor, 'tokenizer') and processor.tokenizer.chat_template:
        processor.chat_template = processor.tokenizer.chat_template
    processor.image_processor.min_pixels=min_pixels
    processor.image_processor.max_pixels=max_pixels
    processor.image_processor.size["shortest_edge"]=min_pixels
    processor.image_processor.size["longest_edge"]=max_pixels

    # 设置输出目录
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    save_dir = os.path.join(output_dir, os.path.splitext(name)[0])
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join('./dataset/overall_prompt.txt'), "r", encoding="utf-8") as f:
        basicqu = f.read()

    input_image = Image.open(image_path)
    im_resized = input_image.resize((512, 512), Image.LANCZOS)

    if args.remove_bg:
        im_resized = remove(im_resized)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": im_resized.convert("RGB"),
                },
                {"type": "text", "text": basicqu},
            ],
        }
    ]

    basicoutput = generate_save(model, messages, save_dir, 'basic_info')
    print(f"[INFO] 基本信息生成完成")
    
    index = 0
    while 'l_' + str(index) in basicoutput:
        index += 1
    print(f"[INFO] 检测到 {index} 个部件")

    allcoord = []
    for part in range(index):
        print(f"[INFO] 处理部件 {part + 1}/{index}...")
        question = "Based on the structured description of l_" + str(part) + ", generate its 3D voxel grid in the following format (voxel grid=32, use numbers from 0 to 32767, merge maximal consecutive runs: 199...216 -> 199-216): 184 198 199-216 230-237..."
        messages1 = addmessage(messages, basicoutput, question)
        output1 = generate_save(model, messages1, save_dir, 'coord_' + str(part), save=True)
        idx_back = dash_str_to_ints(output1)
        voxels_back = voxel_decode(idx_back)
        allcoord.append(voxels_back)
        np.save(os.path.join(save_dir, 'ind_' + str(part) + '.npy'), voxels_back)
        if args.save_part_ply:
            partply = trimesh.points.PointCloud(voxels_back)
            partply.export(os.path.join(save_dir, 'ind_' + str(part) + '.ply'))

    np.save(os.path.join(save_dir, 'allind.npy'), np.concatenate(allcoord))
    print(f"[INFO] VLM 处理完成！结果保存在: {save_dir}")

