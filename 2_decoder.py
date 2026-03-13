import os
os.environ['ATTN_BACKEND'] = 'flash_attn'   # Use flash_attn for GPU acceleration
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import ipdb
import numpy as np
import torch
import trimesh
import argparse

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("./pretrain/decoder")
pipeline.cuda()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="输入图片路径")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录")
    args = parser.parse_args()

    image_path = args.image
    image_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(image_name)[0]
    
    output_subdir = os.path.join(args.output_dir, name_without_ext)
    
    print(f"[INFO] 输入图片: {image_path}")
    print(f"[INFO] 输出目录: {output_subdir}")

    # 检查 VLM 输出是否存在
    allind_path = os.path.join(output_subdir, 'allind.npy')
    if not os.path.exists(allind_path):
        print(f"[ERROR] 找不到 VLM 输出文件: {allind_path}")
        print("[ERROR] 请先运行 1_vlm_demo.py")
        exit(1)

    image = Image.open(image_path)
    
    print(f"[INFO] 加载体素数据...")
    newcoords = np.load(allind_path)
    
    size = 32
    resolution = 64

    newcoords = newcoords + 32 - (size) // 2
    
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, newcoords[:, 0], newcoords[:, 1], newcoords[:, 2]] = 1
    ss = ss.cuda().float().unsqueeze(0)

    print(f"[INFO] TRELLIS 3D 解码中...")
    outputs = pipeline.run_control(ss, image, seed=1)

    print(f"[INFO] 生成 GLB 网格...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.5,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )

    output_glb = os.path.join(output_subdir, 'sample.glb')
    glb.export(output_glb)
    print(f"[INFO] 3D 网格已保存: {output_glb}")




