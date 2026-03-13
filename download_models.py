"""
PhysX-Anything 模型下载脚本
用法: python download_models.py
支持断点续传（已存在且大小正确的文件自动跳过）
"""
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    # ── VLM 权重（Caoza/PhysX-Anything） ──────────────────────────────────
    ("pretrain/vlm/model-00001-of-00004.safetensors",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/model-00001-of-00004.safetensors",
     4968243304),
    ("pretrain/vlm/model-00002-of-00004.safetensors",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/model-00002-of-00004.safetensors",
     4991495816),
    ("pretrain/vlm/model-00003-of-00004.safetensors",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/model-00003-of-00004.safetensors",
     4932751040),
    ("pretrain/vlm/model-00004-of-00004.safetensors",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/model-00004-of-00004.safetensors",
     1691924384),
    # ── VLM 配置/分词器 ───────────────────────────────────────────────────
    ("pretrain/vlm/model.safetensors.index.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/model.safetensors.index.json", 0),
    ("pretrain/vlm/config.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/config.json", 0),
    ("pretrain/vlm/generation_config.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/generation_config.json", 0),
    ("pretrain/vlm/tokenizer_config.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/tokenizer_config.json", 0),
    ("pretrain/vlm/tokenizer.json",
     "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/tokenizer.json", 0),
    ("pretrain/vlm/preprocessor_config.json",
     "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json", 0),
    ("pretrain/vlm/vocab.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/vocab.json", 0),
    ("pretrain/vlm/merges.txt",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/merges.txt", 0),
    ("pretrain/vlm/added_tokens.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/added_tokens.json", 0),
    ("pretrain/vlm/special_tokens_map.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/vlm/special_tokens_map.json", 0),
    # ── Decoder ───────────────────────────────────────────────────────────
    ("pretrain/decoder/pipeline.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/decoder/pipeline.json", 0),
    ("pretrain/decoder/ckpt_new/denoiser_step0350000.pt",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/decoder/ckpt_new/denoiser_step0350000.pt",
     3470728767),
    ("pretrain/decoder/ckpt_new/denoiser_step0350000.json",
     "https://huggingface.co/Caoza/PhysX-Anything/resolve/main/decoder/ckpt_new/denoiser_step0350000.json", 0),
    # ── TRELLIS（microsoft/TRELLIS-image-large） ──────────────────────────
    ("pretrain/trellis/pipeline.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/pipeline.json", 0),
    ("pretrain/trellis/ckpts/slat_enc_swin8_B_64l8_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_enc_swin8_B_64l8_fp16.json", 0),
    ("pretrain/trellis/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors", 0),
    ("pretrain/trellis/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json", 0),
    ("pretrain/trellis/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors", 0),
    ("pretrain/trellis/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json", 0),
    ("pretrain/trellis/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors", 0),
    ("pretrain/trellis/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json", 0),
    ("pretrain/trellis/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors", 0),
    ("pretrain/trellis/ckpts/slat_flow_img_dit_L_64l8p2_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_flow_img_dit_L_64l8p2_fp16.json", 0),
    ("pretrain/trellis/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors", 0),
    ("pretrain/trellis/ckpts/ss_enc_conv3d_16l8_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_enc_conv3d_16l8_fp16.json", 0),
    ("pretrain/trellis/ckpts/ss_enc_conv3d_16l8_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_enc_conv3d_16l8_fp16.safetensors", 0),
    ("pretrain/trellis/ckpts/ss_dec_conv3d_16l8_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.json", 0),
    ("pretrain/trellis/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors", 0),
    ("pretrain/trellis/ckpts/ss_flow_img_dit_L_16l8_fp16.json",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.json", 0),
    ("pretrain/trellis/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors",
     "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors", 0),
]


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(rel_path, url, expected_size):
    dest = os.path.join(BASE, rel_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        actual = os.path.getsize(dest)
        if expected_size == 0 or actual == expected_size:
            print(f"  [跳过] {rel_path}")
            return
        print(f"  [不完整 {human_size(actual)}] {rel_path}，重新下载…")

    print(f"  [下载] {rel_path}")

    def progress(count, block, total):
        if total > 0:
            pct = min(count * block / total * 100, 100)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            sys.stdout.write(f"\r    {bar} {pct:5.1f}%  {human_size(min(count*block,total))}/{human_size(total)}")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
    except Exception as e:
        print(f"\n    [错误] {e}")
        if os.path.exists(dest):
            os.remove(dest)
        raise


if __name__ == "__main__":
    total = len(FILES)
    for i, (path, url, size) in enumerate(FILES, 1):
        print(f"\n[{i}/{total}] {path}")
        download(path, url, size)
    print("\n✓ 全部下载完成！")
