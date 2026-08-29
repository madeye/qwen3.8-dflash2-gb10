from huggingface_hub import snapshot_download
m = "unsloth/Qwen3.8-27B-NVFP4"
print(f"=== downloading {m}", flush=True)
p = snapshot_download(m, max_workers=8)
print(f"=== done {m} -> {p}", flush=True)
print("ALL_DOWNLOADS_COMPLETE", flush=True)
