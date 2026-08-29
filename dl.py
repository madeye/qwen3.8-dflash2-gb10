from huggingface_hub import snapshot_download
for m in ["z-lab/Qwen3.8-27B-DFlash2", "Qwen/Qwen3.8-27B"]:
    print(f"=== downloading {m}", flush=True)
    p = snapshot_download(m, max_workers=8)
    print(f"=== done {m} -> {p}", flush=True)
print("ALL_DOWNLOADS_COMPLETE", flush=True)
