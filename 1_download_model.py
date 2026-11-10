from huggingface_hub import snapshot_download

# Download SmolLM2-360M-Instruct into a local "models" folder.
# We skip extra formats (like ONNX) so the download stays small.
snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    local_dir="models/SmolLM2-360M-Instruct",
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
print("Model downloaded.")
