
import re
from pathlib import Path

# PyTorch 相关包列表
torch_packages = {
    "torch", "torchvision", "torchaudio", "torchtext", "torchdata",
    "pytorch-lightning", "torchmetrics", "transformers", "accelerate",
    "datasets", "tokenizers", "diffusers", "timm", "torchsummary",
    "torch-audio", "torch-vision", "torch-text", "fairseq", "detectron2"
}

req_file = Path("requirements.txt")
if not req_file.exists():
    print("requirements.txt not found, skipping torch package separation")
    exit(0)

lines = [l.strip() for l in req_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]

torch_reqs, other_reqs = [], []
for line in lines:
    pkg_name = re.split(r'[<>=\s\[]', line, 1)[0].lower()
    if pkg_name in torch_packages:
        torch_reqs.append(line)
    else:
        other_reqs.append(line)

if torch_reqs:
    Path("requirements_torch.txt").write_text("\n".join(torch_reqs) + "\n", encoding="utf-8")
    print(f"Generated requirements_torch.txt with {len(torch_reqs)} packages")

if other_reqs:
    Path("requirements_other.txt").write_text("\n".join(other_reqs) + "\n", encoding="utf-8")
    print(f"Generated requirements_other.txt with {len(other_reqs)} packages")
        