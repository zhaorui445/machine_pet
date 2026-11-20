"""Run a small, random-weight Transformer training check on the local GPU.

Uses synthetic data only; does not download model weights or training datasets.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import sentencepiece
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, 128)
        self.position = nn.Embedding(64, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=0.0, batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(128, 256)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        return self.output(self.blocks(hidden, mask=mask))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("电子宠物 · GPU 训练环境检查", flush=True)
    print(f"PyTorch: {torch.__version__} / CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 不可用，请检查 NVIDIA 驱动和 PyTorch CUDA 安装。")

    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    print(f"显卡：{gpu.name}；显存：{gpu.total_memory / 2**30:.2f} GiB")
    print(f"计算能力：{gpu.major}.{gpu.minor}；开始实际训练验证。", flush=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    model = TinyLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 256, (2, 65), device=device)
    before = model.output.weight.detach().clone()
    output_dir = Path(__file__).resolve().parent / "data" / "gpu-check"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "runs" / time.strftime("%Y%m%d-%H%M%S")
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = time.perf_counter()

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for step in range(8):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(tokens[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, 256), tokens[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失出现非有限值。")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            writer.add_scalar("check/loss", losses[-1], step)
            writer.add_scalar("check/grad_norm", grad_norm.item(), step)
            print(f"训练步骤 {step + 1}/8：loss={losses[-1]:.4f}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if torch.equal(before, model.output.weight.detach()):
        raise RuntimeError("模型参数未更新。")

    checkpoint = output_dir / "synthetic-checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 8}, checkpoint)
    restored = torch.load(checkpoint, map_location=device, weights_only=True)
    restored_model = TinyLanguageModel().to(device)
    restored_model.load_state_dict(restored["model"])
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(restored["optimizer"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored_model.state_dict()[key], rtol=0, atol=0)

    # Verify the reloaded optimizer can continue training.
    restored_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = restored_model(tokens[:, :-1])
        resumed_loss = F.cross_entropy(logits.reshape(-1, 256), tokens[:, 1:].reshape(-1))
    scaler.scale(resumed_loss).backward()
    scaler.unscale_(restored_optimizer)
    nn.utils.clip_grad_norm_(restored_model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(restored_optimizer)
    scaler.update()
    if not torch.isfinite(resumed_loss):
        raise RuntimeError("恢复训练失败。")
    torch.cuda.synchronize()

    report = {
        "status": "passed", "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": gpu.name, "compute_capability": f"{gpu.major}.{gpu.minor}",
        "vram_gib": round(gpu.total_memory / 2**30, 2),
        "precision": str(dtype), "parameters": sum(p.numel() for p in model.parameters()),
        "steps": 8, "losses": losses, "elapsed_seconds": round(elapsed, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "checkpoint_reload": "passed", "resumed_training": "passed",
        "numpy": np.__version__, "sentencepiece": sentencepiece.__version__,
        "note": "Synthetic-data environment check only; not a trained pet model or throughput benchmark.",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"通过：{dtype} 前向计算、反向传播、参数更新、保存和恢复训练。")
    print(f"检查报告：{report_path}")
    print("这只是环境验证，不代表已经训练出聊天模型。")


if __name__ == "__main__":
    main()
