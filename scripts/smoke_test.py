"""GPU smoke test: confirm torch sees the A100 inside the container.
Run on a compute node only (never on the login node):

    apptainer exec --nv \\
        --bind $PWD:/workspace --bind $PWD/data:/data --bind $PWD/outputs:/outputs \\
        container/eouq.sif python /workspace/scripts/smoke_test.py
"""
import sys

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible", file=sys.stderr)
        return 1

    n_devices = torch.cuda.device_count()
    print(f"device count: {n_devices}")
    for i in range(n_devices):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"matmul check: output shape {tuple(y.shape)}, sum={y.sum().item():.3e}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
