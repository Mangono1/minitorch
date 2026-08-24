import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from minitorch import Tensor


def test_tensor_shape():
    tensor = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert tensor.shape == (2, 2)


if __name__ == "__main__":
    test_tensor_shape()
    print("Python tensor test: PASS")
