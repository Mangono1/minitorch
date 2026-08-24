"""
MiniTorch Python layer.

V0.1:
The numerical engine lives in C++.
The Python API will be connected to the compiled C++ backend
as the binding layer is introduced.
"""

__version__ = "0.1.0"


class Tensor:
    """
    Temporary Python-facing API placeholder.

    The real Tensor implementation will be exposed through
    the C++ binding in the next step.
    """

    def __init__(self, data, shape=None):
        self.data = data

        if shape is None:
            if not data:
                shape = []
            elif isinstance(data[0], list):
                rows = len(data)
                cols = len(data[0])
                shape = [rows, cols]
            else:
                shape = [len(data)]

        self.shape = tuple(shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"
