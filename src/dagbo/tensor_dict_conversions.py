import torch
from torch import Tensor
from typing import Dict, List

def unpack_to_dict(names: List[str], values: Tensor) -> Dict[str, Tensor]:
    """
    Args:
        names: d-length list
        values: batch_shape * d-dim Tensor
    Returns: d-length Dict[name, value: batch_shape-dim Tensor]
    """
    split_values = torch.unbind(input=values, dim=-1)
    return {name:value for (name, value) in zip(names, split_values)}

def pack_to_tensor(names: List[str], values: Dict[str, Tensor]) -> Tensor:
    """
    Args:
        names: d-length list
        values: d-length Dict[name, value: batch_shape-dim Tensor]
    Returns: batch_shape * d-dim Tensor
        d is has the same order as `names`
    """
    return torch.stack([values[y] for y in names], dim=-1)

