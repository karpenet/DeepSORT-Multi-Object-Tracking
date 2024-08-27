from typing import TypeAlias, NamedTuple, List, Tuple
import numpy as np
import torch


Image: TypeAlias = np.ndarray
Tuple: TypeAlias = Tuple
BBoxList: TypeAlias = List['BBox']
Tensor: TypeAlias = torch.Tensor
FeatureVector: TypeAlias = np.ndarray

class BBox(NamedTuple): 
    x_min: float
    y_min: float
    x_max: float
    y_max: float


