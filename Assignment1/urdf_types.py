from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Joint:
    name: str
    trans: np.ndarray  # with shape (3,) # translation
    rot: np.ndarray  # rotation matrix with shape (3, 3) rotation


@dataclass
class FixedJoint(Joint):
    pass


@dataclass
class RevoluteJoint(Joint):
    axis: np.ndarray
    lower_limit: float # 最小角
    upper_limit: float # 最大角


@dataclass
class Link:
    name: str
    visual_meshes: List[str] # 用于 visualize
