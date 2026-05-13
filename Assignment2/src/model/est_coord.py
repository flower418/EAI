from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config

        # 网络架构
        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64) # 第一个 64 维需要与最后的 1024 维进行拼接
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1088, 512) # 注意这里是堆叠后的
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128, 3) # 预测对应过去以后在 CAD model 上的坐标，所以是 3 维

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape (B, N, 3)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape (B, N, 3)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        B, N, _ = pc.shape

        x = torch.relu(self.bn1(self.fc1(pc).transpose(1, 2)).transpose(1, 2)) # (B, N, 64)
        short_x = x.clone() # 保留下来用于后续拼接，只能用 clone
        x = torch.relu(self.bn2(self.fc2(x).transpose(1, 2)).transpose(1, 2)) # (B, N, 128)
        x = torch.relu(self.bn3(self.fc3(x).transpose(1, 2)).transpose(1, 2)) # (B, N, 1024) # 把它和 short_x 进行拼接

        x, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, 1024)

        x = x.repeat(1, N, 1) # repeat 表示每个维度重复多少次
        x = torch.cat([short_x, x], dim=2) # (B, N, 1088)

        x = torch.relu(self.bn4(self.fc4(x).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn6(self.fc6(x).transpose(1, 2)).transpose(1, 2)) # (B, N, 128)

        scores = self.fc7(x) # (B, N, 3)

        loss = torch.sqrt(torch.sum((scores - coord) ** 2))

        metric = dict(
            loss=loss,
        )

        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape (B, N, 3)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape (B, 3)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape (B, 3, 3)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        B, N, _ = pc.shape

        x = torch.relu(self.bn1(self.fc1(pc).transpose(1, 2)).transpose(1, 2)) # (B, N, 64)
        short_x = x.clone() # 保留下来用于后续拼接，只能用 clone
        x = torch.relu(self.bn2(self.fc2(x).transpose(1, 2)).transpose(1, 2)) # (B, N, 128)
        x = torch.relu(self.bn3(self.fc3(x).transpose(1, 2)).transpose(1, 2)) # (B, N, 1024) # 把它和 short_x 进行拼接

        x, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, 1024)

        x = x.repeat(1, N, 1) # repeat 表示每个维度重复多少次
        x = torch.cat([short_x, x], dim=2) # (B, N, 1088)

        x = torch.relu(self.bn4(self.fc4(x).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn6(self.fc6(x).transpose(1, 2)).transpose(1, 2)) # (B, N, 128)

        cad_points = self.fc7(x) # (B, N, 3)，表示 camera 中每个点在 cad model 上的对应点

        # 中心化
        cad_mean = torch.mean(cad_points, dim=1, keepdim=True) # (B, 1, 3)
        camera_mean = torch.mean(pc, dim=1, keepdim=True) # (B, 1, 3)
        cad_centered = cad_points - cad_mean
        camera_centered = pc - camera_mean

        # SVD
        U, _, V_T = torch.linalg.svd(camera_centered.transpose(1, 2) @ cad_centered) # (B, 3, 3)

        # 计算 R，T
        R = U @ V_T # (B, 3, 3)，初始化，需要加反射修正
        det_R = torch.det(R) # (B,)

        D = torch.eye(3, device=pc.device, dtype=pc.dtype) # 保证类型一致
        D = D.unsqueeze(0).repeat(B, 1, 1) # 增加一个 batch 维度
        D[:, 2, 2] = torch.where(det_R > 0,
                                 torch.tensor(1, dtype=pc.dtype, device=pc.device),
                                 torch.tensor(-1, dtype=pc.dtype, device=pc.device)) # (condition, true_do, false_do)
        R = U @ D @ V_T # (B, 3, 3)
        
        # 注意这里必须对 R 进行转置，因为 points 中的三维向量是行向量，但是默认 rotation matrix 是列向量，所以应该做一个转置对齐：
        T = pc - cad_points @ R.transpose(1, 2) # (B, N, 3)
        # 对所有点求平均得到最终的 translation
        T = torch.mean(T, dim=1, keepdim=True).squeeze(1) # (B, 1, 3)->(B, 3)

        return T, R
