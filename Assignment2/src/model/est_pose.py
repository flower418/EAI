from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64) # 加一个 batchnorm 层，方便训练能够顺利进行。这个参数表示需要 norm 的特征数
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        # 最后一层不需要 batchnorm，另外，由于 relu 和 pooling 没有参数，不需要写进 init，在 forward 直接调用即可
        self.fc6 = nn.Linear(256, 9) # 最终是 9 维的，3 维 translation 加上 6 维 rotation

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape (B, N, 3)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape (B, 3)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape (B, 3, 3)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # 整体架构，先 fc，再 bn，最后 relu
        # 注意 batchnorm 的维度在第 2 个，所以需要把 2，3 维进行 transpose，再转回去
        x = torch.relu(self.bn1(self.fc1(pc).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn2(self.fc2(x).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn3(self.fc3(x).transpose(1, 2)).transpose(1, 2)) # B*N*1024

        # 然后 max pooling
        x, _ = torch.max(x, dim=1) # max 会返回最大值和索引，我们只要最大值，此时 x: (B, 1024)

        x = torch.relu(self.bn4(self.fc4(x))) # 由于变回了 2 维，不需要 transpose，直接就可以 batchnorm
        x = torch.relu(self.bn5(self.fc5(x))) # (B, 256)
        scores = self.fc6(x) # (B, 9): 3-translation, 6-rotation

        trans_vec = scores[:, :3]
        rot_vec = scores[:, 3:]

        # 处理 gt_rot，把它变成 6 维
        # 注意这里 gt_rot 是 3 个维度的，所以应该重新处理
        gt_rot = rot[:, :, :2].reshape(rot.shape[0], -1) # (B, 6)

        trans_loss = torch.sqrt(torch.sum((trans_vec - trans) ** 2))
        rot_loss = torch.sqrt(torch.sum((rot_vec - gt_rot) ** 2))
        loss = trans_loss + rot_loss

        metric = dict(
            loss=loss,
            trans_loss=trans_loss,
            rot_loss=rot_loss
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
        """
        x = torch.relu(self.bn1(self.fc1(pc).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn2(self.fc2(x).transpose(1, 2)).transpose(1, 2))
        x = torch.relu(self.bn3(self.fc3(x).transpose(1, 2)).transpose(1, 2)) # B*N*1024

        # 然后 max pooling
        x, _ = torch.max(x, dim=1) # max 会返回最大值和索引，我们只要最大值，此时 x: (B, 1024)

        x = torch.relu(self.bn4(self.fc4(x)))
        x = torch.relu(self.bn5(self.fc5(x))) # (B, 256)
        pred_vec = self.fc6(x) # (B, 9): 3-translation, 6-rotation

        # 注意维度
        trans_vec = pred_vec[:, :3]
        rot_vec = pred_vec[:, 3:] # (B, 6)

        # 将 rot_vec 拆成 3 个维度
        rot_vec = rot_vec.reshape(-1, 3, 2) # (B, 3, 2)

        col1 = rot_vec[:, :, 0] # (B, 3)
        col2 = rot_vec[:, :, 1] # (B, 3)
        b1 = col1 / torch.norm(col1, dim=1, keepdim=True) # torch.norm 返回 (B, 1)，然后用 col1 除以 norm 就得到 b1: (B, 3)
        temp_b2 = col2 - torch.sum(b1 * col2) * b1
        b2 = temp_b2 / torch.norm(b2, dim=1, keepdim=True) # (B, 3)
        b3 = torch.cross(b1, b2, dim=1) # 按第二个向量维度做叉乘

        rot_mat = torch.stack([b1, b2, b3], dim=2) # 由于需要在第三个维度使用索引表示第 1,2,3 列，所以 stack 的 dim=2
        return trans_vec, rot_mat
