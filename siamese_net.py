import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.typing import Image

class SiameseNetwork(nn.Module):
    """
    Siamese network model.
    """
    def __init__(self) -> None:
        super(SiameseNetwork, self).__init__()

        # Outputs batch X 512 X 1 X 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            # 1X1 filters to increase dimensions
            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
        )

    def forward_once(self, x: Image) -> Image:
        """
        Forward pass for a single input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        output = self.net(x)
        output = torch.squeeze(output)
        return output

    def forward(self, input1: Image, input2: Image, input3: Optional[Image] = None) -> Tuple[Image, Image, Optional[Image]]:
        """
        Forward pass for multiple inputs.

        Args:
            input1 (torch.Tensor): First input tensor.
            input2 (torch.Tensor): Second input tensor.
            input3 (torch.Tensor, optional): Third input tensor.

        Returns:
            tuple: Output tensors.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1, output2, output3

        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin: float = 2.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1: Image, output2: Image, label: Image) -> Image:
        """
        Forward pass for the contrastive loss.

        Args:
            output1 (torch.Tensor): First output tensor.
            output2 (torch.Tensor): Second output tensor.
            label (torch.Tensor): Label tensor.

        Returns:
            torch.Tensor: Loss value.
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        losses = 0.5 * (label.float() * euclidean_distance + (1 + (-1 * label)).float() * F.relu(self.margin - (euclidean_distance + self.eps).sqrt()).pow(2))
        loss_contrastive = torch.mean(losses)
        return loss_contrastive

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample.
    """
    def __init__(self, margin: float) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: Image, positive: Image, negative: Image, size_average: bool = True) -> Image:
        """
        Forward pass for the triplet loss.

        Args:
            anchor (torch.Tensor): Anchor tensor.
            positive (torch.Tensor): Positive tensor.
            negative (torch.Tensor): Negative tensor.
            size_average (bool): Whether to average the loss.

        Returns:
            torch.Tensor: Loss value.
        """
        distance_positive = F.cosine_similarity(anchor, positive)
        distance_negative = F.cosine_similarity(anchor, negative)
        losses = (1 - distance_positive)**2 + (0 - distance_negative)**2
        return losses.mean() if size_average else losses.sum()