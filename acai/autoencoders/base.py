from typing import Any

import torch


class AutoencoderBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ACAI uses L2 here
        self.compute_loss = torch.nn.L1Loss()
        
    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        raise NotImplementedError()