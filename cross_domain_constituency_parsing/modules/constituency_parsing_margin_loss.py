from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class ConstituencyParsingMarginLoss(nn.Module):

    def __init__(self) -> None:
        super(ConstituencyParsingMarginLoss, self).__init__()

    def forward(self, logits: torch.Tensor, pred_event: np.ndarray, gold_event: torch.Tensor) -> torch.Tensor:
        pred_event = torch.tensor(pred_event, device=logits.device)
        pred_score = (pred_event * logits).sum([1, 2, 3])
        gold_score = (gold_event * logits).sum([1, 2, 3])
        loss = F.relu(pred_score - gold_score).sum()

        if not self.training:
            loss = loss + torch.abs(pred_event - gold_event).sum([1, 2, 3]).sum()
        # NOTE: loss should be devided by big batch size!!!
        loss = loss / pred_event.size(0)

        return loss
