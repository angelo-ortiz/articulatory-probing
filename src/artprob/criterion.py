import math
from typing import Optional

import numpy as np
import torch
from cca_zoo.linear import CCA
from jaxtyping import Float
from torch import nn

from artprob.utils.tensor_utils import pearson_correlation


class EMACriterion(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self._store_emas = False
        self._true_emas = []
        self._pred_emas = [[] for _ in range(num_layers)]

    def disable_ema_store(self):
        self._store_emas = False
        self._true_emas = []
        self._pred_emas = [[] for _ in range(self.num_layers)]

    def enable_ema_store(self):
        self._store_emas = True

    def compute_correlations(self) -> Float[torch.Tensor, "num_dims {self.num_layers}"]:
        true_emas = torch.cat(self._true_emas, dim=1).unsqueeze(1)
        pred_emas = torch.stack(
            [torch.cat(p_emas, dim=1) for p_emas in self._pred_emas],
            dim=1
        )

        # compute the correlations by dimension (for all the layers)
        corrs = []
        for t_emas, p_emas in zip(true_emas, pred_emas):
            corrs.append(pearson_correlation(t_emas, p_emas))

        corrs = torch.cat(corrs, dim=0)

        return corrs


class LinearCriterion(EMACriterion):
    """Computes the loss on the zero-padding at the end of the true and predicted EMA,
    and compatible with early stopping's current form.
    """

    def __init__(
            self,
            hidden_sizes: list[int],
            art_dim: int,
            bias: bool = True,
    ):
        super().__init__(len(hidden_sizes))
        self.loss_function = nn.MSELoss(reduction='sum')
        self.fcs = nn.ModuleList(
            [nn.Linear(hs, art_dim, bias=bias) for hs in hidden_sizes]
        )

    def parameters(self, recurse: bool = True):
        for fc in self.fcs:
            yield fc.parameters(recurse)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for fc in self.fcs:
            fc.zero_grad(set_to_none)

    def forward(
            self,
            xs: list[Float[torch.Tensor, "b s2 d2"]],
            ema: Float[torch.Tensor, "b s1 d1"],
            ema_lens: Float[torch.Tensor, "b"]
    ):
        losses = []
        length = min(xs[0].size(1), ema.size(1))
        if length < ema.size(1):
            ema = ema[:, :length].contiguous()
        for i, fc in enumerate(self.fcs):
            pred_ema = fc(xs[i][:, :length])
            losses.append(self.loss_function(pred_ema, ema).unsqueeze(0))

            if self._store_emas:
                self._pred_emas[i].append(
                    pred_ema.detach().view(-1, pred_ema.size(-1)).T
                )

        if self._store_emas:
            self._true_emas.append(ema.detach().view(-1, ema.size(-1)).T)

        return torch.cat(losses, dim=0)


class MultiLinearCriterion(EMACriterion):
    """Does not compute the loss on the zero-padding at the end of the true and
    predicted EMA (used to have EMA tensors with the same sequence size), but
    incompatible with early stopping's current form.
    """

    def __init__(
            self,
            num_layers: int,
            hidden_size: int,  # hypothesis: all the hidden layers have the same size
            art_dim: int,
            bias: bool = True,
    ):
        super().__init__(num_layers)
        self.loss_function = nn.MSELoss(reduction='none')

        self.weight = nn.Parameter(torch.empty(num_layers, art_dim, hidden_size))
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.randn(num_layers, art_dim))
        else:
            self.register_buffer('bias', torch.zeros(1))

    def forward(
            self,
            xs: list[Float[torch.Tensor, "b s2 d2"]],
            ema: Float[torch.Tensor, "b s1 d1"],
            ema_lens: Float[torch.Tensor, "b"]
    ):
        xs = torch.stack(xs, dim=0)
        losses = 0.
        for i, ema_l in enumerate(ema_lens):
            pred_ema = torch.matmul(xs[:, i, :ema_l], self.weight.mT) \
                       + self.bias.unsqueeze(1)
            losses += self.loss_function(pred_ema, ema[i:i + 1, :ema_l]).sum(dim=(1, 2))

            if self._store_emas:
                self._true_emas.append(ema[i, :ema_l].detach().view(-1, ema.size(-1)).T)
                for j in range(self.num_layers):
                    self._pred_emas[j].append(
                        pred_ema[j].detach().view(-1, pred_ema.size(-1)).T
                    )

        return losses


class CCACriterion(EMACriterion):
    def __init__(
            self,
            num_layers: int,
            art_dim: int,
            seed: Optional[int]
    ):
        super().__init__(num_layers)
        self.cca = CCA(
            latent_dimensions=art_dim,
            copy_data=True,
            random_state=np.random.RandomState(seed)
        )

    def compute_correlations(self) -> Float[torch.Tensor, "num_dims {self.num_layers}"]:
        true_emas = torch.cat(self._true_emas, dim=0).numpy()
        pred_emas = [
            torch.cat(p_emas, dim=0).numpy() for p_emas in self._pred_emas
        ]

        cca_scores = []
        for p_emas in pred_emas:
            tr_true_emas, tr_p_emas = self.cca.fit_transform((true_emas, p_emas))

            tr_true_emas = tr_true_emas.astype('float32', casting='same_kind')
            tr_true_emas = torch.from_numpy(tr_true_emas).cuda().T.unsqueeze(1)

            tr_p_emas = tr_p_emas.astype('float32', casting='same_kind')
            tr_p_emas = torch.from_numpy(tr_p_emas).cuda().T.unsqueeze(1)

            # compute the correlations by dimension
            corrs = []
            for _t_emas, _p_emas in zip(tr_true_emas, tr_p_emas):
                corrs.append(pearson_correlation(_t_emas, _p_emas))

            cca_scores.append(torch.cat(corrs, dim=0))

        return torch.cat(cca_scores, dim=1)

    def forward(
            self,
            xs: list[Float[torch.Tensor, "b s2 d2"]],
            ema: Float[torch.Tensor, "b s1 d1"],
            ema_lens: Float[torch.Tensor, "b"]
    ):
        for i in range(self.num_layers):
            if self._store_emas:
                self._pred_emas[i].append(
                    xs[i][:, :ema.size(1)].detach().contiguous()
                    .view(-1, xs[0].size(-1)).cpu()
                )

        if self._store_emas:
            self._true_emas.append(ema.detach().view(-1, ema.size(-1)).cpu())

        return None
