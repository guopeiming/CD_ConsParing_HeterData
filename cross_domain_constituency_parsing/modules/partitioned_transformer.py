import torch
import math
import torch.nn.functional as F

from torch import nn
from allennlp.modules import InputVariationalDropout
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


@Seq2SeqEncoder.register("partitioned_transformer")
class PartitionedTranformer(Seq2SeqEncoder):
    def __init__(
        self, input_size: int, num_layers: int, d_model: int,
        n_head: int = 8, d_qkv: int = 64, d_ff: int = 2048
    ) -> None:
        super(PartitionedTranformer, self).__init__()
        self.layers = nn.ModuleList(
            PartitionedTransformerLayer(d_model, n_head, d_qkv, d_ff)
            for _ in range(num_layers)
        )
        self.linear = nn.Linear(input_size, d_model//2, False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = InputVariationalDropout(0.2)
        self.position_emb = nn.Embedding(512, d_model//2)
        # nn.init.normal_(self.position_emb.weight, 0., 0.1)
        self.d_model = 1024

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x = self.linear(x)
        x = self.dropout(self.linear(x))
        p = torch.arange(0, x.size(-2), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        p = self.position_emb(p)
        # x = self.dropout(torch.cat([x, p], dim=-1))
        x = self.norm(torch.cat([x, p], dim=-1))
        # x = self.norm(self.dropout(torch.cat([x, p], dim=-1)))

        for layer in self.layers:
            x = layer(x, mask)
        return x

    def get_output_dim(self) -> int:
        return self.d_model


class PartitionedTransformerLayer(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, d_qkv: int, d_ff: int,
        ff_dropout: float = 0.1, residual_dropout: float = 0.2, attention_dropout: float = 0.2
    ) -> None:
        super(PartitionedTransformerLayer, self).__init__()

        self.self_attn = PartitionedMultiHeadAttention(d_model, n_head, d_qkv, attention_dropout)
        self.linear1 = PartitionedLinear(d_model, d_ff)
        self.ff_dropout = InputVariationalDropout(ff_dropout)
        self.linear2 = PartitionedLinear(d_ff, d_model)

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.residual_dropout = InputVariationalDropout(residual_dropout)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.self_attn(x, mask=mask)
        residual = self.residual_dropout(residual)
        x = self.norm_attn(x + residual)

        residual = self.linear2(self.ff_dropout(self.activation(self.linear1(x))))
        residual = self.residual_dropout(residual)
        x = self.norm_ff(x + residual)

        return x


class PartitionedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(PartitionedLinear, self).__init__()
        self.linear_c = nn.Linear(in_features // 2, out_features // 2, bias)
        self.linear_p = nn.Linear(in_features // 2, out_features // 2, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c, x_p = torch.chunk(x, 2, dim=-1)
        out_c = self.linear_c(x_c)
        out_p = self.linear_p(x_p)
        return torch.cat([out_c, out_p], dim=-1)


class PartitionedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_qkv: int, attention_dropout: float = 0.2):
        super().__init__()
        self.w_qkv_c = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, d_qkv // 2))
        self.w_qkv_p = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, d_qkv // 2))
        self.w_o_c = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model // 2))
        self.w_o_p = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model // 2))

        self.scaling_factor = 1 / (d_qkv ** 0.5)
        self.dropout = nn.Dropout(attention_dropout)

        self._reset_parameters()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_c, x_p = torch.chunk(x, 2, dim=-1)

        qkv_c = torch.einsum("btf,hfca->bhtca", x_c, self.w_qkv_c)
        qkv_p = torch.einsum("btf,hfca->bhtca", x_p, self.w_qkv_p)
        q_c, k_c, v_c = [c.squeeze(dim=3) for c in torch.chunk(qkv_c, 3, dim=3)]
        q_p, k_p, v_p = [c.squeeze(dim=3) for c in torch.chunk(qkv_p, 3, dim=3)]
        q = torch.cat([q_c, q_p], dim=-1)
        k = torch.cat([k_c, k_p], dim=-1)
        v = torch.cat([v_c, v_p], dim=-1)

        dots = torch.einsum("bhqa,bhka->bhqk", q, k) * self.scaling_factor
        dots.data.masked_fill_(~mask[:, None, None, :], -float("inf"))
        probs = F.softmax(dots, dim=-1)
        probs = self.dropout(probs)

        o = torch.einsum("bhqk,bhka->bhqa", probs, v)
        o_c, o_p = torch.chunk(o, 2, dim=-1)
        out_c = torch.einsum("bhta,haf->btf", o_c, self.w_o_c)
        out_p = torch.einsum("bhta,haf->btf", o_p, self.w_o_p)
        return torch.cat([out_c, out_p], dim=-1)

    def _reset_parameters(self) -> None:
        bound = math.sqrt(3.0) * 0.02
        nn.init.uniform_(self.w_qkv_c, -bound, bound)
        nn.init.uniform_(self.w_qkv_p, -bound, bound)
        nn.init.uniform_(self.w_o_c, -bound, bound)
        nn.init.uniform_(self.w_o_p, -bound, bound)
