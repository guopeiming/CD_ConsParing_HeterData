import torch
from torch.nn.parameter import Parameter

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import (
    SpanExtractorWithSpanWidthEmbedding,
)
from allennlp.nn import util


@SpanExtractor.register("constituency_span")
class ConstituencySpanExtractor(SpanExtractorWithSpanWidthEmbedding):

    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super(ConstituencySpanExtractor, self).__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths,
        )


    def get_output_dim(self) -> int:
        if self._span_width_embedding is not None:
            return self._input_dim + self._span_width_embedding.get_output_dim()
        else:
            return self._input_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        span_starts, span_ends = span_indices[:, :, 0], span_indices[:, :, 1]
        forward_repre = sequence_tensor[:, :-1, :self._input_dim//2]
        backword_repre = sequence_tensor[:, 1:, self._input_dim//2:]
        mix_repre = torch.cat([forward_repre, backword_repre], dim=-1)

        start_embeddings = util.batched_index_select(mix_repre, span_starts)
        end_embeddings = util.batched_index_select(mix_repre, span_ends+1)
        span_embeddings = end_embeddings - start_embeddings

        return span_embeddings
