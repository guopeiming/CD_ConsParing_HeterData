from allennlp.data.fields import AdjacencyField
from allennlp.data.vocabulary import Vocabulary
import logging
from typing import Dict
import torch


logger = logging.getLogger(__name__)

class MyAdjacencyField(AdjacencyField):

    def index(self, vocab: Vocabulary):
        if self.labels is not None:
            token_2_index_vocab = vocab.get_token_to_index_vocabulary(self._label_namespace)
            self._indexed_labels = [
                (vocab.get_token_index(label, self._label_namespace) if label in token_2_index_vocab else 0)
                for label in self.labels
            ]

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens = padding_lengths["num_tokens"]
        tensor = torch.ones(desired_num_tokens, desired_num_tokens, dtype=torch.long) * self._padding_value
        labels = self._indexed_labels or [1 for _ in range(len(self.indices))]

        for index, label in zip(self.indices, labels):
            tensor[index] = label
        return tensor
