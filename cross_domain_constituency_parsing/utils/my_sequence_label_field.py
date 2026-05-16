from allennlp.data.fields import SequenceLabelField
from typing import Dict
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length


class MySequenceLabelField(SequenceLabelField):

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        if self._indexed_labels is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before calling .as_tensor()"
            )
        desired_num_tokens = padding_lengths["num_tokens"]
        padded_tags = pad_sequence_to_length(self._indexed_labels, desired_num_tokens, lambda: -100)
        tensor = torch.LongTensor(padded_tags)
        return tensor
