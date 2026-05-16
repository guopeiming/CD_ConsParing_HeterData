from typing import List
import logging

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer, TokenIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer_mismatched_endpoint")
class PretrainedTransformerMismatchedEndpointIndexer(PretrainedTransformerMismatchedIndexer):

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        output = super().tokens_to_indices(tokens, vocabulary)

        tail_index = len(output["token_ids"])-1
        assert tail_index == output["offsets"][-1][1]+1, "error"
        output["offsets"] = [(0, 0),] + output["offsets"] + [(tail_index, tail_index),]
        output["mask"] = [True] + output["mask"] + [True]

        return output

    def get_empty_token_list(self) -> IndexedTokenList:
        output = self._matched_indexer.get_empty_token_list()
        output["offsets"] = []
        output["wordpiece_mask"] = []
        return output
