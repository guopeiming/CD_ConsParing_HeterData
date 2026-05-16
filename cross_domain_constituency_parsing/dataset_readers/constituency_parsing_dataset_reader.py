from typing import Dict, List, Iterator, Union
import jsonlines
import logging

from cross_domain_constituency_parsing.utils.tree_structure import (
    parse_bracketed_parse_tree,
    Tree,
    get_tree_triples,
    binarization,
    delete_punctuation
)

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    MetadataField,
    Field
)
from cross_domain_constituency_parsing.utils.my_adjacency_field import MyAdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

logger = logging.getLogger(__name__)

PTB_PARENTHESES = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
}


PTB_PARENTHESES_REVERSE = {
    "(": "-LRB-",
    ")": "-RRB-",
    "{": "-LCB-",
    "}": "-RCB-",
    "[": "-LSB-",
    "]": "-RSB-",
}


def data_generator_from_file(file_path: str) -> Iterator[Dict[str, Union[str, List[str]]]]:
    with jsonlines.open(file_path, "r") as reader:
        for line in reader:
            yield line


@DatasetReader.register("base_constituency_parser")
class BaseConstituencyParserDatasetReader(DatasetReader):

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        **kwargs,
    ) -> None:
        super(BaseConstituencyParserDatasetReader, self).__init__(**kwargs)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    def _read(self, file_path: str):
        logger.info("Reading instances from lines in file at: %s", file_path)
        for inst in data_generator_from_file(file_path):
            tree = parse_bracketed_parse_tree(inst["linearized_tree"])

            # This is un-needed and clutters the label space.
            # All the trees also contain a root S node.
            if tree.label == "VROOT" or tree.label == "TOP":
                # assert len(tree.children) == 1, "tree error"
                # tree = tree.children[0]
                if len(tree.children) != 1:
                    tree.label = "S"
                else:
                    tree = tree.children[0]


            inst["tree"] = tree
            inst["postags"] = list(tree.pos_tags())
            assert " ".join(inst["tokens"]) == " ".join(list(tree.leaves())), "sentence error"

            inst.pop("domain")
            inst.pop("task")
            inst.pop("language")
            yield self.text_to_instance(**inst)

    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        postags: List[str],
        linearized_tree: str = None,
        tree: Tree = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}

        token_field = TextField(
            self._tokenizer.tokenize(" ".join([PTB_PARENTHESES.get(token, token) for token in tokens])),
            token_indexers=self._token_indexers
        )
        fields["tokens"] = token_field

        metadata = {"tokens": tokens, "postags": postags}

        if linearized_tree is not None:
            metadata["gold_triples"] = get_tree_triples(delete_punctuation(tree)[0])
            metadata["linearized_tree"] = linearized_tree

            binarized_tree = binarization(tree)
            adj_triples = get_tree_triples(binarized_tree)
            adj_indices, adj_labels = [], []
            for start, end, label in adj_triples:
                adj_indices.append((start, end))
                adj_labels.append(label)
            fields["gold_tree_label"] = MyAdjacencyField(
                adj_indices, token_field, adj_labels, label_namespace="constituency_labels", padding_value=-1)

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
