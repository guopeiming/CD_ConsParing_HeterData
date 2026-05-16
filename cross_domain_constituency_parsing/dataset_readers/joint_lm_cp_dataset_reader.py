from typing import Dict, List, Iterator, Union
import jsonlines
import logging
import numpy as np

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
    Field,
    LabelField
)
from cross_domain_constituency_parsing.utils.my_adjacency_field import MyAdjacencyField
from cross_domain_constituency_parsing.dataset_readers.constituency_parsing_dataset_reader import PTB_PARENTHESES
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.checks import ConfigurationError


logger = logging.getLogger(__name__)


def data_generator_from_file(file_path: str) -> Iterator[Dict[str, Union[str, List[str]]]]:
    with jsonlines.open(file_path, "r") as reader:
        for line in reader:
            yield line


@DatasetReader.register("joint_lm_cp_parser")
class JointLMCPParserDatasetReader(DatasetReader):

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        **kwargs,
    ) -> None:
        super(JointLMCPParserDatasetReader, self).__init__(**kwargs)
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

            if inst["task"] == "constituency_parsing":
                inst["tree"] = tree
                inst["postags"] = list(tree.pos_tags())
                assert " ".join(inst["tokens"]) == " ".join(list(tree.leaves())), "sentence error"

            elif inst["task"] == "language_model":
                tree = parse_bracketed_parse_tree(inst["linearized_tree"])
                inst["tree"] = tree
                inst["postags"] = list(tree.pos_tags())

            else:
                raise ConfigurationError("invalid task")

            yield self.text_to_instance(**inst)

    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        task: str,
        domain: str,
        language: str,
        linearized_tree: str = None,
        postags: List[str] = None,
        tree: Tree = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}

        token_field = TextField(
            self._tokenizer.tokenize(" ".join([PTB_PARENTHESES.get(token, token) for token in tokens])),
            token_indexers=self._token_indexers
        )
        fields["tokens"] = token_field

        fields["task"] = LabelField(task, label_namespace="task_labels")
        fields["domain"] = LabelField(domain, label_namespace="domain_labels")
        fields["language"] = LabelField(language, label_namespace="language_labels")
        metadata = {"tokens": tokens, "postags": postags, "task": task, "domain": domain, "language": language}

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

        if task == "constituency_parsing":
            pass

        elif task == "language_model":
            binarized_tree = binarization(tree)
            adj_triples = get_tree_triples(binarized_tree)
            num_boundary = np.zeros(len(tokens), dtype=np.int64)
            for start, end, label in adj_triples:
                num_boundary[start] += 1
                num_boundary[end] += 1
            metadata["lm_boundary"] = num_boundary / num_boundary.sum()

        else:
            raise ConfigurationError("invalid task")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
