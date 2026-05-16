from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from cross_domain_constituency_parsing.utils.cky import CKY
from cross_domain_constituency_parsing.utils.tree_structure import (
    construct_bracketed_parse_tree, Tree, get_tree_triples, debinarization, delete_punctuation)
from cross_domain_constituency_parsing.metrics.constituency_parsing_f1_score import ConstituencyParsingF1Score
from cross_domain_constituency_parsing.metrics.my_evalb_bracketing_scorer import MyEvalb
from cross_domain_constituency_parsing.modules.constituency_parsing_margin_loss import ConstituencyParsingMarginLoss
from copy import deepcopy
import numpy as np
import logging


logger = logging.getLogger(__name__)


@Model.register("base_constituency_parser")
class BaseConstituencyParser(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        postags_embedding: Embedding,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super(BaseConstituencyParser, self).__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        self.postags_embedding = postags_embedding
        self.var_dropout = InputVariationalDropout(0.2)
        self.encoder = encoder
        self.span_extractor = span_extractor
        self.span_ffn = nn.Linear(self.span_extractor.get_output_dim(), 512)

        self.num_labels = self.vocab.get_vocab_size("constituency_labels")
        # 打分函数
        self.score_label = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.num_labels-1),
        )
        self.margin_loss = ConstituencyParsingMarginLoss()

        self.star_label_index = self.vocab.get_token_index("*", "constituency_labels")
        assert self.star_label_index == 0, "star label index error"

        self.parsing_score = ConstituencyParsingF1Score()
        self.evalb_score = MyEvalb()
        initializer(self)

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        postags: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        res = dict()
        mask = get_text_field_mask(tokens)
        batch_size, seq_len, seq_lens = mask.size(), mask.sum(1)

        embedded_text_input = self.text_field_embedder(tokens)
        embedded_postags = self.postags_embedding(postags)
        embedded_text_input = self.var_dropout(torch.cat([embedded_text_input, embedded_postags], dim=-1))
        encoded_text = self.var_dropout(self.encoder(embedded_text_input, mask))

        triu_indexes = torch.triu_indices(seq_len, seq_len)
        spans = triu_indexes.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1).to(mask.device)
        triu_repre = self.span_extractor(encoded_text, spans, mask)
        triu_repre = self.var_dropout(F.relu(self.span_ffn(triu_repre)))

        triu_logits = self.score_label(triu_repre)
        triu_logits = torch.cat([triu_logits.new_zeros((batch_size, triu_logits.size(1), 1)), triu_logits], dim=-1)
        logits = triu_logits.new_zeros(batch_size, seq_len, seq_len, self.num_labels)
        logits[torch.arange(batch_size).unsqueeze(1), triu_indexes[0:1, :], triu_indexes[1:2, :], :] = triu_logits
        logits[torch.arange(batch_size), 0, seq_lens-1, self.star_label_index] -= 1e9

        if gold_tree_label is not None:
            # gold_event = F.one_hot(gold_tree_label+1, num_classes=self.num_labels+1)
            # gold_event = gold_event[:, :, :, 1:]
            gold_event = F.one_hot(F.relu(gold_tree_label), num_classes=self.num_labels)
            if self.training:
                Haming_augment = 1. - gold_event
                logits = logits + Haming_augment

            # if self.training:
            #     rand_num = np.random.random()
            #     if rand_num < 0.5:
            #         gold_event = F.one_hot(gold_tree_label+1, num_classes=self.num_labels+1)
            #         gold_event = gold_event[:, :, :, 1:]
            #     else:
            #         gold_event = F.one_hot(F.relu(gold_tree_label), num_classes=self.num_labels)
            #     Haming_augment = 1. - gold_event
            #     logits = logits + Haming_augment
            # else:
            #     gold_event = F.one_hot(gold_tree_label+1, num_classes=self.num_labels+1)
            #     gold_event = gold_event[:, :, :, 1:]

        pred_event, pred_split = CKY(logits.data.cpu().numpy(), seq_lens.tolist())
        pred_tree_structure = self.construct_tree_structure(
            pred_event, pred_split,
            [meta["tokens"] for meta in metadata],
            [meta["postags"] for meta in metadata],
            seq_lens
        )
        res["pred_linearized_tree"] = [p_tree_structure.linearize() for p_tree_structure in pred_tree_structure]

        if gold_tree_label is not None:
            res["loss"] = self.margin_loss(logits, pred_event, gold_event)
            self.compute_parsing_performance(pred_tree_structure, metadata)

        return res

    def compute_parsing_performance(self, pred_tree_structure: List[Tree], metadata: List[Dict[str, Any]]) -> None:
        pred_triples = [get_tree_triples(delete_punctuation(p_t_struct)[0]) for p_t_struct in pred_tree_structure]
        gold_triples = [deepcopy(meta["gold_triples"]) for meta in metadata]
        self.parsing_score(pred_triples, gold_triples)

        if not self.training:
            pred_linearized_tree = [p_tree_structure.linearize() for p_tree_structure in pred_tree_structure]
            gold_linearized_tree = [meta["linearized_tree"] for meta in metadata]
            self.evalb_score(pred_linearized_tree, gold_linearized_tree)

    def construct_tree_structure(
        self, pred_event: np.ndarray, pred_split: np.ndarray,
        tokens: List[str], pos_tags: List[str], seq_lens: torch.Tensor
    ) -> List[Tree]:
        tree_structures = list()
        zip_ = zip(pred_event, pred_split, tokens, pos_tags, seq_lens)
        for tree_event, tree_split, tree_tokens, tree_postags, sent_len in zip_:
            assert sent_len == len(tree_tokens), "seq len error"
            tree_label = np.argmax(tree_event, axis=-1)
            assert tree_label[0, sent_len-1] != self.star_label_index, "star label error"

            tree = construct_bracketed_parse_tree(
                tree_label=tree_label,
                tree_split=tree_split,
                tokens=tree_tokens, postags=tree_postags,
                vocab=self.vocab.get_index_to_token_vocabulary("constituency_labels"),
                start=0,
                end=len(tree_tokens)-1
            )
            tree = debinarization(tree)
            tree_structures.append(tree)
        return tree_structures

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.parsing_score.get_metric(reset)
        if not self.training:
            evalb_metrics = self.evalb_score.get_metric(reset)
            metrics.update(evalb_metrics)
        return metrics
