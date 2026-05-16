from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TokenEmbedder
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from cross_domain_constituency_parsing.models.joint_lm_cp_parser import (
    JointLMCPShareParser)
import logging


logger = logging.getLogger(__name__)

@Model.register("joint_lm_cp_ccg_share_span3_parser")
class JointLMCPCCGShareSpan3Parser(JointLMCPShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ccg_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPCCGShareSpan3Parser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.ccg_head = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.vocab.get_vocab_size("ccg_labels"))
        )
        self.ccg_loss_weight = ccg_loss_weight
        self.ccg_loss = nn.CrossEntropyLoss()

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ccg_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None,
        cp_weight: torch.FloatTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ccg_index = torch.nonzero(
            task == self.vocab.get_token_index("ccg_parsing", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ccg = lm_index.size(0), cp_index.size(0), ccg_index.size(0)
        mask = tokens["tokens"]["mask"]

        emb = self.text_field_embedder(domain_ids=domain, task_ids=task, **tokens["tokens"])
        encoded_text = self.encoder(emb, mask)
        encoded_text = torch.cat([encoded_text[..., 0::2], encoded_text[..., 1::2]], dim=-1)

        if num_lm > 0:
            lm_spans = torch.arange(mask.size(1)-2).unsqueeze(0).unsqueeze(-1).expand(num_lm, -1, 2).to(mask.device)
            lm_repre = self.span_extractor(encoded_text[lm_index, :, :], lm_spans)
            lm_logits = self.lm_head(self.lm_proj(lm_repre) + emb[lm_index, 1:-1, :])
            lm_loss = self.lm_loss(lm_logits.permute(0, 2, 1), lm_label[:, 1:-1])
            res["loss"] = lm_loss*self.lm_loss_weight + res["loss"]

        if num_ccg > 0:
            ccg_spans = torch.arange(mask.size(1)-2).unsqueeze(0).unsqueeze(-1).expand(num_ccg, -1, 2).to(mask.device)
            ccg_repre = self.span_extractor(encoded_text[ccg_index, :, :], ccg_spans)
            ccg_logits = self.ccg_head(ccg_repre)
            ccg_loss = self.ccg_loss(ccg_logits.permute(0, 2, 1), ccg_label[ccg_index, :])
            res["loss"] = ccg_loss*self.ccg_loss_weight + res["loss"]

        if num_cp > 0:
            cp_mask = mask[cp_index, :]
            cp_batch_size, cp_seq_len, cp_seq_lens = cp_mask.size(0), cp_mask.size(1)-2, cp_mask.sum(1)-2

            triu_indexes = torch.triu_indices(cp_seq_len, cp_seq_len)
            spans = triu_indexes.transpose(0, 1).unsqueeze(0).expand(cp_batch_size, -1, -1).to(cp_mask.device)
            triu_repre = self.span_extractor(encoded_text[cp_index, :, :], spans)

            if gold_tree_label is not None:
                gold_tree_label = gold_tree_label[cp_index, :, :]

            pred_tree_structure, cp_logits, pred_event, gold_event = self._parsing(
                triu_repre, triu_indexes, gold_tree_label,
                cp_batch_size, cp_seq_len, cp_seq_lens,
                [metadata[i]["tokens"] for i in cp_index],
                [metadata[i]["postags"] for i in cp_index]
            )
            res["pred_linearized_tree"] = [p_tree_structure.linearize() for p_tree_structure in pred_tree_structure]

            if gold_tree_label is not None:
                res["loss"] = res["loss"] + self.margin_loss(cp_logits, pred_event, gold_event)
                self.compute_parsing_performance(pred_tree_structure, [metadata[i] for i in cp_index])

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ccg_loss_weight = alpha
