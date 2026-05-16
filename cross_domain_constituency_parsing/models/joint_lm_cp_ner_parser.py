from typing import Dict, List, Any

import torch
import torch.nn as nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TokenEmbedder
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from cross_domain_constituency_parsing.models.joint_lm_cp_parser import (
    JointLMCPNShareParser, JointLMCPShareParser, JointLMCPQKVNShareParser, JointLMCPQKVShareParser)
import logging


logger = logging.getLogger(__name__)


@Model.register("joint_lm_cp_ner_nshare_parser")
class JointLMCPNERNShareParser(JointLMCPNShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERNShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.ner_head = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.vocab.get_vocab_size("ner_labels"))
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)

        lm_emb, lm_mask, cp_emb, cp_mask, ner_emb, ner_mask = self.text_field_embedder(
            domain_ids=domain, task_ids=task, **tokens["tokens"],
            lm_index=lm_index, cp_index=cp_index, ner_index=ner_index, num_lm=num_lm, num_cp=num_cp, num_ner=num_ner)

        if num_lm > 0:
            lm_logits = self.lm_head(lm_emb)
            lm_loss = self.lm_loss(lm_logits.permute(0, 2, 1), lm_label)
            res["loss"] = lm_loss*self.lm_loss_weight + res["loss"]

        if num_cp > 0:
            cp_batch_size, cp_seq_len, cp_seq_lens = cp_mask.size(0), cp_mask.size(1)-2, cp_mask.sum(1)-2

            encoded_text = self.encoder(cp_emb, cp_mask)
            encoded_text = torch.cat([encoded_text[..., 0::2], encoded_text[..., 1::2]], dim=-1)

            triu_indexes = torch.triu_indices(cp_seq_len, cp_seq_len)
            spans = triu_indexes.transpose(0, 1).unsqueeze(0).expand(cp_batch_size, -1, -1).to(cp_mask.device)
            triu_repre = self.span_extractor(encoded_text, spans)

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

        if num_ner > 0:
            ner_logits = self.ner_head(ner_emb[:, 1:-1, :])
            ner_loss = self.ner_loss(ner_logits.permute(0, 2, 1), ner_label[ner_index, :])
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha


@Model.register("joint_lm_cp_ner_share_parser")
class JointLMCPNERShareParser(JointLMCPShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.ner_proj = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
        )
        self.ner_head = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.vocab.get_vocab_size("ner_labels"))
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)
        mask = tokens["tokens"]["mask"]

        emb = self.text_field_embedder(domain_ids=domain, task_ids=task, **tokens["tokens"])
        encoded_text = self.encoder(emb, mask)
        encoded_text = torch.cat([encoded_text[..., 0::2], encoded_text[..., 1::2]], dim=-1)

        if num_lm > 0:
            lm_logits = self.lm_head(self.lm_proj(encoded_text[lm_index, :, :]) + emb[lm_index, :, :])
            lm_loss = self.lm_loss(lm_logits.permute(0, 2, 1), lm_label)
            res["loss"] = lm_loss*self.lm_loss_weight + res["loss"]

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

        if num_ner > 0:
            ner_logits = self.ner_head(self.ner_proj(encoded_text[ner_index, 1:-1, :]))
            ner_loss = self.ner_loss(ner_logits.permute(0, 2, 1), ner_label[ner_index, :])
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha


@Model.register("joint_lm_cp_ner_share_span1_parser")
class JointLMCPNERShareSpan1Parser(JointLMCPShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERShareSpan1Parser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.ner_head = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.vocab.get_vocab_size("ner_labels")-1)
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()
        assert self.vocab.get_token_index("o", "ner_labels") == 0, "ner label error"

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)
        mask = tokens["tokens"]["mask"]

        emb = self.text_field_embedder(domain_ids=domain, task_ids=task, **tokens["tokens"])
        encoded_text = self.encoder(emb, mask)
        encoded_text = torch.cat([encoded_text[..., 0::2], encoded_text[..., 1::2]], dim=-1)

        if num_lm > 0:
            lm_logits = self.lm_head(self.lm_proj(encoded_text[lm_index, :, :]) + emb[lm_index, :, :])
            lm_loss = self.lm_loss(lm_logits.permute(0, 2, 1), lm_label)
            res["loss"] = lm_loss*self.lm_loss_weight + res["loss"]

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

        if num_ner > 0:
            ner_triu_indexes = torch.triu_indices(mask.size(1)-2, mask.size(1)-2)
            ner_spans = ner_triu_indexes.transpose(0, 1).unsqueeze(0).expand(num_ner, -1, -1).to(mask.device)
            ner_triu_repre = self.span_extractor(encoded_text[ner_index, :, :], ner_spans)
            ner_triu_logits = self.ner_head(ner_triu_repre)
            ner_triu_logits_zeros = ner_triu_logits.new_zeros((num_ner, ner_triu_logits.size(1), 1))
            ner_triu_logits = torch.cat([ner_triu_logits_zeros, ner_triu_logits], dim=-1)
            ner_triu_label = ner_label[ner_index.unsqueeze(1), ner_triu_indexes[0:1, :], ner_triu_indexes[1:2, :]]
            ner_loss = self.ner_loss(ner_triu_logits.permute(0, 2, 1), ner_triu_label)
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha


@Model.register("joint_lm_cp_ner_share_span2_parser")
class JointLMCPNERShareSpan2Parser(JointLMCPShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERShareSpan2Parser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.ner_proj = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
        )
        self.ner_head = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.vocab.get_vocab_size("ner_labels"))
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()
        assert self.vocab.get_token_index("o", "ner_labels") == 0, "ner label error"

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)
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

        if num_ner > 0:
            ner_spans = torch.arange(mask.size(1)-2).unsqueeze(0).unsqueeze(-1).expand(num_ner, -1, 2).to(mask.device)
            ner_repre = self.span_extractor(encoded_text[ner_index, :, :], ner_spans)
            ner_logits = self.ner_head(self.ner_proj(ner_repre))
            ner_loss = self.ner_loss(ner_logits.permute(0, 2, 1), ner_label[ner_index, :])
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha


@Model.register("joint_lm_cp_ner_share_span3_parser")
class JointLMCPNERShareSpan3Parser(JointLMCPShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERShareSpan3Parser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.ner_head = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.vocab.get_vocab_size("ner_labels")-1)
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()
        assert self.vocab.get_token_index("o", "ner_labels") == 0, "ner label error"

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)
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

        if num_ner > 0:
            ner_triu_indexes = torch.triu_indices(mask.size(1)-2, mask.size(1)-2)
            ner_spans = ner_triu_indexes.transpose(0, 1).unsqueeze(0).expand(num_ner, -1, -1).to(mask.device)
            ner_triu_repre = self.span_extractor(encoded_text[ner_index, :, :], ner_spans)
            ner_triu_logits = self.ner_head(ner_triu_repre)
            ner_triu_logits_zeros = ner_triu_logits.new_zeros((num_ner, ner_triu_logits.size(1), 1))
            ner_triu_logits = torch.cat([ner_triu_logits_zeros, ner_triu_logits], dim=-1)
            ner_triu_label = ner_label[ner_index.unsqueeze(1), ner_triu_indexes[0:1, :], ner_triu_indexes[1:2, :]]
            ner_loss = self.ner_loss(ner_triu_logits.permute(0, 2, 1), ner_triu_label)
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha


@Model.register("joint_lm_cp_ner_qkv_nshare_parser")
class JointLMCPNERQKVNShareParser(JointLMCPQKVNShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        share_loss_weight: float,
        domain_loss_weight: float,
        task_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERQKVNShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight,
            share_loss_weight, domain_loss_weight, task_loss_weight, initializer, **kwargs)

        self.ner_head = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.vocab.get_vocab_size("ner_labels"))
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)

        lm_emb, lm_mask, cp_emb, cp_mask, ner_emb, ner_mask, share_hidden, domain_hidden, task_hidden = \
            self.text_field_embedder(
                domain_ids=domain, task_ids=task, **tokens["tokens"], lm_index=lm_index,
                cp_index=cp_index, ner_index=ner_index, num_lm=num_lm, num_cp=num_cp, num_ner=num_ner)

        lm_emb, lm_mask, cp_emb, cp_mask,  = self.text_field_embedder(
            domain_ids=domain, task_ids=task, **tokens["tokens"],
            lm_index=lm_index, cp_index=cp_index, ner_index=ner_index, num_lm=num_lm, num_cp=num_cp, num_ner=num_ner)

        if num_lm > 0:
            lm_logits = self.lm_head(lm_emb)
            lm_loss = self.lm_loss(lm_logits.permute(0, 2, 1), lm_label)
            res["loss"] = lm_loss*self.lm_loss_weight + res["loss"]

        if num_cp > 0:
            cp_batch_size, cp_seq_len, cp_seq_lens = cp_mask.size(0), cp_mask.size(1)-2, cp_mask.sum(1)-2

            encoded_text = self.encoder(cp_emb, cp_mask)
            encoded_text = torch.cat([encoded_text[..., 0::2], encoded_text[..., 1::2]], dim=-1)

            triu_indexes = torch.triu_indices(cp_seq_len, cp_seq_len)
            spans = triu_indexes.transpose(0, 1).unsqueeze(0).expand(cp_batch_size, -1, -1).to(cp_mask.device)
            triu_repre = self.span_extractor(encoded_text, spans)

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

        if num_ner > 0:
            ner_logits = self.ner_head(ner_emb[:, 1:-1, :])
            ner_loss = self.ner_loss(ner_logits.permute(0, 2, 1), ner_label[ner_index, :])
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        share_loss = (
            self.prompt_loss(self.share_domain_cls(self.grl(share_hidden)), domain) + 
            self.prompt_loss(self.share_task_cls(self.grl(share_hidden)), task)
        )
        share_loss = self.share_loss_weight * share_loss
        domain_loss = self.domain_loss_weight * self.prompt_loss(self.domain_cls(domain_hidden), domain)
        task_loss = self.task_loss_weight * self.prompt_loss(self.task_cls(task_hidden), task)
        res["loss"] = res["loss"] + share_loss + domain_loss + task_loss

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha


@Model.register("joint_lm_cp_ner_qkv_share_parser")
class JointLMCPNERQKVShareParser(JointLMCPQKVShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        share_loss_weight: float,
        domain_loss_weight: float,
        task_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERQKVShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight,
            share_loss_weight, domain_loss_weight, task_loss_weight, initializer, **kwargs)

        self.ner_proj = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
        )
        self.ner_head = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.vocab.get_vocab_size("ner_labels"))
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)
        mask = tokens["tokens"]["mask"]

        emb, share_hidden, domain_hidden, task_hidden = self.text_field_embedder(
            domain_ids=domain, task_ids=task, **tokens["tokens"])
        encoded_text = self.encoder(emb, mask)
        encoded_text = torch.cat([encoded_text[..., 0::2], encoded_text[..., 1::2]], dim=-1)

        if num_lm > 0:
            lm_logits = self.lm_head(self.lm_proj(encoded_text[lm_index, :, :]) + emb[lm_index, :, :])
            lm_loss = self.lm_loss(lm_logits.permute(0, 2, 1), lm_label)
            res["loss"] = lm_loss*self.lm_loss_weight + res["loss"]

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

        if num_ner > 0:
            ner_logits = self.ner_head(self.ner_proj(encoded_text[ner_index, 1:-1, :]))
            ner_loss = self.ner_loss(ner_logits.permute(0, 2, 1), ner_label[ner_index, :])
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        share_loss = (
            self.prompt_loss(self.share_domain_cls(self.grl(share_hidden)), domain) + 
            self.prompt_loss(self.share_task_cls(self.grl(share_hidden)), task)
        )
        share_loss = self.share_loss_weight * share_loss
        domain_loss = self.domain_loss_weight * self.prompt_loss(self.domain_cls(domain_hidden), domain)
        task_loss = self.task_loss_weight * self.prompt_loss(self.task_cls(task_hidden), task)
        res["loss"] = res["loss"] + share_loss + domain_loss + task_loss

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha


@Model.register("joint_lm_cp_ner_qkv_share_span3_parser")
class JointLMCPNERQKVShareSpan3Parser(JointLMCPQKVShareParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        ner_loss_weight: float,
        share_loss_weight: float,
        domain_loss_weight: float,
        task_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNERQKVShareSpan3Parser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight,
            share_loss_weight, domain_loss_weight, task_loss_weight, initializer, **kwargs)

        self.ner_head = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.vocab.get_vocab_size("ner_labels")-1)
        )
        self.ner_loss_weight = ner_loss_weight
        self.ner_loss = nn.CrossEntropyLoss()
        assert self.vocab.get_token_index("o", "ner_labels") == 0, "ner label error"

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        ner_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        ner_index = torch.nonzero(
            task == self.vocab.get_token_index("named_entity_recognition", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp, num_ner = lm_index.size(0), cp_index.size(0), ner_index.size(0)
        mask = tokens["tokens"]["mask"]

        emb, share_hidden, domain_hidden, task_hidden = self.text_field_embedder(
            domain_ids=domain, task_ids=task, **tokens["tokens"])
        encoded_text = self.encoder(emb, mask)
        encoded_text = torch.cat([encoded_text[..., 0::2], encoded_text[..., 1::2]], dim=-1)

        if num_lm > 0:
            lm_spans = torch.arange(mask.size(1)-2).unsqueeze(0).unsqueeze(-1).expand(num_lm, -1, 2).to(mask.device)
            lm_repre = self.span_extractor(encoded_text[lm_index, :, :], lm_spans)
            lm_logits = self.lm_head(self.lm_proj(lm_repre) + emb[lm_index, 1:-1, :])
            lm_loss = self.lm_loss(lm_logits.permute(0, 2, 1), lm_label[:, 1:-1])
            res["loss"] = lm_loss*self.lm_loss_weight + res["loss"]

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

        if num_ner > 0:
            ner_triu_indexes = torch.triu_indices(mask.size(1)-2, mask.size(1)-2)
            ner_spans = ner_triu_indexes.transpose(0, 1).unsqueeze(0).expand(num_ner, -1, -1).to(mask.device)
            ner_triu_repre = self.span_extractor(encoded_text[ner_index, :, :], ner_spans)
            ner_triu_logits = self.ner_head(ner_triu_repre)
            ner_triu_logits_zeros = ner_triu_logits.new_zeros((num_ner, ner_triu_logits.size(1), 1))
            ner_triu_logits = torch.cat([ner_triu_logits_zeros, ner_triu_logits], dim=-1)
            ner_triu_label = ner_label[ner_index.unsqueeze(1), ner_triu_indexes[0:1, :], ner_triu_indexes[1:2, :]]
            ner_loss = self.ner_loss(ner_triu_logits.permute(0, 2, 1), ner_triu_label)
            res["loss"] = ner_loss*self.ner_loss_weight + res["loss"]

        share_loss = (
            self.prompt_loss(self.share_domain_cls(self.grl(share_hidden)), domain) + 
            self.prompt_loss(self.share_task_cls(self.grl(share_hidden)), task)
        )
        share_loss = self.share_loss_weight * share_loss
        domain_loss = self.domain_loss_weight * self.prompt_loss(self.domain_cls(domain_hidden), domain)
        task_loss = self.task_loss_weight * self.prompt_loss(self.task_cls(task_hidden), task)
        res["loss"] = res["loss"] + share_loss + domain_loss + task_loss

        return res

    def weight_decay(self, alpha: float) -> None:
        super().weight_decay(alpha)
        self.ner_loss_weight = alpha
