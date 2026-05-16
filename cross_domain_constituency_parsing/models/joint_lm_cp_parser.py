from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TokenEmbedder
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from cross_domain_constituency_parsing.utils.cky import CKY
from cross_domain_constituency_parsing.utils.tree_structure import (
    construct_bracketed_parse_tree, Tree, get_tree_triples, debinarization, delete_punctuation)
from cross_domain_constituency_parsing.metrics.constituency_parsing_f1_score import ConstituencyParsingF1Score
from cross_domain_constituency_parsing.metrics.my_evalb_bracketing_scorer import MyEvalb
from cross_domain_constituency_parsing.modules.GRL import GRL
from cross_domain_constituency_parsing.modules.joint_plm_embedder import JointPLMEmbedder
from cross_domain_constituency_parsing.modules.constituency_parsing_margin_loss import ConstituencyParsingMarginLoss
from copy import deepcopy
import numpy as np
import logging


logger = logging.getLogger(__name__)


class JointLMCPParser(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPParser, self).__init__(vocab, **kwargs)

        self.text_field_embedder: JointPLMEmbedder = text_field_embedder
        self.encoder = encoder
        self.span_extractor = span_extractor

        self.num_labels = self.vocab.get_vocab_size("constituency_labels")
        # 打分函数
        self.score_label = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.num_labels-1),
        )
        self.margin_loss = ConstituencyParsingMarginLoss()

        self.star_label_index = self.vocab.get_token_index("*", "constituency_labels")
        assert self.star_label_index == 0, "star label index error"

        self.parsing_score = ConstituencyParsingF1Score()
        self.evalb_score = MyEvalb()

        self.lm_head = self.text_field_embedder.lm_head
        self.lm_mask_token_id = self.text_field_embedder.tokenizer.mask_token_id
        self.lm_vocab_size = self.text_field_embedder.tokenizer.vocab_size
        self.lm_loss_weight = lm_loss_weight
        self.lm_loss = nn.CrossEntropyLoss()

        initializer(self)

    def language_model_nshare_mask(
        self, tokens: TextFieldTensors, lm_index: torch.Tensor, metadata: Optional[List[Dict[str, Any]]] = None
    ) -> torch.Tensor:
        token_ids = tokens["tokens"]["token_ids"]
        mask = tokens["tokens"]["mask"]
        offset = tokens["tokens"]["offsets"]
        lm_label = token_ids.new_full((lm_index.size(0), token_ids.size(1)), -100)

        for i, j in enumerate(lm_index):
            seq_len = mask[j].sum().item() - 2
            p = metadata[j]["lm_boundary"] if metadata is not None else None
            mask_posi_list = np.random.choice(seq_len, round(seq_len*0.3), replace=False, p=p)
            mask_posi_list = mask_posi_list + 1  # because we do not mask cls and sep
            mask_type_list = np.random.choice(3, mask_posi_list.size, replace=True, p=[0.8, 0.1, 0.1])
    
            for mask_posi, mask_type in zip(mask_posi_list, mask_type_list):
                subword_mask_posi_s, subword_mask_posi_e = offset[j, mask_posi, 0], offset[j, mask_posi, 1]+1
                lm_label[i, subword_mask_posi_s:subword_mask_posi_e] = token_ids[j, subword_mask_posi_s:subword_mask_posi_e]
                if mask_type == 0:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = self.lm_mask_token_id
                elif mask_type == 1:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = np.random.randint(0, high=self.lm_vocab_size)

        tokens["tokens"]["token_ids"] = token_ids
        return lm_label

    def language_model_share_mask(
        self, tokens: TextFieldTensors, lm_index: torch.Tensor, metadata: Optional[List[Dict[str, Any]]] = None
    ) -> torch.Tensor:
        token_ids = tokens["tokens"]["token_ids"]
        mask = tokens["tokens"]["mask"]
        offset = tokens["tokens"]["offsets"]
        lm_label = token_ids.new_full((lm_index.size(0), mask.size(1)), -100)

        for i, j in enumerate(lm_index):
            seq_len = mask[j].sum().item() - 2
            p = metadata[j]["lm_boundary"] if metadata is not None else None
            mask_posi_list = np.random.choice(seq_len, round(seq_len*0.3), replace=False, p=p)
            mask_posi_list = mask_posi_list + 1  # because we do not mask cls and sep
            mask_type_list = np.random.choice(3, mask_posi_list.size, replace=True, p=[0.8, 0.1, 0.1])

            for mask_posi, mask_type in zip(mask_posi_list, mask_type_list):
                subword_mask_posi_s, subword_mask_posi_e = offset[j, mask_posi, 0], offset[j, mask_posi, 1]+1
                lm_label[i, mask_posi] = token_ids[j, subword_mask_posi_s]
                if mask_type == 0:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = self.lm_mask_token_id
                elif mask_type == 1:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = np.random.randint(0, high=self.lm_vocab_size)

        tokens["tokens"]["token_ids"] = token_ids
        return lm_label

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

    def _parsing(
        self, triu_repre: torch.Tensor, triu_indexes: torch.Tensor, gold_tree_label: torch.Tensor,
        batch_size: int, seq_len: int, seq_lens: torch.Tensor, tokens: List[List[str]], postags: List[List[str]]
    ) -> Tuple[List[Tree], torch.Tensor, torch.Tensor, torch.Tensor]:
        triu_logits = self.score_label(triu_repre)
        triu_logits = torch.cat([triu_logits.new_zeros((batch_size, triu_logits.size(1), 1)), triu_logits], dim=-1)
        logits = triu_logits.new_zeros(batch_size, seq_len, seq_len, self.num_labels)
        batch_tensor = torch.arange(batch_size)
        logits[batch_tensor.unsqueeze(1), triu_indexes[0:1, :], triu_indexes[1:2, :], :] = triu_logits
        logits[batch_tensor, 0, seq_lens-1, self.star_label_index] -= 1e9

        if gold_tree_label is not None:
            # gold_event = F.one_hot(gold_tree_label+1, num_classes=self.num_labels+1)
            # gold_event = gold_event[:, :, :, 1:]
            gold_event = F.one_hot(F.relu(gold_tree_label), num_classes=self.num_labels)
            if self.training:
                Haming_augment = 1. - gold_event
                logits = logits + Haming_augment

        pred_event, pred_split = CKY(logits.data.cpu().numpy(), seq_lens.tolist())
        pred_tree_structure = self.construct_tree_structure(pred_event, pred_split, tokens, postags, seq_lens)

        return pred_tree_structure, logits, pred_event, gold_event

    def weight_decay(self, alpha: float) -> None:
        self.lm_loss_weight = alpha


@Model.register("joint_lm_cp_nshare_parser")
class JointLMCPNShareParser(JointLMCPParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPNShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp = lm_index.size(0), cp_index.size(0)

        lm_emb, lm_mask, cp_emb, cp_mask = self.text_field_embedder(
            domain_ids=domain, task_ids=task, **tokens["tokens"],
            lm_index=lm_index, cp_index=cp_index, num_lm=num_lm, num_cp=num_cp)

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

        return res


@Model.register("joint_lm_cp_share_parser")
class JointLMCPShareParser(JointLMCPParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)
        self.lm_proj = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
        )

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp = lm_index.size(0), cp_index.size(0)
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

        return res


@Model.register("joint_lm_cp_share_span3_parser")
class JointLMCPShareSpan3Parser(JointLMCPParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPShareSpan3Parser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)
        self.lm_proj = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
        )

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None,
        cp_weight: torch.FloatTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp = lm_index.size(0), cp_index.size(0)
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

        return res


@Model.register("joint_lm_cp_qkv_nshare_parser")
class JointLMCPQKVNShareParser(JointLMCPParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        share_loss_weight: float,
        domain_loss_weight: float,
        task_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPQKVNShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.grl = GRL()
        self.share_domain_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("domain_labels")),

        )
        self.share_task_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("task_labels")),
        )
        self.domain_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("domain_labels")),

        )
        self.task_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("task_labels")),
        )
        self.share_loss_weight = share_loss_weight
        self.domain_loss_weight = domain_loss_weight
        self.task_loss_weight = task_loss_weight
        self.prompt_loss = nn.CrossEntropyLoss()

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp = lm_index.size(0), cp_index.size(0)

        lm_emb, lm_mask, cp_emb, cp_mask, share_hidden, domain_hidden, task_hidden = self.text_field_embedder(
            domain_ids=domain, task_ids=task, **tokens["tokens"],
            lm_index=lm_index, cp_index=cp_index, num_lm=num_lm, num_cp=num_cp)

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
        self.share_loss_weight = alpha
        self.domain_loss_weight = alpha
        self.task_loss_weight = alpha

    def set_grl_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)


@Model.register("joint_lm_cp_qkv_share_parser")
class JointLMCPQKVShareParser(JointLMCPParser):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder,
        span_extractor: SpanExtractor,
        lm_loss_weight: float,
        share_loss_weight: float,
        domain_loss_weight: float,
        task_loss_weight: float,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super(JointLMCPQKVShareParser, self).__init__(
            vocab, text_field_embedder, encoder, span_extractor, lm_loss_weight, initializer, **kwargs)

        self.lm_proj = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), self.text_field_embedder.get_output_dim()),
            nn.GELU(),
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
        )

        self.grl = GRL()
        self.share_domain_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("domain_labels")),

        )
        self.share_task_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("task_labels")),
        )
        self.domain_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("domain_labels")),

        )
        self.task_cls = nn.Sequential(
            nn.Linear(self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()//3),
            nn.GELU(),
            nn.LayerNorm(self.text_field_embedder.get_output_dim()//3),
            nn.Linear(self.text_field_embedder.get_output_dim()//3, self.vocab.get_vocab_size("task_labels")),
        )
        self.share_loss_weight = share_loss_weight
        self.domain_loss_weight = domain_loss_weight
        self.task_loss_weight = task_loss_weight
        self.prompt_loss = nn.CrossEntropyLoss()

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        task: torch.LongTensor,
        domain: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_tree_label: torch.LongTensor = None,
        lm_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        res = {"loss": torch.tensor(0., device=task.device)}

        lm_index = torch.nonzero(
            task == self.vocab.get_token_index("language_model", "task_labels"), as_tuple=True)[0]
        cp_index = torch.nonzero(
            task == self.vocab.get_token_index("constituency_parsing", "task_labels"), as_tuple=True)[0]
        num_lm, num_cp = lm_index.size(0), cp_index.size(0)
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
        self.share_loss_weight = alpha
        self.domain_loss_weight = alpha
        self.task_loss_weight = alpha

    def set_grl_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)