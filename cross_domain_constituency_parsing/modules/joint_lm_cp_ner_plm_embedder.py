from allennlp.modules.token_embedders import TokenEmbedder
from typing import Tuple, Optional
import torch
from cross_domain_constituency_parsing.modules.joint_plm_embedder import (
    JointPLMKVEmbedder, JointPLMKVMLPEmbedder, JointPLMQKVEmbedder, JointPLMQKVMLPEmbedder
)


@TokenEmbedder.register("joint_lm_cp_ner_plm_kv")
class JointLMCPNERPLMKVEmbedder(JointPLMKVEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
        lm_index: torch.BoolTensor, cp_index: torch.BoolTensor, ner_index: torch.Tensor,
        num_lm: int, num_cp: int, num_ner: int
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        self._flush_prompt()

        lm_hidden, lm_mask = None, None
        if num_lm > 0:
            lm_hidden = plm_output[lm_index, :, :]
            lm_mask = wordpiece_mask[lm_index, :]

        cp_hidden, cp_mask = None, None
        if num_cp > 0:
            cp_mask = mask[cp_index, :]
            cp_len = cp_mask.sum(-1).max()
            cp_mask = cp_mask[:, :cp_len]
            cp_offsets = offsets[cp_index, :cp_len, 0]
            cp_hidden = plm_output[cp_index.unsqueeze(1), cp_offsets, :]

        ner_hidden, ner_mask = None, None
        if num_ner > 0:
            ner_offsets = offsets[ner_index, :, 0]
            ner_hidden = plm_output[ner_index.unsqueeze(1), ner_offsets, :]
            ner_mask = mask[ner_index, :]

        return lm_hidden, lm_mask, cp_hidden, cp_mask, ner_hidden, ner_mask


@TokenEmbedder.register("joint_lm_cp_ner_plm_qkv")
class JointLMCPNERPLMQKVEmbedder(JointPLMQKVEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
        lm_index: torch.BoolTensor, cp_index: torch.BoolTensor, ner_index: torch.Tensor,
        num_lm: int, num_cp: int, num_ner: int
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        share_repres, domain_repres, task_repres = [], [], []
        for atten_layer in self.attention_layers:
            share_repre, domain_repre, task_repre = atten_layer.prompt_layer
            share_repres.append(share_repre)
            domain_repres.append(domain_repre)
            task_repres.append(task_repre)
        share_repres = torch.mean(torch.cat(share_repres, dim=1), dim=1)
        domain_repres = torch.mean(torch.cat(domain_repres, dim=1), dim=1)
        task_repres = torch.mean(torch.cat(task_repres, dim=1), dim=1)
        self._flush_prompt()

        lm_hidden, lm_mask = None, None
        if num_lm > 0:
            lm_hidden = plm_output[lm_index, :, :]
            lm_mask = wordpiece_mask[lm_index, :]

        cp_hidden, cp_mask = None, None
        if num_cp > 0:
            cp_mask = mask[cp_index, :]
            cp_len = cp_mask.sum(-1).max()
            cp_mask = cp_mask[:, :cp_len]
            cp_offsets = offsets[cp_index, :cp_len, 0]
            cp_hidden = plm_output[cp_index.unsqueeze(1), cp_offsets, :]

        ner_hidden, ner_mask = None, None
        if num_ner > 0:
            ner_offsets = offsets[ner_index, :, 0]
            ner_hidden = plm_output[ner_index.unsqueeze(1), ner_offsets, :]
            ner_mask = mask[ner_index, :]

        return lm_hidden, lm_mask, cp_hidden, cp_mask, ner_hidden, ner_mask, share_repres, domain_repres, task_repres


@TokenEmbedder.register("joint_lm_cp_ner_plm_kv_mlp")
class JointLMCPNERPLMKVMLPEmbedder(JointPLMKVMLPEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
        lm_index: torch.BoolTensor, cp_index: torch.BoolTensor, ner_index: torch.Tensor,
        num_lm: int, num_cp: int, num_ner: int
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        self._flush_prompt()

        lm_hidden, lm_mask = None, None
        if num_lm > 0:
            lm_hidden = plm_output[lm_index, :, :]
            lm_mask = wordpiece_mask[lm_index, :]

        cp_hidden, cp_mask = None, None
        if num_cp > 0:
            cp_mask = mask[cp_index, :]
            cp_len = cp_mask.sum(-1).max()
            cp_mask = cp_mask[:, :cp_len]
            cp_offsets = offsets[cp_index, :cp_len, 0]
            cp_hidden = plm_output[cp_index.unsqueeze(1), cp_offsets, :]

        ner_hidden, ner_mask = None, None
        if num_ner > 0:
            ner_offsets = offsets[ner_index, :, 0]
            ner_hidden = plm_output[ner_index.unsqueeze(1), ner_offsets, :]
            ner_mask = mask[ner_index, :]

        return lm_hidden, lm_mask, cp_hidden, cp_mask, ner_hidden, ner_mask


@TokenEmbedder.register("joint_lm_cp_ner_plm_qkv_mlp")
class JointLMCPNERPLMQKVMLPEmbedder(JointPLMQKVMLPEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
        lm_index: torch.BoolTensor, cp_index: torch.BoolTensor, ner_index: torch.Tensor,
        num_lm: int, num_cp: int, num_ner: int
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        share_repres, domain_repres, task_repres = [], [], []
        for atten_layer in self.attention_layers:
            share_repre, domain_repre, task_repre = atten_layer.prompt_layer
            share_repres.append(share_repre)
            domain_repres.append(domain_repre)
            task_repres.append(task_repre)
        share_repres = torch.mean(torch.cat(share_repres, dim=1), dim=1)
        domain_repres = torch.mean(torch.cat(domain_repres, dim=1), dim=1)
        task_repres = torch.mean(torch.cat(task_repres, dim=1), dim=1)
        self._flush_prompt()

        lm_hidden, lm_mask = None, None
        if num_lm > 0:
            lm_hidden = plm_output[lm_index, :, :]
            lm_mask = wordpiece_mask[lm_index, :]

        cp_hidden, cp_mask = None, None
        if num_cp > 0:
            cp_mask = mask[cp_index, :]
            cp_len = cp_mask.sum(-1).max()
            cp_mask = cp_mask[:, :cp_len]
            cp_offsets = offsets[cp_index, :cp_len, 0]
            cp_hidden = plm_output[cp_index.unsqueeze(1), cp_offsets, :]

        ner_hidden, ner_mask = None, None
        if num_ner > 0:
            ner_offsets = offsets[ner_index, :, 0]
            ner_hidden = plm_output[ner_index.unsqueeze(1), ner_offsets, :]
            ner_mask = mask[ner_index, :]

        return lm_hidden, lm_mask, cp_hidden, cp_mask, ner_hidden, ner_mask, share_repres, domain_repres, task_repres


@TokenEmbedder.register("joint_lm_cp_ner_plm_kv_share")
class JointLMCPNERPLMKVShareEmbedder(JointPLMKVEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        self._flush_prompt()

        hidden = plm_output[torch.arange(token_ids.size(0)).unsqueeze(1), offsets[:, :, 0], :]
        return hidden


@TokenEmbedder.register("joint_lm_cp_ner_plm_kv_mlp_share")
class JointLMCPNERPLMKVMLPShareEmbedder(JointPLMKVMLPEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        self._flush_prompt()

        hidden = plm_output[torch.arange(token_ids.size(0)).unsqueeze(1), offsets[:, :, 0], :]
        return hidden


@TokenEmbedder.register("joint_lm_cp_ner_plm_qkv_share")
class JointLMCPNERPLMQKVShareEmbedder(JointPLMQKVEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        share_repres, domain_repres, task_repres = [], [], []
        for atten_layer in self.attention_layers:
            share_repre, domain_repre, task_repre = atten_layer.prompt_layer
            share_repres.append(share_repre)
            domain_repres.append(domain_repre)
            task_repres.append(task_repre)
        share_repres = torch.mean(torch.cat(share_repres, dim=1), dim=1)
        domain_repres = torch.mean(torch.cat(domain_repres, dim=1), dim=1)
        task_repres = torch.mean(torch.cat(task_repres, dim=1), dim=1)
        self._flush_prompt()

        hidden = plm_output[torch.arange(token_ids.size(0)).unsqueeze(1), offsets[:, :, 0], :]
        return hidden, share_repres, domain_repres, task_repres


@TokenEmbedder.register("joint_lm_cp_ner_plm_qkv_mlp_share")
class JointLMCPNERPLMQKVMLPShareEmbedder(JointPLMQKVMLPEmbedder):

    def forward(
        self,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor]:
        self._insert_prompt(domain_ids, task_ids)
        plm_output = self.plm(token_ids, wordpiece_mask, type_ids)["last_hidden_state"]
        share_repres, domain_repres, task_repres = [], [], []
        for atten_layer in self.attention_layers:
            share_repre, domain_repre, task_repre = atten_layer.prompt_layer
            share_repres.append(share_repre)
            domain_repres.append(domain_repre)
            task_repres.append(task_repre)
        share_repres = torch.mean(torch.cat(share_repres, dim=1), dim=1)
        domain_repres = torch.mean(torch.cat(domain_repres, dim=1), dim=1)
        task_repres = torch.mean(torch.cat(task_repres, dim=1), dim=1)
        self._flush_prompt()

        hidden = plm_output[torch.arange(token_ids.size(0)).unsqueeze(1), offsets[:, :, 0], :]
        return hidden, share_repres, domain_repres, task_repres
