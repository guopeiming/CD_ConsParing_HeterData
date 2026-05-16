from allennlp.modules.token_embedders import TokenEmbedder
from typing import Tuple, Optional
import torch
from cross_domain_constituency_parsing.modules.joint_plm_embedder import (
    JointPLMKVEmbedder, JointPLMKVMLPEmbedder, JointPLMQKVEmbedder, JointPLMQKVMLPEmbedder, JointPLMKVMLPOLDEmbedder
)


@TokenEmbedder.register("joint_lm_cp_ner_ccg_plm_kv_mlp_share")
class JointLMCPNERCCGPLMKVMLPShareEmbedder(JointPLMKVMLPEmbedder):

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


@TokenEmbedder.register("joint_lm_cp_ner_ccg_plm_qkv_mlp_share")
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


@TokenEmbedder.register("joint_lm_cp_ner_ccg_plm_kv_mlp_share_old")
class JointLMCPNERCCGPLMKVMLPShareOldEmbedder(JointPLMKVMLPOLDEmbedder):

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
