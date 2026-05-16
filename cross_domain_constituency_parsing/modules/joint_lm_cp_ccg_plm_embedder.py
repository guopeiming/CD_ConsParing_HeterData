from allennlp.modules.token_embedders import TokenEmbedder
from typing import Tuple, Optional
import torch
from cross_domain_constituency_parsing.modules.joint_plm_embedder import JointPLMKVMLPEmbedder


@TokenEmbedder.register("joint_lm_cp_ccg_plm_kv_mlp_share")
class JointLMCPCCGPLMKVMLPShareEmbedder(JointPLMKVMLPEmbedder):

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