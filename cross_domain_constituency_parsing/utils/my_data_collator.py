from typing import List
from allennlp.data.data_loaders.data_collator import DataCollator, allennlp_collate
from allennlp.data.data_loaders.data_loader import TensorDict
from allennlp.data.instance import Instance
import numpy as np
import torch
from transformers import AutoTokenizer


@DataCollator.register("my_data_collator")
class MyDataCollator(DataCollator):

    def __init__(self,  plm_name: str, structure_lm: bool, mlm_probability: float = 0.3) -> None:
        super(MyDataCollator, self).__init__()
        self._mlm_prob = mlm_probability
        self._task_vocab = dict()

        self._structure_lm = structure_lm
        tokenizer = AutoTokenizer.from_pretrained(plm_name)
        self._lm_mask_token_id = tokenizer.mask_token_id
        self._lm_vocab_size = tokenizer.vocab_size

    def __call__(self, instances: List[Instance]) -> TensorDict:
        tensor_dicts = allennlp_collate(instances)
        self._collect_task_vocab(tensor_dicts)

        if "constituency_parsing" in self._task_vocab:
            task_tensor = tensor_dicts["task"]
            for i in range(task_tensor.size(0)):
                if task_tensor[i] != self._task_vocab["constituency_parsing"]:
                    if np.random.rand() <= 0.5:
                        task_tensor[i] = self._task_vocab["constituency_parsing"]

        if "language_model" in self._task_vocab:
            self._language_model_mask(tensor_dicts)

        return tensor_dicts

    def _language_model_mask(self, tensor_dicts: TensorDict) -> None:
        lm_index = torch.nonzero(tensor_dicts["task"] == self._task_vocab["language_model"], as_tuple=True)[0]
        token_ids = tensor_dicts["tokens"]["tokens"]["token_ids"]
        mask = tensor_dicts["tokens"]["tokens"]["mask"]
        offset = tensor_dicts["tokens"]["tokens"]["offsets"]
        metadata = tensor_dicts["metadata"]
        lm_label = token_ids.new_full((lm_index.size(0), mask.size(1)), -100)

        for i, j in enumerate(lm_index):
            seq_len = mask[j].sum().item() - 2
            p = metadata[j]["lm_boundary"] if self._structure_lm else None
            mask_posi_list = np.random.choice(seq_len, round(seq_len*self._mlm_prob), replace=False, p=p)
            mask_posi_list = mask_posi_list + 1  # because we do not mask cls and sep
            mask_type_list = np.random.choice(3, mask_posi_list.size, replace=True, p=[0.8, 0.1, 0.1])

            for mask_posi, mask_type in zip(mask_posi_list, mask_type_list):
                subword_mask_posi_s, subword_mask_posi_e = offset[j, mask_posi, 0], offset[j, mask_posi, 1]+1
                lm_label[i, mask_posi] = token_ids[j, subword_mask_posi_s]
                if mask_type == 0:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = self._lm_mask_token_id
                elif mask_type == 1:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = np.random.randint(0, high=self._lm_vocab_size)

        tensor_dicts["lm_label"] = lm_label
        return lm_label

    def _collect_task_vocab(self, tensor_dicts: TensorDict) -> None:
        for i, metadata in enumerate(tensor_dicts["metadata"]):
            if metadata["task"] not in self._task_vocab:
                self._task_vocab[metadata["task"]] = tensor_dicts["task"][i].item()

    def _language_model_mask_nshare(self, tensor_dicts: TensorDict) -> None:
        lm_index = torch.nonzero(tensor_dicts["task"] == self._task_vocab["language_model"], as_tuple=True)[0]
        token_ids = tensor_dicts["tokens"]["tokens"]["token_ids"]
        mask = tensor_dicts["tokens"]["tokens"]["mask"]
        offset = tensor_dicts["tokens"]["tokens"]["offsets"]
        metadata = tensor_dicts["metadata"]
        lm_label = token_ids.new_full((lm_index.size(0), token_ids.size(1)), -100)

        for i, j in enumerate(lm_index):
            seq_len = mask[j].sum().item() - 2
            p = metadata[j]["lm_boundary"] if self._structure_lm else None
            mask_posi_list = np.random.choice(seq_len, round(seq_len*self._mlm_prob), replace=False, p=p)
            mask_posi_list = mask_posi_list + 1  # because we do not mask cls and sep
            mask_type_list = np.random.choice(3, mask_posi_list.size, replace=True, p=[0.8, 0.1, 0.1])
    
            for mask_posi, mask_type in zip(mask_posi_list, mask_type_list):
                subword_mask_posi_s, subword_mask_posi_e = offset[j, mask_posi, 0], offset[j, mask_posi, 1]+1
                lm_label[i, subword_mask_posi_s:subword_mask_posi_e] = token_ids[j, subword_mask_posi_s:subword_mask_posi_e]
                if mask_type == 0:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = self._lm_mask_token_id
                elif mask_type == 1:
                    token_ids[j, subword_mask_posi_s:subword_mask_posi_e] = np.random.randint(0, high=self._lm_vocab_size)

        tensor_dicts["tokens"]["tokens"]["my_lm_label"] = lm_label
        return lm_label