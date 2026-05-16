from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError
from transformers import AutoModelForMaskedLM, BertModel, BertForMaskedLM, AutoTokenizer, BertTokenizer
from transformers.models.bert.modeling_bert import BertSelfAttention
import torch.nn as nn
from typing import Tuple, Optional, Union
import torch
import math


class JointPLMEmbedder(TokenEmbedder):

    def __init__(self, model_name: str) -> None:
        super(JointPLMEmbedder, self).__init__()
        model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
        self.plm: BertModel = model.bert
        self.lm_head = model.cls
        self.output_dim = self.plm.config.hidden_size
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_output_dim(self) -> int:
        return self.output_dim

    def _prompt_init(
        self, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> Union[nn.ModuleList, Tuple[nn.ModuleList]]:
        raise NotImplementedError()

    def _insert_prompt(self, domain_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
        raise NotImplementedError()

    def _flush_prompt(self) -> None:
        raise NotImplementedError()


class JointPLMKVEmbedder(JointPLMEmbedder):

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        requires_grad: bool,
        share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> None:
        super(JointPLMKVEmbedder, self).__init__(model_name)
        self.vocab = vocab
        self.plm.requires_grad_(requires_grad=requires_grad)

        self.attention_layers, self.prompt_layers = \
            self._prompt_init(share_prompt_len, domain_prompt_len, task_prompt_len)
        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len

    def _prompt_init(
        self, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> Tuple[nn.ModuleList]:
        attention_layers = nn.ModuleList()
        prompt_layers = nn.ModuleList()
        prompt_len = sum([share_prompt_len, domain_prompt_len, task_prompt_len])

        for transf_layer in self.plm.encoder.layer:
            heads = transf_layer.attention.self.num_attention_heads
            dim = transf_layer.attention.self.attention_head_size

            prompts = dict()
            if share_prompt_len > 0:
                prompts["share_k"] = nn.Parameter(
                    nn.init.normal_(torch.empty(heads, share_prompt_len, dim), mean=0., std=1e-3))
                prompts["share_v"] = nn.Parameter(
                    nn.init.normal_(torch.empty(heads, share_prompt_len, dim), mean=0., std=1e-3))

            prompts["domain_k"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("domain_labels"), heads, domain_prompt_len, dim), mean=0., std=1e-3))
            prompts["domain_v"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("domain_labels"), heads, domain_prompt_len, dim), mean=0., std=1e-3))
            prompts["task_k"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("task_labels"), heads, task_prompt_len, dim), mean=0., std=1e-3))
            prompts["task_v"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("task_labels"), heads, task_prompt_len, dim), mean=0., std=1e-3))

            atten_layer = BertSelfAttentionKV(transf_layer.attention.self, prompt_len)
            transf_layer.attention.self = atten_layer
            transf_layer.attention.output.LayerNorm.requires_grad_(requires_grad=True)
            prompts = nn.ParameterDict(prompts)

            attention_layers.append(atten_layer)
            prompt_layers.append(prompts)

        return attention_layers, prompt_layers

    def _insert_prompt(self, domain_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
        batch_size = domain_ids.size(0)
        for prompts, atten_layer in zip(self.prompt_layers, self.attention_layers):
            for k, v in prompts.items():
                if k.startswith("share"):
                    paramter = v.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    setattr(atten_layer, k, paramter)
                elif k.startswith("domain"):
                    paramter = v[domain_ids, :, :, :]
                    setattr(atten_layer, k, paramter)
                elif k.startswith("task"):
                    paramter = v[task_ids, :, :, :]
                    setattr(atten_layer, k, paramter)
                else:
                    raise ConfigurationError("prompt key error")

    def _flush_prompt(self) -> None:
        for atten_layer in self.attention_layers:
            setattr(atten_layer, "share_k", None)
            setattr(atten_layer, "share_v", None)
            setattr(atten_layer, "domain_k", None)
            setattr(atten_layer, "domain_v", None)
            setattr(atten_layer, "task_k", None)
            setattr(atten_layer, "task_v", None)


class BertSelfAttentionKV(nn.Module):

    def __init__(
        self, base: BertSelfAttention, prompt_len: int
    ) -> None:
        super(BertSelfAttentionKV, self).__init__()
        self.base = base
        assert not self.base.is_decoder, "is_decoder error"
        assert self.base.position_embedding_type == "absolute", "position_embedding_type error"
        self.prompt_len = prompt_len
        self.share_k = None
        self.share_v = None
        self.domain_k = None
        self.domain_v = None
        self.task_k = None
        self.task_v = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        assert (past_key_value is None) and (encoder_hidden_states is None), "argument error"
        batch_size, _, seq_len, _ = attention_mask.size()

        prompt_k, prompt_v = [], []
        if self.share_k is not None:
            prompt_k.append(self.share_k)
            prompt_v.append(self.share_v)
        prompt_k.extend([self.domain_k, self.task_k])
        prompt_v.extend([self.domain_v, self.task_v])

        prompt_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.prompt_len))
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=-1)

        # code of BertSelfAttention
        query_layer = self.base.transpose_for_scores(self.base.query(hidden_states))
        key_layer = self.base.transpose_for_scores(self.base.key(hidden_states))
        value_layer = self.base.transpose_for_scores(self.base.value(hidden_states))
        prompt_k.append(key_layer)
        prompt_v.append(value_layer)
        key_layer = torch.cat(prompt_k, dim=2)
        value_layer = torch.cat(prompt_v, dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.base.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.base.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.base.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class JointPLMQKVEmbedder(JointPLMEmbedder):

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        requires_grad: bool,
        share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int, prompt_dim: int
    ) -> None:
        super(JointPLMQKVEmbedder, self).__init__(model_name)
        self.vocab = vocab
        self.plm.requires_grad_(requires_grad=requires_grad)

        self.attention_layers, self.prompt_layers = \
            self._prompt_init(share_prompt_len, domain_prompt_len, task_prompt_len)
        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len

    def _prompt_init(
        self, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> Tuple[nn.ModuleList]:
        attention_layers = nn.ModuleList()
        prompt_layers = nn.ModuleList()

        for transf_layer in self.plm.encoder.layer:
            heads = transf_layer.attention.self.num_attention_heads
            dim = transf_layer.attention.self.attention_head_size

            prompts = dict()
            if share_prompt_len > 0:
                prompts["share_q"] = nn.Parameter(
                    nn.init.normal_(torch.empty(heads, share_prompt_len, dim), mean=0., std=1e-3))
                prompts["share_k"] = nn.Parameter(
                    nn.init.normal_(torch.empty(heads, share_prompt_len, dim), mean=0., std=1e-3))
                prompts["share_v"] = nn.Parameter(
                    nn.init.normal_(torch.empty(heads, share_prompt_len, dim), mean=0., std=1e-3))

            prompts["domain_q"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("domain_labels"), heads, domain_prompt_len, dim), mean=0., std=1e-3))
            prompts["domain_k"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("domain_labels"), heads, domain_prompt_len, dim), mean=0., std=1e-3))
            prompts["domain_v"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("domain_labels"), heads, domain_prompt_len, dim), mean=0., std=1e-3))
            prompts["task_q"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("task_labels"), heads, task_prompt_len, dim), mean=0., std=1e-3))
            prompts["task_k"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("task_labels"), heads, task_prompt_len, dim), mean=0., std=1e-3))
            prompts["task_v"] = nn.Parameter(nn.init.normal_(torch.empty(
                self.vocab.get_vocab_size("task_labels"), heads, task_prompt_len, dim), mean=0., std=1e-3))

            atten_layer = BertSelfAttentionQKV(
                transf_layer.attention.self, share_prompt_len, domain_prompt_len, task_prompt_len)
            transf_layer.attention.self = atten_layer
            transf_layer.attention.output.LayerNorm.requires_grad_(requires_grad=True)
            prompts = nn.ParameterDict(prompts)

            attention_layers.append(atten_layer)
            prompt_layers.append(prompts)

        return attention_layers, prompt_layers

    def _insert_prompt(self, domain_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
        batch_size = domain_ids.size(0)
        for prompts, atten_layer in zip(self.prompt_layers, self.attention_layers):
            for k, v in prompts.items():
                if k.startswith("share"):
                    paramter = v.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    setattr(atten_layer, k, paramter)
                elif k.startswith("domain"):
                    paramter = v[domain_ids, :, :, :]
                    setattr(atten_layer, k, paramter)
                elif k.startswith("task"):
                    paramter = v[task_ids, :, :, :]
                    setattr(atten_layer, k, paramter)
                else:
                    raise ConfigurationError("prompt key error")

    def _flush_prompt(self) -> None:
        for atten_layer in self.attention_layers:
            setattr(atten_layer, "share_q", None)
            setattr(atten_layer, "share_k", None)
            setattr(atten_layer, "share_v", None)
            setattr(atten_layer, "domain_q", None)
            setattr(atten_layer, "domain_k", None)
            setattr(atten_layer, "domain_v", None)
            setattr(atten_layer, "task_q", None)
            setattr(atten_layer, "task_k", None)
            setattr(atten_layer, "task_v", None)
            setattr(atten_layer, "prompt_layer", None)


class BertSelfAttentionQKV(nn.Module):

    def __init__(
        self, base: BertSelfAttention,
        share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> None:
        super(BertSelfAttentionQKV, self).__init__()
        self.base = base
        assert not self.base.is_decoder, "is_decoder error"
        assert self.base.position_embedding_type == "absolute", "position_embedding_type error"

        self.prompt_len = sum([share_prompt_len, domain_prompt_len, task_prompt_len])
        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len

        self.share_q = None
        self.share_k = None
        self.share_v = None
        self.domain_q = None
        self.domain_k = None
        self.domain_v = None
        self.task_q = None
        self.task_k = None
        self.task_v = None

        self.prompt_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        assert (past_key_value is None) and (encoder_hidden_states is None), "argument error"
        attention_mask = attention_mask.expand(-1, -1, attention_mask.size(-1), -1)
        batch_size, _, seq_len, _ = attention_mask.size()

        prompt_q, prompt_k, prompt_v = [], [], []
        if self.share_q is not None:
            prompt_q.append(self.share_q)
            prompt_k.append(self.share_k)
            prompt_v.append(self.share_v)

        prompt_q.append(self.domain_q)
        prompt_k.append(self.domain_k)
        prompt_v.append(self.domain_v)
        prompt_q.append(self.task_q)
        prompt_k.append(self.task_k)
        prompt_v.append(self.task_v)

        prompt_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.prompt_len))
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=-1)
        prompt_mask = attention_mask[:, :, 0:1, :].expand(-1, -1, self.prompt_len, -1)
        # prompt_mask[:, :, :, :self.prompt_len] = -10000.0
        # start_poi, end_poi = 0, 0
        # for length in [self.share_prompt_len, self.domain_prompt_len, self.task_prompt_len]:
        #     start_poi = end_poi
        #     end_poi += length
        #     prompt_mask[:, :, start_poi:end_poi, start_poi:end_poi] = 0.0
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=-2)

        # code of BertSelfAttention
        query_layer = self.base.transpose_for_scores(self.base.query(hidden_states))
        key_layer = self.base.transpose_for_scores(self.base.key(hidden_states))
        value_layer = self.base.transpose_for_scores(self.base.value(hidden_states))
        prompt_q.append(query_layer)
        prompt_k.append(key_layer)
        prompt_v.append(value_layer)
        query_layer = torch.cat(prompt_q, dim=2)
        key_layer = torch.cat(prompt_k, dim=2)
        value_layer = torch.cat(prompt_v, dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.base.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.base.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.base.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        prompt_layer, context_layer = context_layer.split([self.prompt_len, seq_len], dim=1)
        share_layer, domain_layer, task_layer = torch.split(
            prompt_layer, [self.share_prompt_len, self.domain_prompt_len, self.task_prompt_len], dim=1)
        self.prompt_layer = [
            torch.mean(share_layer, dim=1, keepdim=True),
            torch.mean(domain_layer, dim=1, keepdim=True),
            torch.mean(task_layer, dim=1, keepdim=True)
        ]  # batch_size, 1, hidden_dim

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class JointPLMKVMLPEmbedder(JointPLMEmbedder):

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        requires_grad: bool,
        share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int, prompt_dim: int
    ) -> None:
        super(JointPLMKVMLPEmbedder, self).__init__(model_name)
        self.vocab = vocab
        self.plm.requires_grad_(requires_grad=requires_grad)

        self.share_prompt = None
        self.domain_prompt = None
        self.task_prompt = None
        self.share_mlp = None
        self.domain_mlp = None
        self.task_mlp = None
        self.attention_layers = self._prompt_init(share_prompt_len, domain_prompt_len, task_prompt_len, prompt_dim)

        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len
        self.prompt_dim = prompt_dim

    def _prompt_init(
        self, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int, prompt_dim: int
    ) -> nn.ModuleList:
        attention_layers = nn.ModuleList()

        emb_dim = self.output_dim
        if share_prompt_len > 0:
            self.share_prompt = nn.Parameter(nn.init.normal_(torch.empty(share_prompt_len, emb_dim)))
            self.share_mlp = nn.Sequential(nn.Linear(emb_dim, prompt_dim), nn.Tanh())
        self.domain_prompt = nn.Embedding(self.vocab.get_vocab_size("domain_labels"), domain_prompt_len*emb_dim)
        self.task_prompt = nn.Embedding(self.vocab.get_vocab_size("task_labels"), task_prompt_len*emb_dim)
        self.domain_mlp = nn.Sequential(nn.Linear(emb_dim, prompt_dim), nn.Tanh())
        self.task_mlp = nn.Sequential(nn.Linear(emb_dim, prompt_dim), nn.Tanh())
        # nn.init.normal_(self.share_prompt, mean=0., std=1e-3)
        # nn.init.normal_(self.domain_prompt.weight, mean=0., std=1e-3)
        # nn.init.normal_(self.task_prompt.weight, mean=0., std=1e-3)

        for transf_layer in self.plm.encoder.layer:
            atten_layer = BertSelfAttentionKVMLP(
                transf_layer.attention.self, prompt_dim, share_prompt_len, domain_prompt_len, task_prompt_len)
            transf_layer.attention.self = atten_layer
            transf_layer.attention.output.LayerNorm.requires_grad_(requires_grad=True)
            attention_layers.append(atten_layer)

        return attention_layers

    def _insert_prompt(self, domain_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
        batch_size = domain_ids.size(0)
        for atten_layer in self.attention_layers:
            if self.share_prompt is not None:
                share_prompt = self.share_mlp(self.share_prompt)
                share_prompt = share_prompt.unsqueeze(0).expand(batch_size, -1, -1)
                setattr(atten_layer, "share_prompt", share_prompt)

            domain_prompt = self.domain_prompt(domain_ids)
            domain_prompt = domain_prompt.unsqueeze(-1).view(batch_size, self.domain_prompt_len, self.output_dim)
            domain_prompt = self.domain_mlp(domain_prompt)
            setattr(atten_layer, "domain_prompt", domain_prompt)

            task_prompt = self.task_prompt(task_ids)
            task_prompt = task_prompt.unsqueeze(-1).view(batch_size, self.task_prompt_len, self.output_dim)
            task_prompt = self.task_mlp(task_prompt)
            setattr(atten_layer, "task_prompt", task_prompt)

    def _flush_prompt(self) -> None:
        for atten_layer in self.attention_layers:
            setattr(atten_layer, "share_prompt", None)
            setattr(atten_layer, "domain_prompt", None)
            setattr(atten_layer, "task_prompt", None)


class BertSelfAttentionKVMLP(nn.Module):

    def __init__(
        self, base: BertSelfAttention,
        prompt_dim: int, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> None:
        super(BertSelfAttentionKVMLP, self).__init__()
        self.base = base
        assert not self.base.is_decoder, "is_decoder error"
        assert self.base.position_embedding_type == "absolute", "position_embedding_type error"
        self.num_heads, self.head_size = self.base.num_attention_heads, self.base.attention_head_size

        self.prompt_len = sum([share_prompt_len, domain_prompt_len, task_prompt_len])
        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len

        self.share_prompt = None
        self.domain_prompt = None
        self.task_prompt = None
        self.share_mlp = None
        if share_prompt_len > 0:
            self.share_mlp = nn.Sequential(
                # nn.Linear(self.num_heads*self.head_size, prompt_dim),
                # nn.Tanh(),
                nn.Linear(prompt_dim, self.num_heads*self.head_size*2)
            )
        self.domain_mlp = nn.Sequential(
            # nn.Linear(self.num_heads*self.head_size, prompt_dim),
            # nn.Tanh(),
            nn.Linear(prompt_dim, self.num_heads*self.head_size*2)
        )
        self.task_mlp = nn.Sequential(
            # nn.Linear(self.num_heads*self.head_size, prompt_dim),
            # nn.Tanh(),
            nn.Linear(prompt_dim, self.num_heads*self.head_size*2)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        assert (past_key_value is None) and (encoder_hidden_states is None), "argument error"
        attention_mask = attention_mask.expand(-1, -1, attention_mask.size(-1), -1)
        batch_size, _, seq_len, _ = attention_mask.size()

        prompt_k, prompt_v = [], []
        if self.share_prompt is not None:
            share_prompt = self.share_mlp(self.share_prompt).unsqueeze(-1)
            share_prompt = share_prompt.view(batch_size, self.share_prompt_len, self.num_heads, self.head_size*2)
            share_prompt_k, share_prompt_v = torch.split(share_prompt, self.head_size, dim=-1)
            prompt_k.append(share_prompt_k.permute(0, 2, 1, 3))
            prompt_v.append(share_prompt_v.permute(0, 2, 1, 3))

        domain_prompt = self.domain_mlp(self.domain_prompt).unsqueeze(-1)
        domain_prompt = domain_prompt.view(batch_size, self.domain_prompt_len, self.num_heads, self.head_size*2)
        domain_prompt_k, domain_prompt_v = torch.split(domain_prompt, self.head_size, dim=-1)
        prompt_k.append(domain_prompt_k.permute(0, 2, 1, 3))
        prompt_v.append(domain_prompt_v.permute(0, 2, 1, 3))
        task_prompt = self.task_mlp(self.task_prompt).unsqueeze(-1)
        task_prompt = task_prompt.view(batch_size, self.task_prompt_len, self.num_heads, self.head_size*2)
        task_prompt_k, task_prompt_v = torch.split(task_prompt, self.head_size, dim=-1)
        prompt_k.append(task_prompt_k.permute(0, 2, 1, 3))
        prompt_v.append(task_prompt_v.permute(0, 2, 1, 3))

        prompt_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.prompt_len))
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=-1)

        # code of BertSelfAttention
        query_layer = self.base.transpose_for_scores(self.base.query(hidden_states))
        key_layer = self.base.transpose_for_scores(self.base.key(hidden_states))
        value_layer = self.base.transpose_for_scores(self.base.value(hidden_states))
        prompt_k.append(key_layer)
        prompt_v.append(value_layer)
        key_layer = torch.cat(prompt_k, dim=2)
        value_layer = torch.cat(prompt_v, dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.base.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.base.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.base.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class JointPLMQKVMLPEmbedder(JointPLMEmbedder):

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        requires_grad: bool,
        share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int, prompt_dim: int
    ) -> None:
        super(JointPLMQKVMLPEmbedder, self).__init__(model_name)
        self.vocab = vocab
        self.plm.requires_grad_(requires_grad=requires_grad)

        self.share_prompt = None
        self.domain_prompt = None
        self.task_prompt = None
        self.share_mlp = None
        self.domain_mlp = None
        self.task_mlp = None
        self.attention_layers = self._prompt_init(share_prompt_len, domain_prompt_len, task_prompt_len, prompt_dim)

        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len
        self.prompt_dim = prompt_dim

    def _prompt_init(
        self, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int, prompt_dim: int
    ) -> nn.ModuleList:
        attention_layers = nn.ModuleList()

        emb_dim = self.plm.config.hidden_size
        if share_prompt_len > 0:
            self.share_prompt = nn.Parameter(nn.init.normal_(torch.empty(share_prompt_len, emb_dim)))
            self.share_mlp = nn.Sequential(nn.Linear(emb_dim, prompt_dim), nn.Tanh())
        self.domain_prompt = nn.Embedding(self.vocab.get_vocab_size("domain_labels"), domain_prompt_len*emb_dim)
        self.task_prompt = nn.Embedding(self.vocab.get_vocab_size("task_labels"), task_prompt_len*emb_dim)
        self.domain_mlp = nn.Sequential(nn.Linear(emb_dim, prompt_dim), nn.Tanh())
        self.task_mlp = nn.Sequential(nn.Linear(emb_dim, prompt_dim), nn.Tanh())
        # nn.init.normal_(self.share_prompt, mean=0., std=1e-3)
        # nn.init.normal_(self.domain_prompt.weight, mean=0., std=1e-3)
        # nn.init.normal_(self.task_prompt.weight, mean=0., std=1e-3)

        for transf_layer in self.plm.encoder.layer:
            atten_layer = BertSelfAttentionQKVMLP(
                transf_layer.attention.self, prompt_dim, share_prompt_len, domain_prompt_len, task_prompt_len)
            transf_layer.attention.self = atten_layer
            transf_layer.attention.output.LayerNorm.requires_grad_(requires_grad=True)
            attention_layers.append(atten_layer)

        return attention_layers

    def _insert_prompt(self, domain_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
        batch_size = domain_ids.size(0)
        for atten_layer in self.attention_layers:
            if self.share_prompt is not None:
                share_prompt = self.share_mlp(self.share_prompt)
                share_prompt = share_prompt.unsqueeze(0).expand(batch_size, -1, -1)
                setattr(atten_layer, "share_prompt", share_prompt)

            domain_prompt = self.domain_prompt(domain_ids)
            domain_prompt = domain_prompt.unsqueeze(-1).view(batch_size, self.domain_prompt_len, self.output_dim)
            domain_prompt = self.domain_mlp(domain_prompt)
            setattr(atten_layer, "domain_prompt", domain_prompt)

            task_prompt = self.task_prompt(task_ids)
            task_prompt = task_prompt.unsqueeze(-1).view(batch_size, self.task_prompt_len, self.output_dim)
            task_prompt = self.task_mlp(task_prompt)
            setattr(atten_layer, "task_prompt", task_prompt)

    def _flush_prompt(self) -> None:
        for atten_layer in self.attention_layers:
            setattr(atten_layer, "share_prompt", None)
            setattr(atten_layer, "domain_prompt", None)
            setattr(atten_layer, "task_prompt", None)
            setattr(atten_layer, "prompt_layer", None)


class BertSelfAttentionQKVMLP(nn.Module):

    def __init__(
        self, base: BertSelfAttention,
        prompt_dim: int, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> None:
        super(BertSelfAttentionQKVMLP, self).__init__()
        self.base = base
        assert not self.base.is_decoder, "is_decoder error"
        assert self.base.position_embedding_type == "absolute", "position_embedding_type error"
        self.num_heads, self.head_size = self.base.num_attention_heads, self.base.attention_head_size

        self.prompt_len = sum([share_prompt_len, domain_prompt_len, task_prompt_len])
        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len

        self.share_prompt = None
        self.domain_prompt = None
        self.task_prompt = None
        self.share_mlp = None
        if share_prompt_len > 0:
            self.share_mlp = nn.Sequential(
                # nn.Linear(self.num_heads*self.head_size, prompt_dim),
                # nn.Tanh(),
                nn.Linear(prompt_dim, self.num_heads*self.head_size*3)
            )
        self.domain_mlp = nn.Sequential(
            # nn.Linear(self.num_heads*self.head_size, prompt_dim),
            # nn.Tanh(),
            nn.Linear(prompt_dim, self.num_heads*self.head_size*3)
        )
        self.task_mlp = nn.Sequential(
            # nn.Linear(self.num_heads*self.head_size, prompt_dim),
            # nn.Tanh(),
            nn.Linear(prompt_dim, self.num_heads*self.head_size*3)
        )

        self.prompt_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        assert (past_key_value is None) and (encoder_hidden_states is None), "argument error"
        attention_mask = attention_mask.expand(-1, -1, attention_mask.size(-1), -1)
        batch_size, _, seq_len, _ = attention_mask.size()

        prompt_q, prompt_k, prompt_v = [], [], []
        if self.share_prompt is not None:
            share_prompt = self.share_mlp(self.share_prompt).unsqueeze(-1)
            share_prompt = share_prompt.view(batch_size, self.domain_prompt_len, self.num_heads, self.head_size*3)
            share_prompt_q, share_prompt_k, share_prompt_v = torch.split(share_prompt, self.head_size, dim=-1)
            prompt_q.append(share_prompt_q.permute(0, 2, 1, 3))
            prompt_k.append(share_prompt_k.permute(0, 2, 1, 3))
            prompt_v.append(share_prompt_v.permute(0, 2, 1, 3))

        domain_prompt = self.domain_mlp(self.domain_prompt).unsqueeze(-1)
        domain_prompt = domain_prompt.view(batch_size, self.domain_prompt_len, self.num_heads, self.head_size*3)
        domain_prompt_q, domain_prompt_k, domain_prompt_v = torch.split(domain_prompt, self.head_size, dim=-1)
        prompt_q.append(domain_prompt_q.permute(0, 2, 1, 3))
        prompt_k.append(domain_prompt_k.permute(0, 2, 1, 3))
        prompt_v.append(domain_prompt_v.permute(0, 2, 1, 3))
        task_prompt = self.task_mlp(self.task_prompt).unsqueeze(-1)
        task_prompt = task_prompt.view(batch_size, self.task_prompt_len, self.num_heads, self.head_size*3)
        task_prompt_q, task_prompt_k, task_prompt_v = torch.split(task_prompt, self.head_size, dim=-1)
        prompt_q.append(task_prompt_q.permute(0, 2, 1, 3))
        prompt_k.append(task_prompt_k.permute(0, 2, 1, 3))
        prompt_v.append(task_prompt_v.permute(0, 2, 1, 3))

        prompt_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.prompt_len))
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=-1)
        prompt_mask = attention_mask[:, :, 0:1, :].expand(-1, -1, self.prompt_len, -1)
        # prompt_mask[:, :, :, :self.prompt_len] = -10000.0
        # start_poi, end_poi = 0, 0
        # for length in [self.share_prompt_len, self.domain_prompt_len, self.task_prompt_len]:
        #     start_poi = end_poi
        #     end_poi += length
        #     prompt_mask[:, :, start_poi:end_poi, start_poi:end_poi] = 0.0
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=-2)

        # code of BertSelfAttention
        query_layer = self.base.transpose_for_scores(self.base.query(hidden_states))
        key_layer = self.base.transpose_for_scores(self.base.key(hidden_states))
        value_layer = self.base.transpose_for_scores(self.base.value(hidden_states))
        prompt_q.append(query_layer)
        prompt_k.append(key_layer)
        prompt_v.append(value_layer)
        query_layer = torch.cat(prompt_q, dim=2)
        key_layer = torch.cat(prompt_k, dim=2)
        value_layer = torch.cat(prompt_v, dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.base.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.base.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.base.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        prompt_layer, context_layer = context_layer.split([self.prompt_len, seq_len], dim=1)
        share_layer, domain_layer, task_layer = torch.split(
            prompt_layer, [self.share_prompt_len, self.domain_prompt_len, self.task_prompt_len], dim=1)
        self.prompt_layer = [
            torch.mean(share_layer, dim=1, keepdim=True),
            torch.mean(domain_layer, dim=1, keepdim=True),
            torch.mean(task_layer, dim=1, keepdim=True)
        ]  # batch_size, 1, hidden_dim

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class JointPLMKVMLPOLDEmbedder(JointPLMEmbedder):

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        requires_grad: bool,
        share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int, prompt_dim: int
    ) -> None:
        super(JointPLMKVMLPOLDEmbedder, self).__init__(model_name)
        self.vocab = vocab
        self.plm.requires_grad_(requires_grad=requires_grad)

        self.share_prompt = None
        self.domain_prompt = None
        self.task_prompt = None
        self.share_mlp = None
        self.domain_mlp = None
        self.task_mlp = None
        self.attention_layers = self._prompt_init(share_prompt_len, domain_prompt_len, task_prompt_len, prompt_dim)

        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len
        self.prompt_dim = prompt_dim

    def _prompt_init(
        self, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int, prompt_dim: int
    ) -> nn.ModuleList:
        attention_layers = nn.ModuleList()

        emb_dim = self.output_dim
        if share_prompt_len > 0:
            self.share_prompt = nn.Parameter(nn.init.normal_(torch.empty(share_prompt_len, emb_dim)))
        self.domain_prompt = nn.Embedding(self.vocab.get_vocab_size("domain_labels"), domain_prompt_len*emb_dim)
        self.task_prompt = nn.Embedding(self.vocab.get_vocab_size("task_labels"), task_prompt_len*emb_dim)
        # nn.init.normal_(self.share_prompt, mean=0., std=1e-3)
        # nn.init.normal_(self.domain_prompt.weight, mean=0., std=1e-3)
        # nn.init.normal_(self.task_prompt.weight, mean=0., std=1e-3)

        for transf_layer in self.plm.encoder.layer:
            atten_layer = BertSelfAttentionKVMLPOLD(
                transf_layer.attention.self, prompt_dim, share_prompt_len, domain_prompt_len, task_prompt_len)
            transf_layer.attention.self = atten_layer
            transf_layer.attention.output.LayerNorm.requires_grad_(requires_grad=True)
            attention_layers.append(atten_layer)

        return attention_layers

    def _insert_prompt(self, domain_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
        batch_size = domain_ids.size(0)
        for atten_layer in self.attention_layers:
            if self.share_prompt is not None:
                share_prompt = self.share_prompt.unsqueeze(0).expand(batch_size, -1, -1)
                setattr(atten_layer, "share_prompt", share_prompt)

            domain_prompt = self.domain_prompt(domain_ids)
            domain_prompt = domain_prompt.unsqueeze(-1).view(batch_size, self.domain_prompt_len, self.output_dim)
            setattr(atten_layer, "domain_prompt", domain_prompt)

            task_prompt = self.task_prompt(task_ids)
            task_prompt = task_prompt.unsqueeze(-1).view(batch_size, self.task_prompt_len, self.output_dim)
            setattr(atten_layer, "task_prompt", task_prompt)

    def _flush_prompt(self) -> None:
        for atten_layer in self.attention_layers:
            setattr(atten_layer, "share_prompt", None)
            setattr(atten_layer, "domain_prompt", None)
            setattr(atten_layer, "task_prompt", None)


class BertSelfAttentionKVMLPOLD(nn.Module):

    def __init__(
        self, base: BertSelfAttention,
        prompt_dim: int, share_prompt_len: int, domain_prompt_len: int, task_prompt_len: int
    ) -> None:
        super(BertSelfAttentionKVMLPOLD, self).__init__()
        self.base = base
        assert not self.base.is_decoder, "is_decoder error"
        assert self.base.position_embedding_type == "absolute", "position_embedding_type error"
        self.num_heads, self.head_size = self.base.num_attention_heads, self.base.attention_head_size

        self.prompt_len = sum([share_prompt_len, domain_prompt_len, task_prompt_len])
        self.share_prompt_len = share_prompt_len
        self.domain_prompt_len = domain_prompt_len
        self.task_prompt_len = task_prompt_len

        self.share_prompt = None
        self.domain_prompt = None
        self.task_prompt = None
        self.share_mlp = None
        if share_prompt_len > 0:
            self.share_mlp = nn.Sequential(
                nn.Linear(self.num_heads*self.head_size, prompt_dim),
                nn.Tanh(),
                nn.Linear(prompt_dim, self.num_heads*self.head_size*2)
            )
        self.domain_mlp = nn.Sequential(
            nn.Linear(self.num_heads*self.head_size, prompt_dim),
            nn.Tanh(),
            nn.Linear(prompt_dim, self.num_heads*self.head_size*2)
        )
        self.task_mlp = nn.Sequential(
            nn.Linear(self.num_heads*self.head_size, prompt_dim),
            nn.Tanh(),
            nn.Linear(prompt_dim, self.num_heads*self.head_size*2)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        assert (past_key_value is None) and (encoder_hidden_states is None), "argument error"
        attention_mask = attention_mask.expand(-1, -1, attention_mask.size(-1), -1)
        batch_size, _, seq_len, _ = attention_mask.size()

        prompt_k, prompt_v = [], []
        if self.share_prompt is not None:
            share_prompt = self.share_mlp(self.share_prompt).unsqueeze(-1)
            share_prompt = share_prompt.view(batch_size, self.share_prompt_len, self.num_heads, self.head_size*2)
            share_prompt_k, share_prompt_v = torch.split(share_prompt, self.head_size, dim=-1)
            prompt_k.append(share_prompt_k.permute(0, 2, 1, 3))
            prompt_v.append(share_prompt_v.permute(0, 2, 1, 3))

        domain_prompt = self.domain_mlp(self.domain_prompt).unsqueeze(-1)
        domain_prompt = domain_prompt.view(batch_size, self.domain_prompt_len, self.num_heads, self.head_size*2)
        domain_prompt_k, domain_prompt_v = torch.split(domain_prompt, self.head_size, dim=-1)
        prompt_k.append(domain_prompt_k.permute(0, 2, 1, 3))
        prompt_v.append(domain_prompt_v.permute(0, 2, 1, 3))
        task_prompt = self.task_mlp(self.task_prompt).unsqueeze(-1)
        task_prompt = task_prompt.view(batch_size, self.task_prompt_len, self.num_heads, self.head_size*2)
        task_prompt_k, task_prompt_v = torch.split(task_prompt, self.head_size, dim=-1)
        prompt_k.append(task_prompt_k.permute(0, 2, 1, 3))
        prompt_v.append(task_prompt_v.permute(0, 2, 1, 3))

        prompt_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.prompt_len))
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=-1)

        # code of BertSelfAttention
        query_layer = self.base.transpose_for_scores(self.base.query(hidden_states))
        key_layer = self.base.transpose_for_scores(self.base.key(hidden_states))
        value_layer = self.base.transpose_for_scores(self.base.value(hidden_states))
        prompt_k.append(key_layer)
        prompt_v.append(value_layer)
        key_layer = torch.cat(prompt_k, dim=2)
        value_layer = torch.cat(prompt_v, dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.base.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.base.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.base.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
