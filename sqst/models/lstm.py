"""
LSTM Language Model implementation and registration for squeal_lstm_5m.
"""

from typing import Generator, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput


class LSTMConfig(PretrainedConfig):
    """Configuration class for squeal_lstm_5m language model."""
    model_type = "lstm_lm"
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size: int = 4000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        use_cache: bool = False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # The HF KV-cache protocol is not supported by this model; the recurrent
        # state is handled by `generate_recurrent` instead. Force-disable it even
        # if a stored config.json says otherwise.
        self.use_cache = False


def _apply_repetition_penalty(scores: torch.Tensor, seen: Iterable[int], penalty: float) -> torch.Tensor:
    """
    Repetition penalty with Hugging Face semantics: positive logits are divided by
    ``penalty`` and negative logits multiplied by it, so repeated tokens always
    become less likely. Returns a new tensor.
    """
    if penalty == 1.0:
        return scores
    ids = sorted(set(int(t) for t in seen))
    if not ids:
        return scores
    idx = torch.tensor(ids, dtype=torch.long, device=scores.device)
    picked = scores.index_select(-1, idx)
    picked = torch.where(picked < 0, picked * penalty, picked / penalty)
    return scores.index_copy(-1, idx, picked)


def _top_p_filter(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keeps the smallest set of tokens whose probability mass >= top_p."""
    sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()  # always keep the first token over the threshold
    remove[..., 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, -float("inf"))
    return torch.full_like(scores, -float("inf")).scatter(-1, sorted_idx, sorted_scores)


class LSTMLanguageModel(PreTrainedModel, GenerationMixin):
    """
    Recurrent LSTM language model for squeal_lstm_5m.
    Supports both standard Hugging Face generation and fast recurrent step-by-step state caching.
    """
    config_class = LSTMConfig
    tied_weights_keys = []
    _tied_weights_keys = []

    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=config.pad_token_id
        )
        self.rnn = nn.LSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.hidden_dim, config.embed_dim, bias=False)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embedding = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        **kwargs
    ) -> Union[CausalLMOutput, Tuple[CausalLMOutput, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        Args:
            input_ids: Tensor of shape (batch, seq_len)
            labels: Optional labels for CrossEntropy loss
            attention_mask: Accepted for HF compatibility (ignored)
            state: Optional tuple (h, c) for recurrent continuation
            return_state: If True, also returns the updated (h, c) hidden states
        """
        x = self.dropout(self.embedding(input_ids))
        if state is not None:
            out, new_state = self.rnn(x, state)
        else:
            out, new_state = self.rnn(x)

        out = self.dropout(out)
        logits = self.lm_head(self.proj(out))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = CausalLMOutput(loss=loss, logits=logits)
        if return_state:
            return output, new_state
        return output

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs) -> dict:
        """Required by Hugging Face GenerationMixin."""
        return {"input_ids": input_ids}

    @torch.no_grad()
    def generate_recurrent(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> Generator[int, None, None]:
        """
        Step-by-step recurrent token generation that carries the LSTM state (h, c)
        instead of re-running the whole sequence at every step (O(n) instead of O(n^2)).
        Yields generated token ids one by one. Batch size must be 1.
        """
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("generate_recurrent supports a single sequence of shape (1, seq_len).")
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        was_training = self.training
        self.eval()  # dropout must be off while sampling
        try:
            # 1. Warm-up RNN with the prompt tokens
            out, state = self.rnn(self.embedding(input_ids))
            logits = self.lm_head(self.proj(out[:, -1, :]))

            seen = set(input_ids[0].tolist())
            vocab_size = logits.size(-1)

            for _ in range(max_new_tokens):
                # float32 avoids fp16 overflow when dividing by a small temperature
                scores = _apply_repetition_penalty(logits.float(), seen, repetition_penalty)

                if not do_sample or temperature <= 0.01:
                    next_token = scores.argmax(dim=-1, keepdim=True)
                else:
                    scores = scores / temperature
                    if top_k > 0:
                        kth = torch.topk(scores, min(top_k, vocab_size)).values[:, -1:]
                        scores = scores.masked_fill(scores < kth, -float("inf"))
                    if top_p is not None and 0.0 < top_p < 1.0:
                        scores = _top_p_filter(scores, top_p)
                    probs = torch.softmax(scores, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = int(next_token.item())
                if token_id == eos_token_id:
                    break

                yield token_id
                seen.add(token_id)

                # Advance recurrent state with the single new token
                out_step, state = self.rnn(self.embedding(next_token), state)
                logits = self.lm_head(self.proj(out_step[:, -1, :]))
        finally:
            if was_training:
                self.train()


def register_lstm_architecture() -> None:
    """
    Registers the custom LSTMConfig and LSTMLanguageModel in Hugging Face Auto classes.
    Safe to call multiple times.
    """
    try:
        AutoConfig.register("lstm_lm", LSTMConfig)
    except (ValueError, KeyError):
        pass

    try:
        AutoModelForCausalLM.register(LSTMConfig, LSTMLanguageModel)
    except (ValueError, KeyError):
        pass


# Automatically register on module import
register_lstm_architecture()
