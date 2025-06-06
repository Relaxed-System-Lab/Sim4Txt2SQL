import torch
from typing import List

from ..orchestrator import _BatchedPenalizer, _TokenIDs


class BatchedMinNewTokensPenalizer(_BatchedPenalizer):
    """
    Min new tokens penalizer penalizes tokens based on the length of the output.
    """

    min_new_tokens: torch.Tensor = None
    stop_token_penalties: torch.Tensor = None
    len_output_tokens: torch.Tensor = None

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.min_new_tokens > 0 for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.min_new_tokens = torch.tensor(
            data=[
                req.sampling_params.min_new_tokens for req in self.orchestrator.reqs()
            ],
            dtype=torch.int32,
            device=self.orchestrator.device,
        ).unsqueeze_(1)

        padded_stop_token_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=[
                torch.tensor(
                    data=(
                        list(
                            (req.sampling_params.stop_token_ids or set())
                            | (req.tokenizer.additional_stop_token_ids or set())
                            | {req.tokenizer.eos_token_id}
                        )
                    ),
                    dtype=torch.int64,
                    device=self.orchestrator.device,
                )
                for req in self.orchestrator.reqs()
            ],
            batch_first=True,
            padding_value=self.orchestrator.vocab_size,
        )
        self.stop_token_penalties = torch.zeros(
            size=(self.orchestrator.batch_size(), self.orchestrator.vocab_size + 1),
            dtype=torch.float32,
            device=self.orchestrator.device,
        ).scatter_add_(
            dim=1,
            index=padded_stop_token_ids,
            src=torch.full_like(
                input=padded_stop_token_ids,
                dtype=torch.float32,
                fill_value=float("-inf"),
                device=self.orchestrator.device,
            ),
        )[
            :, : self.orchestrator.vocab_size
        ]

        self.len_output_tokens = torch.zeros(
            size=(self.orchestrator.batch_size(), 1),
            dtype=torch.int32,
            device=self.orchestrator.device,
        )

    def _teardown(self):
        self.min_new_tokens = None
        self.stop_token_penalties = None
        self.len_output_tokens = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        pass

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        self.len_output_tokens += 1

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        mask = (self.len_output_tokens < self.min_new_tokens).expand_as(logits)
        logits[mask] += self.stop_token_penalties[mask]
        return logits

    def _filter(self, indices_to_keep: List[int], indices_tensor_to_keep: torch.Tensor):
        self.min_new_tokens = self.min_new_tokens[indices_tensor_to_keep]
        self.stop_token_penalties = self.stop_token_penalties[indices_tensor_to_keep]
        self.len_output_tokens = self.len_output_tokens[indices_tensor_to_keep]

    def _merge(self, their: "BatchedMinNewTokensPenalizer"):
        self.min_new_tokens = torch.cat(
            [self.min_new_tokens, their.min_new_tokens], dim=0
        )
        self.stop_token_penalties = torch.cat(
            [self.stop_token_penalties, their.stop_token_penalties], dim=0
        )
        self.len_output_tokens = torch.cat(
            [self.len_output_tokens, their.len_output_tokens], dim=0
        )
