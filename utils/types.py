from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple


Metric = Dict[str, Any]


@dataclass
class VeriFreeTrajectoryData:
    prompt: str
    prompt_ids: List[int]
    response: str
    response_ids: List[int]
    response_logprobs: List[float]
    gt_text: str
    think_text: str
    answer_text: str
    think_ids: List[int]
    answer_ids: List[int]
    gt_action_logprobs: List[float] = None
    logp_reward: float = None
    reparam_ratio: float = None
    loss_mask: bool = True
    no_eos: bool = False
    end_with_eos: bool = False
    info: Metric = None
