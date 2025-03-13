import enum
from datetime import datetime
from typing import Dict
from dataclasses import dataclass
from typing import List, Optional

class REQ_STATUS(enum.Enum):
    PENDING = 1
    SCHEDULED = 2
    PREFILL = 3
    GENERATE = 4
    EXIT = 5


class GenerationRequest:
    def __init__(self, req_id: str, model: str, input_length: int, output_length: int, 
                 arrive_at: float, parent_request: Optional['Text2SQLRequest'] = None):
        self.req_id = req_id
        self.model = model
        self.input_length = input_length
        self.output_length = output_length
        self.arrive_at = arrive_at
        self.status = REQ_STATUS.PENDING
        self.prefill_time = None
        self.parent_request = parent_request

        self.generated_tokens = 0
        self.prefill_finished_at = None
        self.generation_finished_at = None

    def set_generation_finished_at(self, finished_at: float):
        self.generation_finished_at = finished_at

    def set_prefill_finished_at(self, finished_at: float):
        self.prefill_finished_at = finished_at

    def _prefill(self):
        self.status = REQ_STATUS.PREFILL

    def _decode(self) -> bool:
        self.generated_tokens += 1
        if self.generated_tokens == self.output_length:
            self._stop()
            return True
        return False

    def _stop(self):
        pass

    def __str__(self):
        return f"Request {self.req_id} for model {self.model} with input length {self.input_length} and output length {self.output_length} arrived at {self.arrive_at}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict:
        return {
            "req_id": self.req_id,
            "model": self.model,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "arrive_at": self.arrive_at,
            "prefill_time": self.prefill_time,
            "generated_tokens": self.generated_tokens,
            "prefill_finished_at": self.prefill_finished_at,
            "generation_finished_at": self.generation_finished_at,
        }



@dataclass
class Text2SQLRequest:
    req_id: str
    current_stage: int
    total_stages: int
    total_time: float
    stages_config: List[dict]  # 每个阶段的input_length和output_length配置
    current_request: Optional[GenerationRequest] = None
    
    def __init__(self, req_id: str, stages_config: List[dict]):
        self.req_id = req_id
        self.current_stage = 0
        self.total_stages = len(stages_config)
        self.stages_config = stages_config
        self.total_time = 0.0
        self.current_request = None
    
    def create_next_stage_request(self, model, arrive_at: float) -> Optional[GenerationRequest]:
        """创建下一阶段的请求"""
        stage_config = self.stages_config[self.current_stage]
        self.current_request = GenerationRequest(
            req_id=f"{self.req_id}_stage_{self.current_stage}",
            model=model,
            input_length=stage_config["input_length"],
            output_length=stage_config["output_length"],
            arrive_at=arrive_at,
            parent_request=self
        )
        return self.current_request
    
    def update_stage(self, current_time: float):
        """更新阶段和总时间"""
        if self.current_request and self.current_request.generation_finished_at is not None:
            self.total_time += (self.current_request.generation_finished_at - 
                              self.current_request.arrive_at)
        self.current_stage += 1
        self.current_request = None