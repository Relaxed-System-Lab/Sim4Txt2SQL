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
    gen_requests_config: List[dict]
    current_requests: Optional[GenerationRequest] = None
    stage_lst: List[str] = None
    request_counter: int = 0
    
    def __init__(self, req_id: str, gen_requests_config: List[dict]):
        self.req_id = req_id
        self.current_stage = 0
        self.stage_lst = []
        self.gen_requests_config = gen_requests_config
        self.total_time = 0.0
        self.current_requests = []
        self.request_counter = 0
        self._initialize_stages()
        self.total_stages = len(self.stage_lst)
    
    def _initialize_stages(self):
        """Initialize the stages and maintain the order"""
        consecutive_steps = {"generate_candidate_llama-agent", "evaluate"}
        seen_steps = set()
        for stage_config in self.gen_requests_config:
            step = stage_config["step"]
            if step in consecutive_steps:
                if step not in seen_steps:
                    self.stage_lst.append(step)
                    seen_steps.add(step)
            else:
                self.stage_lst.append(step)
        # for stage_config in self.gen_requests_config:
        #     step = stage_config["step"]
        #     self.stage_lst.append(step)
    
    def create_current_stage_requests(self, model, arrive_at: float) -> List[Optional[GenerationRequest]]:
        """创建下一阶段的请求"""
        current_step = self.stage_lst[self.current_stage]
        if current_step == "generate_candidate_llama-agent" or current_step == "evaluate":
            for stage_config in self.gen_requests_config:
                if stage_config["step"] == current_step:
                    next_request = GenerationRequest(
                        req_id=f"{self.req_id}_req_{self.request_counter}",
                        model=model,
                        input_length=stage_config["input_length"],
                        output_length=stage_config["output_length"],
                        arrive_at=arrive_at,
                        parent_request=self
                    )
                    self.current_requests.append(next_request)
                    self.request_counter += 1
        else:
            next_request = GenerationRequest(
                req_id=f"{self.req_id}_req_{self.request_counter}",
                model=model,
                input_length=self.gen_requests_config[self.request_counter]["input_length"],
                output_length=self.gen_requests_config[self.request_counter]["output_length"],
                arrive_at=arrive_at,
                parent_request=self
            )
            self.current_requests.append(next_request)
            self.request_counter += 1
        return self.current_requests
        # next_request = GenerationRequest(
        #     req_id=f"{self.req_id}_req_{self.request_counter}",
        #     model=model,
        #     input_length=self.gen_requests_config[self.request_counter]["input_length"],
        #     output_length=self.gen_requests_config[self.request_counter]["output_length"],
        #     arrive_at=arrive_at,
        #     parent_request=self
        # )
        # self.current_requests.append(next_request)
        # self.request_counter += 1
        # return self.current_requests
    
    def update_stage(self, request, current_time: float):
        """更新阶段和总时间"""
        self.current_requests.remove(request)
        if self.current_requests == []:
            self.total_time += (current_time - request.arrive_at)
            self.current_stage += 1
