from copy import deepcopy
from collections import deque
from simulator.internal.analyzer import ModelAnalyzer
from simulator.core.trace import TraceEvent
from simulator.core.memory_planner import MemoryPlanner
from simulator.internal.configs.hardware_params import hardware_params
from typing import List, Deque, Optional

from .request import GenerationRequest
from .global_waitlist import GlobalWaitlist  # Import global waitlist
from .obtain_latency import build_latency_dict

TEMPLATE_TOKENS = {"Information Retriever": 213,
                  "extract_keywords": 630, 
                  "generate_candidate_llama-agent1": 9158,
                  "generate_candidate_llama-agent": 5292,
                  "revise": 3059,
                  "unit_tester": 213,
                  "generate_unit_test": 436,
                  "evaluate": 261}

class LLMEngineSJF:
    def __init__(self, engine_id, model_name, hardware_name, w_bit, a_bit, kv_bit):
        self.engine_id = engine_id
        self.model_name = model_name
        self.hardware_name = hardware_name
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.analyzer = ModelAnalyzer(
            model_id=model_name,
            hardware=hardware_name,
            config_file="simulator/internal/configs/llama.py",
            source="huggingface",
        )
        self.global_waitlist = GlobalWaitlist.get_instance()
        self.waiting: Deque[GenerationRequest] = deque()
        self.running: Deque[GenerationRequest] = deque()
        self.finished: List[GenerationRequest] = []
        self.memory_planner = MemoryPlanner(
            self.analyzer.model_params,
            hardware_params[hardware_name],
            w_bit,
            a_bit,
            kv_bit,
        )
        self.memory_planner.print_status()
        self.latency_dict = build_latency_dict(hardware_name)
        self.finished_requests: int = 0
        self.configure()

    def configure(self):
        pass

    def add_request(self, request: GenerationRequest):
        self.waiting.append(request)

    def update_elapsed_time(self, current_time: float, hardware_name: str):
        for request in self.waiting:
            request.elapsed_time = current_time - request.arrive_at
            request.update_urgency(hardware_name)

    def get_highest_priority_request(self, waitlist) -> Optional[GenerationRequest]:
        # Implements Shortest-Job-First (SJF) scheduling using calculate_empirical_time()
        if not waitlist:
            return None
        # Find the request with the shortest empirical time
        sjf_request = None
        min_empirical_time = float("inf")
        for request in waitlist:
            empirical_time = request.calculate_empirical_time(self.hardware_name)
            if empirical_time < min_empirical_time:
                min_empirical_time = empirical_time
                sjf_request = request
        if sjf_request and self.memory_planner.can_allocate_request(sjf_request):
            return sjf_request
        return None

    def _prefill(self, request: GenerationRequest, start_at: float):
        self.memory_planner.allocate(request)
        memory_event = self.memory_event(start_at)
        if start_at < request.arrive_at:
            start_at = request.arrive_at
        self.running.append(request)
        request._prefill()
        # if request.step in {req.step for req in self.running}:
        #     prefill_result = self.analyzer.analyze(
        #         seqlen=request.input_length-TEMPLATE_TOKENS[request.step],
        #         batchsize=1,
        #         w_bit=self.w_bit,
        #         a_bit=self.a_bit,
        #         kv_bit=self.kv_bit,
        #     )
        # else:

        # prefill_result = self.analyzer.analyze(
        #     seqlen=request.input_length,
        #     batchsize=1,
        #     w_bit=self.w_bit,
        #     a_bit=self.a_bit,
        #     kv_bit=self.kv_bit,
        # )
        # prefill_time = prefill_result["total_results"]["prefill"]["inference_time"]
        prefill_time = self.latency_dict[(request.input_length, request.output_length)]["prefill_latency"]
        request.set_prefill_finished_at(start_at + prefill_time)
        if request.output_length == 1:
            request.set_generation_finished_at(start_at + prefill_time)
            self.memory_planner.free([request.req_id])
        return prefill_time + start_at, [request], memory_event

    def _decode(self, requests: List[GenerationRequest], start_at: float):
        max_batch_size = len(requests)
        decode_time = []
        finished_requests_in_this_batch = []
        executable_requests = []
        for req in requests:
            if self.memory_planner.can_allocate_request(req):
                self.memory_planner.allocate(req)
                executable_requests.append(req)
        batch_size = len(executable_requests)
        memory_event = self.memory_event(start_at)
        for req in executable_requests:
            if start_at < req.arrive_at:
                start_at = req.arrive_at
            # decode_result = self.analyzer.analyze(
            #     req.input_length + req.generated_tokens,
            #     batchsize=max_batch_size,
            #     w_bit=self.w_bit,
            #     a_bit=self.a_bit,
            #     kv_bit=self.kv_bit,
            # )
            # decode_time.append(
            #     decode_result["total_results"]["decode"]["inference_time"]
            # )
            decode_time.append(
                self.latency_dict[(req.input_length, req.output_length)]["per_token_decode_latency"]
            )
        finished_at = max(decode_time) + start_at
        finished_lst = []
        for req in executable_requests:
            finished = req._decode() #Check if the request is finished
            if finished:
                req.set_generation_finished_at(finished_at)
                self.finished_requests += 1
                self.running.remove(req)
                self.finished.append(req)
                finished_requests_in_this_batch.append(req.req_id)
                finished_lst.append(req)
        self.memory_planner.free(finished_requests_in_this_batch)
        return finished_at, executable_requests, memory_event, finished_lst

    def step(self, start_at: float):
        handled_requests = []
        # Fetch the highest-priority request from the global waitlist
        self.update_elapsed_time(start_at, self.hardware_name)
        next_request = self.get_highest_priority_request(self.waiting)

        if next_request:
            self.waiting.remove(next_request)
            handled_requests = [next_request.req_id]
            prefill_end_at, handled_requests, memory_event = self._prefill(
                next_request, start_at
            )
            return (
                self.create_event(
                    "prefill", handled_requests, start_at, prefill_end_at
                ),
                [],
                prefill_end_at,
                memory_event,
            )
        elif len(self.running) > 0:
            decode_finished_at, handled_requests, memory_event, finished_lst = self._decode(
                list(self.running), start_at
            )
            return (
                self.create_event(
                    "decode", handled_requests, start_at, decode_finished_at
                ),
                finished_lst,
                decode_finished_at,
                memory_event,
            )
        else:
            return None, [], start_at + 0.0001, None

    def create_event(self, phase, handled_requests, start_at, end_at):
        complete_events = []
        handled_requests = [req.req_id for req in handled_requests]
        for req in handled_requests:
            complete = TraceEvent(
                name=f"{phase}-{req}",
                cat=f"{phase,req}",
                ph="X",
                pid=self.engine_id,
                tid=0,
                ts=int(start_at * 1000 * 1000),  # convert to microseconds
                dur=int((end_at - start_at) * 1000 * 1000),
            )
            complete_events.append(complete)
        return complete_events

    def memory_event(self, start_at):
        return TraceEvent(
            name="block usage",
            ph="C",
            ts=start_at * 1e6,
            pid=self.engine_id,
            tid=0,
            cat="memory",
            args={
                "used": self.memory_planner._allocated_blocks,
                "free": self.memory_planner._max_num_blocks
                - self.memory_planner._allocated_blocks,
            },
        )

    @property
    def empty(self):
        return len(self.waiting) == 0 and len(self.running) == 0
