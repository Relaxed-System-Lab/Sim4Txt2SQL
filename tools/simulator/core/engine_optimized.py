from copy import deepcopy
from collections import deque
from simulator.internal.analyzer import ModelAnalyzer
from simulator.core.trace import TraceEvent
from simulator.core.memory_planner import MemoryPlanner
from simulator.internal.configs.hardware_params import hardware_params
from typing import List, Deque, Optional

from .request import GenerationRequest
from simulator.core.global_waitlist import GlobalWaitlist  # Import global waitlist


TEMPLATE_TOKENS = {"Information Retriever": 10,
                  "extract_keywords": 4.05, 
                  "generate_candidate_llama-agent": 130.5,
                  "revise": 80,
                  "unit_tester": 1.94,
                  "generate_unit_test": 32.82,
                  "evaluate": 4.32}

class LLMEngine:
    def __init__(self, w1, w2, engine_id, model_name, hardware_name, w_bit, a_bit, kv_bit):
        self.w1 = w1
        self.w2 = w2
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
        self.finished_requests: int = 0
        self.configure()

    def configure(self):
        pass

    def get_highest_priority_request(self, waitlist) -> Optional[GenerationRequest]:
        if not waitlist:
            return None
        prior_request = waitlist[0]
        highest_priority = prior_request.priority
        for request in waitlist:
            if request.step in {req.step for req in self.running}:
                saved_prefill_result = self.analyzer.analyze(
                    seqlen=TEMPLATE_TOKENS[request.step],
                    batchsize=1,
                    w_bit=self.w_bit,
                    a_bit=self.a_bit,
                    kv_bit=self.kv_bit,
                )
                saved_prefill_time = saved_prefill_result["total_results"]["prefill"]["inference_time"]
            else:
                saved_prefill_time = 0
            priority = self.w1 * request.urgency + self.w2 * saved_prefill_time
            if priority > highest_priority:
                prior_request = request
                highest_priority = priority
        if self.memory_planner.can_allocate_request(prior_request):
            return prior_request
        return None

    def _prefill(self, request: GenerationRequest, start_at: float):
        self.memory_planner.allocate(request)
        memory_event = self.memory_event(start_at)
        if start_at < request.arrive_at:
            start_at = request.arrive_at
        self.running.append(request)
        request._prefill()
        if request.step in {req.step for req in self.running}:
            prefill_result = self.analyzer.analyze(
                seqlen=request.input_length-TEMPLATE_TOKENS[request.step],
                batchsize=1,
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                kv_bit=self.kv_bit,
            )
        else:
            prefill_result = self.analyzer.analyze(
                seqlen=request.input_length,
                batchsize=1,
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                kv_bit=self.kv_bit,
            )
        prefill_time = prefill_result["total_results"]["prefill"]["inference_time"]
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
            decode_result = self.analyzer.analyze(
                req.input_length + req.generated_tokens,
                batchsize=max_batch_size,
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                kv_bit=self.kv_bit,
            )
            decode_time.append(
                decode_result["total_results"]["decode"]["inference_time"]
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
        self.global_waitlist.update_elapsed_time(start_at)
        next_request = self.get_highest_priority_request(self.global_waitlist.waitlist)

        if next_request:
            self.global_waitlist.remove_request(next_request)
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
        return len(self.global_waitlist) == 0 and len(self.running) == 0
