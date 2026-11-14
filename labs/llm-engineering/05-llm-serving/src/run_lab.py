#!/usr/bin/env python3
"""Chapter 05: standard-library simulator for paged KV allocation and continuous batching.

This is a teaching model, not vLLM source code, a CUDA kernel, or a performance benchmark.
"""
from __future__ import annotations
import csv, hashlib, json, platform, socket
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE, VISUALS, REPORTS = ROOT / "evidence", ROOT / "evidence" / "visuals", ROOT / "reports"


@dataclass
class Block:
    block_id: int
    tokens: list[int] = field(default_factory=list)
    ref_count: int = 0
    cache_key: str | None = None
    last_used: int = 0


@dataclass
class SequenceState:
    request_id: str
    tokens: list[int]
    block_ids: list[int]


class KVBlockPool:
    def __init__(self, num_blocks: int, block_size: int, model_revision: str = "toy-model-v1"):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks, self.block_size = num_blocks, block_size
        self.model_revision = model_revision
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids = deque(range(num_blocks))
        self.prefix_index: dict[str, int] = {}
        self.clock = self.prefix_cache_hits = self.prefix_cache_misses = self.evictions = 0

    def _touch(self, block: Block):
        self.clock += 1
        block.last_used = self.clock

    def _key(self, tokens: list[int]) -> str:
        payload = self.model_revision + "|" + ",".join(map(str, tokens))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _allocate(self) -> Block:
        if self.free_ids:
            block = self.blocks[self.free_ids.popleft()]
            self._touch(block)
            return block
        candidates = [b for b in self.blocks if b.ref_count == 0 and b.cache_key is not None]
        if not candidates:
            raise RuntimeError("KV block pool exhausted: every block is in use")
        victim = min(candidates, key=lambda b: b.last_used)
        if self.prefix_index.get(victim.cache_key) == victim.block_id:
            del self.prefix_index[victim.cache_key]
        victim.tokens.clear()
        victim.cache_key = None
        victim.ref_count = 0
        self.evictions += 1
        self._touch(victim)
        return victim

    def _lookup(self, key: str) -> Block | None:
        block_id = self.prefix_index.get(key)
        if block_id is None:
            self.prefix_cache_misses += 1
            return None
        block = self.blocks[block_id]
        if block.cache_key != key or len(block.tokens) != self.block_size:
            self.prefix_index.pop(key, None)
            self.prefix_cache_misses += 1
            return None
        block.ref_count += 1
        self._touch(block)
        self.prefix_cache_hits += 1
        return block

    def create_sequence(self, request_id: str, tokens: list[int]) -> SequenceState:
        state = SequenceState(request_id, list(tokens), [])
        prior: list[int] = []
        for offset in range(0, len(tokens), self.block_size):
            chunk = list(tokens[offset:offset + self.block_size])
            full = len(chunk) == self.block_size
            prefix = prior + chunk
            key = self._key(prefix) if full else None
            block = self._lookup(key) if key else None
            if block is None:
                block = self._allocate()
                block.tokens, block.ref_count = chunk, 1
                if key:
                    block.cache_key = key
                    self.prefix_index[key] = block.block_id
                self._touch(block)
            state.block_ids.append(block.block_id)
            prior = prefix
        return state

    def append_token(self, state: SequenceState, token_id: int):
        state.tokens.append(token_id)
        if not state.block_ids or len(self.blocks[state.block_ids[-1]].tokens) == self.block_size:
            block = self._allocate()
            block.tokens, block.ref_count = [token_id], 1
            state.block_ids.append(block.block_id)
            self._touch(block)
            return
        block = self.blocks[state.block_ids[-1]]
        if block.ref_count != 1:
            raise RuntimeError("Cannot append to a shared partial block")
        block.tokens.append(token_id)
        self._touch(block)
        if len(block.tokens) == self.block_size:
            key = self._key(state.tokens)
            if key not in self.prefix_index:
                block.cache_key = key
                self.prefix_index[key] = block.block_id

    def release_sequence(self, state: SequenceState):
        for block_id in state.block_ids:
            block = self.blocks[block_id]
            if block.ref_count < 1:
                raise RuntimeError("Reference-count underflow")
            block.ref_count -= 1
            self._touch(block)
            if block.ref_count == 0 and block.cache_key is None:
                block.tokens.clear()
                self.free_ids.append(block.block_id)
        state.block_ids.clear()

    def active_unique_blocks(self, states: list[SequenceState]) -> set[int]:
        return {bid for state in states for bid in state.block_ids}

    def snapshot(self, states: list[SequenceState]) -> dict:
        active = self.active_unique_blocks(states)
        return {
            "pool_blocks": self.num_blocks,
            "block_size_tokens": self.block_size,
            "active_unique_blocks": len(active),
            "referenced_blocks": sum(b.ref_count > 0 for b in self.blocks),
            "cached_full_blocks": sum(b.cache_key is not None for b in self.blocks),
            "free_blocks": len(self.free_ids),
            "active_tail_waste_slots": sum(
                self.block_size - len(self.blocks[i].tokens) for i in active
            ),
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "evictions": self.evictions,
            "block_tables": {s.request_id: list(s.block_ids) for s in states},
        }


@dataclass
class Request:
    request_id: str
    arrival_step: int
    prompt_tokens: int
    output_tokens: int
    prefill_done: int = 0
    generated: int = 0
    phase: str = "WAITING"
    first_scheduled_step: int | None = None
    completion_step: int | None = None
    prefill_steps: int = 0
    decode_steps: int = 0


def run_scheduler(workload: list[dict], token_budget: int = 6, max_active: int = 3):
    """Decode-first, round-robin chunked-prefill teaching approximation."""
    if token_budget < max_active:
        raise ValueError("token_budget must be >= max_active for this scenario")
    requests = {
        x["request_id"]: Request(
            x["request_id"], x["arrival_step"], x["prompt_tokens"], x["output_tokens"]
        ) for x in workload
    }
    arrivals: dict[int, list[Request]] = {}
    for req in requests.values():
        arrivals.setdefault(req.arrival_step, []).append(req)
    waiting, running = deque(), []
    events, steps = [], []
    step = max_running_seen = 0

    while not all(req.phase == "FINISHED" for req in requests.values()):
        if step > 1000:
            raise RuntimeError("Scheduler did not finish")
        for req in sorted(arrivals.get(step, []), key=lambda r: r.request_id):
            waiting.append(req)
            events.append(dict(step=step, request_id=req.request_id, event="ARRIVE",
                               phase="WAITING", tokens_scheduled=0,
                               prompt_done=0, generated=0))

        budget_left = token_budget
        prefill_count = decode_count = 0

        # Existing decoders receive one token before new prefill work.
        for req in list(running):
            if req.phase == "DECODE" and budget_left:
                req.generated += 1
                req.decode_steps += 1
                budget_left -= 1
                decode_count += 1
                events.append(dict(step=step, request_id=req.request_id,
                                   event="DECODE_TOKEN", phase="DECODE",
                                   tokens_scheduled=1, prompt_done=req.prefill_done,
                                   generated=req.generated))
                if req.generated >= req.output_tokens:
                    req.phase = "FINISHED"
                    req.completion_step = step
                    events.append(dict(step=step, request_id=req.request_id,
                                       event="FINISH", phase="FINISHED",
                                       tokens_scheduled=0, prompt_done=req.prefill_done,
                                       generated=req.generated))

        # Completed requests release their running slots immediately.
        running = [r for r in running if r.phase != "FINISHED"]
        admitted = []
        while waiting and len(running) < max_active:
            req = waiting.popleft()
            req.phase = "PREFILL"
            req.first_scheduled_step = step
            running.append(req)
            admitted.append(req)
            events.append(dict(step=step, request_id=req.request_id,
                               event="ADMIT", phase="PREFILL",
                               tokens_scheduled=0, prompt_done=0, generated=0))

        # Use the remaining budget for chunked prefill in round-robin order.
        candidates = [r for r in running if r.phase == "PREFILL"]
        while budget_left and candidates:
            made_progress = False
            for req in list(candidates):
                if not budget_left:
                    break
                req.prefill_done += 1
                req.prefill_steps += 1
                budget_left -= 1
                prefill_count += 1
                made_progress = True
                events.append(dict(step=step, request_id=req.request_id,
                                   event="PREFILL_TOKEN", phase="PREFILL",
                                   tokens_scheduled=1, prompt_done=req.prefill_done,
                                   generated=req.generated))
                if req.prefill_done == req.prompt_tokens:
                    req.phase = "DECODE"
                    candidates.remove(req)
                    events.append(dict(step=step, request_id=req.request_id,
                                       event="PREFILL_COMPLETE", phase="DECODE",
                                       tokens_scheduled=0, prompt_done=req.prefill_done,
                                       generated=req.generated))
            if not made_progress:
                break

        max_running_seen = max(max_running_seen, len(running))
        scheduled = prefill_count + decode_count
        steps.append(dict(
            step=step, token_budget=token_budget, prefill_tokens=prefill_count,
            decode_tokens=decode_count, total_scheduled_tokens=scheduled,
            budget_unused=token_budget - scheduled, running_after_step=len(running),
            waiting_after_step=len(waiting),
            active_ids=";".join(r.request_id for r in running),
            admitted_ids=";".join(r.request_id for r in admitted),
        ))
        step += 1

    request_rows = [
        dict(
            request_id=r.request_id, arrival_step=r.arrival_step,
            first_scheduled_step=r.first_scheduled_step,
            queue_wait_steps=r.first_scheduled_step - r.arrival_step,
            prompt_tokens=r.prompt_tokens, output_tokens=r.output_tokens,
            prefill_scheduler_steps=r.prefill_steps, decode_scheduler_steps=r.decode_steps,
            completion_step=r.completion_step, phase=r.phase,
        )
        for r in sorted(requests.values(), key=lambda x: x.request_id)
    ]
    checks = {
        "all_requests_completed": all(r["phase"] == "FINISHED" for r in request_rows),
        "per_step_token_budget_respected": all(r["total_scheduled_tokens"] <= token_budget for r in steps),
        "max_active_requests_respected": all(r["running_after_step"] <= max_active for r in steps),
        "every_request_admitted_once": sum(e["event"] == "ADMIT" for e in events) == len(requests),
        "every_prompt_token_processed": sum(r["prefill_tokens"] for r in steps) == sum(x.prompt_tokens for x in requests.values()),
        "every_output_token_generated": sum(r["decode_tokens"] for r in steps) == sum(x.output_tokens for x in requests.values()),
        "continuous_admission_used": any(e["event"] == "ADMIT" and e["step"] > 0 for e in events),
    }
    metrics = dict(
        token_budget_per_step=token_budget, max_active_requests=max_active,
        scheduler_steps=len(steps), peak_running_requests=max_running_seen,
        total_prompt_tokens=sum(x.prompt_tokens for x in requests.values()),
        total_output_tokens=sum(x.output_tokens for x in requests.values()),
        mean_queue_wait_steps=sum(r["queue_wait_steps"] for r in request_rows) / len(request_rows),
        checks=checks,
        policy_note="Teaching approximation: one decode token per active decoder, then round-robin prompt tokens within the remaining budget. Not vLLM's implementation or a latency model.",
    )
    return events, steps, request_rows, metrics


def fragmentation_metrics():
    lengths, block_size, max_reservation = [23, 77, 12, 49], 16, 128
    live = sum(lengths)
    static_reserved = len(lengths) * max_reservation
    static_waste = static_reserved - live
    paged_capacity = sum(((n + block_size - 1) // block_size) * block_size for n in lengths)
    paged_waste = paged_capacity - live
    return dict(
        sequence_lengths=lengths, block_size_tokens=block_size,
        static_max_tokens_per_sequence=max_reservation, live_tokens=live,
        static_reserved_slots=static_reserved, static_wasted_slots=static_waste,
        paged_allocated_slots=paged_capacity, paged_tail_waste_slots=paged_waste,
        wasted_slot_reduction_percent=100 * (1 - paged_waste / static_waste),
        note="Illustrative capacity arithmetic, not measured GPU memory or a production benchmark.",
    )


def make_block_svg():
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="570" viewBox="0 0 1200 570" role="img" aria-labelledby="title desc">
<title id="title">Paged KV block mapping and prefix sharing</title><desc id="desc">Two request block tables map the same first two logical token blocks to shared physical blocks; divergent suffix blocks are private.</desc>
<rect width="1200" height="570" fill="#fbfaf7"/><text x="54" y="54" font-family="Arial" font-size="24" font-weight="700" fill="#192630">Logical sequence positions → non-contiguous physical KV blocks</text>
<text x="54" y="95" font-family="Arial" font-size="17" font-weight="700" fill="#17394b">Request A · prompt [A B C D E F G H I]</text><text x="54" y="165" font-family="Arial" font-size="17" font-weight="700" fill="#7b412e">Request B · prompt [A B C D E F G H X Y]</text>
<g font-family="Arial" font-size="15" text-anchor="middle"><rect x="54" y="112" width="110" height="40" rx="4" fill="#28566b"/><text x="109" y="138" fill="white">logical 0</text><rect x="172" y="112" width="110" height="40" rx="4" fill="#28566b"/><text x="227" y="138" fill="white">logical 1</text><rect x="290" y="112" width="110" height="40" rx="4" fill="#a9573d"/><text x="345" y="138" fill="white">logical 2</text><rect x="54" y="182" width="110" height="40" rx="4" fill="#28566b"/><text x="109" y="208" fill="white">logical 0</text><rect x="172" y="182" width="110" height="40" rx="4" fill="#28566b"/><text x="227" y="208" fill="white">logical 1</text><rect x="290" y="182" width="110" height="40" rx="4" fill="#a9573d"/><text x="345" y="208" fill="white">logical 2</text></g>
<path d="M109 152 L500 300 M227 152 L700 300 M345 152 L900 300 M109 222 L500 300 M227 222 L700 300 M345 222 L1020 430" stroke="#647078" stroke-width="2.5" fill="none"/>
<text x="54" y="272" font-family="Arial" font-size="14" fill="#647078">Same complete prefix → same physical blocks; divergent suffix → separate block</text>
<g font-family="Arial" font-size="16" text-anchor="middle"><rect x="430" y="300" width="140" height="86" rx="5" fill="#d7e7e8" stroke="#28566b" stroke-width="2"/><text x="500" y="333" fill="#17394b" font-weight="700">Physical #0</text><text x="500" y="359" fill="#34444d" font-size="13">A B C D</text><text x="500" y="377" fill="#28566b" font-size="12">ref_count = 2</text><rect x="630" y="300" width="140" height="86" rx="5" fill="#d7e7e8" stroke="#28566b" stroke-width="2"/><text x="700" y="333" fill="#17394b" font-weight="700">Physical #1</text><text x="700" y="359" fill="#34444d" font-size="13">E F G H</text><text x="700" y="377" fill="#28566b" font-size="12">ref_count = 2</text><rect x="830" y="300" width="140" height="86" rx="5" fill="#f2ded4" stroke="#a9573d" stroke-width="2"/><text x="900" y="333" fill="#6c4032" font-weight="700">Physical #2</text><text x="900" y="359" fill="#34444d" font-size="13">I · · ·</text><text x="900" y="377" fill="#a9573d" font-size="12">A only</text><rect x="950" y="412" width="140" height="86" rx="5" fill="#f2ded4" stroke="#a9573d" stroke-width="2"/><text x="1020" y="445" fill="#6c4032" font-weight="700">Physical #3</text><text x="1020" y="471" fill="#34444d" font-size="13">X Y · ·</text><text x="1020" y="489" fill="#a9573d" font-size="12">B only</text></g>
<text x="54" y="530" font-family="Arial" font-size="13" fill="#647078">Conceptual block table · block size = 4 tokens · physical blocks need not be adjacent</text></svg>"""


def make_scheduler_svg():
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="500" viewBox="0 0 1100 500" role="img" aria-labelledby="title desc">
<title id="title">Continuous batching scheduler loop</title><desc id="desc">A scheduler iterates through running decode requests, fills remaining token budget with prompt chunks, admits waiting requests into freed slots and repeats.</desc>
<rect width="1100" height="500" fill="#fbfaf7"/><text x="54" y="54" font-family="Arial" font-size="24" font-weight="700" fill="#192630">Every iteration rebuilds the active batch</text>
<rect x="54" y="100" width="270" height="100" rx="7" fill="#d7e7e8" stroke="#28566b" stroke-width="2"/><text x="189" y="137" font-family="Arial" font-size="18" text-anchor="middle" font-weight="700" fill="#17394b">1 · Decode first</text><text x="189" y="166" font-family="Arial" font-size="14" text-anchor="middle" fill="#34444d">One token per active decoder</text>
<path d="M324 150 H396 M386 142 L398 150 L386 158" stroke="#647078" stroke-width="3" fill="none"/>
<rect x="402" y="100" width="270" height="100" rx="7" fill="#f2ded4" stroke="#a9573d" stroke-width="2"/><text x="537" y="137" font-family="Arial" font-size="18" text-anchor="middle" font-weight="700" fill="#6c4032">2 · Fill token budget</text><text x="537" y="166" font-family="Arial" font-size="14" text-anchor="middle" fill="#34444d">Chunk pending Prefill work</text>
<path d="M672 150 H744 M734 142 L746 150 L734 158" stroke="#647078" stroke-width="3" fill="none"/>
<rect x="750" y="100" width="296" height="100" rx="7" fill="#e8e6d9" stroke="#737047" stroke-width="2"/><text x="898" y="137" font-family="Arial" font-size="18" text-anchor="middle" font-weight="700" fill="#4b4a2e">3 · Admit waiting requests</text><text x="898" y="166" font-family="Arial" font-size="14" text-anchor="middle" fill="#34444d">Reuse slots freed by completion</text>
<path d="M898 200 V280 H537 V212" stroke="#647078" stroke-width="2.5" fill="none"/><path d="M529 224 L537 210 L545 224" stroke="#647078" stroke-width="2.5" fill="none"/>
<rect x="280" y="306" width="514" height="84" rx="7" fill="#17394b"/><text x="537" y="340" font-family="Arial" font-size="18" font-weight="700" text-anchor="middle" fill="white">Repeat scheduler iteration</text><text x="537" y="366" font-family="Arial" font-size="14" text-anchor="middle" fill="#d7e7e8">Per-step limits: active sequences + scheduled token budget</text>
<text x="54" y="454" font-family="Arial" font-size="13" fill="#647078">Conceptual policy, not a trace of a particular vLLM release or a measured workload.</text></svg>"""


def make_fragmentation_svg(m):
    maximum = max(m["static_wasted_slots"], 1)
    sw = 650 * m["static_wasted_slots"] / maximum
    pw = 650 * m["paged_tail_waste_slots"] / maximum
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="440" viewBox="0 0 1000 440" role="img" aria-labelledby="title desc">'
        '<title id="title">Static reservation vs paged allocation</title><desc id="desc">Four variable-length sequences show lower unused-slot arithmetic with 16-token blocks than reserving 128 token slots for every request.</desc>'
        '<rect width="1000" height="440" fill="#fbfaf7"/>'
        '<text x="54" y="54" font-family="Arial" font-size="23" font-weight="700" fill="#192630">Reserve for maximum length vs allocate on demand</text>'
        '<text x="54" y="88" font-family="Arial" font-size="14" fill="#647078">Four contexts · live tokens = 161 · block size = 16 · static reservation = 128 per request</text>'
        '<text x="54" y="164" font-family="Arial" font-size="16" fill="#34444d">Static reservation · wasted slots</text>'
        f'<rect x="54" y="180" width="{sw:.1f}" height="44" rx="4" fill="#a9573d"/><text x="{68+sw:.1f}" y="209" font-family="Arial" font-size="16" fill="#192630">{m["static_wasted_slots"]}</text>'
        '<text x="54" y="272" font-family="Arial" font-size="16" fill="#34444d">Paged allocation · tail waste</text>'
        f'<rect x="54" y="288" width="{pw:.1f}" height="44" rx="4" fill="#28566b"/><text x="{68+pw:.1f}" y="317" font-family="Arial" font-size="16" fill="#192630">{m["paged_tail_waste_slots"]}</text>'
        f'<text x="54" y="390" font-family="Arial" font-size="14" fill="#647078">Arithmetic reduction in unused slots: {m["wasted_slot_reduction_percent"]:.2f}% · not a measured serving gain.</text></svg>'
    )


def make_architecture_svg():
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="580" viewBox="0 0 1200 580" role="img" aria-labelledby="title desc">
<title id="title">Simplified vLLM V1 serving architecture</title>
<desc id="desc">Client requests enter the API Server, are scheduled by Engine Core which manages KV cache blocks, and are executed by GPU Workers. Metrics are exposed to Prometheus and Grafana.</desc>
<rect width="1200" height="580" fill="#fbfaf7"/>
<text x="54" y="54" font-family="Arial" font-size="24" font-weight="700" fill="#192630">Serving responsibilities and control path</text>
<rect x="54" y="135" width="160" height="88" rx="7" fill="#e8e6d9" stroke="#737047" stroke-width="2"/>
<text x="134" y="171" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="#4b4a2e">Client /</text><text x="134" y="195" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="#4b4a2e">Gateway</text>
<path d="M214 179 H276 M266 171 L278 179 L266 187" stroke="#647078" stroke-width="3" fill="none"/>
<rect x="282" y="135" width="190" height="88" rx="7" fill="#d7e7e8" stroke="#28566b" stroke-width="2"/>
<text x="377" y="171" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="#17394b">API Server</text><text x="377" y="195" text-anchor="middle" font-family="Arial" font-size="13" fill="#34444d">HTTP · auth · input</text>
<path d="M472 179 H528 M518 171 L530 179 L518 187" stroke="#647078" stroke-width="3" fill="none"/>
<rect x="534" y="96" width="310" height="190" rx="8" fill="#edf0e8" stroke="#737047" stroke-width="2"/>
<text x="689" y="130" text-anchor="middle" font-family="Arial" font-size="19" font-weight="700" fill="#4b4a2e">Engine Core</text>
<rect x="560" y="150" width="258" height="52" rx="5" fill="#28566b"/><text x="689" y="182" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="white">Scheduler · token budget</text>
<rect x="560" y="214" width="258" height="52" rx="5" fill="#a9573d"/><text x="689" y="246" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="white">KV Cache Manager · blocks</text>
<path d="M844 179 H902 M892 171 L904 179 L892 187" stroke="#647078" stroke-width="3" fill="none"/>
<rect x="908" y="135" width="238" height="88" rx="7" fill="#f2ded4" stroke="#a9573d" stroke-width="2"/>
<text x="1027" y="171" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="#6c4032">GPU Workers</text><text x="1027" y="195" text-anchor="middle" font-family="Arial" font-size="13" fill="#34444d">Forward passes · kernels</text>
<path d="M1027 223 V322 H689 V286" stroke="#647078" stroke-width="2.5" fill="none"/><path d="M681 298 L689 284 L697 298" stroke="#647078" stroke-width="2.5" fill="none"/>
<rect x="532" y="338" width="314" height="74" rx="6" fill="#17394b"/><text x="689" y="368" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="white">KV block pool</text><text x="689" y="391" text-anchor="middle" font-family="Arial" font-size="13" fill="#d7e7e8">Allocation · sharing · release · eviction</text>
<path d="M689 412 V458 H377 V223" stroke="#647078" stroke-width="2" stroke-dasharray="7 6" fill="none"/>
<rect x="54" y="458" width="260" height="72" rx="6" fill="#e3ecec" stroke="#28566b" stroke-width="2"/><text x="184" y="488" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#17394b">Metrics endpoint</text><text x="184" y="512" text-anchor="middle" font-family="Arial" font-size="13" fill="#34444d">Prometheus → Grafana</text>
<path d="M377 223 V422 H314 V494" stroke="#28566b" stroke-width="2" stroke-dasharray="6 6" fill="none"/>
<text x="875" y="492" font-family="Arial" font-size="13" fill="#647078">DP coordinator may be present in multi-rank deployments.</text>
<text x="54" y="559" font-family="Arial" font-size="12" fill="#647078">Simplified role diagram · process counts and communication paths vary with DP / TP / PP configuration.</text>
</svg>"""


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    for folder in (EVIDENCE, VISUALS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)

    fragmentation = fragmentation_metrics()
    (REPORTS / "fragmentation-metrics.json").write_text(
        json.dumps(fragmentation, indent=2) + "\n", encoding="utf-8"
    )

    # Same first eight tokens, then divergent tails. Full four-token blocks can
    # be reused; partial blocks intentionally are not entered in the prefix map.
    pool = KVBlockPool(num_blocks=8, block_size=4)
    common = [101, 102, 103, 104, 105, 106, 107, 108]
    req_a = pool.create_sequence("A", common + [109])
    req_b = pool.create_sequence("B", common + [201, 202])
    initial_snapshot = pool.snapshot([req_a, req_b])
    shared_first_two = req_a.block_ids[:2] == req_b.block_ids[:2]
    ref_count_two_before_release = all(
        pool.blocks[bid].ref_count == 2 for bid in req_a.block_ids[:2]
    )
    unique_a_b = len(pool.active_unique_blocks([req_a, req_b]))
    unshared_reference = 3 + 3

    pool.release_sequence(req_a)
    shared_survives = all(pool.blocks[bid].ref_count == 1 for bid in req_b.block_ids[:2])
    for token in (203, 204, 205):
        pool.append_token(req_b, token)
    block_table_after_decode = pool.snapshot([req_b])
    req_c = pool.create_sequence("C", common + [201, 202, 203, 204, 999])
    third_request_shares = req_c.block_ids[:3] == req_b.block_ids[:3]
    final_allocator_snapshot = pool.snapshot([req_b, req_c])

    block_rows = []
    for state in (req_b, req_c):
        for logical_i, block_id in enumerate(state.block_ids):
            b = pool.blocks[block_id]
            block_rows.append(dict(
                request_id=state.request_id, logical_block_index=logical_i,
                physical_block_id=block_id, token_ids=";".join(map(str, b.tokens)),
                tokens_in_block=len(b.tokens), block_capacity=pool.block_size,
                ref_count=b.ref_count, cached_full_block=b.cache_key is not None,
            ))
    write_csv(EVIDENCE / "block-table-evidence.csv", block_rows)

    workload = [
        dict(request_id="A", arrival_step=0, prompt_tokens=9, output_tokens=4),
        dict(request_id="B", arrival_step=1, prompt_tokens=6, output_tokens=3),
        dict(request_id="C", arrival_step=2, prompt_tokens=5, output_tokens=2),
        dict(request_id="D", arrival_step=3, prompt_tokens=7, output_tokens=3),
    ]
    events, scheduler_steps, request_rows, scheduler_metrics = run_scheduler(workload)
    write_csv(EVIDENCE / "scheduler-events.csv", events)
    write_csv(EVIDENCE / "scheduler-steps.csv", scheduler_steps)
    write_csv(EVIDENCE / "request-outcomes.csv", request_rows)

    allocator_checks = {
        "identical_full_prefix_blocks_are_shared": shared_first_two,
        "shared_prefix_reference_count_starts_at_two": ref_count_two_before_release,
        "shared_blocks_survive_first_request_release": shared_survives,
        "later_request_reuses_three_full_prefix_blocks": third_request_shares,
        "physical_blocks_lower_than_unshared_reference": unique_a_b < unshared_reference,
        "partial_tail_blocks_are_not_shared": req_b.block_ids[-1] != req_c.block_ids[-1],
    }
    checks = {
        **allocator_checks,
        **scheduler_metrics["checks"],
        "paged_tail_waste_lower_than_static_reservation": (
            fragmentation["paged_tail_waste_slots"] < fragmentation["static_wasted_slots"]
        ),
    }

    metrics = {
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "dependencies": "Python standard library only",
        "device": "CPU", "cost_eur": 0,
        "experiments": {
            "allocator": {
                "block_size": pool.block_size, "pool_blocks": pool.num_blocks,
                "request_a_initial_block_table": initial_snapshot["block_tables"]["A"],
                "request_b_initial_block_table": initial_snapshot["block_tables"]["B"],
                "initial_unique_physical_blocks_a_b": unique_a_b,
                "unshared_reference_blocks_a_b": unshared_reference,
                "snapshot_after_b_decode": block_table_after_decode,
                "prefix_cache_hits_after_c": pool.prefix_cache_hits,
                "prefix_cache_evictions": pool.evictions,
                "final_snapshot": final_allocator_snapshot,
            },
            "fragmentation": fragmentation,
            "scheduler": scheduler_metrics,
        },
        "checks": checks,
        "limitations": [
            "Teaching simulator, not vLLM source code, a CUDA kernel or GPU execution.",
            "Block IDs and scheduler rounds are logical examples, not hardware measurements.",
            "Scheduler uses a simplified decode-first, round-robin prefill policy.",
            "No model weights, tokenizer, CUDA, GPU, external dataset, API or hosted service.",
            "Production behaviour depends on model architecture, vLLM version, hardware, workload and configuration.",
        ],
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (VISUALS / "paged-block-mapping.svg").write_text(make_block_svg(), encoding="utf-8")
    (VISUALS / "continuous-batching-loop.svg").write_text(make_scheduler_svg(), encoding="utf-8")
    (VISUALS / "static-vs-paged-fragmentation.svg").write_text(
        make_fragmentation_svg(fragmentation), encoding="utf-8"
    )
    (VISUALS / "vllm-serving-architecture.svg").write_text(
        make_architecture_svg(), encoding="utf-8"
    )

    report = f"""# Lab report — Paged KV allocation and continuous batching

- Host: {socket.gethostname()}
- OS: {platform.platform()}
- Python: {platform.python_version()}
- Dependencies: Python standard library only
- Device: CPU · no GPU, CUDA, model weights, tokenizer, API or cloud service
- Cost: 0 €

## Experiment 1 — static reservation vs paged allocation

Workload context lengths: {fragmentation["sequence_lengths"]} tokens.
- Total live tokens: {fragmentation["live_tokens"]}.
- Static max-length reservation: {fragmentation["static_reserved_slots"]} token slots.
- Static unused slots: {fragmentation["static_wasted_slots"]}.
- Paged allocation, block size {fragmentation["block_size_tokens"]}: {fragmentation["paged_allocated_slots"]} slots.
- Last-block tail waste: {fragmentation["paged_tail_waste_slots"]} slots.
- Arithmetic reduction in unused slots: {fragmentation["wasted_slot_reduction_percent"]:.2f}%.

This is capacity arithmetic, not measured GPU memory or a production benchmark. Uniform pages remove the need for one contiguous per-request allocation, but do not remove internal tail slack, metadata or all memory overhead.

## Experiment 2 — physical block tables and prefix reuse

- Request A: prompt length 9; initial block table: {initial_snapshot["block_tables"]["A"]}.
- Request B: prompt length 10; initial block table: {initial_snapshot["block_tables"]["B"]}.
- Shared complete-prefix blocks: {req_b.block_ids[:2]}.
- Unique physical blocks for A and B: {unique_a_b}; naive unshared reference: {unshared_reference}.
- After A releases its reference, B's shared blocks keep reference count 1.
- B appends tokens and creates a third full block.
- Request C reuses three complete prefix blocks from B.
- Prefix-cache hits after C: {pool.prefix_cache_hits}; evictions: {pool.evictions}.

Only complete blocks are indexed in this teaching model. Real vLLM prefix-cache hashing, block lifecycle, block groups, eviction and connector behaviour are more involved and version-dependent.

## Experiment 3 — continuous batching with a token budget

- Per-step token budget: {scheduler_metrics["token_budget_per_step"]}.
- Maximum active requests: {scheduler_metrics["max_active_requests"]}.
- Scheduler iterations: {scheduler_metrics["scheduler_steps"]}.
- Peak concurrent requests: {scheduler_metrics["peak_running_requests"]}.
- Prompt tokens processed: {scheduler_metrics["total_prompt_tokens"]}.
- Output tokens generated: {scheduler_metrics["total_output_tokens"]}.
- Mean queue wait: {scheduler_metrics["mean_queue_wait_steps"]:.2f} simulated scheduler steps.

The simulator admits waiting work when a running request completes, within the active-request limit. Each iteration schedules decode first, then uses remaining token budget for round-robin prompt chunks. A step is a logical scheduler iteration, not milliseconds.

## Automated checks

"""
    for name, passed in checks.items():
        report += f"- {name}: {'PASS' if passed else 'FAIL'}\n"
    report += f"\n**LAB STATUS: {'PASS' if all(checks.values()) else 'FAIL'}**\n\n"
    report += """## Boundaries

- This is a deliberately small reference simulator, not vLLM source code and not PagedAttention CUDA execution.
- It validates block allocation, reference-count safety, complete-block prefix reuse, tail-fragmentation arithmetic and token-budget admission.
- It does not measure tokens/s, TTFT, TPOT, GPU utilisation, network cost or production scaling.
- Real vLLM V1 uses its own KV cache manager, scheduler, kernels, memory planner, admission/preemption and observability. Consult versioned vendor documentation before tuning production flags.
- The simulator uses deterministic integer token IDs and no real model or customer data.
"""
    (REPORTS / "lab-report.md").write_text(report, encoding="utf-8")

    print("=== LLM SERVING LAB SUMMARY ===")
    print(f"host={socket.gethostname()} python={platform.python_version()} dependencies=stdlib device=cpu")
    print("fragmentation=" + json.dumps({
        "static_waste_slots": fragmentation["static_wasted_slots"],
        "paged_tail_waste_slots": fragmentation["paged_tail_waste_slots"],
        "waste_reduction_percent": round(fragmentation["wasted_slot_reduction_percent"], 2),
    }))
    print("allocator=" + json.dumps({
        "a_b_unique_blocks": unique_a_b,
        "unshared_reference_blocks": unshared_reference,
        "prefix_cache_hits": pool.prefix_cache_hits,
        "evictions": pool.evictions,
    }))
    print("scheduler=" + json.dumps({
        "steps": scheduler_metrics["scheduler_steps"],
        "max_active": scheduler_metrics["max_active_requests"],
        "mean_queue_wait_steps": round(scheduler_metrics["mean_queue_wait_steps"], 2),
        "prompt_tokens": scheduler_metrics["total_prompt_tokens"],
        "output_tokens": scheduler_metrics["total_output_tokens"],
    }))
    print("checks=" + json.dumps(checks, ensure_ascii=False))
    print("LAB STATUS: " + ("PASS" if all(checks.values()) else "FAIL"))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
