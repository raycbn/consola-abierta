# Lab report — Paged KV allocation and continuous batching

- Host: PCRAY
- OS: Windows-11-10.0.26300-SP0
- Python: 3.12.10
- Dependencies: Python standard library only
- Device: CPU · no GPU, CUDA, model weights, tokenizer, API or cloud service
- Cost: 0 €

## Experiment 1 — static reservation vs paged allocation

Workload context lengths: [23, 77, 12, 49] tokens.
- Total live tokens: 161.
- Static max-length reservation: 512 token slots.
- Static unused slots: 351.
- Paged allocation, block size 16: 192 slots.
- Last-block tail waste: 31 slots.
- Arithmetic reduction in unused slots: 91.17%.

This is capacity arithmetic, not measured GPU memory or a production benchmark. Uniform pages remove the need for one contiguous per-request allocation, but do not remove internal tail slack, metadata or all memory overhead.

## Experiment 2 — physical block tables and prefix reuse

- Request A: prompt length 9; initial block table: [0, 1, 2].
- Request B: prompt length 10; initial block table: [0, 1, 3].
- Shared complete-prefix blocks: [0, 1].
- Unique physical blocks for A and B: 4; naive unshared reference: 6.
- After A releases its reference, B's shared blocks keep reference count 1.
- B appends tokens and creates a third full block.
- Request C reuses three complete prefix blocks from B.
- Prefix-cache hits after C: 5; evictions: 0.

Only complete blocks are indexed in this teaching model. Real vLLM prefix-cache hashing, block lifecycle, block groups, eviction and connector behaviour are more involved and version-dependent.

## Experiment 3 — continuous batching with a token budget

- Per-step token budget: 6.
- Maximum active requests: 3.
- Scheduler iterations: 10.
- Peak concurrent requests: 3.
- Prompt tokens processed: 27.
- Output tokens generated: 12.
- Mean queue wait: 0.50 simulated scheduler steps.

The simulator admits waiting work when a running request completes, within the active-request limit. Each iteration schedules decode first, then uses remaining token budget for round-robin prompt chunks. A step is a logical scheduler iteration, not milliseconds.

## Automated checks

- identical_full_prefix_blocks_are_shared: PASS
- shared_prefix_reference_count_starts_at_two: PASS
- shared_blocks_survive_first_request_release: PASS
- later_request_reuses_three_full_prefix_blocks: PASS
- physical_blocks_lower_than_unshared_reference: PASS
- partial_tail_blocks_are_not_shared: PASS
- all_requests_completed: PASS
- per_step_token_budget_respected: PASS
- max_active_requests_respected: PASS
- every_request_admitted_once: PASS
- every_prompt_token_processed: PASS
- every_output_token_generated: PASS
- continuous_admission_used: PASS
- paged_tail_waste_lower_than_static_reservation: PASS

**LAB STATUS: PASS**

## Boundaries

- This is a deliberately small reference simulator, not vLLM source code and not PagedAttention CUDA execution.
- It validates block allocation, reference-count safety, complete-block prefix reuse, tail-fragmentation arithmetic and token-budget admission.
- It does not measure tokens/s, TTFT, TPOT, GPU utilisation, network cost or production scaling.
- Real vLLM V1 uses its own KV cache manager, scheduler, kernels, memory planner, admission/preemption and observability. Consult versioned vendor documentation before tuning production flags.
- The simulator uses deterministic integer token IDs and no real model or customer data.
