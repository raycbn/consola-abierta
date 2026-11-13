# Transformer Internals 01 — TinyGPT Lab

This folder contains the runnable experiment and the exact artifacts used in the
Consola Abierta article about Transformer internals.

## Objective

Reproduce the mechanics of a small decoder-only Transformer using PyTorch:
token and position embeddings, Q/K/V projections, scaled dot-product attention,
a causal mask, multi-head attention, Pre-LayerNorm residual blocks, a
feed-forward network, a language-modeling head, Cross Entropy, Backpropagation
and autoregressive generation.

This is a teaching experiment. It is not a production LLM and it is not a
performance benchmark.

## Tested host

- Windows 11 Home, build 26300
- Intel Core i5-1155G7, 4 cores / 8 logical processors
- 16 GB RAM
- Intel Iris Xe integrated graphics (not used)
- Python 3.12.10
- PyTorch 2.14.0+cpu
- Device: CPU, four PyTorch threads
- No CUDA or Intel XPU available in this Python installation

## Model configuration and measured run

- Character vocabulary: 66 symbols
- Training / validation corpus: 7,571 / 2,413 characters
- Context length: 96 characters
- Embedding width: 64
- Transformer blocks: 2
- Attention heads: 4 (16 dimensions per head)
- Feed-forward width: 4 × embedding width
- Pre-LayerNorm residual blocks
- Learned absolute position embeddings
- Tied token-embedding and output-projection weights
- Dropout: 0.10
- Optimizer: AdamW, learning rate 0.0005, weight decay 0.01
- 600 optimizer steps, batch size 16
- Trainable parameters: 110,464
- Wall time: 53.383 seconds (measured on the host above)
- Train loss: 4.2293 → 2.1660
- Validation loss: 4.2222 → 2.3091

## Run

From PowerShell:

    python -m pip install -r requirements.txt
    .\run.ps1

Or run a shorter smoke test:

    .\run.ps1 -Steps 50

The script reads data/corpus_es.txt and writes to evidence, models and reports. The requirements file points to the official PyTorch CPU wheel index to reproduce the CPU-only environment:
  https://pytorch.org/get-started/previous-versions/
A different CPU, PyTorch version or numerical kernel can change exact losses
and timing; compare the trends and invariant checks, not wall time alone.

## Automated checks

1. Output logits have shape B × T × vocabulary.
2. Each attention row sums to one after softmax (maximum absolute error recorded).
3. Future positions receive exactly zero attention mass.
4. Changing future tokens does not change logits for the unchanged prefix.
5. Manual attention agrees numerically with
   torch.nn.functional.scaled_dot_product_attention.
6. Backpropagation produces gradients and the optimizer updates parameters.
7. Training and held-out validation losses decrease in the recorded run.

Recorded checks:
- Attention row-sum maximum error: 2.3841858e-7
- Maximum attention mass in future positions: 0
- Prefix-logit delta after future-token perturbation: 0
- Manual vs PyTorch SDPA maximum absolute difference: 2.9802322e-7

## Interpretation and limitations

- The tokenizer operates on individual Unicode characters. Real LLMs commonly
  use subword or byte-level tokenizers. Character tokenization creates longer
  sequences and a very different vocabulary/cost profile.
- The corpus is short, authored for this experiment and domain-specific.
  Validation text is separate and is never used for gradient updates.
- The model has only 110k parameters. It is not a general-purpose LLM; its
  losses are not comparable to external model benchmarks.
- CPU-only execution was intentional because this host has no CUDA or XPU.
- The manual attention code is transparent for inspection. Fused kernels can
  avoid materialising the full attention matrix and have different performance
  characteristics.
- The character output is only a sanity check that the autoregressive loop
  runs. It is not a claim of fluent generation or semantic quality.
- Attention heatmaps show operation-level weights; they are not, by themselves,
  a complete explanation of a model's internal reasoning.

## Evidence files

- evidence/training-log.csv: each optimizer step, loss and gradient norm
- evidence/loss-history.json: periodic train/validation metrics
- evidence/full-run-console.txt: captured terminal output
- evidence/generated-sample.txt: generated character sample
- reports/metrics.json: environment, configuration, numeric results and tests
- reports/lab-report.md: human-readable report
- models/tiny-gpt-es.pt: checkpoint and vocabulary
- assets/labs/transformer-internals/training-validation-loss.svg: measured loss curve
- assets/labs/transformer-internals/causal-mask.svg: causal visibility matrix
- assets/labs/transformer-internals/attention-heads-heatmap.svg: measured attention weights
- assets/labs/transformer-internals/decoder-only-architecture.svg: decoder flow
- assets/labs/transformer-internals/attention-dataflow.svg: Q/K/V tensor flow

## Source references

- Vaswani et al., Attention Is All You Need (NeurIPS 2017):
  https://arxiv.org/abs/1706.03762
- Radford et al., Language Models are Unsupervised Multitask Learners (2019):
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- PyTorch scaled_dot_product_attention:
  https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- Official PyTorch 2.14.0 CPU wheel installation instructions:
  https://pytorch.org/get-started/previous-versions/
- PyTorch CrossEntropyLoss:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Karpathy nanoGPT reference implementation:
  https://github.com/karpathy/nanoGPT/blob/master/model.py
