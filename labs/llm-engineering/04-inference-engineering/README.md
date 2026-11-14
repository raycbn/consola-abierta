# LLM Inference — Laboratorio 04

Laboratorio CPU-only para estudiar las dos fases de inferencia autoregresiva y la reutilización del estado de atención.

## Qué demuestra

1. Prefill procesa todas las posiciones del prompt en paralelo y construye los pares Key/Value de cada capa.
2. Decode procesa una nueva posición y consulta los K/V anteriores junto con los del token actual.
3. Los logits de cada paso incremental coinciden numéricamente con recalcular el prefijo completo.
4. La longitud de la KV Cache aumenta una posición por token incorporado.
5. El tamaño real de los tensores K/V coincide con la fórmula de memoria.
6. La ruta con cache evita volver a pasar todo el prefijo por las capas del modelo en cada paso.

La ejecución contiene siete comprobaciones automáticas. Registra el error absoluto máximo en prefill y en cada paso de decode.

## Entorno y coste

- Python 3.12.
- PyTorch 2.14.0.
- Windows, macOS o Linux.
- CPU; el laboratorio fuerza CPU y float32.
- No descarga pesos de modelos ni datasets.
- Sin CUDA, APIs, cloud, endpoints alojados ni servicios de pago.
- Coste: 0 €.

Instala PyTorch CPU desde el índice oficial:

    python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

## Ejecutar

En Windows PowerShell:

    .\run.ps1

O manualmente:

    python -m py_compile .\src\run_lab.py
    python -u .\src\run_lab.py

## Diseño experimental

Se construye un decoder-only Transformer didáctico de tres capas, cuatro attention heads, d_model=64 y batch size 2. Los pesos se inicializan de forma determinista y aleatoria. No se entrena: el objetivo es aislar el mecanismo de inferencia, no medir calidad lingüística.

El prompt tiene 24 posiciones y se incorporan 12 tokens de continuación fijados previamente. Para cada posición nueva se comparan los logits de la ruta con KV Cache contra un forward completo sobre el prefijo acumulado. Ambas rutas usan las mismas capas y pesos.

El código implementa manualmente la máscara causal durante prefill. En decode no aplica una máscara triangular sobre la matriz de atención: el query tiene longitud uno y las claves disponibles son exclusivamente las posiciones pasadas y la actual. Una máscara causal no cuadrada aplicada sin atender a su alineación puede ser incorrecta; el test comprueba la equivalencia con el forward completo.

## Fórmula de memoria

Para un decoder con full attention:

    KV_bytes = 2 × layers × batch × tokens × KV_heads × head_dim × bytes_per_element

El factor 2 representa Key y Value. Con GQA/MQA, KV_heads puede ser menor que el número de query heads. Sliding-window y arquitecturas híbridas pueden limitar la longitud almacenada en determinadas capas.

El CSV de memoria usa un ejemplo ilustrativo de 32 capas, batch 1, ocho KV heads, head_dim 128 y dos bytes por elemento (FP16/BF16). No es una configuración atribuida a un modelo concreto.

## Salidas

- reports/metrics.json: entorno, configuración, comprobaciones y errores máximos.
- reports/lab-report.md: resumen legible de la ejecución.
- evidence/cache-equivalence.csv: error numérico en prefill y cada decode.
- evidence/token-position-comparison.csv: recuento estructural con y sin cache.
- evidence/kv-cache-memory.csv: cálculo de memoria para varias longitudes.
- evidence/visuals/prefill-decode-flow.svg: flujo de las dos fases.
- evidence/visuals/token-position-comparison.svg: comparación estructural.
- evidence/visuals/kv-cache-memory.svg: crecimiento de memoria.

## Límites

- Los pesos son aleatorios. No se evalúa generación de texto ni calidad de un LLM.
- El test de CPU comprueba corrección, no aceleración real de GPU.
- El recuento de posiciones no equivale a FLOPs ni a latencia.
- La cache evita recalcular proyecciones y estados anteriores, pero cada nuevo query sigue atendiendo al contexto KV acumulado.
- No se estudian PagedAttention, continuous batching, prefix caching, speculative decoding, offloading ni cuantización del cache; se reservan para siguientes capítulos o documentación específica.
- La fórmula no incluye fragmentación del allocator, metadatos, workspaces, pesos ni activaciones.

## Referencias primarias

- Hugging Face Transformers, [Cache strategies](https://huggingface.co/docs/transformers/en/kv_cache).
- NVIDIA, [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/).
- NVIDIA, [Co-Designing AI Model Attention for Fast, Interactive Long-Context Inference](https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference).
- PyTorch, [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
- vLLM, [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/).

Reproducido localmente el 14 de noviembre de 2026.
