# LLM Serving — Laboratorio 05

Simulador didáctico de dos mecanismos de serving: asignación de KV Cache mediante bloques físicos y admisión continua de peticiones bajo un token budget.

## Objetivos

1. Comparar reserva de memoria por longitud máxima con asignación on-demand de bloques.
2. Mostrar cómo una block table traduce bloques lógicos de secuencia a IDs de bloques físicos no contiguos.
3. Reutilizar bloques completos de un prefijo común entre solicitudes, con reference counting.
4. Liberar referencias sin destruir bloques que siguen siendo utilizados por otra petición.
5. Simular continuous batching: admitir solicitudes nuevas cuando finalizan otras, sin esperar a que termine un lote estático entero.
6. Simular chunked prefill bajo un presupuesto de tokens por iteración y priorizar trabajo de decode.
7. Exportar trazas CSV, métricas JSON, informe y diagramas SVG.

## Coste y dependencias

- Python 3.12 o compatible.
- Solo Python standard library.
- Windows, macOS o Linux.
- No instala vLLM ni descarga pesos de modelos.
- No necesita CUDA, GPU, Docker, API, cloud, datasets o servicios de pago.
- Coste: 0 €.

## Ejecutar

En PowerShell:

    .\run.ps1

Manual:

    python -m py_compile .\src\run_lab.py
    python -u .\src\run_lab.py

## Experimento 1 — asignación de bloques y fragmentación

La reserva estática aparta para cada secuencia un máximo conocido, aunque todavía no haya generado tokens. La asignación paginada divide la memoria KV en bloques de tamaño fijo y reserva nuevos bloques según crece el contexto.

Para cada longitud T y tamaño de bloque B:

    required_blocks = ceil(T / B)
    allocated_slots = required_blocks × B
    tail_waste = allocated_slots - T

Con bloques uniformes, cualquier bloque libre sirve sin exigir una zona contigua de memoria física. Esto elimina la fragmentación externa del esquema de bloques, pero queda slack en el último bloque de cada secuencia y siguen existiendo metadatos, allocator overhead y límites de capacidad.

## Experimento 2 — block table y prefix caching

Cada solicitud mantiene una lista de IDs de bloques físicos. Dos solicitudes con un prefijo completo idéntico pueden apuntar a los mismos bloques, siempre que coincidan el modelo/revisión y el contenido relevante. El simulador calcula un hash del prefijo acumulado para cada bloque completo.

- Los bloques completos se añaden al índice de prefijos.
- Los bloques compartidos incrementan ref_count.
- Liberar una petición decrementa referencias.
- Un bloque compartido no se puede reutilizar mientras su ref_count sea mayor que cero.
- Los bloques cacheados con ref_count=0 son candidatos a evicción LRU.
- Los bloques finales parciales no se publican en el índice de prefijos.

Es una versión pedagógica de la idea. El KV cache manager de vLLM V1 tiene más detalles: hashes, free queue, bloques duplicados, eviction y lifecycle específicos de la implementación.

## Experimento 3 — continuous batching

El simulador utiliza:

- Token budget por iteración: 6.
- Máximo de solicitudes activas: 3.
- Decode-first: hasta un token de salida para cada petición que está generando.
- Chunked prefill: los tokens de prompt restantes se reparten en round-robin dentro del presupuesto disponible.
- Admisión dinámica: cuando una petición termina, la siguiente solicitud en espera ocupa el hueco en la misma iteración.

El scheduler real de vLLM V1 no debe asumirse idéntico a este pseudocódigo. La documentación actual describe un presupuesto de tokens por paso, solicitudes en ejecución y waiting, chunked prefill y políticas de scheduling configurables. Usa esta simulación para entender los invariantes, no para predecir latencia.

## Salidas

- reports/metrics.json: resumen global y checks.
- reports/fragmentation-metrics.json: reserva estática vs bloques.
- reports/lab-report.md: informe de ejecución.
- evidence/block-table-evidence.csv: mapeo lógico/físico y reference counts.
- evidence/scheduler-events.csv: eventos por solicitud y paso.
- evidence/scheduler-steps.csv: presupuesto de tokens y concurrencia por paso.
- evidence/request-outcomes.csv: espera y finalización por solicitud.
- evidence/visuals/paged-block-mapping.svg: block tables y prefijo compartido.
- evidence/visuals/continuous-batching-loop.svg: esquema del scheduler.
- evidence/visuals/static-vs-paged-fragmentation.svg: slack de reserva.
- evidence/visuals/vllm-serving-architecture.svg: arquitectura simplificada de responsabilidades.

## Interpretar resultados

El informe separa el ahorro aritmético de slots vacíos del rendimiento. No conviertas el porcentaje de slots desperdiciados en una afirmación sobre tokens/s o TTFT. No se ejecuta un modelo real, no hay GPU y cada paso del scheduler es una iteración lógica, no una unidad de tiempo.

## Documentación primaria

- vLLM, https://docs.vllm.ai/en/latest/design/arch_overview/
- vLLM, https://docs.vllm.ai/en/latest/design/paged_attention/
- vLLM, https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- vLLM, https://docs.vllm.ai/en/latest/configuration/optimization/
- vLLM, https://docs.vllm.ai/en/latest/configuration/engine_args/
- vLLM, https://docs.vllm.ai/en/latest/design/metrics/
- vLLM Security, https://docs.vllm.ai/en/stable/usage/security/ (CVE-2025-46570, cache_salt, multi-tenant isolation).
- Kwon et al., https://doi.org/10.1145/3600006.3613165, SOSP 2023.

Documentación reconsultada el 14 de noviembre de 2026.
