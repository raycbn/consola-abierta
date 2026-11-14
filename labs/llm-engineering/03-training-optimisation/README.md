# Cómo aprende un LLM — Laboratorio 03

Laboratorio CPU-only para estudiar el objetivo autoregresivo y el ciclo de optimización de un decoder-only Transformer.

## Qué comprueba

1. La alineación de next-token prediction: la entrada y el target se desplazan una posición.
2. La equivalencia numérica entre CrossEntropyLoss(logits, targets) y log_softmax + NLLLoss.
3. La derivada de Cross Entropy respecto a los logits: (softmax(logits) - one_hot(targets)) / N, contrastada con autograd en float64.
4. El weight decay desacoplado de AdamW mediante una actualización escalar controlada.
5. El efecto de clip_grad_norm_ sobre el global gradient norm.
6. La forma y los límites del schedule lineal warmup + cosine decay.
7. Una comparación controlada entre AdamW con learning rate constante y AdamW con warmup + cosine.

La ejecución automatizada contiene doce aserciones booleanas: alineación, equivalencia y gradiente de Cross Entropy, AdamW, clipping, cuatro propiedades del schedule, reducción de train/validation loss en ambos runs y finitud de las pérdidas finales.

## Entorno y coste

- Python 3.12.
- PyTorch 2.14.0.
- Windows, macOS o Linux.
- CPU suficiente; el laboratorio fuerza el dispositivo CPU.
- Sin CUDA/XPU, checkpoints descargados, APIs, cloud ni servicios de pago.
- Coste: 0 €.

Instala PyTorch CPU desde el índice oficial:

    python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

## Ejecutar

En Windows PowerShell:

    .\run.ps1

O manualmente:

    python -m py_compile .\src\run_lab.py
    python -u .\src\run_lab.py

## Diseño de la comparación

Los dos experimentos parten de los mismos pesos iniciales y consumen los mismos minibatches precomputados en el mismo orden. Comparten arquitectura, corpus, batch size, AdamW, weight decay, gradient clipping y número de actualizaciones.

- constant: learning rate fijo de 5e-4.
- warmup_cosine: warmup lineal durante 24 updates, seguido de cosine decay desde 5e-4 hasta 5e-5.
- Cada política ejecuta 240 updates.
- La loss de validación se calcula con ventanas fijas y nunca interviene en el backward pass ni en las actualizaciones.

Este diseño permite comparar las dos políticas dentro de este experimento, no probar que un schedule sea superior en cualquier modelo, corpus o presupuesto de entrenamiento.

## Datos

Se reutiliza una copia local del corpus técnico en español creado para el capítulo 01. El tokenizer es deliberadamente character-level para reducir dependencias y hacer visibles los IDs y targets. No representa los tokenizers subword utilizados habitualmente por los LLM actuales.

El corpus tiene secciones de entrenamiento y validación separadas explícitamente. La validación no se usa para actualizar pesos. El alfabeto se construye con los caracteres observados en ambos splits para que la codificación sea válida; esto no incorpora ejemplos de validación al cálculo de gradientes, aunque sí revela qué caracteres aparecen en el split reservado. Es una simplificación didáctica; en un pipeline real, tokenizer/vocabulary se ajusta con entrenamiento y se define cómo manejar tokens desconocidos.

## Salidas

- reports/metrics.json: entorno, configuración, métricas, resultados y checks.
- reports/lab-report.md: resumen legible.
- evidence/training-comparison.csv: loss de minibatch, learning rate y grad norm pre-clipping en cada update.
- evidence/loss-history.json: train/validation loss en los puntos de evaluación.
- evidence/visuals/training-policy-comparison.svg: curvas de ambos experimentos.
- evidence/visuals/warmup-cosine-schedule.svg: curva de la tasa de aprendizaje.

## Límites

- El corpus es corto y de dominio técnico; no permite inferir calidad general de lenguaje.
- El modelo es una implementación didáctica pequeña, no un LLM de producción.
- Se comparan dos políticas con una semilla y un presupuesto concretos. Para una conclusión estadística harían falta varias semillas, más datos y un diseño de evaluación más amplio.
- La loss mide ajuste probabilístico al siguiente carácter en este corpus; no demuestra comprensión, factualidad, utilidad ni seguridad.
- El experimento no mide GPU throughput, memoria de aceleradores ni escalabilidad distribuida.

## Referencias

- Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- Loshchilov y Hutter (2017), [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- PyTorch 2.14, [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).
- PyTorch 2.14, [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
- PyTorch 2.14, [CosineAnnealingLR](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).
- PyTorch 2.14, [clip_grad_norm_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html).
- Hugging Face Transformers, [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling).

Reproducido localmente el 14 de noviembre de 2026.