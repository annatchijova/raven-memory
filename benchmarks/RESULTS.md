# Resultados de rendimiento — optimizaciones Fase 2

**Fecha:** 2026-08-29 · **Harness:** `benchmarks/perf.py --sizes 1000,5000 --queries 50`
· contenedor Linux x86_64, Python 3.11, embeddings dummy deterministas (seed 42).

Los números son de una sola corrida en un contenedor compartido: úsalos para
comparar **antes vs. después en la misma máquina**, no como cifras absolutas.

## Antes (HEAD previo al sprint 4) vs. después (2.1–2.4)

| Métrica | n | Antes | Después | Mejora |
|---|---:|---:|---:|---:|
| Store (memorias/s) | 1 000 | 404 | 703 | 1.7× |
| Store (memorias/s) | 5 000 | 133 | 602 | **4.5×** |
| Recall p50 (ms) | 1 000 | 33.9 | 14.1 | 2.4× |
| Recall p50 (ms) | 5 000 | 128.1 | 23.8 | **5.4×** |
| Recall p95 (ms) | 5 000 | 142.4 | 28.8 | 4.9× |

La latencia de recall dejó de escalar linealmente con el corpus
(34→128 ms entre 1k y 5k antes; 14→24 ms después).

## Qué cambió (plan 2.1–2.4)

- **2.1** — el hop de descubrimiento se registra durante el BFS de expansión;
  se eliminó el re-BFS por candidato (`_hop_distance`) del camino caliente.
- **2.2** — índice direccional de cell_links en memoria, actualizado en cada
  escritura de link; el recall ya no relee la tabla completa (41 664 filas por
  recall a 5k).
- **2.3** — escrituras por lotes en una transacción: activaciones del top-k,
  pesos STDP del turno y pares de contradicción de un store.
- **2.4** — `export_graph` cuenta y ordena en SQL (`ORDER BY recall_count
  LIMIT`) en vez de materializar toda la tabla.

## Lo que NO mejoró (honesto, con ítem de plan asignado)

- **Primer recall** (~0.7 s a 1k, ~10 s a 5k): dominado por la construcción
  del grafo k-NN sobre el KDTree en 384 dimensiones (una consulta k=7 sobre
  los n puntos). Sin cambios en este sprint — es exactamente el ítem **2.6**
  (backend ANN opcional). Mitigación actual: el rebuild es perezoso y ocurre
  una vez por arranque/mutación, no por consulta.
- ~~**Tamaño del audit log** (~11 KB por recall)~~ — **resuelto después en el
  mismo sprint** (ítem **2.5**, esquema v3): el audit log persiste ahora el
  SHA-256 del embedding en columna propia en vez del vector JSON completo.
  Medido: **11 217 → 2 792 bytes por recall (4×)**. Las filas legacy siguen
  verificando por recomputación; la cadena mixta legacy+v3 tiene test propio.

## Reproducir

```bash
python benchmarks/perf.py --sizes 1000,5000 --queries 50 --json results.json
```
