# Benchmark de calidad — campo raven vs. top-k plano

**Fecha:** 2026-08-29 · **Harness:** `benchmarks/quality.py` · 20 clusters × 10
memorias + 20 pares de contradicción (una afirmación validada por el usuario,
una sin verificar), embeddings sintéticos con ground truth conocido.

## Resultados (dos semillas independientes)

| Métrica | baseline top-k | campo raven |
|---|---:|---:|
| recall@5 (queries de cluster) | 0.587 / 0.570 | **0.693 / 0.693** |
| La afirmación validada aparece primero | 7/20 · 12/20 | **20/20 · 20/20** |
| Contradicción suprimida del top-k | 0/20 · 0/20 | 7/20 · 12/20 |
| Verdad validada presente (regla de rescate) | n/a | **20/20 · 20/20** |

(formato: semilla 42 / semilla 7)

## Lectura honesta

- **recall@5**: las dinámicas del campo no degradan la recuperación básica —
  la mejoran (~+0.11 absoluto): la expansión por saltos recupera miembros del
  cluster que el corte top-k plano deja fuera.
- **validated-first (la métrica estrella)**: con dos afirmaciones casi
  empatadas en similitud, el baseline decide por ruido de embedding (7-12 de
  20). El campo colapsa alrededor de la verdad validada en **20/20** — esto
  es exactamente el comportamiento "collapse around truth" del README, ahora
  medido.
- **Supresión**: parcial por diseño (7-12 de 20 vs 0 del baseline). El enlace
  INHIBITORY se dispara cuando el BFS atraviesa la celda validada; si la
  query cae primero en la celda contradictoria, la regla de rescate protege a
  la validada pero la contradictoria sigue visible (con ranking inferior —
  cubierto por validated-first). No es un fallo: suprimir siempre exigiría
  inhibición retroactiva, y preferimos el invariante simple y auditable.
- **Rescate**: 20/20 — ninguna verdad validada fue silenciada por su
  contradicción no verificada, en ninguna semilla.

## Límites de la metodología

Este benchmark mide las **mecánicas del campo** (boosts de estado, supresión,
rescate, expansión) contra un baseline de coseno puro **sobre los mismos
vectores**, con relevancia definida por construcción. No mide calidad
semántica de embeddings — esa es una propiedad del modelo de embeddings, no
del campo. Evaluación sobre datasets públicos de memoria conversacional
(LongMemEval, LoCoMo) requiere embeddings reales y queda como trabajo futuro
(plan 4.1, segunda parte).

## Reproducir

```bash
python benchmarks/quality.py --seed 42
python benchmarks/quality.py --seed 7
```
