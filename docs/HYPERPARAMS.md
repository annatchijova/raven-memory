# Sensibilidad de hiperparámetros (plan 4.3)

**Harness:** `benchmarks/sweep.py` (one-at-a-time sobre `benchmarks/quality.py`,
seed 42). El objetivo no es auto-tunear sino saber qué perillas importan y por
qué cada default sobrevive.

```
param              value   recall@5  validated-first
----------------------------------------------------
HOP_LAMBDA          0.0       0.690            20/20
HOP_LAMBDA         0.15*      0.693            20/20
HOP_LAMBDA          0.5       0.687             7/20
HOP_LAMBDA          1.0       0.630             7/20
RECENCY_WEIGHT      0.0       0.693            20/20
RECENCY_WEIGHT     0.05*      0.693            20/20
RECENCY_WEIGHT      0.3       0.517            20/20
RESONANT_BOOST      0.0       0.693            20/20
RESONANT_BOOST      0.5*      0.693            20/20
RESONANT_BOOST      1.5       0.693            20/20
```

## Conclusiones

- **`HOP_LAMBDA = 0.15` está en una meseta segura** (0.0–0.15 rinde igual).
  A partir de ~0.5 el decay aplasta a las memorias a 1 salto y la afirmación
  validada pierde contra su contradicción cuando la query cae en la celda
  contradictoria (20/20 → 7/20). El default queda validado; no subirlo.
- **`RECENCY_WEIGHT = 0.05` es inocuo pero no aporta en este corpus**
  (idéntico a 0.0). A 0.3 el bono de recencia aplasta la similitud y el
  recall@5 cae de 0.69 a 0.52 — es la perilla más peligrosa de subir. Se
  mantiene en 0.05 por diseño para cargas reales con estructura temporal,
  pero con una advertencia documentada: este benchmark no aporta evidencia a
  su favor, solo evidencia de que valores altos dañan.
- **`RESONANT_BOOST` no queda ejercitado por este harness** (los links
  automáticos del corpus son solo INHIBITORY; no hay links RESONANT
  manuales). Sin efecto medible en ninguna dirección — esto NO es evidencia
  de que sobre, es un hueco del harness. Trabajo futuro: escenario con
  links RESONANT explícitos.

## Reproducir

```bash
python benchmarks/sweep.py --seed 42
```
