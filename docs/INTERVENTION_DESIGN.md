# InterventionSpec — sondas causales sobre el recall (diseño, pre-implementación)

**Estado:** v1 implementado (`AdaptiveMemoryEngine.intervene`,
`raven/intervention.py`, `tests/test_intervention.py`). Un solo modo
(`suppress`) y una sola etapa (`field`); el resto está nombrado acá para fijar
la forma del tipo y es **rechazado en runtime**, no ignorado.

**Origen:** inspirado en la inhibición óptica reversible de la optogenética
(loss-of-function transitorio en vez de lesión permanente). La metáfora se puede
descartar entera sin que el feature pierda sentido: lo que queda es *atribución
causal sobre un retrieval determinista y auditable*.

---

## 1. El claim que tenemos derecho a hacer

Hay dos grafos causales distintos y sólo uno es nuestro:

```
(A)  intervención → retrieval RAVEN → contexto recuperado        ← determinista, sellable
(B)  intervención → retrieval RAVEN → contexto LLM → respuesta   ← incluye al modelo
```

Este módulo mide **(A)**, y el resultado se llama **retrieval causal influence**.
No autoriza la frase "esta memoria causó la respuesta del agente": entre el
contexto recuperado y la respuesta hay un modelo con varianza propia, muestreo, y
un prompt que no controlamos acá.

Definición operacional, falsable:

```
I(c, q) = D( R(q) , R(q | do(c = 0)) )
```

donde `R` es el resultado determinista de `recall()` y `D` se reporta
**descompuesto**, nunca colapsado a un escalar (ver §7).

Toda superficie pública (API, MCP, docs) usa el término `retrieval_causal_influence`.
Cualquier redacción que sugiera causalidad sobre la salida del agente es un bug de
documentación, no una imprecisión menor.

---

## 2. Por qué un tipo y no parámetros sueltos

La alternativa rechazada es extender la firma de `recall()`:

```python
recall(..., suppress_cells=[], excite_cells=[], excitation_tau=..., seed_cells=[], ...)
```

Las extensiones naturales ya se conocen (excitación, boost transitorio, engram
seed), así que el crecimiento de la firma es predecible y ninguna combinación de
flags sueltos es serializable, sellable ni validable como unidad. La intervención
entra como **un objeto**, se valida como un objeto, y se sella como un objeto.

```python
@dataclass(frozen=True)
class InterventionSpec:
    mode:    str            # v1: sólo "suppress"
    targets: List[str]      # memory_ids (identidad pública y estable)
    stage:   str = "field"  # "field" | "readout"   (ver §3)
```

### Regla de cierre (no negociable)

El parser **rechaza ruidosamente** todo `mode` o `stage` desconocido, y todo
`target` que no resuelva a una celda activa. Nunca ignora silenciosamente un
campo que no entiende.

Razón: una intervención silenciosamente ignorada produce `R(q) == R(q|do(c))`,
es decir **Δ = 0**, que es exactamente el resultado que se lee como "esta memoria
no tiene influencia causal". Un no-op silencioso y un hallazgo negativo real son
indistinguibles en la salida. Un cliente nuevo contra un servidor viejo
produciría evidencia falsa, no un error. Esto es el caso honest-degradation del
repo aplicado a la sonda: si no se pudo intervenir, se dice; no se devuelve un
delta plausible.

---

## 3. `stage` — dónde se aplica la supresión (esto define qué significa el Δ)

Una celda participa del recall en tres puntos distintos
(`memory_engine.py:1409`, `1428-1453`, `1475-1482`). Suprimir en cada uno da un
Δ distinto, y sin nombrarlo el número es ininterpretable:

| punto | efecto de suprimir ahí |
|---|---|
| seed (KDTree) | la celda no puede ser el punto de entrada al campo |
| propagación (BFS) | la celda no retransmite activación a sus vecinos ni dispara sus links |
| readout (scoring) | la celda no entra al ranking, pero sí propagó |

- `stage: "field"` **(default v1)** = los tres a la vez. Es el análogo honesto de
  la inhibición óptica: la neurona no dispara, así que tampoco transmite. Es el
  modo que hace interesante la ablación combinatoria, porque mata rutas además de
  nodos.
- `stage: "readout"` = sólo el tercero. Es una máscara de lectura: responde
  "¿cuánto aportaba esta memoria por sí misma?" sin alterar la topología de
  activación.

Las dos preguntas son legítimas y **distintas**: `field` mide influencia
estructural, `readout` mide contribución directa. Mezclarlas es la forma más
fácil de publicar un Δ que no significa nada.

v1 implementa `field`. `readout` queda nombrado y rechazado por la regla de cierre
hasta que se implemente.

---

## 4. Separación de autoridad: probe ≠ stimulation ≠ learning

Una intervención que pueda alimentar `_update_stdp()` deja de ser una sonda y pasa
a ser una **primitiva de escritura indirecta**: permite fabricar asociaciones que
nunca vinieron del uso. Ese es el análogo computacional de la implantación de
engramas falsos, y es el riesgo de seguridad principal de todo este módulo.

`INTERVENTION` en v1, como invariante de autoridad:

```
INTERVENTION
    ├─ read-only respecto de SQLite
    ├─ sin actualización de STDP
    ├─ sin reinforcement
    ├─ sin transición de estado
    ├─ sin links persistentes
    ├─ sin update de last_activation / total_recalls
    └─ sellada en la cadena de auditoría
```

El renglón de `last_activation` no es cosmético: `recall()` normal escribe
timestamps de activación (`memory_engine.py:1643`) que alimentan el
`recency_bonus`. Una sonda que los tocara cambiaría el scoring de los recalls
siguientes — la medición modificaría el sistema medido.

### Cómo se hace estructural y no una promesa

`recall()` hoy tiene cinco efectos de escritura entrelazados con el scoring:
`_update_stdp`, `update_activations`, `store_audit`, `store_alert`, y la
degradación a `FORGOTTEN` bajo `RAVEN_STYLO_ENFORCE=1` (`memory_engine.py:1508-1514`),
que además muta `_active_cells` y `_kdtree_dirty`.

Enhebrar un flag `read_only=True` por todo eso es frágil: alcanza con olvidarse de
un branch. El diseño es extraer un núcleo puro:

```
_recall_core(query, ..., intervention=None) -> (results, diagnostics)   # cero escrituras
recall(...)      = _recall_core(...) + efectos (STDP, activaciones, audit)
intervene(...)   = _recall_core(baseline) + _recall_core(perturbado) + UN audit
```

Así el read-only no se sostiene por disciplina sino porque la ruta de intervención
no tiene acceso al código que escribe. El chequeo estilométrico enforce queda
**duro-desactivado** dentro del core cuando hay intervención (una sonda no puede
cuarentenar memorias).

Una futura operación `stimulate`, explícitamente distinta y con su propio nombre
en la auditoría, podría permitir plasticidad. No es v1 y no comparte tipo con
`InterventionSpec`.

---

## 5. Por qué es UNA llamada al engine y no dos recalls del cliente

El Δ sólo es atribuible a la intervención si las dos ramas corren bajo condiciones
idénticas. Dos llamadas separadas a `recall()` **no** lo garantizan:

1. `recall()` toma `now = time.time()` por llamada (`memory_engine.py:1473`) y el
   `recency_bonus` depende de `now`. Dos llamadas → dos `now` → scores distintos
   por una razón que no es la intervención.
2. El primer `recall()` escribe `last_activation` de sus top-k, con lo cual la
   segunda llamada arranca sobre un campo ya modificado por la primera.
3. El KDTree se reconstruye perezosamente (`_ensure_kdtree`); un `store()`
   concurrente entre ambas ramas cambia el campo bajo los pies.
4. `_update_stdp` de la primera llamada altera los pesos sinápticos que la segunda
   usa en el término `synaptic_boost`.

Conclusión de diseño: **la intervención es una sola operación del engine**, bajo
una sola adquisición del lock `@_synchronized`, con **un único `now` fijado y
compartido por ambas ramas**. Es la única forma de que "determinismo del delta"
sea un test que puede fallar por la razón correcta.

Precondición adicional para el Δ sobre *rangos*: `results.sort()` ordena sólo por
`final_score` (`memory_engine.py:1636`), sin desempate explícito. En la ruta normal
el orden de inserción viene de `load_memories(... ORDER BY cell_id ASC)`, así que
los empates se resuelven de forma estable y consistente entre ramas. Pero la ruta
de fragmentación para >999 celdas activas (`memory_engine.py:681-701`) no aplica
`ORDER BY`, y ahí el orden de inserción depende de los límites de chunk, que se
corren al cambiar el conjunto de celdas. Con campos grandes, un desplazamiento de
rango podría ser un artefacto de ordenamiento y no un efecto causal. **v1 agrega
un desempate explícito por `memory_id`** antes de reportar cualquier métrica de
rango.

---

## 6. Binding de auditoría (implica schema v4)

Una sonda no sellada no sirve como evidencia. Pero `compute_audit_hash()`
(`memory_engine.py:232`) hashea un payload fijo `{ts, op, query, cells, results,
qemb_sha256}`, y `verify_audit_chain()` lo **recomputa desde columnas
persistidas**. De ahí salen tres restricciones duras:

1. La `InterventionSpec` entra al payload hasheado **sólo si se persiste como
   columna propia**. Si no, la cadena deja de ser recomputable y se rompe la
   propiedad central del repo. → migración a `SCHEMA_VERSION = 4`, columna
   `intervention TEXT`.
2. Lo que se sella es el **conjunto de targets ya resuelto** (memory_id → cell_id),
   no un selector. Un selector tipo "todas las del topic X" no es reproducible: su
   resultado cambia con el contenido del campo. El audit guarda lo que
   efectivamente se suprimió.
3. **Compatibilidad hacia atrás:** las filas viejas no tienen la columna. El campo
   nuevo debe quedar *ausente* del dict JSON cuando es `None` — no `"intervention":
   null` — para que el payload canónico de una fila legacy sea byte-idéntico al
   esquema actual y las cadenas existentes sigan verificando. Es el mismo patrón
   que ya usa `qemb_sha256` para las filas pre-v3.

`operation` pasa a ser un parámetro real en `_build_audit()` (hoy está hardcodeado
`"recall"` en dos lugares, líneas 1888 y 1893) y las sondas escriben
`"recall_intervention"`. Un auditor tiene que poder separar recalls reales de
sondas sin leer el payload.

Una intervención escribe **una** entrada, no cero y no dos: ambas ramas y el Δ
viven en el mismo payload sellado.

---

## 7. La métrica Δ

Se reporta descompuesta:

- **disappeared** — memorias en `R(q)` ausentes de `R(q|do(c))`
- **appeared** — memorias que entran al top-k al liberarse lugar
- **rank_displacement** — Σ|Δrank| sobre la intersección (requiere el desempate de §5)
- **score_delta** — Δ`final_score` por memoria sobreviviente

**Trampa metodológica a evitar:** comparar dos top-k truncados hace que el Δ sea
hipersensible en el borde. Una memoria que estaba en el puesto k+1 y "aparece" no
es el mismo hallazgo que una del puesto 1 que desaparece. El Δ se computa sobre el
**conjunto completo de candidatos scoreados**, y recién después se reporta
proyectado a k.

---

## 8. Ablación combinatoria (capa por encima, no en el engine)

El valor no está en suprimir celdas de a una:

```
suppress(A)    → sin cambios
suppress(B)    → sin cambios
suppress(A,B)  → desaparece X      ⇒ redundancia causal (rutas paralelas)

suppress(A)    → desaparece X
suppress(B)    → desaparece X      ⇒ dos dependencias necesarias en una ruta
```

Esto vive en `raven/intervention.py` como runner sobre la primitiva, no dentro del
engine. El power set explota, así que v1 toma conjuntos explícitos provistos por
quien llama, más un helper para todos los pares de las top-N celdas activadas.

---

## 9. Fuzzing de invariantes — y por qué el oráculo hoy no alcanza

El objetivo interesante es la regla de rescate:

> *una verdad validada no puede ser silenciada por un claim no verificado*
> (`memory_engine.py:1455-1463`)

Fault injection dirigido: buscar un conjunto de intervención bajo el cual una
memoria `REINFORCED` desaparezca del resultado.

**Pero el oráculo ingenuo produce falsos positivos.** Una supresión *puede*
legítimamente eliminar una memoria `REINFORCED`: si se suprime su propia celda, o
si se suprimen todas las celdas de la única ruta que la conecta con el seed. Eso
no viola el invariante — el invariante habla de silenciamiento **por inhibición**,
no de inalcanzabilidad.

La forma falsable es:

> Ninguna supresión de celdas **no-`REINFORCED`**, que no incluya la celda de la
> memoria objetivo, puede causar que esa memoria quede excluida **por la vía
> `f_inhib`**.

Lo que obliga a un requisito concreto: hay que poder distinguir *"salió porque era
inalcanzable"* de *"salió porque quedó inhibida"*. Hoy `recall()` lleva esos
motivos sólo como **contadores agregados** (`f_state`, `f_estilo`, `f_inhib`,
líneas 1471-1482). Para el fuzzer hacen falta **por memoria**.

Así que `_recall_core()` devuelve `diagnostics` con un motivo de exclusión por
candidato. Sin eso el fuzzer sólo puede verificar "corrió", que es exactamente la
clase de test que no puede fallar.

---

## 10. No-objetivos de v1

- `excite` / boost transitorio con τ propia — el tipo lo contempla, el código lo rechaza.
- `seed_set` / reactivación de engramas — requiere separar "seed por similitud
  semántica" de "seed por historia de activación explícita"; es un cambio en el
  punto de entrada al campo (línea 1409), no una intervención sobre la propagación.
  Merece su propio diseño.
- `stimulate` con plasticidad — requiere su propio modelo de autoridad.
- Cualquier claim sobre la salida del agente (§1).

---

## 11. Plan de tests

1. **No-mutación** — snapshot completo de la DB (hash del archivo + estados +
   pesos sinápticos + `last_activation` + conteo de audit) antes y después de una
   intervención. Sólo puede diferir la única fila nueva de `audit_log`.
2. **Equivalencia con baseline** — `intervene(spec=None)` debe producir resultados
   idénticos a `recall()` sobre el mismo campo y el mismo `now`. Control negativo:
   con `targets=[]` el Δ tiene que ser exactamente cero.
3. **Determinismo del Δ** — la misma intervención sobre el mismo campo produce el
   mismo Δ bit a bit, entre corridas y entre procesos.
4. **Binding de auditoría** — la entrada sellada recomputa desde columnas; alterar
   el spec guardado rompe la verificación; las cadenas legacy siguen verificando
   después de la migración v4.
5. **Control negativo de la regla de cierre** — un `mode` desconocido tiene que
   *fallar*, no devolver Δ=0. Este test protege contra el modo de falla de §2 y es
   el que más fácil se olvida.
6. **Oráculo del invariante** — un caso construido donde el rescate debe sostenerse,
   y un caso donde la memoria `REINFORCED` sale legítimamente por inalcanzabilidad,
   comprobando que el fuzzer los distingue.

---

## 12. v1 — lo que efectivamente quedó

| Gate | Dónde |
|---|---|
| 1. `_recall_core()` sin efectos | `memory_engine.py` — devuelve `pending_alerts` / `enforce_forget` en vez de escribirlos |
| 2. Un snapshot y un `now` para ambas ramas | `intervene()` los fija antes de las dos llamadas al core |
| 3. `stage="field"` únicamente | `INTERVENTION_STAGES`; `readout` reservado y rechazado |
| 4. Modo/etapa desconocidos → error | `InterventionError`, incluidas claves extra en `from_dict` |
| 5. Targets resueltos en el payload v4 | `_resolve_targets()` → `{memory_id, cell_id}` sellado |
| 6. `None` preserva hashes históricos | clave omitida, no `null` — test con oráculo independiente |
| 7. Orden determinista | `sort(key=(-final_score, memory_id))` |
| 8. Diagnostics por memoria | `ExclusionReason`, `absence_reason()` |
| 9. La sonda no toca STDP/activaciones/estados/links | comparación de estado persistente antes/después |
| 10. Comportamiento futuro indistinguible | dos campos gemelos, uno sondeado, recalls posteriores idénticos |

Los cinco controles negativos (romper la supresión del bypass sináptico, hacer
que la sonda escriba activaciones, serializar `null`, quitar el desempate,
dejar que una celda silenciada siembre la búsqueda) fallan el test que les
corresponde y sólo ese.

### Hallazgo empírico: la influencia se manifiesta como descenso, no como borrado

La primera corrida real de la sonda contradice una suposición del diseño. Sobre
un campo sintético de 42 celdas en 3 clusters, 12 combinaciones
(memoria observada × conjunto suprimido):

| criterio | resultado |
|---|---|
| sólo desaparición | 12 / 12 `NO_DEPENDENCE` |
| desplazamiento de rango ≥ 1 | 6 `MULTIPLE_NECESSARY`, 3 `SINGLE_NECESSARY`, 3 `NO_DEPENDENCE` |

La causa está en `_rebuild_kdtree()`: el grafo k-NN se **simetriza**
(`cell_neighbors[n].add(cell)`), así que toda celda activa conserva K aristas y
casi nunca queda topológicamente aislada. Suprimir celdas rara vez borra una
memoria del campo; lo que hace es correrla de lugar.

Consecuencia práctica: `dropped()` es un criterio demasiado grueso para este
motor. `Ablation.moved(min_rank_shift=n)` es el que lleva la señal, y
`classify_dependence()` lo toma como parámetro.

`REDUNDANT_PATHS` —la firma que justifica la ablación combinatoria— está
implementado y su lógica testeada, pero **no se observó todavía en un campo
real**. Queda como hipótesis abierta, no como capacidad demostrada.

### Fuera del alcance de v1

La primitiva es de motor: no hay endpoint REST ni herramienta MCP. Exponerla
es la extensión obvia, y cuando se haga, la superficie pública tiene que usar
`retrieval_causal_influence` y nunca sugerir causalidad sobre la respuesta del
agente (§1).
