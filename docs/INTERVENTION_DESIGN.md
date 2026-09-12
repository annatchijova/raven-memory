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

---

## 13. Pasada adversarial sobre v1 (`tests/test_intervention_properties.py`)

Tres propiedades metamórficas más pureza, sobre el commit congelado. Dos
hallazgos de producto salieron de acá, ninguno introducido por la sonda.

### Hallazgo A — `now` está fijo dentro de una sonda, no entre sondas

`intervene()` toma `time.time()` una vez por llamada. Dentro de la sonda las dos
ramas lo comparten, así que el término de recencia se cancela en la resta y el
**Δ es exactamente reproducible**. Pero dos sondas distintas corren a `now`
distintos, así que sus *scores absolutos* difieren (~1e-8 por llamada) y no son
comparables bit a bit.

Consecuencia para el runner futuro: un barrido que quiera comparar sondas
*entre sí* —no sólo el Δ interno de cada una— necesita un `now` inyectable.
Hoy no existe. El Δ de cada sonda no está afectado.

### Hallazgo B — `recency_bonus` no acota edad negativa (pre-existente en `main`)

`recency_bonus = 0.05 · exp(−ln2 · age / 24h)` con `age < 0` **crece sin
límite** en vez de decaer. Con `last_activation` en el futuro:

| desfasaje | score top |
|---|---:|
| +1 día | 1.1 |
| +10 días | 52.2 |
| +30 días | 5.4e7 |
| +90 días | 6.2e25 |
| ~+2.8 años | `OverflowError` |

Verificado idéntico en `e539df8` (= `main`), así que **no es regresión del
refactor**: la pasada adversarial lo destapó.

Alcanzable sin tocar la base a mano: `portability.import_field()` copia
`last_activation` literal, así que un campo importado desde una máquina con el
reloj adelantado —o un salto de NTP hacia atrás— produce ranking basura **en
silencio**, sin `degraded`, sin warning. Es exactamente el modo de falla que el
principio de degradación honesta del repo existe para prohibir.

**Estado: arreglado** (`age = max(0.0, now - mem.last_activation)`), con
`tests/test_recency_bounds.py` escrito **en rojo antes del fix**. Los tests
expresan la propiedad, no los números que el código con bug producía:

```
age_efectiva            = max(0, now − last_activation)
0 ≤ recency_bonus ≤ RECENCY_WEIGHT      para cualquier timestamp
futuro(+1s … +50 años)  ≡ now           en contribución de recencia
ranking                 : un timestamp futuro no compra rango
import_field(futuro)    : fidelidad del dato + scoring acotado
```

El test de ranking deriva su tolerancia del reloj transcurrido en la propia
corrida (`RECENCY_WEIGHT·(1−exp(−ln2·elapsed/24h))`) en vez de usar un épsilon
elegido a mano, así que sólo admite la diferencia que el paso del tiempo
explica. Sin el clamp reporta: *future timestamp gained 5.369e+07 of recency
bonus; elapsed wall-clock (0.021s) explains at most 8.229e-09*.

### Lo que el clamp NO hace — pendiente deliberado

El clamp arregla la **integridad del ranking**. No diagnostica el skew.

Si el motor observa `last_activation > now`, el resultado ya contiene evidencia
de una anomalía temporal, y convertirla en silencio en una activación normal de
`now` es raro en un sistema que predica degradación honesta. Lo que faltaría:

```
effective_age    = 0
recency_bonus    = RECENCY_WEIGHT
temporal_anomaly = FUTURE_LAST_ACTIVATION        ← no implementado
```

No se implementó ahora a propósito: toca diagnostics y probablemente schema, y
exige decidir una tolerancia `ε` tal que `last_activation − now > ε ⇒ skew`.
Los relojes reales tienen jitter y precisiones distintas, así que `ε` tiene que
ser una constante **documentada y testeada**, no un número mágico descubierto
por accidente. Eso es un commit aparte, no una extensión de este.

Responsabilidades separadas, y el test lo fija: `import_field()` preserva el
timestamp tal cual por fidelidad y portabilidad; interpretarlo de forma segura
es trabajo del scorer.

### Hallazgo A — no es un bug, es un requisito del runner

El `now` por llamada (§13.A) **no se arregla todavía**. El primitive es correcto
para el claim que hace hoy: dentro de una sonda el contrafactual está bien
construido porque ambas ramas comparten `now`.

Pero un runner que quiera comparar una *matriz* de intervenciones bajo un mismo
estado experimental necesita algo más que un reloj inyectable — necesita
declarar cuál es la unidad experimental:

```
ExperimentContext
    now
    query embedding
    field snapshot / identity
    …
```

Pinnear sólo `now` daría una falsa sensación de snapshot si queda cualquier otro
estado contextual variable. Se decide al diseñar el runner, no antes.

### Lo que los controles negativos dijeron de los tests mismos

| control | resultado |
|---|---|
| el core muta `cell_neighbors` | rojo en pureza |
| el core contamina `_alerted_memories` | rojo en pureza |
| las ramas corren con parámetros distintos | rojo en sham |
| supresión dependiente del orden | **verde — no detectado** |
| se quita el `sort` de `_resolve_targets` | rojo en ambos tests de orden |

La conmutatividad conductual **no puede fallar**: `_resolve_targets` ordena por
`cell_id` y el core recibe un `FrozenSet`, así que el orden queda normalizado
dos veces antes de alcanzar cualquier lógica. El test se reescribió para afirmar
esos dos puntos de normalización —que sí se rompen detectablemente— en vez de
sostener una tautología en verde.

Tercer test inútil encontrado por control negativo, en el lote escrito
específicamente para ser riguroso.

### Pureza: sin rastro

`structural_digest()` recorre `vars(engine)` genéricamente (un atributo nuevo
queda cubierto solo; una lista escrita a mano deja de testear lo que olvida).
Tras una batería de sondas —vacía, simples, pares, con `current_turn_memories`—
ningún atributo del engine difiere. El core **aliasea** estructuras del engine
en vez de copiarlas (`all_cell_links = self._cell_links_index`), así que "no
ejecutó SQL" no habría alcanzado como evidencia.

---

## 14. Qué observa realmente `structural_digest()` (`tests/test_digest_coverage.py`)

El gate de pureza demuestra hoy algo más estrecho de lo que su nombre sugiere:

> no cambió nada **que `structural_digest()` sepa representar**

y no

> no cambió el estado estructural del engine.

Los dos producen exactamente el mismo verde. `_canon_value()` falla **abierto**:
lo que no reconoce se vuelve `("opaque", type_name)` —una etiqueta de tipo sin
contenido— y todo lo alcanzable únicamente a través de ese nodo desaparece de la
comparación.

### Inventario sobre una instancia real

369 nodos alcanzables (profundidad ≤ 6, ≤ 40 hijos por nodo): 346 representados
por valor, 23 opacos, 27 alcanzables sólo a través de un opaco. El cociente no
es una medida física; lo que importa es *cuáles*.

| opaco | origen |
|---|---|
| `_db`, `_lock`, `kdtree`, `stylometric`, `_spectral` | exclusión declarada |
| `LinkType`, `EnumType`, `function`, `type` | código y singletons internados, inertes |
| **`_author_profiles[k]` (`AuthorStyleProfile`, `deque`)** | **ceguera accidental** |

`kdtree` es una exclusión declarada **con consecuencia**: una sonda que
corrompiera el índice en el lugar sería invisible. `_db._prev_audit_hash` —la
cabeza de la cadena de auditoría— también queda escondido bajo una exclusión
declarada.

Las exclusiones ahora llevan justificación escrita y un test falla si
`OPAQUE_ATTRS` y las justificaciones se separan, o si aparece un opaco nuevo que
no sea código ni enum. Ignorancia accidental convertida en decisión auditable.

### Cuatro cegueras, confirmadas empíricamente

| mutación | gate |
|---|---|
| objeto opaco inyectado, mutado por el core durante la sonda | **verde** |
| `_author_profiles[k]._samples[0].avg_sentence_length: 8.0 → 999.0` | **verde** |
| aliasing roto: dos atributos al mismo objeto → copia de igual valor | **verde** |
| mutación transitoria `S0 → S1 → S0` | **verde** |
| control positivo: mutación persistente representable | rojo (detecta) |

La segunda no es artificial: es un fingerprint vivo de la ventana rodante que
alimenta la comparación forense estilométrica.

Quedan congeladas como tests de caracterización marcados `KNOWN_BLINDNESS`,
verdes hoy. **No son propiedades.** Cuando aterrice el witness hay que
**invertirlas**, no borrarlas.

### Tres claims distintos, ninguno llamado `pure=True`

```
before == after                    → sin mutación de valor persistente observada
alias_graph_before == after        → sin mutación de topología de identidad observada
write_log == []                    → no hubo mutación por las rutas instrumentadas
```

El digest actual intenta el primero. El tercero no es alcanzable por
introspección before/after —ninguna comparación de estado final puede ver
`S0 → S1 → S0`— y necesita spies sobre las operaciones mutantes
(`__setitem__`, `add`, `append`). El test de mutación transitoria queda verde
incluso después del witness, y el witness **no debe** reclamar cubrirlo.

### El arreglo obvio no funciona

Descender ingenuamente a `__dict__` desde `_canon_value()` produce
`RecursionError`: el grafo tiene ciclos (`LinkType.__objclass__` → la clase enum
→ su `__dict__` → los miembros → vuelta).

La detección de ciclos no es una optimización, es requisito de corrección — y es
la misma tabla de nodos indexada por `id()` que hace falta para responder la
pregunta de topología de identidad. Las dos necesidades convergen en la misma
estructura, que es el argumento a favor del `StateWitness` explícito y en contra
de un `structural_digest()` cada vez más mágico.

**No implementado a propósito.** Establecer qué puede afirmar el instrumento va
antes de cambiar lo que hace.

---

## 15. `StateWitness` (`tests/state_witness.py`)

**Alcance del claim, con precisión:** esto es un *structural witness sobre el
subconjunto soportado de estado propiedad del engine, excluyendo subsistemas
semánticos*. **No** es un "engine state witness". Mientras `_db` y `kdtree`
queden afuera el gate conserva blind spots conocidos y relevantes —
`_prev_audit_hash` entre ellos, que no es un detalle: una sonda que escribiera
estado de auditoría y restaurara todo lo demás podría parecer persistentemente
limpia.

Instrumento sucesor de `structural_digest()`, que **no se arregla** — §14 queda
como su autopsia. No reemplaza todavía ningún gate.

### Frontera semántica, no mecánica

Un walker irrestricto de "todo objeto Python alcanzable" no captura estado del
engine, captura reachability del intérprete: instancia → Enum → clase →
funciones → globals → módulos. La versión ingenua reventaba con `RecursionError`
en `LinkType.__objclass__`.

```
DESCEND               dict / list / tuple / set / frozenset / deque,
                      dataclasses, instancias con __dict__
TERMINAL_BY_VALUE     None/bool/int/float/str/bytes, miembros de Enum, ndarray
TERMINAL_BY_IDENTITY  funciones, clases, módulos
EXCLUDED              sólo por path + justificación escrita
UNKNOWN + mutable     → UnrepresentableState(path, type)   ← falla, no resume
```

Los miembros de `Enum` como terminal por valor son lo que corta la recursión de
`__objclass__` sin necesidad de auditar el intérprete.

**Criterio de aceptación, fijado antes de escribir código:** si el witness
encuentra un objeto mutable propiedad del engine y no sabe representarlo, la
prueba falla nombrando path y tipo. En la primera corrida contra un engine real
falló en `engine._db.db_path (PosixPath)` — exactamente el comportamiento
buscado.

### Valor e identidad son dos claims, no uno

```
value_signature    path → (tipo, valor canónico)      qué ES el estado
alias_signature    (path_canónico(padre), label,      qué referencias apuntan
                    path_canónico(hijo))              al MISMO objeto
```

`value_signature` se indexa por **path**, no por nodo: dos paths que comparten
un objeto y dos paths con copias de igual valor producen la misma firma, que es
justamente lo que hace que el claim de valor sea independiente del cableado.
`alias_signature` nombra los nodos por su path canónico (más corto, desempate
lexicográfico) porque `id()` no es comparable entre dos witness.

Sobre el caso que el digest viejo declaraba idéntico:

```
value_state_equal    = True     ← las copias tienen igual valor
alias_topology_equal = False    ← dejaron de compartir objeto
```

Canonicalización de valor y construcción del grafo **no comparten reglas**: un
`set` es independiente del orden como valor, un `list`/`deque` no, y la
asociación `key → hijo` de un dict sobrevive en las etiquetas de arista, no en
el valor.

### Ciclos: terminan sin perder la arista de vuelta

Detectar un ciclo no puede significar descartarlo — la back-edge **es** estado.
El witness registra la arista y no re-camina el nodo. Testeado con un ciclo
`a → b → a` y con un dict autorreferente.

### Las tres inversiones, y la que no

| | `structural_digest()` | `StateWitness` |
|---|---|---|
| objeto opaco mutado por el core | ciego | **detecta** (valor) |
| `_author_profiles[k]._samples[0]` | ciego | **detecta** (valor) |
| aliasing roto, valor igual | ciego | **detecta** (topología) |
| `S0 → S1 → S0` | ciego | **sigue ciego, correctamente** |

La transitoria no es alcanzable por ninguna comparación before/after. Necesita
un `MutationJournal` que instrumente superficies de escritura, y su claim
tendría que estar acotado a *"ninguna mutación por las superficies
instrumentadas"* — nunca *"ninguna mutación"*. El witness no reclama cubrirla.

### Pendiente: witnesses semánticos

`_db` y `kdtree` quedan excluidos **con justificación**, no por dificultad:
necesitan witnesses semánticos, no recursión estructural dentro de las
internals de sqlite o scipy.

```
DBWitness        audit_head (_prev_audit_hash), fingerprints de tablas relevantes
KDTreeWitness    indexed_cell_ids, fingerprint de vectores fuente, dirty state
```

`_prev_audit_hash` importa particularmente: si una sonda pudiera mover la cabeza
de la cadena de auditoría y el gate no lo viera, sería exactamente el falso
"read-only" que todo esto existe para eliminar.

Se implementan **después** de que el structural witness esté estable, y recién
entonces se decide qué gate reemplaza a cuál.

---

## 16. Ronda adversarial sobre `StateWitness` (`tests/test_state_witness_attacks.py`)

Veredicto: arquitectura correcta, **no promovible a oracle de pureza del
engine**. Tres defectos confirmados, congelados como `KNOWN_DEFECT` —verdes hoy,
afirmando que la falla está presente— para invertir cuando se arreglen. Nada
arreglado en esta ronda.

### Defecto 1 — el grafo de identidad está contaminado por átomos internados

El ataque de canonical-path renaming **cayó**, y por un mecanismo más profundo
que el naming. Agregar **un** alias nuevo a un objeto compartido renombró **19
aristas no relacionadas**.

Causa raíz, confirmada: CPython internea los enteros chicos, así que el `1`
dentro de un dict recién adosado **es** el `1` de `_active_cells` —el mismo
objeto—. El grafo registra "estos dos paths comparten un objeto" para átomos de
valor, que no es topología del engine. El path canónico (más corto, desempate
lexicográfico) convierte eso en un renombre global: el alias nuevo gana el path
canónico de los átomos internados y toda arista que los nombre se reetiqueta.

Control que aísla la causa: el mismo ataque con un payload sin enteros chicos
produce muchísimo menos churn.

**El arreglo no es un esquema de nombres mejor**: los terminales de valor no
deben participar del grafo de identidad. Simulado —restringir el grafo a nodos
mutables `DESCEND`— las dos caracterizaciones del defecto se invierten y las
tres propiedades reales (discriminación de ciclos, alias merge, multiplicidad)
**siguen verdes**. Dirección validada, no aplicada: la decisión de
representación es previa a construir cualquier cosa encima.

Hasta entonces, `persistent_alias_topology_equal` compara una **renderización
rooted y path-labelled** de la topología, no la topología.

### Defecto 2 — observar ejecuta código del objeto observado

`repr()` corre sobre miembros de `set` y claves de `dict`; un sort por
comparación puede llegar a `__eq__`. Un objeto con `__repr__` con efectos
laterales los dispara: 5 llamadas en el experimento. Un instrumento de pureza
que ejecuta comportamiento arbitrario mientras observa **puede ser él mismo
fuente de mutación**.

### Defecto 3 — claves por `repr()` filtran direcciones de heap

`engine.dk → ('dict', ['<KeyObj object at 0x7f1f0d6790d0>'])`. Estable dentro de
un proceso para el mismo objeto, pero no reproducible entre procesos, y dos
claves iguales-por-valor se leen como estados distintos. Además el objeto-clave
**nunca es un nodo**: hay un segundo grafo de identidad escondido dentro de las
etiquetas de arista.

### Lo que sobrevivió al ataque

| propiedad | estado |
|---|---|
| ciclos topológicamente distintos con payload idéntico | **discrimina** (valor igual, topología distinta) |
| alias merge (dos iguales independientes → compartido) | detecta |
| multiplicidad 3 → 2+1, con clases de equivalencia preservadas | detecta |
| estabilidad bajo allocation noise | estable |
| mutable no soportado enterrado en profundidad | falla cerrado |
| mutable no soportado alcanzable por dos aliases | falla cerrado |

La discriminación de ciclos recién ahora está **demostrada**. Antes el claim era
"termina y conserva la back-edge", que no es discriminación.

### Decisiones tomadas en esta ronda

- **Nombres de claim**: `persistent_value_state_equal`,
  `persistent_alias_topology_equal`, `persistent_root_identity_equal`. Ninguna
  API se llama `is_pure()`, ni antes ni después del `MutationJournal`.
- **`PurePath` y otros value objects con `__slots__`** → terminales por valor,
  con razón escrita. "Tipo Python desconocido" no es lo mismo que "estado
  mutable desconocido"; sólo el segundo debe fallar cerrado. El fallo original
  en `engine._db.db_path (PosixPath)` fue un control accidental útil, y se
  resuelve por decisión declarada, no aflojando la heurística.

### Orden pendiente

`atacar StateWitness` (hecho) → arreglar defectos 1–3 → congelar →
`DBWitness` → `KDTreeWitness` → witness compuesto → **recién entonces** discutir
reemplazo de gate → `MutationJournal` aparte.

---

## 17. `StateWitness` reconstruido — frontera de identidad antes que representación

Los tres defectos de §16 quedaron invertidos. El orden fue el que corresponde:
frontera semántica de identidad → grafo de identidad → representación canónica.
Estabilizar los paths canónicos primero habría sido arreglar la representación de
un grafo con aristas espurias.

### `tracks_identity` es un concepto aparte de `descends`

```
VALUE_TERMINAL              descends=False  tracks_identity=False
IDENTITY_TERMINAL           descends=False  tracks_identity=False
OWNED_MUTABLE_CONTAINER     descends=True   tracks_identity=True
OWNED_IMMUTABLE_CONTAINER   descends=True   tracks_identity=False   ← los separa
OWNED_OBJECT                descends=True   tracks_identity=True
UNSUPPORTED_MUTABLE         → UnrepresentableState
```

El contenedor inmutable es el caso que impide que sean el mismo predicado. Sin
él la coincidencia se habría vuelto arquitectura por accidente.

**Invariante que prohíbe la tercera categoría accidental:** todo mutable
soportado propiedad del engine o participa del identity tracking, o está
excluido como subsistema semántico con witness propio. *"Mutable, representado
sólo por valor, identidad ignorada"* —la forma exacta que tenía la ceguera— no
puede existir, y hay un test que recorre todos los nodos para comprobarlo.

### Observar no ejecuta nada que el objeto observado defina

Ni `__repr__`, `__str__`, `__eq__`, `__lt__`, `__hash__`, ni properties ni
descriptores: `vars()` en vez de `getattr`. El dispatch es por **tipo exacto**,
porque una subclase de `str` o de `dict` puede sobrescribir justamente los
protocolos que se asumirían seguros.

Los miembros de un `set` se convierten primero a tokens inertes y se ordenan los
**tokens**, nunca los objetos.

### Las claves de dict son tokens, no strings

```
("str", …) ("int", …) ("bytes", …) ("tuple", (…)) ("enum", Tipo, miembro) ("path", …)
```

Una clave fuera del dominio de valor soportado **falla cerrado**. Eso mata a la
vez el leak de `0x7f…` y el segundo grafo de identidad que vivía escondido
dentro de las etiquetas de arista.

### `SAFE_VALUE_TYPES` es un registro, no una heurística

Cada entrada es una decisión con razón escrita. **`__slots__` no es criterio de
pertenencia**: no implica inmutabilidad ni semántica de valor, y un objeto con
`__slots__` no registrado falla cerrado.

### Hallazgo del propio arreglo: las raíces no eran aristas

Una metamórfica nueva (`engine.b = engine.a` sobre un mutable) reveló que
`alias_signature()` no veía el aliasing **a nivel raíz**: las raíces estaban
fuera del grafo. Los tests de merge y multiplicidad pasaban indirectamente, por
los hijos — frágil. El engine es ahora un nodo y cada atributo raíz es una
arista real.

### Residual declarado, no tapado

El path canónico renombra el subárbol del objeto aliaseado, porque genuinamente
adquirió un camino más corto. El delta reportado **no es mínimo**: toda arista
que cambia nombra al objeto aliaseado. El churn espurio desapareció —el estado
preexistente del engine queda intacto, `pristine ⊆ after`— pero en ese sentido el
claim sigue comparando una **renderización rooted** de la topología.

### `StateSnapshot` vs `StateComparison`

`root_identity` es una relación entre dos capturas del mismo proceso, no estado
canonicalizable, así que vive en el comparador y no en el snapshot. Hay un test
estructural que lo verifica.

### Controles negativos

| control | resultado |
|---|---|
| terminales de valor vuelven al identity graph | 8 tests rojos |
| claves vía `repr()` otra vez | rojo |
| ordenar objetos del set en vez de tokens | rojo |
| `tracks_identity == descends` | rojo |
| raíces fuera del grafo | rojo |
| `__slots__` tratado como value-semantic | **verde — test impreciso** |

El último obligó a corregir un test: el objeto con `__slots__` igual fallaba,
pero por un **segundo** guard (canonicalizador ausente), no porque la
clasificación lo rechazara. Defensa en profundidad es bienvenida; un test que no
distingue qué capa aguantó, no. Ahora afirma `classify(...) ==
UNSUPPORTED_MUTABLE` directamente.

### Sigue fuera de alcance

`S0 → S1 → S0`, sin cambios y sin reclamo. Y `_db` / `kdtree` siguen excluidos,
así que esto continúa siendo un *structural witness sobre el subconjunto
soportado*, no un witness del estado del engine. Ningún gate reemplazado.
