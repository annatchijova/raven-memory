# Plan de mejora — raven-memory

**Fecha:** 2026-08-29 · **Base:** v1.1 (`db379f2`) · **Estado:** propuesta

Este documento es el resultado de una revisión completa del código (motor, API,
cliente Qwen, consolidador, spectral, MCP, tests e infraestructura). Está
organizado en fases por prioridad: cada ítem incluye el problema concreto, la
referencia al código y un criterio de aceptación verificable.

---

## Resumen ejecutivo

raven-memory tiene un núcleo sólido y bien documentado: la cadena de auditoría
recomputable, la consolidación atómica, la degradación honesta del proveedor de
embeddings y la regla de rescate epistémico son puntos fuertes reales. Las
mejoras de mayor impacto ahora son:

1. **Un bug P0 real**: el módulo spectral nunca se carga en ningún punto de
   entrada real — la resonancia/coherencia que documenta el README está
   silenciosamente muerta en producción.
2. **Corrección de concurrencia y estado compartido en la API**: el event loop
   se bloquea en cada recall, la caché de `/recall` sirve resultados obsoletos
   tras cada escritura, y todos los clientes HTTP comparten una única
   conversación e historial STDP.
3. **Infraestructura de ingeniería**: no hay CI, ni packaging, ni pinning de
   dependencias — todo lo demás depende de esto para no regresionar.
4. **Validación cuantitativa**: el proyecto afirma que el campo supera al
   top-k plano, pero no hay ningún benchmark que lo mida. Es la mejora que más
   credibilidad aporta.

---

## Fase 0 — Bugs y correcciones críticas

### 0.1 El módulo spectral nunca se importa (P0)

`raven/memory_engine.py:44` hace `from spectral import SpectralField,
SpectralStore`, pero `spectral.py` vive dentro del paquete `raven/` y ningún
punto de entrada (api_server, mcp_server, tests, run_all) añade `raven/` a
`sys.path` — todos importan `raven.memory_engine`. Resultado: el `ImportError`
se traga en el `try/except`, `_SPECTRAL_AVAILABLE = False`, y toda la
funcionalidad spectral (resonancia, coherencia, persistencia del campo, el
rebuild del consolidador en `raven/sleep_consolidator.py:467`) está muerta en
cada despliegue real. El fallo es un `logger.debug` — invisible, lo contrario
de la filosofía de "degradación honesta" del proyecto.

- **Arreglo:** import relativo (`from .spectral import ...`) con fallback a
  absoluto para ejecución directa; en el consolidador, `from raven.spectral
  import ...`. Subir el log de ausencia a `WARNING`.
- **Nota:** el comentario en `memory_engine.py:37-41` dice que spectral debe
  importarse antes que numpy para fijar las vars de BLAS single-thread — al
  arreglar el import, verificar que esa garantía de determinismo se preserva
  (o documentar que ya no aplica).
- **Aceptación:** un test que arranque el engine vía `raven.memory_engine` y
  afirme `_SPECTRAL_AVAILABLE is True`; `/recall` devuelve `resonance_score`
  distinto de 0.0 con un campo construido.

### 0.2 La API bloquea el event loop en cada recall (P0)

`api_server.py:445` (`/recall`) y `:327` (`POST /memories`) son `async def`
pero llaman código síncrono: `orch.process_message()` encadena embedding local
(torch) + HTTP a Qwen con timeout de 30 s + reintentos. Durante ese tiempo el
event loop entero está congelado: ni health checks, ni WebSocket, ni otras
peticiones. Con el rate limit de 30/min basta un cliente lento para degradar
todo el servidor.

- **Arreglo:** mover el trabajo síncrono a threadpool (`def` en vez de
  `async def`, o `run_in_executor`). El orquestador pasa a ser accedido desde
  varios hilos → proteger su estado mutable (`_turn_history`,
  `_conversation_history`) con un lock, o resolver junto con 0.4.
- **Aceptación:** test de concurrencia: un `/recall` lento (mock de Qwen con
  sleep) no impide que `/health` responda en <100 ms.

### 0.3 La caché de `/recall` sirve resultados obsoletos y crece sin límite (P0)

`api_server.py:202` — `_recall_cache` no se invalida nunca: tras almacenar,
reforzar u olvidar una memoria, la misma query devuelve el resultado viejo
(con `stats` y `audit_log` viejos incluidos), lo que contradice directamente
la propuesta de valor ("el campo cambia con cada interacción"). Además es un
dict sin límite de tamaño ni TTL → fuga de memoria bajo tráfico variado, y
cachea también respuestas degradadas (dummy embeddings).

- **Arreglo:** invalidación por generación (un contador que incrementan
  store/reinforce/forget/link, incluido en la clave), TTL + tamaño máximo
  (LRU), y no cachear cuando `embedding_provider.degraded` o `recall_error`.
- **Aceptación:** test API: store A → recall Q → store B (relevante a Q) →
  recall Q devuelve B; la caché no supera N entradas.

### 0.4 Todos los clientes HTTP comparten una sola conversación (P0)

`MemoryAgentOrchestrator` mantiene `_turn_history` y `_conversation_history`
globales al proceso (`raven/qwen_client.py:338-339`), y la API instancia un
único orquestador. Dos usuarios concurrentes mezclan: el historial de
conversación que se inyecta al LLM (fuga de información entre clientes), y las
señales STDP (co-activaciones de un usuario refuerzan sinapsis del otro).
`RecallRequest` ni siquiera expone `session_id`; el store del orquestador
siempre usa `session_id="agent"`.

- **Arreglo:** estado por sesión — `session_id` en `RecallRequest`, y un mapa
  `session_id → (turn_history, conversation_history)` con expiración, o un
  orquestador ligero por sesión sobre el engine compartido.
- **Aceptación:** test: dos sesiones intercaladas no comparten historial de
  conversación ni de turnos.

### 0.5 Endpoints de lectura sin autenticación (P1, seguridad)

Con `RAVEN_API_TOKEN` configurado, los endpoints mutadores exigen token, pero
`GET /memories`, `GET /memories/{id}`, `GET /audit`, `GET /alerts` y
`GET /stats` no (`api_server.py:370-544`) — cualquiera puede leer el contenido
completo de la memoria y el rastro de auditoría (que incluye previews de
contenido y las queries de los usuarios). Solo `GET /graph` está protegido.

- **Arreglo:** `Depends(require_token)` también en lecturas (manteniendo el
  modo abierto sin token para la demo local). `/health` puede quedar público
  pero sin volcar la configuración de CORS/rate-limit cuando hay token.
- **Extras del mismo barrido:** comparación de token en tiempo constante
  (`secrets.compare_digest`, `api_server.py:77` y `:559`); confiar en
  `X-Forwarded-For` solo detrás de proxy declarado (`RAVEN_TRUST_PROXY=1`),
  porque hoy cualquier cliente directo evade el rate limit falsificando la
  cabecera (`api_server.py:113`).
- **Aceptación:** con token configurado, toda lectura sin token → 401; test
  de spoofing de `X-Forwarded-For` sin proxy declarado no rota el bucket.

### 0.6 Inconsistencia de variable de entorno en el MCP server (P1)

`mcp_server.py:72` lee `QWEN_API_KEY`, pero el cliente, el README y la API
usan `DASHSCOPE_API_KEY`. Quien configure `DASHSCOPE_API_KEY` para el MCP
server (como indica toda la documentación) obtiene silenciosamente el tier
dummy.

- **Arreglo:** aceptar `DASHSCOPE_API_KEY` (primario) con `QWEN_API_KEY` como
  alias retrocompatible; loguear qué tier quedó activo al arrancar.
- **Aceptación:** test unitario de resolución de config.

### 0.7 Perfil estilométrico basado en una sola muestra (P1)

`_author_fingerprints` guarda la **primera** huella vista por autor y no se
actualiza jamás (`memory_engine.py:962-964`, `:1062-1063`). Consecuencias: el
perfil "histórico" es una muestra única y arbitraria (en el arranque, la de
menor `cell_id`); un autor legítimo con un primer texto atípico genera falsos
positivos que **destruyen datos de recall** (auto-FORGOTTEN en plena lectura,
`memory_engine.py:1211-1227`); y el orden de carga determina el veredicto.

- **Arreglo:** perfil rodante (media incremental por autor sobre las últimas N
  muestras del mismo idioma), mínimo de M muestras antes de activar el
  enforcement, y separar detección de acción: por defecto alertar
  (`ForensicAlert`) sin degradar a FORGOTTEN salvo opt-in
  (`RAVEN_STYLO_ENFORCE=1`). Documentar que recall() muta estado.
- **Aceptación:** test: 5 textos variados del mismo autor no generan alerta;
  un cambio real de estilo sí; sin opt-in, el estado no cambia.

---

## Fase 1 — Infraestructura de ingeniería

### 1.1 CI en GitHub Actions (P0 de proceso)

No hay `.github/workflows/`. Nada impide que un push rompa los 20 tests.

- Workflow con matriz Python 3.11/3.12: instalar deps core (sin torch, usando
  el tier dummy determinista — los tests ya no dependen de
  sentence-transformers), correr pytest, lint y build de Docker.
- **Aceptación:** badge verde en README; un PR con un test roto falla.

### 1.2 Empaquetado real del proyecto (P1)

`raven/__init__.py` está vacío, no hay `pyproject.toml`, y los entry points
dependen de hacks de `sys.path` (`mcp_server.py:47`, `run_all.py:22`,
`tests/test_suite.py:18`). El proyecto no es `pip install`-able, que es la
forma en que un agente real lo consumiría.

- `pyproject.toml` con extras: `raven-memory[local-embeddings]`, `[api]`,
  `[demo]`, `[mcp]`; exports en `raven/__init__.py`
  (`AdaptiveMemoryEngine`, `MemoryAgentOrchestrator`, …); scripts de consola
  (`raven-api`, `raven-mcp`, `raven-consolidate`); eliminar los hacks de path.
- **Aceptación:** `pip install -e . && python -c "from raven import
  AdaptiveMemoryEngine"` funciona desde cualquier CWD; tests corren vía
  `pytest` sin `sys.path.insert`.

### 1.3 Migrar la suite a pytest y ampliar cobertura (P1)

`tests/test_suite.py` usa un runner artesanal (`run_all()` con contador
manual). Ya se instala pytest en requirements pero no se usa.

- Convertir a pytest (fixtures con `tmp_path` para DBs), mantener
  `run_all.py` como envoltorio. Añadir los tests que hoy no existen:
  qwen_client (fallback de 3 tiers con mocks, sanitización de historial,
  presupuesto de contexto), api_server (TestClient: auth, rate limit, caché,
  CORS), sleep_consolidator (dry-run, merge atómico con crash simulado,
  continuidad de cadena), mcp_server (sanitización de inputs) — hoy la
  cobertura fuera del engine es ~0.
- **Aceptación:** `pytest -q` verde; cobertura reportada en CI (objetivo
  inicial ≥70 % en `raven/`).

### 1.4 Dependencias reproducibles y versionado de esquema (P2)

- `requirements.txt` usa `>=` sin lockfile → instalaciones no reproducibles
  (el propio comentario del archivo aspira a "reproducible install"). Añadir
  constraints/lock (uv o pip-tools) manteniendo rangos en el metadata del
  paquete.
- Las migraciones de esquema son `try/except ALTER` ad-hoc
  (`memory_engine.py:441-444`, `:487-493`). Adoptar `PRAGMA user_version` con
  migraciones numeradas idempotentes — prerequisito para cualquier cambio de
  esquema de las fases siguientes.
- **Aceptación:** instalar dos veces produce el mismo entorno; abrir una DB
  v1.1 con el código nuevo migra y deja `user_version` correcto.

---

## Fase 2 — Rendimiento y escalabilidad

Todas con benchmarks antes/después (ver 4.2) — ninguna se fusiona sin números.

### 2.1 Eliminar el doble BFS por recall (P1)

`recall()` ya hace un BFS de expansión, pero después llama
`_hop_distance(query_cell, mem.cell_id)` **por cada candidato**
(`memory_engine.py:1232`), que es otro BFS completo desde la semilla — O(n·V)
por recall. La distancia de salto ya se conoce en el momento en que el BFS de
expansión alcanza cada celda.

- **Arreglo:** registrar `hop` por celda durante la expansión
  (`{cell_id: hop}` en vez de solo el set `activated_cells`) y eliminar
  `_hop_distance` del camino caliente.
- **Aceptación:** mismos resultados en la suite; benchmark de recall con 10k
  memorias mejora de forma medible.

### 2.2 No cargar todos los cell_links en cada recall (P1)

`load_all_cell_links_indexed()` (`memory_engine.py:1136`) lee la tabla
completa de links en cada recall — O(total_links) aunque la expansión toque
20 celdas. Tras meses de auto-links de contradicciones esto crece sin techo.

- **Arreglo:** mantener el índice de links en memoria (ya existe la mitad:
  `_resonant_neighbors`, `_linked_pairs`) actualizado en
  `store_cell_link`/`create_cell_link`, con recarga completa solo en
  `_load_from_db()`; o consultar por lotes solo las celdas de la frontera.
- **Aceptación:** recalls no escalan con el total de links (benchmark).

### 2.3 Batching de escrituras post-recall (P2)

Cada recall hace: 1 UPDATE por resultado top-k (`update_activation`,
`memory_engine.py:1341-1342`) + 1 UPDATE por memoria previa en STDP
(`_update_stdp`, `:1385`) + el INSERT de auditoría — cada uno con su propia
conexión y commit (`MemoryStore._connect` abre conexión nueva por llamada).
Son ~10-30 transacciones por recall.

- **Arreglo:** una transacción por recall para todas las escrituras
  derivadas; valorar una conexión persistente por hilo
  (`threading.local`) en vez de conexión-por-llamada.
- **Aceptación:** latencia p50 de recall baja (benchmark); la suite y el
  consolidador concurrente siguen verdes (WAL + busy_timeout intactos).

### 2.4 `export_graph` y arranque sin materializar toda la tabla (P2)

- `export_graph()` llama `load_memories()` completo para ordenar por
  `recall_count` (`memory_engine.py:1481-1485`) — hacerlo en SQL
  (`ORDER BY recall_count DESC LIMIT ?`).
- `_load_from_db()` carga todas las memorias (embeddings incluidos) al
  arrancar. Aceptable hasta ~10⁵; documentar el límite y, para la fase de
  escala, cargar embeddings solo de celdas activas.
- **Aceptación:** export de un corpus de 50k con `max_nodes=1000` no carga
  50k filas (verificable con contador de filas leídas o tiempo).

### 2.5 Adelgazar el audit log (P2)

Cada recall persiste el embedding completo de la query como JSON
(`store_audit`, `memory_engine.py:774`) — ~4-8 KB por entrada que solo se usa
para recomputar `qemb_sha256`. La cadena puede sellar el hash directamente.

- **Arreglo (requiere 1.4):** nueva versión de esquema que almacene
  `qemb_sha256` en columna propia y no el vector; `verify_audit_chain()`
  distingue entradas v1 (recomputa desde el vector) y v2 (usa el hash
  almacenado) para no romper cadenas existentes.
- **Aceptación:** cadenas mixtas v1/v2 verifican; crecimiento del audit log
  por recall cae un orden de magnitud.

### 2.6 Backend ANN opcional para corpus grandes (P3)

KDTree en 384 dimensiones degenera a fuerza bruta (maldición de la
dimensionalidad) y el rebuild es completo en cada dirty. Para el roadmap de
escala: interfaz de índice enchufable (KDTree por defecto; hnswlib como extra
opcional) manteniendo el contrato de "solo celdas activas".

- **Aceptación:** benchmark 100k memorias: recall p50 < 100 ms con ANN;
  resultados equivalentes (recall@k vs KDTree ≥ 0.95).

---

## Fase 3 — Producto y funcionalidad

### 3.1 Consolidación sin reinicio (P1)

Hoy el consolidador exige reiniciar el engine para refrescar el índice
(`sleep_consolidator.py:379-382` lo admite). Es el mayor roce operativo.

- Endpoint `POST /consolidate` (autenticado) que ejecute la consolidación
  in-process y llame `_load_from_db()` + `rebuild_spectral_field()` al
  terminar, con lock para excluir recalls durante el swap; mantener el CLI
  para uso offline. Añadir herramienta MCP `raven_consolidate`.
- **Aceptación:** test: store duplicados → consolidate vía API → recall
  inmediato ve el nodo consolidado, sin reiniciar.

### 3.2 Export/import del campo de memoria (P2)

No hay forma de respaldar, migrar o compartir un campo salvo copiar el
SQLite. Añadir `raven-export` / `raven-import` (JSONL: memorias, links,
audit) documentando qué se preserva (la cadena de auditoría es verificable
tras un import completo; un import parcial la rompe y debe decirlo — el
mismo principio de honestidad que ya aplica el consolidador).

- **Aceptación:** export → import en DB nueva → misma respuesta a las
  queries de la suite; `verify_audit_chain` reporta el estado real.

### 3.3 Observabilidad (P2)

Con la API desplegada en ECS no hay métricas: añadir `/metrics` Prometheus
(latencia de recall, tier de embeddings activo, `dummy_fallbacks`, tamaño de
caché, clientes WS, MSS) y logging estructurado opcional (JSON). El caso de
uso concreto: enterarse de que el proveedor cayó a dummy **sin** mirar logs.

- **Aceptación:** `curl /metrics` expone las series; una caída a dummy es
  visible como métrica.

### 3.4 Mejoras MCP (P3)

- Alinear env vars (0.6) y añadir `raven_consolidate` (3.1) y
  `raven_verify_chain` como herramienta separada de `raven_audit_trail`.
- Exponer `topic`/`claim` también en `raven_recall` (hoy solo en store), y
  documentar en las descripciones de herramienta el patrón contradicción →
  reinforce → collapse, que es el diferencial del producto para un agente.
- **Aceptación:** flujo completo store-contradicción → reinforce → recall
  ejecutable solo con herramientas MCP, cubierto por un test.

### 3.5 Limpieza de repo (P3)

- `demo_killer.py` → `demo/gradio_demo.py` (el nombre actual no ayuda en
  contextos profesionales); consolidar `site/index_v1.0.html` y páginas
  duplicadas; añadir CHANGELOG.md y CONTRIBUTING.md; recortar el README
  (~630 líneas) moviendo el detalle de mecanismos a `docs/ARCHITECTURE.md`.

---

## Fase 4 — Validación científica del campo

Es la inversión de mayor retorno en credibilidad: el README afirma que el
campo es mejor que `top-k` plano, pero no hay ningún número que lo respalde.

### 4.1 Benchmark de calidad de recall (P1)

- Harness `benchmarks/` con dos condiciones sobre el mismo corpus y
  embeddings: (a) baseline coseno top-k puro, (b) campo completo (estados +
  links + STDP + hop decay). Métricas: recall@k / MRR, y métricas propias del
  diferencial: tasa de supresión correcta de contradicciones, tasa de rescate
  de verdades validadas, precisión tras reinforce.
- Corpus: el corpus inglés existente (`load_english_corpus.sh`) + un corpus
  sintético de contradicciones etiquetadas; valorar un subconjunto de un
  benchmark público de memoria conversacional (p. ej. LongMemEval/LoCoMo)
  para comparabilidad externa.
- **Resultado honesto:** publicar los números en README aunque alguna
  condición no gane — es coherente con la sección "honest boundary" y la
  hace creíble.
- **Aceptación:** `python -m benchmarks.run` reproduce la tabla del README
  con seed fija.

### 4.2 Benchmark de rendimiento (P1, prerequisito de la Fase 2)

- Script que puebla 1k/10k/100k memorias (embeddings dummy deterministas) y
  mide: latencia de store, latencia de recall (p50/p95), tiempo de rebuild
  del KDTree, tamaño de DB y crecimiento del audit log. Corre en CI en la
  escala pequeña para detectar regresiones.
- **Aceptación:** los PRs de la Fase 2 citan estos números antes/después.

### 4.3 Calibrar los hiperparámetros con el benchmark (P3)

`HOP_LAMBDA=0.15`, boost RESONANT `+0.5`, peso sináptico `×0.3`,
`RECENCY_WEIGHT=0.05` (`memory_engine.py:63-78`) son constantes con nombre
pero sin justificación empírica. Con 4.1 en pie, hacer un barrido y
documentar la sensibilidad — o descubrir que algún término (p. ej. recency)
no aporta y simplificar el scoring.

---

## Orden de ejecución sugerido

| Sprint | Ítems | Tema |
|---|---|---|
| 1 | 0.1, 0.2, 0.3, 1.1 | Bugs P0 + CI para que nada regresione |
| 2 | 0.4, 0.5, 0.6, 1.3 | Seguridad/estado compartido + suite pytest |
| 3 | 1.2, 1.4, 0.7 | Packaging, lockfile, esquema, estilometría |
| 4 | 4.2, 2.1, 2.2, 2.3 | Benchmark de rendimiento y optimizaciones del camino caliente |
| 5 | 3.1, 3.3, 2.4, 2.5 | Consolidación en caliente, métricas, dieta del audit log |
| 6 | 4.1, 4.3 | Benchmark de calidad y calibración |
| 7+ | 2.6, 3.2, 3.4, 3.5 | Escala ANN, export/import, MCP, limpieza |

Los ítems 0.x son independientes entre sí y paralelizables. Nada de la Fase 2
debería fusionarse antes de que exista 4.2 (medir antes de optimizar).

---

## Qué NO cambiar

Decisiones actuales que esta revisión valida y que conviene proteger con
tests, no "mejorar":

- **SQLite + WAL como único almacenamiento** — correcto para el modelo de
  despliegue (proceso único + consolidador); no migrar a un servidor de BD.
- **La regla de rescate** (una verdad validada no puede ser silenciada por
  una afirmación no verificada) — es el invariante epistémico distintivo.
- **Forgetting como exclusión, nunca borrado** — sustenta la historia
  forense completa.
- **Contradicciones por metadata (`topic`/`claim`), no por NLI** — el
  determinismo es una feature; un modelo de inferencia lo destruiría.
- **Spectral fuera del ranking** (metadata de solo lectura) — mantenerlo así
  incluso después de arreglar 0.1.
- **La degradación ruidosa del proveedor de embeddings** — extenderla (0.1,
  3.3), jamás silenciarla.
