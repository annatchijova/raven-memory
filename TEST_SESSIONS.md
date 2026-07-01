# Test Sessions — Raven Memory contra Qwen Cloud

Sesiones de prueba para ejecutar contra el deploy real (`8.222.219.67:8012`)
antes de grabar el video demo. Objetivo: encontrar bugs con la API de Qwen
real, no con embeddings locales.

**Prerequisito**: el contenedor debe estar corriendo con `RAVEN_USE_LOCAL_EMBEDDINGS=0`
y `DASHSCOPE_API_KEY` configurada. Verificar con la sesión 0.

---

## Sesión 0 — Verificación de entorno

```bash
# 1. Health check — embedding_provider.active debe decir "qwen_api", NO "dummy" ni "local"
curl -s http://8.222.219.67:8012/health | python3 -m json.tool

# Verificar estos campos en la respuesta:
#   "embedding_provider.active" == "qwen_api"
#   "embedding_provider.degraded" == false
#   "status" == "ok"

# 2. Stats iniciales
curl -s http://8.222.219.67:8012/stats | python3 -m json.tool
```

**Si `embedding_provider.active` dice "dummy"**: la API key no está llegando
al contenedor o Qwen rechaza las requests. Revisar logs:
```bash
# Desde la instancia de Alibaba:
docker logs <container_id> --tail 50
```

---

## Sesión 1 — Store + Recall básico

Objetivo: verificar que store y recall funcionan end-to-end con Qwen embeddings + LLM.

```bash
# 1. Almacenar una memoria
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "VIGÍA es un sistema de análisis semiótico forense que opera con aritmética racional.",
    "layer": "semantic",
    "state": "NEUTRAL",
    "metadata": {"topic": "vigia", "claim": "semiotic"}
  }' | python3 -m json.tool

# Anotar el memory_id de la respuesta: _______________

# 2. Recall — debe recuperar la memoria recién guardada
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qué es VIGÍA?",
    "top_k": 5,
    "hops": 2
  }' | python3 -m json.tool

# VERIFICAR:
#   [ ] qwen_response.content NO dice "[OFFLINE MODE"
#   [ ] recalled_memories contiene la memoria guardada
#   [ ] embedding_provider.active == "qwen_api"
#   [ ] latency_ms < 10000 (si tarda más de 10s, hay problema de red)
```

---

## Sesión 2 — Memorias contradictorias + collapse

Objetivo: probar el feature principal del sistema — dos memorias contradictorias,
reinforcement de una, y observar el cambio de scores.

```bash
# 1. Insertar contradicción A
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "VIGÍA es completamente determinista. No usa machine learning en ninguna de sus capas.",
    "layer": "semantic",
    "state": "NEUTRAL",
    "metadata": {"topic": "vigia_nature", "claim": "deterministic"}
  }' | python3 -m json.tool
# Anotar memory_id A: _______________

# 2. Insertar contradicción B
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "VIGÍA usa embeddings y similitud coseno. Es técnicamente un sistema de ML híbrido.",
    "layer": "semantic",
    "state": "NEUTRAL",
    "metadata": {"topic": "vigia_nature", "claim": "ml_hybrid"}
  }' | python3 -m json.tool
# Anotar memory_id B: _______________

# 3. Recall ANTES del reinforcement — ambas deberían tener scores similares
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Es VIGÍA un sistema de machine learning?",
    "top_k": 5,
    "hops": 2
  }' | python3 -m json.tool
# Anotar scores de A y B: A=_____ B=_____

# 4. Reinforce A (deterministic)
curl -s -X POST http://8.222.219.67:8012/memories/MEMORY_ID_A/reinforce | python3 -m json.tool

# 5. Recall DESPUÉS del reinforcement — A debería tener score > B
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Es VIGÍA un sistema de machine learning?",
    "top_k": 5,
    "hops": 2
  }' | python3 -m json.tool

# VERIFICAR:
#   [ ] Score de A subió (state_boost = 1.5)
#   [ ] Score de B se mantuvo o bajó
#   [ ] qwen_response menciona que hay conflicto pero prioriza A
#   [ ] MSS cambió en stats
```

---

## Sesión 3 — Forget + verificar que no desaparece

Objetivo: verificar que forget reduce score pero no borra la memoria.

```bash
# 1. Forget B (ml_hybrid)
curl -s -X POST http://8.222.219.67:8012/memories/MEMORY_ID_B/forget | python3 -m json.tool

# 2. Recall — B debería aparecer con score muy bajo o no aparecer en top_k=3
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Es VIGÍA un sistema de machine learning?",
    "top_k": 5,
    "hops": 2
  }' | python3 -m json.tool

# 3. Pero la memoria sigue existiendo
curl -s http://8.222.219.67:8012/memories/MEMORY_ID_B | python3 -m json.tool

# VERIFICAR:
#   [ ] B tiene state: "FORGOTTEN" (state_boost = 0.5)
#   [ ] B aún existe en GET /memories/{id}
#   [ ] A domina el recall
```

---

## Sesión 4 — Multilingual (español + inglés)

Objetivo: verificar que Qwen embeddings manejan bien textos en ambos idiomas.

```bash
# 1. Memoria en inglés
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The SANS hackathon deadline is June 15, 2026. Rob T. Lee approved the dataset.",
    "layer": "episodic",
    "state": "REINFORCED",
    "metadata": {"topic": "hackathon", "language": "en"}
  }' | python3 -m json.tool

# 2. Recall en español sobre un tema en inglés
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cuándo es la fecha límite del hackathon de SANS?",
    "top_k": 5,
    "hops": 2
  }' | python3 -m json.tool

# 3. Recall en inglés
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When is the SANS hackathon deadline?",
    "top_k": 5,
    "hops": 2
  }' | python3 -m json.tool

# VERIFICAR:
#   [ ] Ambos recalls encuentran la memoria en inglés
#   [ ] Qwen responde en el idioma de la query
#   [ ] No hay errores de encoding (caracteres rotos)
```

---

## Sesión 5 — STDP (co-activación sináptica)

Objetivo: verificar que memorias que se recuperan juntas forman links sinápticos.

```bash
# 1. Dos memorias relacionadas
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Rob T. Lee aprobó el dataset de VIGÍA para la competencia de SANS.",
    "layer": "episodic",
    "metadata": {"topic": "hackathon", "person": "Rob Lee"}
  }' | python3 -m json.tool

curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "El deadline del hackathon SANS es el 15 de junio de 2026 a las 14:00 PDT.",
    "layer": "episodic",
    "metadata": {"topic": "hackathon", "event": "deadline"}
  }' | python3 -m json.tool

# 2. Query que debería activar ambas (turn 1)
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qué dijo Rob sobre VIGÍA y SANS?",
    "top_k": 5,
    "hops": 2,
    "store_interaction": true
  }' | python3 -m json.tool

# 3. Query diferente que debería beneficiarse del link (turn 2)
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cuándo es el deadline de SANS?",
    "top_k": 5,
    "hops": 2,
    "store_interaction": true
  }' | python3 -m json.tool

# VERIFICAR:
#   [ ] En turn 2, algún resultado tiene source: "synaptic"
#   [ ] synaptic_boost > 0 en al menos un resultado
#   [ ] El grafo muestra links: GET /graph
curl -s http://8.222.219.67:8012/graph | python3 -m json.tool
```

---

## Sesión 6 — Audit trail + integridad de cadena

Objetivo: verificar que la cadena de hashes es íntegra después de varias operaciones.

```bash
# Después de todas las sesiones anteriores:
curl -s http://8.222.219.67:8012/audit?limit=20 | python3 -m json.tool

# VERIFICAR:
#   [ ] chain_intact: true
#   [ ] hash_integrity: true
#   [ ] issues: [] (lista vacía)
#   [ ] Cada entry tiene operation, audit_hash, prev_hash
#   [ ] Las operaciones reflejan lo que hiciste (RECALL, STORE, etc.)
```

---

## Sesión 7 — Edge cases

Objetivo: buscar bugs con inputs inusuales.

```bash
# 1. Query vacía
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "", "top_k": 5}' | python3 -m json.tool
# Esperado: debería devolver algo razonable o un error claro, no un 500

# 2. Texto muy largo (>2000 chars)
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d "{
    \"content\": \"$(python3 -c "print('A' * 3000)")\",
    \"layer\": \"semantic\"
  }" | python3 -m json.tool
# Esperado: se guarda o error limpio, no crash

# 3. Caracteres especiales / Unicode
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Señales débiles: ñ, ü, é, 中文, 日本語, العربية, emoji: 🦅🧠⚡",
    "layer": "semantic"
  }' | python3 -m json.tool

# 4. JSON metadata inválido (debería fallar limpio)
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "test",
    "metadata": "esto no es un dict"
  }' | python3 -m json.tool
# Esperado: 422 validation error, no 500

# 5. top_k extremo
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 20, "hops": 4}' | python3 -m json.tool

# 6. Memory ID que no existe
curl -s -X POST http://8.222.219.67:8012/memories/nonexistent-id-12345/reinforce | python3 -m json.tool
# Esperado: 404, no 500

# VERIFICAR:
#   [ ] Ningún caso devuelve HTTP 500
#   [ ] Los errores tienen mensajes claros (detail field)
#   [ ] El servidor sigue respondiendo después de cada edge case
```

---

## Sesión 8 — Latencia y estabilidad

Objetivo: verificar que el sistema no se degrada con uso sostenido.

```bash
# 1. Ráfaga de 10 recalls seguidos
for i in $(seq 1 10); do
  echo "--- Request $i ---"
  time curl -s -X POST http://8.222.219.67:8012/recall \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"test query $i\", \"top_k\": 5, \"hops\": 2}" \
    -o /dev/null -w "HTTP %{http_code} — %{time_total}s\n"
done

# VERIFICAR:
#   [ ] Ninguna request devuelve 429 (rate limit) con 10 requests — el límite es 30/min
#   [ ] Los tiempos de respuesta son consistentes (no crecen)
#   [ ] No hay 500s

# 2. Health después de la ráfaga
curl -s http://8.222.219.67:8012/health | python3 -m json.tool
# embedding_provider.degraded sigue en false
```

---

## Sesión 9 — Alertas forenses (si aplica)

Objetivo: verificar que la estilometría funciona con Qwen embeddings reales.

```bash
# 1. Memoria con estilo de Anna
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "El sistema VIGÍA es completamente determinista. No usa floats ni probabilidades. Es puro análisis semiótico forense con aritmética racional.",
    "layer": "semantic",
    "author_id": "anna",
    "metadata": {"topic": "vigia_nature"}
  }' | python3 -m json.tool

# 2. Memoria con estilo radicalmente diferente, mismo author_id (simula tampering)
curl -s -X POST http://8.222.219.67:8012/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Bueno o sea lo que pasa es que todo el tema ese va asi: cada cosa que entra sale y listo no hay mas vuelta que darle porque al final del dia todo se reduce a eso nada mas!!!",
    "layer": "semantic",
    "author_id": "anna",
    "metadata": {"topic": "vigia_nature"}
  }' | python3 -m json.tool

# 3. Recall para triggear check de estilometría
curl -s -X POST http://8.222.219.67:8012/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "Qué es VIGÍA?", "top_k": 10, "hops": 2}' | python3 -m json.tool

# 4. Verificar alertas
curl -s http://8.222.219.67:8012/alerts | python3 -m json.tool

# VERIFICAR:
#   [ ] Se generó al menos una alerta de tampering
#   [ ] mismatch_score > 0.5
#   [ ] action_taken describe qué hizo el sistema
```

---

## Checklist post-testing

Después de correr todas las sesiones:

- [ ] Ningún HTTP 500 encontrado
- [ ] `embedding_provider.active` siempre fue `"qwen_api"` (nunca cayó a dummy)
- [ ] La cadena de audit está intacta (`chain_intact: true`)
- [ ] El LLM respondió en el idioma correcto
- [ ] Los scores de reinforced > neutral > forgotten
- [ ] El sistema sigue respondiendo después de ~30+ requests
- [ ] Listo para grabar video demo
