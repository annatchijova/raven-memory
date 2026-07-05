#!/bin/bash
# raven-memory — single-topic English corpus (dense, for stress-testing recall)
HOST="${1:-http://8.222.219.67:8012}"
COUNT=0

store() {
  local content="$1"; local layer="$2"; local state="${3:-NEUTRAL}"; local meta="$4"
  resp=$(curl -s -X POST "$HOST/memories" -H "Content-Type: application/json" \
    -d "{\"content\":\"$content\",\"layer\":\"$layer\",\"state\":\"$state\",\"metadata\":$meta}")
  COUNT=$((COUNT+1))
  mid=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('memory_id','?')[:30])" 2>/dev/null)
  prov=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('embedding_provider','?'))" 2>/dev/null)
  if [ "$mid" = "?" ] || [ -z "$mid" ]; then
    echo "  [$COUNT] FAILED: $resp"
  else
    echo "  [$COUNT] $mid... | $layer | $state | $prov"
  fi
  sleep 0.6
}

echo "=== raven-memory: single-topic English corpus ==="

store "raven-memory represents each stored item as a point inside a Voronoi diagram. The cell a memory occupies determines which other memories are its geometric neighbours." \
  "semantic" "REINFORCED" '{"topic":"voronoi","claim":"cell_assignment"}'

store "Voronoi cell boundaries in raven-memory are recomputed incrementally as new memories are inserted, using a KDTree rebuild only when the tree's balance factor degrades past a threshold." \
  "semantic" "NEUTRAL" '{"topic":"voronoi","claim":"incremental_rebuild"}'

store "A memory's ternary state multiplies its base similarity score: REINFORCED applies ×1.5, NEUTRAL applies ×1.0, FORGOTTEN applies ×0.5. States are reversible, not deletions." \
  "semantic" "REINFORCED" '{"topic":"ternary_states","claim":"multiplier_values"}'

store "Some early design notes for raven-memory considered a binary state model (active/inactive) instead of ternary. This was abandoned because it could not represent partial distrust." \
  "semantic" "NEUTRAL" '{"topic":"ternary_states","claim":"binary_was_rejected"}'

store "STDP synaptic boosts in raven-memory accumulate every time two memories are co-activated within the same recall. The boost value is stored per edge in the cell_links table, not per memory." \
  "semantic" "REINFORCED" '{"topic":"stdp","claim":"edge_storage"}'

store "The STDP synaptic boost in raven-memory has no upper cap in the current implementation — repeated co-activation can drive the boost arbitrarily high, which the team has flagged as a known limitation." \
  "semantic" "NEUTRAL" '{"topic":"stdp","claim":"uncapped_known_limitation"}'

store "The spectral module in raven-memory computes a truncated SVD over the memory embedding matrix. The resulting eigen-modes describe resonance between memories, not raw cosine similarity." \
  "semantic" "REINFORCED" '{"topic":"spectral","claim":"svd_eigen_modes"}'

store "raven-memory's spectral field is rebuilt lazily: it only recomputes the SVD decomposition when is_stale() detects that enough new memories were inserted since the last rebuild." \
  "semantic" "NEUTRAL" '{"topic":"spectral","claim":"lazy_rebuild"}'

store "Sleep consolidation in raven-memory clusters episodic memories by embedding similarity above a threshold, then fuses each cluster into a single semantic node with a recency-weighted centroid." \
  "semantic" "REINFORCED" '{"topic":"sleep","claim":"cluster_and_fuse"}'

store "Sleep consolidation in raven-memory runs as an offline batch job. It is not triggered automatically on any schedule inside the running server — an operator must invoke it manually or via cron." \
  "semantic" "NEUTRAL" '{"topic":"sleep","claim":"manual_trigger"}'

store "Every store, recall, and consolidation operation in raven-memory appends an entry to a SHA-256 hash chain, where each entry's hash depends on the previous entry's hash." \
  "semantic" "REINFORCED" '{"topic":"audit","claim":"hash_chain_structure"}'

store "The audit chain in raven-memory is verified on read by recomputing hashes from the stored payloads and comparing them against the recorded hash column, not by trusting the stored hash blindly." \
  "semantic" "REINFORCED" '{"topic":"audit","claim":"recompute_on_verify"}'

store "raven-memory's recall function walks outward from the query's nearest Voronoi cell through neighbouring cells, applying exponential hop decay e^(-0.15 × hops) to each additional step." \
  "semantic" "REINFORCED" '{"topic":"recall","claim":"hop_decay_formula"}'

store "raven-memory's recall never performs a global top-k scan over the entire embedding space. It only considers memories reachable within the configured hop limit from the query cell." \
  "semantic" "REINFORCED" '{"topic":"recall","claim":"local_only_no_global_scan"}'

store "raven-memory falls back through three embedding tiers: local sentence-transformers first, then the Qwen Cloud API, then a deterministic dummy vector if both previous tiers are unavailable." \
  "semantic" "REINFORCED" '{"topic":"embeddings","claim":"three_tier_fallback"}'

store "When raven-memory falls back to dummy embeddings, recall results become semantically meaningless random projections. The system logs this loudly and reports degraded:true on /health." \
  "semantic" "REINFORCED" '{"topic":"embeddings","claim":"dummy_is_loud_not_silent"}'

store "raven-memory stores all data in a single SQLite file using WAL mode. There is no distributed storage layer, and no plan to add one — the design targets a single agent's memory, not a shared cluster." \
  "semantic" "REINFORCED" '{"topic":"storage","claim":"sqlite_single_file"}'

store "raven-memory can scale horizontally across multiple nodes by partitioning memories along Voronoi cell boundaries, allowing each node to own a distinct region of the embedding space." \
  "semantic" "NEUTRAL" '{"topic":"storage","claim":"horizontal_scaling_false"}'

echo ""
echo "Done. $COUNT memories loaded."
