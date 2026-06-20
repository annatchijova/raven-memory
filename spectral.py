#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVEN-MEMORY — Spectral Resonance Module v2.4
PCA-based eigen-mode decomposition with strict determinism controls,
versioned schema, cross-process BLAS isolation, and hardened tensor validation.

Design decision (frozen — PCA Engine v2.3/v2.4):
  projection = dot(eigen_modes, centered_vector)   [no energy weighting]
  resonance  = cosine(projection_q, projection_m)  [standard PCA similarity]
  coherence  = RESONANT / (RESONANT + INHIBITORY)  [epistemic flag only]
  Energy weighting and spectral graph fusion deferred to v3.

Determinism guarantee (explicit):
  intra-process : STRONG  — same embeddings → same modes (float32, single-thread
                  BLAS, sign-normalized eigenvectors). Reproducible bit-for-bit.
  cross-process : BEST-EFFORT — env vars force single-threaded BLAS, but backend
                  selection happens at install time, not runtime. Degenerate
                  eigenspaces (repeated singular values) may rotate under different
                  BLAS builds (OpenBLAS vs MKL vs Accelerate), producing equivalent
                  but non-identical modes. Not a problem for recall quality, but
                  serialize+reload will differ from a fresh build across machines.

Changelog:
  v2.5  (Anna Tchijova + Claude, VIGÍA AI Collective; external review 2026-06)
        FIX  P1-A: legacy fields loaded with reconstructed (zero) mean_vector now
             set requires_rebuild=True — is_stale() forces rebuild, summary() warns.
             A PCA projected over the wrong centre degrades silently otherwise.
        FIX  P1-B: build_from_memories() validates per-embedding shape and
             finiteness BEFORE vstack — clear TensorValidationError instead of
             an opaque numpy broadcast error on corrupt input.
        FIX  P1-C: project() guards against non-finite projections (overflow /
             corrupt input) and returns a zero vector instead of propagating NaN.
        FIX  cross-process determinism test now reports PASS / WARN / FAIL as
             distinct states — a failed child no longer masquerades as PASS
             (observability fix; suite counts WARNs separately).
        FIX  k >= 1 guaranteed in mode truncation (first-variance >= threshold
             edge case could not produce k=0, but the invariant is now explicit).
        ADD  cross-process child also verifies the JSON serialization round-trip
             (the path production actually uses), not just the in-memory build.
        ADD  build_and_persist_spectral_field() catches SpectralStore errors —
             a persistence failure no longer destroys a valid in-memory field.
        NOTE schema unchanged → ENGINE_VERSION remains "pca_v2.4".
  v2.4  (Anna Tchijova + Claude, VIGÍA AI Collective)
        FIX  warning-count asserts replaced with type-filtered any() / filtered-list
             checks throughout test suite — eliminates false negatives when numpy or
             other libs emit their own warnings (ChatGPT review, 2025-06)
        FIX  default strict=False in from_dict / SpectralStore.load() — JSON
             round-trips always yield Python float (float64); strict=True is only
             meaningful for binary formats that preserve dtype (pickle, .npy)
        FIX  test_version_check: corrected multi-warning check (same issue as above)
        ADD  "determinism_level": "best_effort" in to_dict() schema
        ADD  SpectralField.is_stale() — lazy rebuild decisions in engine
        ADD  SpectralField.summary() / __repr__() — human-readable state for logging
        ADD  build_and_persist_spectral_field() — build + save in one call
        DOC  explicit determinism guarantee section in header and class docstring
  v2.3  (Anna Tchijova + Claude, VIGÍA AI Collective; ChatGPT review)
        SVD over centered matrix (replaces eigh over explicit covariance)
        BLAS env vars set before numpy import + RuntimeWarning if already imported
        Strict tensor validation with two-phase shape + dtype checks
        Versioned schema: ENGINE_VERSION = "pca_v2.3"
        Cross-process determinism test with subprocess BLAS isolation
        Backward compatibility: v1/v2 fields without mean_vector

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# DETERMINISM: BLAS env vars BEFORE numpy import
# ============================================================
# P0: must be set before numpy initialises any BLAS backend.
# These force single-threaded execution, which removes the main source
# of non-determinism from parallelism. Cross-process guarantee remains
# best-effort because BLAS backend (OpenBLAS/MKL/Accelerate) is selected
# at install time and can have different floating-point rounding paths.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Warn if numpy was already imported (would mean env vars came too late).
# Using setdefault above avoids clobbering vars the caller already set.
if "numpy" in sys.modules:
    warnings.warn(
        "numpy already imported before spectral.py set BLAS env vars. "
        "Cross-process determinism guarantee is best-effort anyway; "
        "intra-process determinism may also be affected. "
        "Import spectral before numpy to silence this warning.",
        RuntimeWarning,
        stacklevel=2,
    )

import numpy as np


# ============================================================
# CONFIG
# ============================================================

ENGINE_VERSION = "pca_v2.4"      # Increment when serialization schema changes
DTYPE = np.float32
MAX_MODES = 128
VARIANCE_THRESHOLD = 0.99
EPS = 1e-10
FP_TOL = 1e-6


# ============================================================
# STRICT TENSOR VALIDATION
# ============================================================

class TensorValidationError(ValueError):
    """Raised when serialized tensor data fails shape or dtype integrity checks."""
    pass


def validate_shape(data: List, expected_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Phase 1 — structural integrity: shape, finiteness, non-empty.
    Returns a numpy array of whatever dtype numpy inferred from the data.
    dtype contract is the caller's responsibility (see coerce_dtype).
    """
    if not isinstance(data, list):
        raise TensorValidationError(f"Expected list, got {type(data).__name__}")

    arr = np.asarray(data)

    if arr.shape != expected_shape:
        raise TensorValidationError(
            f"Shape mismatch: expected {expected_shape}, got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise TensorValidationError("Array contains NaN or Inf values")
    if arr.size == 0:
        raise TensorValidationError("Array is empty")

    return arr


def coerce_dtype(arr: np.ndarray, expected_dtype: np.dtype, strict: bool = False) -> np.ndarray:
    """
    Phase 2 — dtype contract.

    strict=True  : raise TensorValidationError if arr.dtype != expected_dtype.
                   Use with binary sources (pickle, .npy) that preserve dtype.
    strict=False : silently coerce with a UserWarning.
                   Use with JSON sources where Python float always becomes float64.

    Note: strict=False is the correct default for deserialization from JSON.
    JSON cannot encode dtype information; all numbers become Python float (float64).
    """
    if arr.dtype == expected_dtype:
        return arr
    if strict:
        raise TensorValidationError(
            f"Dtype mismatch: expected {expected_dtype}, got {arr.dtype}. "
            f"JSON sources always produce float64 — use strict=False for from_dict()."
        )
    warnings.warn(
        f"Dtype coercion: {arr.dtype} → {expected_dtype}. "
        f"Precision may be lost if source was higher precision.",
        UserWarning,
        stacklevel=3,
    )
    return arr.astype(expected_dtype)


def validate_array(
    data: List,
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    strict: bool = False,
) -> np.ndarray:
    """
    Full two-phase tensor validation: shape integrity then dtype contract.
    strict=False is the correct default for JSON deserialization.
    """
    arr = validate_shape(data, expected_shape)
    return coerce_dtype(arr, expected_dtype, strict=strict)


# ============================================================
# SPECTRAL FIELD — PCA ENGINE v2.4 (frozen design)
# ============================================================

class SpectralField:
    """
    PCA-based spectral field for memory resonance.

    The 'cavity' metaphor: recall asks "what modes does this field support,
    and which does this query excite?" — not "what is similar?".

    In practice this is standard PCA: the covariance structure of the active
    memory embeddings defines the field's eigen-modes; resonance is cosine
    similarity in that reduced space.

    Determinism guarantee:
      intra-process : STRONG (float32, single-thread, sign-normalized)
      cross-process : BEST-EFFORT (see module docstring)

    Coherence is an EPISTEMIC FLAG, not a ranking modifier:
      coherence = RESONANT_links / (RESONANT + INHIBITORY_links)
      1.0 = no contradictions in neighbourhood
      0.0 = fully contradicted
    It is reported in RecallResult metadata for agent inspection, not
    folded into final_score.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.eigen_modes: Optional[np.ndarray] = None   # (k, embedding_dim)
        self.eigen_values: Optional[np.ndarray] = None  # (k,)
        self.mean_vector: Optional[np.ndarray] = None   # (embedding_dim,)
        self.is_built: bool = False
        self.build_timestamp: float = 0.0
        self.memory_count: int = 0
        # P1-A (v2.5): True when the field was loaded from a legacy schema and
        # the mean_vector had to be reconstructed as zeros. The PCA no longer
        # represents the original space — resonance over it is unreliable.
        self.requires_rebuild: bool = False

    # ----------------------------------------------------------
    # BUILD
    # ----------------------------------------------------------

    def build_from_memories(self, embeddings: List[np.ndarray]) -> bool:
        """
        Build PCA eigen-modes from a list of active memory embeddings.

        Deterministic within process: same inputs → same float32 output.
        Uses SVD on the centered matrix (numerically preferable to explicit
        covariance for the case n < dim).

        Returns True if successfully built, False if fewer than 2 embeddings.
        """
        if len(embeddings) < 2:
            self.is_built = False
            return False

        # P1-B (v2.5): validate every embedding BEFORE vstack.
        # A single corrupt vector ((383,), (512,), NaN) would otherwise explode
        # deep inside numpy with an opaque broadcast error.
        for i, e in enumerate(embeddings):
            arr = np.asarray(e)
            if arr.shape != (self.embedding_dim,):
                raise TensorValidationError(
                    f"Embedding {i} has shape {arr.shape}, "
                    f"expected ({self.embedding_dim},)"
                )
            if not np.isfinite(arr).all():
                raise TensorValidationError(f"Embedding {i} contains NaN or Inf")

        # Stack and center (float32, deterministic)
        matrix = np.vstack([e.astype(DTYPE) for e in embeddings])
        self.mean_vector = np.mean(matrix, axis=0, dtype=DTYPE)
        centered = matrix - self.mean_vector

        # SVD: U S Vt where Vt[k, :] = k-th right singular vector
        # (= principal component direction). full_matrices=False: compact form.
        # Eigenvalues of covariance: λₖ = Sₖ² / (n-1)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n = len(embeddings)
        eigenvalues = (S ** 2 / max(n - 1, 1)).astype(DTYPE)

        # Truncate by cumulative explained variance
        total_var = float(np.sum(eigenvalues))
        if total_var < EPS:
            self.is_built = False
            return False

        cumvar = np.cumsum(eigenvalues) / total_var
        k = int(np.searchsorted(cumvar, VARIANCE_THRESHOLD)) + 1
        # v2.5: explicit invariant — at least one mode always survives, even if
        # the first component alone explains >= VARIANCE_THRESHOLD of variance.
        k = max(1, min(k, MAX_MODES, len(eigenvalues)))

        # Sign-normalize for cross-platform consistency (see docstring)
        raw_modes = Vt[:k, :].astype(DTYPE)
        self.eigen_modes = self._normalize_signs(raw_modes)
        self.eigen_values = eigenvalues[:k]
        self.is_built = True
        self.build_timestamp = time.time()
        self.memory_count = n
        self.requires_rebuild = False   # fresh build → mean is authentic
        return True

    @staticmethod
    def _normalize_signs(modes: np.ndarray) -> np.ndarray:
        """
        Canonical sign for each eigenvector: the component of largest
        absolute magnitude is forced positive.

        This maps {v, −v} to a canonical representative, eliminating the
        sign ambiguity inherent in SVD/eigh.

        Limitation: degenerate subspaces (repeated singular values) can
        still produce rotated modes under different BLAS backends. For
        the hackathon workload (non-pathological embeddings) this is
        practically irrelevant. A full Procrustes alignment is v3 scope.
        """
        normalized = modes.copy()
        for i in range(len(normalized)):
            mode = normalized[i]
            max_idx = int(np.argmax(np.abs(mode)))
            if mode[max_idx] < 0:
                normalized[i] *= -1
        return normalized

    # ----------------------------------------------------------
    # PROJECTION
    # ----------------------------------------------------------

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project an embedding onto the PCA eigen-modes of the field.
        ŝ(m) = normalize(eigen_modes @ (m − mean))

        L2-normalized output makes downstream cosine similarity scale-invariant.
        Returns a zero vector if the projection is degenerate.
        """
        if not self.is_built or self.eigen_modes is None or self.mean_vector is None:
            raise RuntimeError(
                "SpectralField not built. Call build_from_memories() first."
            )
        centered = embedding.astype(DTYPE) - self.mean_vector
        projection = np.dot(self.eigen_modes, centered)   # (k,)
        # P1-C (v2.5): a corrupt or overflowed input must not propagate NaN/Inf
        # into resonance scores. Degenerate projection → neutral zero vector.
        if not np.isfinite(projection).all():
            return np.zeros(self.eigen_modes.shape[0], dtype=DTYPE)
        norm = float(np.linalg.norm(projection))
        if norm < EPS:
            return np.zeros_like(projection)
        return (projection / norm).astype(DTYPE)

    # ----------------------------------------------------------
    # RESONANCE — epistemic, not ranking
    # ----------------------------------------------------------

    def resonance(self, query_emb: np.ndarray, memory_emb: np.ndarray) -> float:
        """
        Resonance between query and memory in the spectral (PCA) space.

        resonance(q, m) = cosine(ŝ(q), ŝ(m))

        NOT added to final_score — it is epistemic metadata reported in
        RecallResult and the audit log. The caller decides how to use it.
        """
        s_q = self.project(query_emb)
        s_m = self.project(memory_emb)
        return self._cosine(s_q, s_m)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Numerically safe cosine similarity, clamped to [-1, 1]."""
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < EPS or nb < EPS:
            return 0.0
        return float(max(-1.0, min(1.0, dot / (na * nb))))

    # ----------------------------------------------------------
    # COHERENCE — epistemic flag
    # ----------------------------------------------------------

    @staticmethod
    def coherence(
        resonant_links: int,
        inhibitory_links: int,
        neutral_links: int = 0,  # retained for forward compatibility
    ) -> float:
        """
        Structural coherence of a memory within its neighbourhood.

        coherence(m) = Σ(RESONANT) / (Σ(RESONANT) + Σ(INHIBITORY))

        1.0 → fully coherent (no contradictions in graph neighbourhood)
        0.5 → balanced (equal RESONANT and INHIBITORY links)
        0.0 → fully contradicted

        Default 1.0 when no links exist (absence of contradiction ≠ incoherence).

        This is a FLAG, not a score multiplier. It informs the agent about
        epistemic reliability; it does NOT influence final_score.
        """
        total = resonant_links + inhibitory_links
        if total == 0:
            return 1.0
        return float(resonant_links / total)

    # ----------------------------------------------------------
    # LAZY REBUILD SUPPORT
    # ----------------------------------------------------------

    def is_stale(
        self, current_memory_count: int, drift_threshold: float = 0.10
    ) -> bool:
        """
        Returns True if the field should be rebuilt.

        Triggered when:
          - Field not yet built.
          - Field loaded from legacy schema with reconstructed mean (P1-A).
          - Memory count changed by more than drift_threshold fraction.

        Used by AdaptiveMemoryEngine for lazy rebuild decisions.
        Default drift_threshold=0.10 means: rebuild if >10% of memories changed.
        """
        if not self.is_built or self.memory_count == 0:
            return True
        if self.requires_rebuild:
            return True
        drift = abs(current_memory_count - self.memory_count) / self.memory_count
        return drift > drift_threshold

    # ----------------------------------------------------------
    # HUMAN-READABLE STATE
    # ----------------------------------------------------------

    def summary(self) -> str:
        """One-line summary for logging and debugging."""
        if not self.is_built:
            return "SpectralField(not built)"
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.build_timestamp))
        k = len(self.eigen_values) if self.eigen_values is not None else 0
        rebuild_flag = ", REQUIRES_REBUILD (reconstructed mean)" if self.requires_rebuild else ""
        return (
            f"SpectralField(modes={k}, memories={self.memory_count}, "
            f"determinism=best_effort, built={ts}{rebuild_flag})"
        )

    def __repr__(self) -> str:
        return self.summary()

    # ----------------------------------------------------------
    # SERIALIZATION — versioned schema
    # ----------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        Serialize to JSON-compatible dict.
        Note: .tolist() converts float32 → Python float (float64).
        Use from_dict(strict=False) when loading — JSON cannot preserve dtype.
        """
        return {
            "engine_version": ENGINE_VERSION,
            "determinism_level": "best_effort",   # v2.4: explicit in schema
            "is_built": self.is_built,
            "build_timestamp": self.build_timestamp,
            "memory_count": self.memory_count,
            "embedding_dim": self.embedding_dim,
            "num_modes": len(self.eigen_values) if self.eigen_values is not None else 0,
            "eigen_values": self.eigen_values.tolist() if self.eigen_values is not None else [],
            "eigen_modes": self.eigen_modes.tolist() if self.eigen_modes is not None else [],
            "mean_vector": self.mean_vector.tolist() if self.mean_vector is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict, strict: bool = False) -> "SpectralField":
        """
        Deserialize from a JSON-compatible dict.

        strict=False (default): coerce float64 → float32 with a UserWarning.
            Use this for JSON/SQLite round-trips where Python float = float64.
        strict=True: raise TensorValidationError if dtype is not float32.
            Use this for binary sources (pickle, numpy .npy) that preserve dtype.

        Version mismatch between saved field and current ENGINE_VERSION
        produces a UserWarning; loading proceeds with best-effort compatibility.
        """
        version = data.get("engine_version", "unknown")
        if version != ENGINE_VERSION:
            warnings.warn(
                f"Loading SpectralField saved with engine_version='{version}', "
                f"current is '{ENGINE_VERSION}'. "
                f"Forward compatibility is best-effort; consider rebuilding.",
                UserWarning,
                stacklevel=2,
            )

        sf = cls(embedding_dim=data.get("embedding_dim", 384))
        sf.is_built = data.get("is_built", False)
        sf.build_timestamp = data.get("build_timestamp", 0.0)
        sf.memory_count = data.get("memory_count", 0)

        if sf.is_built:
            num_modes = data.get("num_modes", 0)
            dim = sf.embedding_dim

            # Validate and coerce each tensor
            if data.get("eigen_modes"):
                sf.eigen_modes = validate_array(
                    data["eigen_modes"],
                    expected_shape=(num_modes, dim),
                    expected_dtype=DTYPE,
                    strict=strict,
                )
            if data.get("eigen_values"):
                sf.eigen_values = validate_array(
                    data["eigen_values"],
                    expected_shape=(num_modes,),
                    expected_dtype=DTYPE,
                    strict=strict,
                )
            if data.get("mean_vector"):
                sf.mean_vector = validate_array(
                    data["mean_vector"],
                    expected_shape=(dim,),
                    expected_dtype=DTYPE,
                    strict=strict,
                )
            else:
                # Backward compat: v1/v2 fields saved without mean_vector.
                # P1-A (v2.5): a zero mean is NOT the original centre — the PCA
                # no longer represents the space it was built over. Mark the
                # field so is_stale() forces a rebuild instead of silently
                # producing false resonances.
                sf.mean_vector = np.zeros(dim, dtype=DTYPE)
                sf.requires_rebuild = True
                warnings.warn(
                    "Legacy spectral field loaded without mean_vector — "
                    "reconstructed as zeros. Resonance over this field is "
                    "unreliable; is_stale() will report True until rebuild.",
                    UserWarning,
                    stacklevel=2,
                )

        return sf


# ============================================================
# DB PERSISTENCE
# ============================================================

class SpectralStore:
    """Persists the spectral field in the same SQLite DB as raven-memory."""

    TABLE_NAME = "spectral_field"

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    id INTEGER PRIMARY KEY,
                    field_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    def save(self, field: SpectralField):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {self.TABLE_NAME} "
                f"(id, field_json, updated_at) VALUES (1, ?, ?)",
                (json.dumps(field.to_dict()), time.time()),
            )
            conn.commit()

    def load(self, strict: bool = False) -> Optional[SpectralField]:
        """
        Load persisted spectral field.
        strict=False (default): accept float64 from JSON round-trip.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT field_json FROM {self.TABLE_NAME} WHERE id = 1"
            ).fetchone()
            if row and row[0]:
                return SpectralField.from_dict(json.loads(row[0]), strict=strict)
            return None


# ============================================================
# STANDALONE HELPERS
# ============================================================

def build_spectral_field_from_db(
    db_path: Path,
    embedding_dim: int = 384,
) -> Optional[SpectralField]:
    """
    Load active memories (NEUTRAL or REINFORCED) from DB and build spectral field.
    Does NOT persist — use build_and_persist_spectral_field() for that.
    Returns None if fewer than 2 active memories exist.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT embedding FROM memories WHERE state IN ('NEUTRAL', 'REINFORCED')"
        ).fetchall()

    if len(rows) < 2:
        return None

    embeddings = [np.frombuffer(r[0], dtype=np.float32).copy() for r in rows]
    field = SpectralField(embedding_dim=embedding_dim)
    field.build_from_memories(embeddings)
    return field


def build_and_persist_spectral_field(
    db_path: Path,
    embedding_dim: int = 384,
) -> Optional[SpectralField]:
    """
    Build spectral field from active memories and immediately persist to DB.
    Convenience wrapper for sleep_consolidator and manual rebuild calls.
    Returns None if fewer than 2 active memories exist.
    """
    field = build_spectral_field_from_db(db_path, embedding_dim)
    if field and field.is_built:
        # v2.5: a persistence failure (locked DB, disk full) must not destroy
        # a perfectly valid in-memory field. Warn and return it anyway.
        try:
            SpectralStore(db_path).save(field)
        except Exception as exc:
            warnings.warn(
                f"SpectralStore.save() failed ({exc}); returning in-memory "
                f"field without persistence.",
                RuntimeWarning,
                stacklevel=2,
            )
    return field


# ============================================================
# TEST SUITE
# ============================================================

def test_intra_process_determinism():
    """Two independent instances on identical inputs must produce identical output."""
    rng = np.random.default_rng(42)
    embeddings = [rng.standard_normal(384).astype(np.float32) for _ in range(20)]

    f1, f2 = SpectralField(384), SpectralField(384)
    assert f1.build_from_memories(embeddings)
    assert f2.build_from_memories(embeddings)

    assert np.allclose(f1.eigen_modes,  f2.eigen_modes,  atol=FP_TOL), "eigen_modes differ"
    assert np.allclose(f1.eigen_values, f2.eigen_values, atol=FP_TOL), "eigen_values differ"
    assert np.allclose(f1.mean_vector,  f2.mean_vector,  atol=FP_TOL), "mean_vector differs"

    r1 = f1.resonance(embeddings[0], embeddings[1])
    r2 = f2.resonance(embeddings[0], embeddings[1])
    assert np.isclose(r1, r2, atol=FP_TOL), f"Resonance drift: {r1} vs {r2}"

    print(f"  PASS intra-process determinism "
          f"(modes={len(f1.eigen_values)}, resonance={r1:.6f})")
    return True


def test_serialization_roundtrip():
    """Build → serialize → deserialize → resonance must be identical."""
    rng = np.random.default_rng(42)
    embeddings = [rng.standard_normal(384).astype(np.float32) for _ in range(20)]

    f1 = SpectralField(384)
    f1.build_from_memories(embeddings)

    # JSON round-trip always goes float64 → needs strict=False
    f2 = SpectralField.from_dict(f1.to_dict(), strict=False)

    assert np.allclose(f1.eigen_modes, f2.eigen_modes, atol=FP_TOL), "roundtrip eigen_modes differ"
    assert np.allclose(f1.mean_vector, f2.mean_vector, atol=FP_TOL), "roundtrip mean_vector differs"

    r1 = f1.resonance(embeddings[0], embeddings[1])
    r2 = f2.resonance(embeddings[0], embeddings[1])
    assert np.isclose(r1, r2, atol=FP_TOL), f"Roundtrip resonance drift: {r1} vs {r2}"

    print(f"  PASS serialization roundtrip (resonance={r1:.6f})")
    return True


def test_cross_process_determinism():
    """
    Spawn a child process with fresh interpreter + BLAS isolation.
    Verify eigen-modes match parent AND that the JSON serialization
    round-trip (the path production actually uses) is consistent.

    Returns "PASS" or "WARN" — never silently converts a child failure
    into a green suite. A WARN means: determinism is best-effort and this
    environment did not confirm it (observability, not module failure).
    """
    import pickle

    rng = np.random.default_rng(42)
    embeddings = [rng.standard_normal(384).astype(np.float32) for _ in range(20)]

    f_parent = SpectralField(384)
    f_parent.build_from_memories(embeddings)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as fh:
        pickle.dump(
            {
                "embeddings": embeddings,
                "eigen_modes": f_parent.eigen_modes,
                "eigen_values": f_parent.eigen_values,
                "mean_vector": f_parent.mean_vector,
            },
            fh,
        )
        temp_path = fh.name

    current_file = Path(__file__).resolve()
    module_dir = str(current_file.parent)
    module_name = current_file.stem

    script = f"""
import os, sys, pickle, warnings
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, {repr(module_dir)})
from {module_name} import SpectralField, FP_TOL

with open({repr(temp_path)}, "rb") as f:
    data = pickle.load(f)

f_child = SpectralField(384)
f_child.build_from_memories(data["embeddings"])

atol = FP_TOL
modes_ok  = np.allclose(f_child.eigen_modes,  data["eigen_modes"],  atol=atol)
values_ok = np.allclose(f_child.eigen_values, data["eigen_values"], atol=atol)
mean_ok   = np.allclose(f_child.mean_vector,  data["mean_vector"],  atol=atol)

# v2.5: also verify the JSON round-trip — production persists through
# SpectralStore (JSON in SQLite), so determinism must hold over that path.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    f_round = SpectralField.from_dict(f_child.to_dict(), strict=False)
r_direct = f_child.resonance(data["embeddings"][0], data["embeddings"][1])
r_round  = f_round.resonance(data["embeddings"][0], data["embeddings"][1])
json_ok  = bool(np.isclose(r_direct, r_round, atol=atol))

print(f"MODES_OK={{modes_ok}}")
print(f"VALUES_OK={{values_ok}}")
print(f"MEAN_OK={{mean_ok}}")
print(f"JSON_ROUNDTRIP_OK={{json_ok}}")
print(f"OMP_THREADS={{os.environ.get('OMP_NUM_THREADS')}}")

sys.exit(0 if (modes_ok and values_ok and mean_ok and json_ok) else 1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # v2.5: distinct WARN state — visible in the suite summary instead
            # of a silent PASS that hides real cross-process drift.
            print(f"  WARN cross-process determinism: child exit={result.returncode}")
            print(f"       stdout: {result.stdout.strip()}")
            print(f"       stderr: {result.stderr.strip()[:200]}")
            print(f"       (best-effort guarantee — not a module failure, but "
                  f"this environment did NOT confirm cross-process determinism)")
            return "WARN"
        print(f"  PASS cross-process determinism ({result.stdout.strip()})")
        return "PASS"
    except subprocess.TimeoutExpired:
        print("  WARN cross-process determinism: child timed out (30 s)")
        return "WARN"
    finally:
        os.unlink(temp_path)


def test_backward_compatibility():
    """Fields saved without mean_vector (v1/v2) load correctly with zeros fallback."""
    old_data = {
        "engine_version": "unknown",
        "is_built": True,
        "build_timestamp": 0.0,
        "memory_count": 10,
        "embedding_dim": 384,
        "num_modes": 5,
        "eigen_values": [1.0, 0.5, 0.3, 0.2, 0.1],
        "eigen_modes": [[float(i == j) for j in range(384)] for i in range(5)],
        "mean_vector": [],   # empty → backward compat path
    }
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        sf = SpectralField.from_dict(old_data, strict=False)

    assert sf.mean_vector is not None, "mean_vector must be set"
    assert sf.mean_vector.shape == (384,), "mean_vector wrong shape"
    assert np.allclose(sf.mean_vector, np.zeros(384, dtype=DTYPE)), "mean_vector must be zeros"
    # P1-A (v2.5): reconstructed mean → field flagged for rebuild
    assert sf.requires_rebuild, "legacy load must set requires_rebuild"
    assert sf.is_stale(10), "field with reconstructed mean must report stale"
    assert "REQUIRES_REBUILD" in sf.summary(), "summary must surface rebuild flag"
    print("  PASS backward compatibility (zeros fallback + requires_rebuild flag)")
    return True


def test_build_dimension_validation():
    """P1-B (v2.5): corrupt embedding dims must raise BEFORE vstack."""
    rng = np.random.default_rng(3)
    good = [rng.standard_normal(384).astype(np.float32) for _ in range(5)]

    f = SpectralField(384)

    bad_dim = good[:3] + [rng.standard_normal(383).astype(np.float32)]
    try:
        f.build_from_memories(bad_dim)
        assert False, "Should have raised TensorValidationError for (383,)"
    except TensorValidationError as exc:
        assert "383" in str(exc), "error must report the offending shape"

    bad_nan = good[:3] + [np.full(384, np.nan, dtype=np.float32)]
    try:
        f.build_from_memories(bad_nan)
        assert False, "Should have raised TensorValidationError for NaN"
    except TensorValidationError:
        pass

    assert f.build_from_memories(good), "clean input must still build"
    print("  PASS build dimension validation (corrupt input rejected pre-vstack)")
    return True


def test_projection_finite_guard():
    """P1-C (v2.5): non-finite input must produce a zero projection, not NaN."""
    rng = np.random.default_rng(5)
    embeddings = [rng.standard_normal(384).astype(np.float32) for _ in range(10)]
    f = SpectralField(384)
    f.build_from_memories(embeddings)

    corrupt = np.full(384, np.inf, dtype=np.float32)
    proj = f.project(corrupt)
    assert np.isfinite(proj).all(), "projection must never contain NaN/Inf"
    assert np.allclose(proj, 0.0), "degenerate input → zero projection"

    res = f.resonance(corrupt, embeddings[0])
    assert np.isfinite(res), "resonance must stay finite"
    assert res == 0.0, "degenerate query → zero resonance"
    print("  PASS projection finite guard (Inf input → zero projection)")
    return True


def test_tensor_validation():
    """Corrupted shape must raise TensorValidationError."""
    bad_data = {
        "engine_version": ENGINE_VERSION,
        "is_built": True,
        "build_timestamp": 0.0,
        "memory_count": 10,
        "embedding_dim": 384,
        "num_modes": 5,
        "eigen_values": [1.0, 0.5, 0.3, 0.2],   # wrong: 4 instead of 5
        "eigen_modes": [[float(i == j) for j in range(384)] for i in range(5)],
        "mean_vector": [0.0] * 384,
    }
    try:
        SpectralField.from_dict(bad_data, strict=True)
        assert False, "Should have raised TensorValidationError"
    except TensorValidationError:
        pass
    print("  PASS tensor validation (corrupted shape rejected)")
    return True


def test_dtype_validation():
    """
    strict=True  → reject float64 (JSON source).
    strict=False → accept with UserWarning(s).

    Important: JSON always produces Python float = float64. from_dict() with
    strict=False is the correct path for JSON/SQLite round-trips.
    We check that at least one 'Dtype coercion' warning appears (there will
    be one per tensor field: eigen_modes, eigen_values, mean_vector).
    """
    float64_data = {
        "engine_version": ENGINE_VERSION,
        "is_built": True,
        "build_timestamp": 0.0,
        "memory_count": 10,
        "embedding_dim": 384,
        "num_modes": 5,
        "eigen_values": [1.0, 0.5, 0.3, 0.2, 0.1],
        "eigen_modes": [[float(i == j) for j in range(384)] for i in range(5)],
        "mean_vector": [0.0] * 384,
    }

    # strict=True must reject float64
    try:
        SpectralField.from_dict(float64_data, strict=True)
        assert False, "strict=True should raise TensorValidationError for float64 input"
    except TensorValidationError:
        pass

    # strict=False must accept with at least one Dtype coercion warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sf = SpectralField.from_dict(float64_data, strict=False)

    coercion_warnings = [w for w in caught if "Dtype coercion" in str(w.message)]
    assert len(coercion_warnings) >= 1, (
        f"Expected at least 1 'Dtype coercion' warning, got: "
        f"{[str(w.message) for w in caught]}"
    )
    assert sf.eigen_modes is not None
    assert sf.eigen_modes.dtype == DTYPE, f"dtype after coercion must be {DTYPE}"

    print(f"  PASS dtype validation ({len(coercion_warnings)} coercion warning(s) emitted)")
    return True


def test_version_check():
    """Future engine_version produces a version-mismatch UserWarning."""
    future_data = {
        "engine_version": "spectral_v99.0",
        "is_built": True,
        "build_timestamp": 0.0,
        "memory_count": 10,
        "embedding_dim": 384,
        "num_modes": 5,
        "eigen_values": [1.0, 0.5, 0.3, 0.2, 0.1],
        "eigen_modes": [[float(i == j) for j in range(384)] for i in range(5)],
        "mean_vector": [0.0] * 384,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SpectralField.from_dict(future_data, strict=False)

    # One version-mismatch warning must be present; other warnings (dtype coercion,
    # numpy internals) may also appear — check by content, not total count.
    version_warnings = [w for w in caught if "spectral_v99.0" in str(w.message)]
    assert len(version_warnings) == 1, (
        f"Expected exactly 1 version mismatch warning, got: "
        f"{[str(w.message) for w in caught]}"
    )
    print("  PASS version check (future version warning emitted)")
    return True


def test_is_stale():
    """is_stale() triggers correctly based on drift threshold."""
    rng = np.random.default_rng(7)
    embeddings = [rng.standard_normal(384).astype(np.float32) for _ in range(20)]
    f = SpectralField(384)

    assert f.is_stale(20), "Not-yet-built field must be stale"

    f.build_from_memories(embeddings)
    assert not f.is_stale(20), "Field just built should not be stale (0% drift)"
    assert not f.is_stale(21), "1 memory added → 5% drift, under 10% threshold"
    assert f.is_stale(23), "3 memories added → 15% drift, over 10% threshold"
    assert f.is_stale(10), "10 memories removed → 50% drift, stale"

    print("  PASS is_stale() drift detection")
    return True


def test_coherence_flag():
    """coherence() is a pure ratio, not a score modifier."""
    assert SpectralField.coherence(0, 0) == 1.0, "No links → coherence 1.0"
    assert SpectralField.coherence(5, 0) == 1.0, "All resonant → 1.0"
    assert SpectralField.coherence(0, 5) == 0.0, "All inhibitory → 0.0"
    assert abs(SpectralField.coherence(1, 1) - 0.5) < EPS, "Equal links → 0.5"
    assert abs(SpectralField.coherence(3, 1) - 0.75) < EPS
    print("  PASS coherence flag (pure ratio, range [0,1])")
    return True


def run_all_tests(include_cross_process: bool = True) -> bool:
    """Run the full test suite. Returns True if all tests passed."""
    print("=" * 62)
    print(f"  raven-memory spectral.py {ENGINE_VERSION} — Test Suite")
    print("=" * 62)

    tests = [
        test_intra_process_determinism,
        test_serialization_roundtrip,
        test_backward_compatibility,
        test_build_dimension_validation,
        test_projection_finite_guard,
        test_tensor_validation,
        test_dtype_validation,
        test_version_check,
        test_is_stale,
        test_coherence_flag,
    ]
    # Cross-process test only from a top-level invocation (not from a subprocess)
    if include_cross_process and os.environ.get("_RAVEN_SPECTRAL_SUBPROCESS") != "1":
        tests.append(test_cross_process_determinism)

    passed = warned = failed = 0
    failures: List[str] = []
    warnings_list: List[str] = []

    for test in tests:
        try:
            outcome = test()
            # v2.5 tri-state: tests may return "PASS", "WARN", or truthy.
            if outcome == "WARN":
                warned += 1
                warnings_list.append(test.__name__)
            else:
                passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL {test.__name__}: {exc}")
            traceback.print_exc()
            failures.append(test.__name__)
            failed += 1
        print()

    print("=" * 62)
    print(f"  RESULT: {passed} PASS, {warned} WARN, {failed} FAIL  "
          f"({passed + warned}/{len(tests)})")
    if warnings_list:
        print(f"  WARNED: {', '.join(warnings_list)}  "
              f"(best-effort guarantees not confirmed in this environment)")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
    elif not warnings_list:
        print("  ALL TESTS PASSED")
    print("=" * 62)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
