"""
BioMCP — Biological Entity Resolver
=====================================
Resolves any biological identifier to its canonical cross-database form.

Problem: "p53", "TP53", "tumor protein p53", "P04637", "ENSG00000141510"
all refer to the same biological entity. Without resolution, every tool
call is a separate, disconnected lookup.

Solution: The EntityResolver maintains a unified canonical identity for
every biological entity encountered during the session, enabling:
  - Automatic deduplication across tool calls
  - Correct cross-referencing between databases
  - Enriched context from multiple ID systems

Architecture:
  Input: any identifier (symbol, accession, alias, common name)
      ↓
  HGNC + UniProt + Ensembl parallel resolution
      ↓
  Canonical BioEntity (all IDs unified)
      ↓
  Cached in EntityRegistry for session reuse
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Entity dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BioEntity:
    """
    Fully resolved biological entity with IDs from all major databases.
    Once resolved, this object can be used as input to any BioMCP tool
    without additional identifier lookups.
    """
    # ── Primary canonical form ──────────────────────────────────────────────
    canonical_symbol:   str            # HGNC-approved gene symbol (e.g. "TP53")
    canonical_name:     str            # Full name ("Tumor protein p53")
    entity_class:       str            # "gene" | "protein" | "drug" | "disease"

    # ── Cross-database identifiers ──────────────────────────────────────────
    hgnc_id:            str   = ""     # HGNC:11998
    ncbi_gene_id:       str   = ""     # 7157
    ensembl_gene_id:    str   = ""     # ENSG00000141510
    uniprot_accession:  str   = ""     # P04637
    refseq_mrna:        str   = ""     # NM_000546
    refseq_protein:     str   = ""     # NP_000537
    chembl_target_id:   str   = ""     # CHEMBL4523582
    omim_id:            str   = ""     # 191170
    pharmgkb_id:        str   = ""     # PA37140
    disgenet_id:        str   = ""

    # ── Aliases seen across databases ───────────────────────────────────────
    aliases:            list[str]  = field(default_factory=list)
    organism:           str        = "Homo sapiens"
    chromosome:         str        = ""
    resolution_sources: list[str]  = field(default_factory=list)
    confidence:         float      = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_symbol":  self.canonical_symbol,
            "canonical_name":    self.canonical_name,
            "entity_class":      self.entity_class,
            "cross_references": {
                "hgnc_id":           self.hgnc_id,
                "ncbi_gene_id":      self.ncbi_gene_id,
                "ensembl_gene_id":   self.ensembl_gene_id,
                "uniprot_accession": self.uniprot_accession,
                "refseq_mrna":       self.refseq_mrna,
                "refseq_protein":    self.refseq_protein,
                "chembl_target_id":  self.chembl_target_id,
                "omim_id":           self.omim_id,
            },
            "aliases":            self.aliases,
            "organism":           self.organism,
            "chromosome":         self.chromosome,
            "resolution_sources": self.resolution_sources,
            "confidence":         self.confidence,
        }

    def best_tool_inputs(self) -> dict[str, str]:
        """Return the best identifier to use for each BioMCP tool."""
        return {
            "search_pubmed":              self.canonical_symbol,
            "get_gene_info":             self.canonical_symbol,
            "get_protein_info":          self.uniprot_accession or self.canonical_symbol,
            "get_alphafold_structure":   self.uniprot_accession,
            "get_drug_targets":          self.canonical_symbol,
            "get_gene_disease_associations": self.canonical_symbol,
            "get_reactome_pathways":     self.canonical_symbol,
            "get_gene_variants":         self.canonical_symbol,
            "search_gene_expression":    self.canonical_symbol,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entity Registry — session-scoped cache
# ─────────────────────────────────────────────────────────────────────────────

class EntityRegistry:
    """
    Session-scoped registry of resolved biological entities.
    Thread-safe with asyncio.Lock.
    """

    def __init__(self) -> None:
        self._registry: dict[str, BioEntity] = {}   # canonical_symbol → entity
        self._alias_map: dict[str, str]       = {}   # any alias → canonical_symbol
        self._lock = asyncio.Lock()

    async def register(self, entity: BioEntity) -> None:
        async with self._lock:
            self._registry[entity.canonical_symbol] = entity
            # Register all known aliases
            for alias in [entity.canonical_symbol] + entity.aliases:
                self._alias_map[alias.lower().strip()] = entity.canonical_symbol
            if entity.ncbi_gene_id:
                self._alias_map[entity.ncbi_gene_id] = entity.canonical_symbol
            if entity.uniprot_accession:
                self._alias_map[entity.uniprot_accession.lower()] = entity.canonical_symbol
            if entity.ensembl_gene_id:
                self._alias_map[entity.ensembl_gene_id.lower()] = entity.canonical_symbol

    def lookup(self, query: str) -> BioEntity | None:
        """Case-insensitive lookup by any identifier or alias."""
        key = query.lower().strip()
        canonical = self._alias_map.get(key)
        return self._registry.get(canonical) if canonical else None

    def all_entities(self) -> list[BioEntity]:
        return list(self._registry.values())

    def summary(self) -> dict[str, Any]:
        return {
            "total_entities": len(self._registry),
            "entities": [
                {
                    "symbol": e.canonical_symbol,
                    "class":  e.entity_class,
                    "ids_resolved": sum(1 for x in [
                        e.ncbi_gene_id, e.ensembl_gene_id,
                        e.uniprot_accession, e.omim_id, e.chembl_target_id,
                    ] if x),
                }
                for e in self._registry.values()
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entity Resolver — the core resolution engine
# ─────────────────────────────────────────────────────────────────────────────

class EntityResolver:
    """
    Resolves biological identifiers to canonical cross-database entities.

    Resolution strategy (parallel):
      1. NCBI Gene E-utilities (authoritative for human genes)
      2. UniProt ID mapping API (protein cross-references)
      3. Ensembl xref API (genomic coordinates + IDs)
    """

    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    async def resolve(
        self,
        query: str,
        hint_type: str = "gene",    # "gene" | "protein" | "drug" | "disease"
    ) -> BioEntity:
        """
        Resolve a query string to a canonical BioEntity.
        Returns from cache if already resolved this session.
        """
        # ── Fast path: check registry first ──────────────────────────────────
        cached = self._registry.lookup(query)
        if cached:
            logger.debug(f"[EntityResolver] Cache hit: {query} → {cached.canonical_symbol}")
            return cached

        # ── Resolution pipeline (parallel) ───────────────────────────────────
        logger.info(f"[EntityResolver] Resolving '{query}' as {hint_type}")

        ncbi_task    = asyncio.create_task(self._resolve_via_ncbi(query))
        uniprot_task = asyncio.create_task(self._resolve_via_uniprot(query))
        ensembl_task = asyncio.create_task(self._resolve_via_ensembl(query))

        results = await asyncio.gather(
            ncbi_task, uniprot_task, ensembl_task,
            return_exceptions=True,
        )

        entity = _merge_resolution_results(query, hint_type, results)
        await self._registry.register(entity)
        logger.info(f"[EntityResolver] Resolved: {query} → {entity.canonical_symbol} "
                    f"(UniProt:{entity.uniprot_accession}, NCBI:{entity.ncbi_gene_id})")
        return entity

    async def resolve_batch(
        self,
        queries: list[str],
        hint_type: str = "gene",
    ) -> list[BioEntity]:
        """Resolve multiple identifiers in parallel."""
        tasks = [self.resolve(q, hint_type) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        entities = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[EntityResolver] Batch resolution failed: {r}")
            else:
                entities.append(r)
        return entities

    # ── Individual resolvers ──────────────────────────────────────────────────

    async def _resolve_via_ncbi(self, query: str) -> dict[str, Any]:
        """Resolve via NCBI Gene E-utilities."""
        try:
            from biomcp.utils import get_http_client, ncbi_params
            client = await get_http_client()

            # esearch
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=ncbi_params({
                    "db":   "gene",
                    "term": f"{query}[Gene Name] AND Homo sapiens[Organism]",
                    "retmax": 1,
                }),
            )
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return {}

            # esummary
            summ = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=ncbi_params({"db": "gene", "id": ids[0]}),
            )
            summ.raise_for_status()
            gd = summ.json().get("result", {}).get(ids[0], {})

            return {
                "source":         "NCBI Gene",
                "canonical_symbol": gd.get("name", query).upper(),
                "canonical_name": gd.get("description", ""),
                "ncbi_gene_id":   ids[0],
                "chromosome":     gd.get("chromosome", ""),
                "aliases":        [a.strip() for a in gd.get("otheraliases", "").split(",") if a.strip()],
                "organism":       gd.get("organism", {}).get("scientificname", "Homo sapiens"),
            }
        except Exception as exc:
            logger.debug(f"[EntityResolver] NCBI resolution failed for '{query}': {exc}")
            return {}

    async def _resolve_via_uniprot(self, query: str) -> dict[str, Any]:
        """Resolve via UniProt ID mapping API."""
        try:
            from biomcp.utils import get_http_client
            client = await get_http_client()

            resp = await client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query":  f"gene:{query} AND organism_id:9606 AND reviewed:true",
                    "format": "json",
                    "size":   1,
                    "fields": "accession,gene_names,protein_name,xref_refseq",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return {}

            entry = results[0]
            acc   = entry.get("primaryAccession", "")
            genes = entry.get("genes", [])
            gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""
            rec_name  = (entry.get("proteinDescription", {})
                         .get("recommendedName", {})
                         .get("fullName", {})
                         .get("value", ""))

            # Extract RefSeq cross-references
            refseq_mrna    = ""
            refseq_protein = ""
            for xref in entry.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "RefSeq":
                    for prop in xref.get("properties", []):
                        if "NM_" in xref.get("id", ""):
                            refseq_mrna = xref["id"]
                        if "NP_" in xref.get("id", ""):
                            refseq_protein = xref["id"]

            return {
                "source":             "UniProt",
                "canonical_symbol":   gene_name.upper() if gene_name else query.upper(),
                "canonical_name":     rec_name,
                "uniprot_accession":  acc,
                "refseq_mrna":        refseq_mrna,
                "refseq_protein":     refseq_protein,
            }
        except Exception as exc:
            logger.debug(f"[EntityResolver] UniProt resolution failed for '{query}': {exc}")
            return {}

    async def _resolve_via_ensembl(self, query: str) -> dict[str, Any]:
        """Resolve via Ensembl xref API."""
        try:
            from biomcp.utils import get_http_client
            client = await get_http_client()

            resp = await client.get(
                f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{query}",
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            hits = [e for e in resp.json() if e.get("type") == "gene"]
            if not hits:
                return {}

            ensembl_id = hits[0].get("id", "")

            # Get OMIM cross-reference from Ensembl
            xref_resp = await client.get(
                f"https://rest.ensembl.org/xrefs/id/{ensembl_id}",
                headers={"Accept": "application/json"},
                params={"external_db": "MIM_GENE"},
            )
            omim_id = ""
            if xref_resp.status_code == 200:
                for x in xref_resp.json():
                    if "MIM" in x.get("dbname", "") or "OMIM" in x.get("dbname", ""):
                        omim_id = x.get("primary_id", "")
                        break

            return {
                "source":          "Ensembl",
                "ensembl_gene_id": ensembl_id,
                "omim_id":         omim_id,
            }
        except Exception as exc:
            logger.debug(f"[EntityResolver] Ensembl resolution failed for '{query}': {exc}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Merge helper
# ─────────────────────────────────────────────────────────────────────────────

def _merge_resolution_results(
    query: str,
    hint_type: str,
    results: list[Any],
) -> BioEntity:
    """Merge results from multiple resolvers into one canonical BioEntity."""
    merged: dict[str, Any] = {}
    sources: list[str] = []

    for result in results:
        if isinstance(result, Exception) or not result:
            continue
        sources.append(result.get("source", ""))
        for key, val in result.items():
            if key == "source":
                continue
            if key == "aliases":
                merged.setdefault("aliases", [])
                merged["aliases"].extend(val)
            elif not merged.get(key):   # first non-empty value wins for scalars
                merged[key] = val

    # Deduplicate aliases
    aliases = list(set(merged.get("aliases", [])))

    # Fall back to sanitized query if no canonical symbol resolved
    canonical = (merged.get("canonical_symbol") or query).upper().strip()
    canonical = re.sub(r"[^A-Z0-9\-]", "", canonical) or query.upper()

    return BioEntity(
        canonical_symbol=canonical,
        canonical_name=merged.get("canonical_name", ""),
        entity_class=hint_type,
        ncbi_gene_id=merged.get("ncbi_gene_id", ""),
        ensembl_gene_id=merged.get("ensembl_gene_id", ""),
        uniprot_accession=merged.get("uniprot_accession", ""),
        refseq_mrna=merged.get("refseq_mrna", ""),
        refseq_protein=merged.get("refseq_protein", ""),
        omim_id=merged.get("omim_id", ""),
        chromosome=merged.get("chromosome", ""),
        aliases=aliases,
        organism=merged.get("organism", "Homo sapiens"),
        resolution_sources=[s for s in sources if s],
        confidence=min(1.0, 0.3 * len([s for s in sources if s])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: EntityRegistry | None = None
_RESOLVER: EntityResolver | None = None
_INIT_LOCK = asyncio.Lock()


async def get_resolver() -> EntityResolver:
    """Return the session-scoped entity resolver singleton."""
    global _REGISTRY, _RESOLVER
    if _RESOLVER is None:
        async with _INIT_LOCK:
            if _RESOLVER is None:
                _REGISTRY = EntityRegistry()
                _RESOLVER = EntityResolver(_REGISTRY)
    return _RESOLVER


async def get_registry() -> EntityRegistry:
    """Return the session-scoped entity registry."""
    await get_resolver()    # ensure init
    return _REGISTRY        # type: ignore[return-value]
