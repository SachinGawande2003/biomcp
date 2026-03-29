"""
BioMCP — Session Knowledge Graph (SKG)
=======================================
The most architecturally novel component of BioMCP.

Every tool call feeds into a live, in-memory biological knowledge graph
that persists for the lifetime of the MCP session. As Claude queries
more databases, the graph grows richer — enabling cross-database
connection discovery, contradiction detection, and compound insights
that no single tool call could produce.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │           Session Knowledge Graph (SKG)             │
  │                                                     │
  │  Gene ──TARGETS──► Drug ──TREATS──► Disease         │
  │   │                                    │            │
  │   └──IN_PATHWAY──► Pathway ◄──LINKED───┘            │
  │   │                                                 │
  │   └──EXPRESSED_IN──► Tissue ──HAS_VARIANT──► SNP    │
  └─────────────────────────────────────────────────────┘

Node types: Gene, Protein, Drug, Disease, Pathway, Tissue,
            Variant, ClinicalTrial, Publication, Organism

Edge types: TARGETS, TREATS, IN_PATHWAY, EXPRESSED_IN,
            ASSOCIATED_WITH, INTERACTS_WITH, HAS_VARIANT,
            PUBLISHED_IN, MENTIONED_IN, CONTRADICTS, CONFIRMS
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator
from uuid import uuid4

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Node and Edge Types
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    GENE          = "Gene"
    PROTEIN       = "Protein"
    DRUG          = "Drug"
    DISEASE       = "Disease"
    PATHWAY       = "Pathway"
    TISSUE        = "Tissue"
    VARIANT       = "Variant"
    CLINICAL_TRIAL= "ClinicalTrial"
    PUBLICATION   = "Publication"
    ORGANISM      = "Organism"


class EdgeType(str, Enum):
    TARGETS         = "TARGETS"
    TREATS          = "TREATS"
    IN_PATHWAY      = "IN_PATHWAY"
    EXPRESSED_IN    = "EXPRESSED_IN"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    INTERACTS_WITH  = "INTERACTS_WITH"
    HAS_VARIANT     = "HAS_VARIANT"
    PUBLISHED_IN    = "PUBLISHED_IN"
    MENTIONED_IN    = "MENTIONED_IN"
    CONTRADICTS     = "CONTRADICTS"
    CONFIRMS        = "CONFIRMS"
    ENCODES         = "ENCODES"
    PART_OF         = "PART_OF"


@dataclass
class SKGNode:
    """A node in the Session Knowledge Graph."""
    node_id:    str
    node_type:  NodeType
    label:      str                        # Human-readable name
    properties: dict[str, Any] = field(default_factory=dict)
    aliases:    list[str]      = field(default_factory=list)
    sources:    list[str]      = field(default_factory=list)  # which APIs provided this
    created_at: float          = field(default_factory=time.monotonic)
    confidence: float          = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":         self.node_id,
            "type":       self.node_type.value,
            "label":      self.label,
            "properties": self.properties,
            "aliases":    self.aliases,
            "sources":    self.sources,
            "confidence": self.confidence,
        }


@dataclass
class SKGEdge:
    """A directed edge in the Session Knowledge Graph."""
    edge_id:    str
    edge_type:  EdgeType
    source_id:  str
    target_id:  str
    properties: dict[str, Any] = field(default_factory=dict)
    evidence:   list[str]      = field(default_factory=list)
    score:      float          = 1.0
    created_at: float          = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":         self.edge_id,
            "type":       self.edge_type.value,
            "source":     self.source_id,
            "target":     self.target_id,
            "properties": self.properties,
            "evidence":   self.evidence,
            "score":      self.score,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Session Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

class SessionKnowledgeGraph:
    """
    Thread-safe in-memory biological knowledge graph for an MCP session.

    Automatically populated by tool calls via the @auto_index decorator.
    Supports:
      - Entity lookup by label, alias, or canonical ID
      - Path finding between biological entities
      - Contradiction detection across data sources
      - Cross-database connection discovery
      - Full session export with provenance
    """

    def __init__(self) -> None:
        self._nodes: dict[str, SKGNode] = {}
        self._edges: dict[str, SKGEdge] = {}
        self._lock  = asyncio.Lock()

        # Indices for fast lookup
        self._label_index:  dict[str, str]        = {}   # lowercase label → node_id
        self._alias_index:  dict[str, str]         = {}   # lowercase alias → node_id
        self._type_index:   dict[NodeType, set[str]] = defaultdict(set)
        self._adj_out:      dict[str, list[str]]   = defaultdict(list)  # node → edge_ids
        self._adj_in:       dict[str, list[str]]   = defaultdict(list)  # node ← edge_ids
        self._tool_calls:   list[dict[str, Any]]   = []

        logger.info("🧬 Session Knowledge Graph initialised")

    # ── Node operations ───────────────────────────────────────────────────────

    async def upsert_node(
        self,
        label:      str,
        node_type:  NodeType,
        properties: dict[str, Any] | None = None,
        aliases:    list[str]      | None = None,
        source:     str            = "unknown",
        confidence: float          = 1.0,
    ) -> SKGNode:
        """
        Add or update a node. Merges properties if node already exists.
        Returns the canonical node (existing or newly created).
        """
        async with self._lock:
            key = label.strip().lower()

            # Check if node exists by label or alias
            existing_id = (
                self._label_index.get(key)
                or next(
                    (self._alias_index[a.lower()] for a in (aliases or [])
                     if a.lower() in self._alias_index),
                    None,
                )
            )

            if existing_id and existing_id in self._nodes:
                node = self._nodes[existing_id]
                # Merge
                node.properties.update(properties or {})
                node.aliases = list(set(node.aliases + (aliases or [])))
                if source not in node.sources:
                    node.sources.append(source)
                node.confidence = max(node.confidence, confidence)
                for alias in (aliases or []):
                    self._alias_index[alias.lower()] = existing_id
                logger.debug(f"[SKG] Merged node: {label} ({node_type.value})")
                return node

            # Create new node
            node_id = f"{node_type.value}:{uuid4().hex[:8]}"
            node = SKGNode(
                node_id=node_id,
                node_type=node_type,
                label=label.strip(),
                properties=properties or {},
                aliases=aliases or [],
                sources=[source],
                confidence=confidence,
            )
            self._nodes[node_id] = node
            self._label_index[key] = node_id
            self._type_index[node_type].add(node_id)
            for alias in (aliases or []):
                self._alias_index[alias.lower()] = node_id

            logger.debug(f"[SKG] Added node: {label} ({node_type.value})")
            return node

    async def upsert_edge(
        self,
        source_label: str,
        source_type:  NodeType,
        edge_type:    EdgeType,
        target_label: str,
        target_type:  NodeType,
        properties:   dict[str, Any] | None = None,
        evidence:     list[str]      | None = None,
        score:        float          = 1.0,
        source:       str            = "unknown",
    ) -> SKGEdge | None:
        """
        Add a directed edge between two entities.
        Auto-creates nodes if they don't exist.
        """
        src_node = await self.upsert_node(source_label, source_type, source=source)
        tgt_node = await self.upsert_node(target_label, target_type, source=source)

        async with self._lock:
            # De-duplicate edges by (src, type, tgt)
            dup_key = f"{src_node.node_id}:{edge_type.value}:{tgt_node.node_id}"
            for eid in self._adj_out[src_node.node_id]:
                e = self._edges.get(eid)
                if e and e.edge_type == edge_type and e.target_id == tgt_node.node_id:
                    e.score = max(e.score, score)
                    e.evidence.extend(ev for ev in (evidence or []) if ev not in e.evidence)
                    e.properties.update(properties or {})
                    return e

            edge_id = f"E:{uuid4().hex[:8]}"
            edge = SKGEdge(
                edge_id=edge_id,
                edge_type=edge_type,
                source_id=src_node.node_id,
                target_id=tgt_node.node_id,
                properties=properties or {},
                evidence=evidence or [],
                score=score,
            )
            self._edges[edge_id] = edge
            self._adj_out[src_node.node_id].append(edge_id)
            self._adj_in[tgt_node.node_id].append(edge_id)

            logger.debug(f"[SKG] Edge: {source_label} --{edge_type.value}--> {target_label}")
            return edge

    # ── Query operations ──────────────────────────────────────────────────────

    def find_node(self, label: str) -> SKGNode | None:
        """Find a node by label or alias (case-insensitive)."""
        key = label.strip().lower()
        nid = self._label_index.get(key) or self._alias_index.get(key)
        return self._nodes.get(nid) if nid else None

    def get_neighbors(
        self,
        node_id:   str,
        edge_type: EdgeType | None = None,
        direction: str = "out",   # "out" | "in" | "both"
    ) -> list[tuple[SKGNode, SKGEdge]]:
        """Return neighboring nodes with their connecting edges."""
        results: list[tuple[SKGNode, SKGEdge]] = []
        edge_ids: list[str] = []

        if direction in ("out", "both"):
            edge_ids.extend(self._adj_out.get(node_id, []))
        if direction in ("in", "both"):
            edge_ids.extend(self._adj_in.get(node_id, []))

        for eid in edge_ids:
            edge = self._edges.get(eid)
            if not edge:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbor = self._nodes.get(neighbor_id)
            if neighbor:
                results.append((neighbor, edge))
        return results

    def find_paths(
        self,
        start_label: str,
        end_label:   str,
        max_depth:   int = 4,
    ) -> list[list[dict[str, Any]]]:
        """
        BFS path finding between two entities.
        Returns all paths up to max_depth hops.
        """
        start = self.find_node(start_label)
        end   = self.find_node(end_label)
        if not start or not end:
            return []

        paths: list[list[dict[str, Any]]] = []
        queue: list[tuple[str, list[dict]]] = [(start.node_id, [{"node": start.to_dict()}])]
        visited_per_path: list[set[str]] = [set()]

        while queue and len(paths) < 10:
            node_id, path = queue.pop(0)
            visited = visited_per_path[len(queue)]

            if node_id == end.node_id and len(path) > 1:
                paths.append(path)
                continue
            if len(path) > max_depth:
                continue

            for neighbor, edge in self.get_neighbors(node_id, direction="out"):
                if neighbor.node_id not in visited:
                    new_visited = visited | {node_id}
                    new_path    = path + [
                        {"edge": edge.to_dict()},
                        {"node": neighbor.to_dict()},
                    ]
                    queue.append((neighbor.node_id, new_path))
                    visited_per_path.append(new_visited)

        return paths

    def detect_contradictions(self) -> list[dict[str, Any]]:
        """
        Scan the graph for contradictory information across data sources.
        E.g. Drug X shows high affinity in ChEMBL but low in another source.
        """
        contradictions: list[dict[str, Any]] = []

        for node in self._nodes.values():
            if len(node.sources) < 2:
                continue

            # Check for conflicting property values across sources
            if "function" in node.properties and isinstance(node.properties.get("function"), list):
                funcs = node.properties["function"]
                if len(set(str(f)[:50] for f in funcs)) > 1:
                    contradictions.append({
                        "type":         "CONFLICTING_ANNOTATION",
                        "entity":       node.label,
                        "entity_type":  node.node_type.value,
                        "sources":      node.sources,
                        "details":      "Multiple functional annotations from different databases",
                        "severity":     "LOW",
                    })

        # Check for CONTRADICTS edges
        for edge in self._edges.values():
            if edge.edge_type == EdgeType.CONTRADICTS:
                src = self._nodes.get(edge.source_id)
                tgt = self._nodes.get(edge.target_id)
                if src and tgt:
                    contradictions.append({
                        "type":     "EXPLICIT_CONTRADICTION",
                        "entity_a": src.label,
                        "entity_b": tgt.label,
                        "evidence": edge.evidence,
                        "severity": "HIGH",
                    })

        return contradictions

    def find_unexpected_connections(self, min_path_length: int = 2) -> list[dict[str, Any]]:
        """
        Discover non-obvious multi-hop connections between entities.
        These are the biological insights no single tool call could surface.
        """
        connections: list[dict[str, Any]] = []
        node_list = list(self._nodes.values())

        # Look for entities connected through ≥2 intermediate nodes
        for i, node_a in enumerate(node_list[:20]):   # cap for performance
            for node_b in node_list[i + 1:20]:
                if node_a.node_type == node_b.node_type:
                    continue  # same type connections less interesting
                paths = self.find_paths(node_a.label, node_b.label, max_depth=3)
                for path in paths:
                    if len(path) >= min_path_length * 2 + 1:
                        connections.append({
                            "from":        node_a.label,
                            "from_type":   node_a.node_type.value,
                            "to":          node_b.label,
                            "to_type":     node_b.node_type.value,
                            "path_length": (len(path) - 1) // 2,
                            "path":        [
                                p.get("node", {}).get("label", "") or
                                p.get("edge", {}).get("type", "")
                                for p in path
                            ],
                            "insight": (
                                f"{node_a.label} and {node_b.label} are connected through "
                                f"{(len(path) - 1) // 2} intermediate biological entities"
                            ),
                        })
        return connections[:15]  # Top 15 most interesting

    def record_tool_call(self, tool_name: str, args: dict, result_summary: str) -> None:
        """Record a tool call for provenance tracking."""
        self._tool_calls.append({
            "tool":      tool_name,
            "args":      args,
            "summary":   result_summary,
            "timestamp": time.time(),
        })

    def snapshot(self) -> dict[str, Any]:
        """Return full graph snapshot for the get_session_knowledge_graph tool."""
        nodes_by_type: dict[str, list[dict]] = defaultdict(list)
        for node in self._nodes.values():
            nodes_by_type[node.node_type.value].append(node.to_dict())

        return {
            "summary": {
                "total_nodes":     len(self._nodes),
                "total_edges":     len(self._edges),
                "tool_calls_made": len(self._tool_calls),
                "node_type_counts": {k: len(v) for k, v in nodes_by_type.items()},
            },
            "nodes_by_type":          nodes_by_type,
            "edges":                  [e.to_dict() for e in self._edges.values()],
            "contradictions":         self.detect_contradictions(),
            "unexpected_connections": self.find_unexpected_connections(),
        }

    def export_provenance(self) -> dict[str, Any]:
        """
        Export full research session with provenance, citations, and
        reproducibility script — for use in publications.
        """
        all_sources: set[str] = set()
        for node in self._nodes.values():
            all_sources.update(node.sources)

        bibtex = _generate_bibtex(all_sources)

        return {
            "session_metadata": {
                "total_entities":     len(self._nodes),
                "total_relationships":len(self._edges),
                "data_sources_used":  sorted(all_sources),
                "tool_calls":         self._tool_calls,
            },
            "entities": {
                node.label: {
                    "type":     node.node_type.value,
                    "sources":  node.sources,
                    "key_facts": {k: v for k, v in node.properties.items() if k != "raw"},
                }
                for node in self._nodes.values()
            },
            "relationships": [e.to_dict() for e in self._edges.values()],
            "citations": {
                "bibtex":     bibtex,
                "data_access_date": time.strftime("%Y-%m-%d"),
                "fair_principles": {
                    "Findable":   "All entities include source database URLs",
                    "Accessible": "Data retrieved via public REST APIs",
                    "Interoperable": "IDs cross-referenced across databases",
                    "Reusable":   "Reproducibility script included below",
                },
            },
            "reproducibility_script": _generate_repro_script(self._tool_calls),
        }

    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "calls": len(self._tool_calls),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton — one graph per server process / session
# ─────────────────────────────────────────────────────────────────────────────

_SKG: SessionKnowledgeGraph | None = None
_SKG_LOCK = asyncio.Lock()


async def get_skg() -> SessionKnowledgeGraph:
    """Return the module-level session graph, creating it on first access."""
    global _SKG
    if _SKG is None:
        async with _SKG_LOCK:
            if _SKG is None:
                _SKG = SessionKnowledgeGraph()
    return _SKG


def reset_skg() -> None:
    """Reset the graph (useful for testing or explicit session reset)."""
    global _SKG
    _SKG = None


# ─────────────────────────────────────────────────────────────────────────────
# Auto-indexing decorator — attach to tool functions
# ─────────────────────────────────────────────────────────────────────────────

def auto_index(extractor_fn: Any) -> Any:
    """
    Decorator factory — automatically extracts entities from tool results
    and populates the Session Knowledge Graph.

    Usage::
        @auto_index(extract_pubmed_entities)
        async def search_pubmed(...): ...

    The extractor_fn receives (result: dict) and returns list of graph operations.
    """
    from functools import wraps

    def decorator(func: Any) -> Any:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            try:
                skg = await get_skg()
                await extractor_fn(skg, result, kwargs or {})
            except Exception as exc:
                logger.warning(f"[SKG] Auto-index failed for {func.__name__}: {exc}")
            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Entity extractors — one per tool category
# ─────────────────────────────────────────────────────────────────────────────

async def index_pubmed_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract entities from PubMed search results."""
    for article in result.get("articles", []):
        pmid = article.get("pmid", "")
        if not pmid:
            continue
        await skg.upsert_node(
            label=f"PMID:{pmid}",
            node_type=NodeType.PUBLICATION,
            properties={
                "title":   article.get("title", ""),
                "year":    article.get("year", ""),
                "journal": article.get("journal", ""),
                "doi":     article.get("doi", ""),
                "url":     article.get("url", ""),
            },
            source="PubMed",
        )
        # Index MeSH terms as potential disease/gene nodes
        for mesh in article.get("mesh_terms", []):
            if any(w in mesh.lower() for w in ("cancer","disease","disorder","syndrome","tumor")):
                await skg.upsert_node(mesh, NodeType.DISEASE, source="PubMed/MeSH")
            elif any(w in mesh.lower() for w in ("gene","protein","receptor","kinase","enzyme")):
                await skg.upsert_node(mesh, NodeType.PROTEIN, source="PubMed/MeSH")


async def index_gene_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract gene entity from NCBI Gene result."""
    if "error" in result:
        return
    symbol = result.get("symbol", "")
    if not symbol:
        return
    await skg.upsert_node(
        label=symbol,
        node_type=NodeType.GENE,
        properties={
            "ncbi_gene_id": result.get("gene_id", ""),
            "full_name":    result.get("full_name", ""),
            "chromosome":   result.get("chromosome", ""),
            "location":     result.get("location", ""),
            "summary":      result.get("summary", "")[:500],
        },
        aliases=result.get("aliases", []),
        source="NCBI Gene",
    )


async def index_protein_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract protein entity and link to diseases."""
    if "error" in result:
        return
    accession = result.get("accession", "")
    if not accession:
        return
    protein_node = await skg.upsert_node(
        label=accession,
        node_type=NodeType.PROTEIN,
        properties={
            "full_name":   result.get("full_name", ""),
            "function":    result.get("function", "")[:300],
            "sequence_length": result.get("sequence_length", 0),
        },
        aliases=result.get("gene_names", []),
        source="UniProt",
    )
    # Link gene → encodes → protein
    for gene_name in result.get("gene_names", []):
        await skg.upsert_edge(
            source_label=gene_name,
            source_type=NodeType.GENE,
            edge_type=EdgeType.ENCODES,
            target_label=accession,
            target_type=NodeType.PROTEIN,
            source="UniProt",
        )
    # Link protein → associated_with → disease
    for disease in result.get("diseases", []):
        disease_name = disease.get("name", "")
        if disease_name:
            await skg.upsert_edge(
                source_label=accession,
                source_type=NodeType.PROTEIN,
                edge_type=EdgeType.ASSOCIATED_WITH,
                target_label=disease_name,
                target_type=NodeType.DISEASE,
                source="UniProt",
            )


async def index_drug_targets_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract drug→gene relationships from ChEMBL result."""
    gene = result.get("gene", "")
    if not gene:
        return
    for drug in result.get("drugs", []):
        drug_name = drug.get("molecule_name") or drug.get("molecule_chembl_id", "")
        if not drug_name:
            continue
        await skg.upsert_edge(
            source_label=drug_name,
            source_type=NodeType.DRUG,
            edge_type=EdgeType.TARGETS,
            target_label=gene,
            target_type=NodeType.GENE,
            properties={
                "activity_type":  drug.get("activity_type", ""),
                "activity_value": drug.get("activity_value", ""),
                "activity_units": drug.get("activity_units", ""),
                "chembl_id":      drug.get("molecule_chembl_id", ""),
            },
            score=0.9,
            source="ChEMBL",
        )


async def index_disease_associations_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract gene→disease relationships from Open Targets."""
    gene = result.get("gene", "")
    if not gene:
        return
    for assoc in result.get("associations", []):
        disease_name = assoc.get("disease_name", "")
        if disease_name:
            await skg.upsert_edge(
                source_label=gene,
                source_type=NodeType.GENE,
                edge_type=EdgeType.ASSOCIATED_WITH,
                target_label=disease_name,
                target_type=NodeType.DISEASE,
                properties={"overall_score": assoc.get("overall_score", 0)},
                score=assoc.get("overall_score", 0),
                source="Open Targets",
            )


async def index_pathways_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract gene→pathway relationships from Reactome."""
    gene = result.get("gene", "")
    if not gene:
        return
    for pathway in result.get("pathways", [])[:20]:
        pathway_name = pathway.get("name", "")
        if pathway_name:
            await skg.upsert_edge(
                source_label=gene,
                source_type=NodeType.GENE,
                edge_type=EdgeType.IN_PATHWAY,
                target_label=pathway_name,
                target_type=NodeType.PATHWAY,
                properties={"reactome_id": pathway.get("reactome_id", "")},
                source="Reactome",
            )


async def index_clinical_trials_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract drug→disease relationships from clinical trials."""
    for study in result.get("studies", []):
        nct_id = study.get("nct_id", "")
        if not nct_id:
            continue
        await skg.upsert_node(
            label=nct_id,
            node_type=NodeType.CLINICAL_TRIAL,
            properties={
                "title":  study.get("title", ""),
                "phase":  study.get("phase", []),
                "status": study.get("status", ""),
            },
            source="ClinicalTrials.gov",
        )
        # Link interventions → treats → conditions
        for interv in study.get("interventions", []):
            drug_name = interv.get("name", "")
            if not drug_name:
                continue
            for condition in study.get("conditions", []):
                await skg.upsert_edge(
                    source_label=drug_name,
                    source_type=NodeType.DRUG,
                    edge_type=EdgeType.TREATS,
                    target_label=condition,
                    target_type=NodeType.DISEASE,
                    evidence=[nct_id],
                    source="ClinicalTrials.gov",
                )


async def index_variants_result(
    skg: SessionKnowledgeGraph,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Extract gene→variant relationships from Ensembl."""
    gene = result.get("gene", "")
    if not gene:
        return
    for variant in result.get("variants", [])[:30]:
        vid = variant.get("id", "")
        if vid:
            await skg.upsert_edge(
                source_label=gene,
                source_type=NodeType.GENE,
                edge_type=EdgeType.HAS_VARIANT,
                target_label=vid,
                target_type=NodeType.VARIANT,
                properties={
                    "consequence_types":    variant.get("consequence_types", []),
                    "clinical_significance":variant.get("clinical_significance", []),
                },
                source="Ensembl",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_BIBTEX = {
    "PubMed":          "@misc{pubmed, title={PubMed}, howpublished={\\url{https://pubmed.ncbi.nlm.nih.gov}}, note={Accessed {DATE}}}",
    "UniProt":         "@article{uniprot2023, title={UniProt: the Universal Protein Knowledgebase in 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1052}}",
    "NCBI Gene":       "@misc{ncbi_gene, title={NCBI Gene}, howpublished={\\url{https://www.ncbi.nlm.nih.gov/gene}}, note={Accessed {DATE}}}",
    "ChEMBL":          "@article{chembl2023, title={The ChEMBL Database in 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkad1004}}",
    "Reactome":        "@article{reactome2022, title={The Reactome Pathway Knowledgebase 2022}, journal={Nucleic Acids Research}, year={2022}, doi={10.1093/nar/gkab1028}}",
    "Open Targets":    "@article{opentargets2023, title={The Open Targets Platform: supporting systematic drug-target identification and prioritisation}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1046}}",
    "Ensembl":         "@article{ensembl2023, title={Ensembl 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac958}}",
    "ClinicalTrials.gov": "@misc{clinicaltrials, title={ClinicalTrials.gov}, howpublished={\\url{https://clinicaltrials.gov}}, note={Accessed {DATE}}}",
    "AlphaFold":       "@article{jumper2021, title={Highly accurate protein structure prediction with AlphaFold}, journal={Nature}, year={2021}, doi={10.1038/s41586-021-03819-2}}",
    "STRING":          "@article{string2023, title={The STRING database in 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1000}}",
    "GTEx":            "@article{gtex2020, title={The GTEx Consortium atlas of genetic regulatory effects across human tissues}, journal={Science}, year={2020}, doi={10.1126/science.aaz1776}}",
    "OMIM":            "@misc{omim, title={Online Mendelian Inheritance in Man (OMIM)}, howpublished={\\url{https://omim.org}}, note={Accessed {DATE}}}",
    "DisGeNET":        "@article{disgenet2020, title={The DisGeNET knowledge platform for disease genomics}, journal={Nucleic Acids Research}, year={2020}, doi={10.1093/nar/gkz1021}}",
    "GWAS Catalog":    "@article{gwas2023, title={The NHGRI-EBI GWAS Catalog}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1010}}",
    "cBioPortal":      "@article{cbioportal2023, title={The cBioPortal for Cancer Genomics}, journal={Cancer Discovery}, year={2023}, doi={10.1158/2159-8290.CD-23-0125}}",
    "PharmGKB":        "@article{pharmgkb2023, title={PharmGKB: the Pharmacogenomics Knowledgebase}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1009}}",
}


def _generate_bibtex(sources: set[str]) -> str:
    date_str = time.strftime("%Y-%m-%d")
    entries: list[str] = []
    for source in sorted(sources):
        template = _SOURCE_BIBTEX.get(source)
        if template:
            entries.append(template.replace("{DATE}", date_str))
    return "\n\n".join(entries)


def _generate_repro_script(tool_calls: list[dict]) -> str:
    lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated BioMCP reproducibility script."""',
        "# Re-run all tool calls from your research session",
        "# Install: pip install biomcp",
        "# Run: python repro_script.py",
        "",
        "import asyncio",
        "from biomcp.tools import ncbi, proteins, pathways, advanced",
        "",
        "async def main():",
    ]
    for call in tool_calls:
        tool = call.get("tool", "")
        args = call.get("args", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        lines.append(f"    # {call.get('summary', '')}")
        lines.append(f"    await {tool}({args_str})")
        lines.append("")
    lines.append("asyncio.run(main())")
    return "\n".join(lines)
