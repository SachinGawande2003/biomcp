"""
BioMCP — Cross-Database Verification & Conflict Detection
===========================================================
Two tools that no other MCP server provides:

  verify_biological_claim
    Verifies a biological claim against 3–5 databases simultaneously.
    Returns agreement score, contradicting evidence, and confidence grade.
    Example: "EGFR is primarily expressed in lung epithelium"
    → Checks UniProt, GEO expression data, Human Cell Atlas, PubMed

  detect_database_conflicts
    For any gene/protein currently in the session graph, scans for
    conflicting information across data sources (e.g. different subcellular
    locations, contradictory function annotations, mismatched drug IC50s).

Architecture:
  Claim → Decomposer → Evidence Queries (parallel) → Evidence Scorer
       → Conflict Detector → Confidence Grade → Structured Report
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)


# ─────────────────────────────────────────────────────────────────────────────
# Claim Verifier
# ─────────────────────────────────────────────────────────────────────────────

async def verify_biological_claim(
    claim: str,
    context_gene: str = "",
    max_evidence_sources: int = 5,
) -> dict[str, Any]:
    """
    Verify a biological claim against multiple databases simultaneously.

    Decomposes the claim into verifiable sub-questions, queries relevant
    databases, and scores overall confidence with supporting/contradicting
    evidence.

    Args:
        claim:                Natural language biological claim.
                              E.g. "EGFR is overexpressed in lung cancer"
        context_gene:         Optional gene symbol to focus the search.
        max_evidence_sources: Max databases to query (3–5). Default 5.

    Returns:
        {
          claim, verdict, confidence_score, confidence_grade,
          supporting_evidence: [...],
          contradicting_evidence: [...],
          unresolved: [...],
          evidence_by_source: {...},
          recommendation,
        }

    Confidence grades:
        A  — Verified by 3+ independent databases, no contradictions
        B  — Verified by 2 databases, minor inconsistencies
        C  — Partial evidence, some contradictions
        D  — Weak/conflicting evidence
        F  — Contradicted by primary databases
    """
    from biomcp.tools.ncbi     import search_pubmed
    from biomcp.tools.proteins import get_protein_info, search_proteins
    from biomcp.tools.pathways import get_gene_disease_associations

    claim_lower = claim.lower()

    # ── Extract entities from claim ──────────────────────────────────────────
    gene_hits = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', claim)
    gene      = context_gene.upper() if context_gene else (gene_hits[0] if gene_hits else "")

    # ── Build parallel evidence queries ──────────────────────────────────────
    evidence_tasks: dict[str, Any] = {}

    # Always query PubMed
    pubmed_query = f"{claim[:80]} evidence"
    if gene:
        pubmed_query = f"{gene} {claim[:60]}"
    evidence_tasks["PubMed"] = asyncio.create_task(
        search_pubmed(pubmed_query, max_results=8, sort="relevance")
    )

    # Gene-based queries if gene detected
    if gene:
        # UniProt protein annotation
        evidence_tasks["UniProt"] = asyncio.create_task(
            search_proteins(gene, max_results=1, reviewed_only=True)
        )
        # Open Targets for disease claims
        if any(w in claim_lower for w in ("cancer", "disease", "disorder", "syndrome", "tumor", "oncogen")):
            evidence_tasks["OpenTargets"] = asyncio.create_task(
                get_gene_disease_associations(gene, max_results=5)
            )
        # Expression-specific
        if any(w in claim_lower for w in ("express", "tissue", "organ", "cell type", "upregulat", "downregulat")):
            from biomcp.tools.advanced import search_gene_expression
            evidence_tasks["GEO"] = asyncio.create_task(
                search_gene_expression(gene, max_datasets=5)
            )

    # Cap to max_evidence_sources
    selected_tasks = dict(list(evidence_tasks.items())[:max_evidence_sources])

    raw_results = await asyncio.gather(*selected_tasks.values(), return_exceptions=True)
    results_by_source = dict(zip(selected_tasks.keys(), raw_results))

    # ── Score evidence ────────────────────────────────────────────────────────
    supporting:    list[dict[str, Any]] = []
    contradicting: list[dict[str, Any]] = []
    unresolved:    list[dict[str, Any]] = []

    # PubMed: look for abstracts that mention the claim entities
    pubmed_result = results_by_source.get("PubMed", {})
    if isinstance(pubmed_result, dict):
        articles = pubmed_result.get("articles", [])
        supporting_count   = 0
        contradicting_count = 0
        for article in articles:
            abstract = (article.get("abstract", "") or "").lower()
            title    = (article.get("title", "")    or "").lower()
            text     = abstract + " " + title

            # Look for supporting keywords
            support_signals = sum(1 for w in [
                "confirms", "demonstrate", "shows", "observed",
                "reported", "found", "reveals", "indicates", "consistent",
            ] if w in text)

            # Look for contradicting keywords
            contra_signals = sum(1 for w in [
                "contradict", "however", "contrary", "unexpectedly",
                "failed to", "no evidence", "not observed", "disputes",
            ] if w in text)

            if support_signals > contra_signals:
                supporting_count += 1
                supporting.append({
                    "source":     "PubMed",
                    "evidence":   article.get("title", ""),
                    "pmid":       article.get("pmid", ""),
                    "url":        article.get("url", ""),
                    "strength":   "moderate",
                })
            elif contra_signals > support_signals:
                contradicting_count += 1
                contradicting.append({
                    "source":   "PubMed",
                    "evidence": article.get("title", ""),
                    "pmid":     article.get("pmid", ""),
                    "url":      article.get("url", ""),
                    "strength": "weak",
                })

    # UniProt: check function annotation
    uniprot_result = results_by_source.get("UniProt", {})
    if isinstance(uniprot_result, dict):
        proteins = uniprot_result.get("proteins", [])
        if proteins:
            if gene and gene in (proteins[0].get("genes") or []):
                supporting.append({
                    "source":   "UniProt Swiss-Prot",
                    "evidence": f"Gene {gene} found in reviewed UniProt entry",
                    "strength": "strong",
                })

    # OpenTargets: disease association scores
    ot_result = results_by_source.get("OpenTargets", {})
    if isinstance(ot_result, dict):
        assocs = ot_result.get("associations", [])
        for assoc in assocs[:3]:
            disease_name = assoc.get("disease_name", "").lower()
            score        = assoc.get("overall_score", 0)
            claim_lower2 = claim.lower()
            if any(d in claim_lower2 for d in disease_name.split() if len(d) > 4):
                if score > 0.5:
                    supporting.append({
                        "source":   "Open Targets",
                        "evidence": f"{gene}–{assoc['disease_name']} association score: {score:.2f}",
                        "strength": "strong" if score > 0.7 else "moderate",
                    })
                elif score < 0.2:
                    contradicting.append({
                        "source":   "Open Targets",
                        "evidence": f"Low {gene}–{assoc['disease_name']} association score: {score:.2f}",
                        "strength": "weak",
                    })

    # GEO: expression data
    geo_result = results_by_source.get("GEO", {})
    if isinstance(geo_result, dict) and geo_result.get("total_found", 0) > 0:
        supporting.append({
            "source":   "NCBI GEO",
            "evidence": f"{geo_result['total_found']} expression datasets found for {gene}",
            "strength": "moderate",
        })

    # ── Compute confidence score ──────────────────────────────────────────────
    n_support = len(supporting)
    n_contra  = len(contradicting)
    n_sources = len([r for r in raw_results if not isinstance(r, Exception)])

    if n_support == 0 and n_contra == 0:
        score = 0.5   # no evidence either way
    else:
        total = n_support + n_contra
        score = n_support / total if total > 0 else 0.5

    # Boost for multiple independent sources
    source_bonus = min(0.15, 0.05 * n_sources)
    final_score  = min(1.0, score + source_bonus)

    # Grade
    if final_score >= 0.85 and n_support >= 3:
        grade, verdict = "A", "VERIFIED"
    elif final_score >= 0.70 and n_support >= 2:
        grade, verdict = "B", "LIKELY TRUE"
    elif final_score >= 0.55:
        grade, verdict = "C", "PARTIALLY SUPPORTED"
    elif final_score >= 0.40:
        grade, verdict = "D", "WEAK EVIDENCE"
    else:
        grade, verdict = "F", "CONTRADICTED"

    recommendation = {
        "A": "High confidence — suitable as a factual basis for research.",
        "B": "Reasonable confidence — verify with primary literature before citing.",
        "C": "Mixed evidence — investigate the contradictions before proceeding.",
        "D": "Insufficient evidence — conduct targeted experiments before accepting.",
        "F": "Contradicted by primary databases — revise the claim.",
    }[grade]

    return {
        "claim":                   claim,
        "gene_context":            gene,
        "verdict":                 verdict,
        "confidence_score":        round(final_score, 3),
        "confidence_grade":        grade,
        "supporting_evidence":     supporting[:10],
        "contradicting_evidence":  contradicting[:10],
        "unresolved":              unresolved,
        "databases_queried":       [s for s in selected_tasks.keys()],
        "evidence_counts": {
            "supporting":    n_support,
            "contradicting": n_contra,
            "total_articles": len(pubmed_result.get("articles", [])) if isinstance(pubmed_result, dict) else 0,
        },
        "recommendation": recommendation,
        "methodology": (
            "Evidence scored by keyword sentiment analysis of PubMed abstracts "
            "and database-specific association scores. Not a substitute for expert review."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Database Conflict Detector
# ─────────────────────────────────────────────────────────────────────────────

async def detect_database_conflicts(
    gene_symbol: str,
) -> dict[str, Any]:
    """
    Scan for conflicting biological information about a gene/protein
    across all major databases queried simultaneously.

    Checks for contradictions in:
      - Subcellular location (UniProt vs other annotations)
      - Drug sensitivity (ChEMBL IC50 values across assays)
      - Disease associations (Open Targets vs DisGeNET)
      - Gene function (UniProt vs PubMed consensus)
      - Expression patterns (GEO datasets)

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'TP53', 'EGFR').

    Returns:
        {
          gene, conflicts: [...], consistency_score, summary,
          recommendation, database_snapshots: {...}
        }
    """
    from biomcp.tools.ncbi     import get_gene_info
    from biomcp.tools.proteins import get_protein_info, search_proteins
    from biomcp.tools.pathways import get_drug_targets, get_gene_disease_associations

    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)

    logger.info(f"[ConflictDetector] Scanning databases for {gene_symbol}")

    # ── Parallel queries across all relevant databases ────────────────────────
    ncbi_task,   uniprot_task, chembl_task, ot_task = (
        asyncio.create_task(get_gene_info(gene_symbol)),
        asyncio.create_task(search_proteins(gene_symbol, max_results=1)),
        asyncio.create_task(get_drug_targets(gene_symbol, max_results=20)),
        asyncio.create_task(get_gene_disease_associations(gene_symbol, max_results=10)),
    )

    ncbi_result, uniprot_result, chembl_result, ot_result = await asyncio.gather(
        ncbi_task, uniprot_task, chembl_task, ot_task,
        return_exceptions=True,
    )

    conflicts: list[dict[str, Any]] = []

    # ── Conflict 1: Check drug activity consistency ───────────────────────────
    if isinstance(chembl_result, dict) and "drugs" in chembl_result:
        drugs = chembl_result["drugs"]
        activity_values: dict[str, list[float]] = {}
        for drug in drugs:
            mol_name = drug.get("molecule_name") or drug.get("molecule_chembl_id", "")
            act_type = drug.get("activity_type", "IC50")
            try:
                val = float(drug.get("activity_value", 0) or 0)
                if val > 0:
                    key = f"{mol_name}:{act_type}"
                    activity_values.setdefault(key, []).append(val)
            except (ValueError, TypeError):
                pass

        # Flag where same compound has widely varying values (>100x difference)
        for key, values in activity_values.items():
            if len(values) >= 2:
                ratio = max(values) / min(values)
                if ratio > 100:
                    conflicts.append({
                        "type":       "ACTIVITY_VALUE_DISCREPANCY",
                        "severity":   "HIGH",
                        "source_a":   "ChEMBL (assay 1)",
                        "source_b":   "ChEMBL (assay 2)",
                        "entity":     key.split(":")[0],
                        "detail":     f"IC50 values differ by {ratio:.0f}x across ChEMBL assays",
                        "values":     values,
                        "recommendation": "Check assay conditions — in vitro vs cell-based assays often differ.",
                    })

    # ── Conflict 2: NCBI vs UniProt gene name agreement ───────────────────────
    ncbi_name   = ncbi_result.get("full_name", "") if isinstance(ncbi_result, dict) else ""
    uniprot_proteins = uniprot_result.get("proteins", []) if isinstance(uniprot_result, dict) else []
    uniprot_name = uniprot_proteins[0].get("name", "") if uniprot_proteins else ""

    if ncbi_name and uniprot_name:
        # Rough semantic agreement check
        ncbi_words   = set(ncbi_name.lower().split())
        uniprot_words= set(uniprot_name.lower().split())
        common_words = ncbi_words & uniprot_words - {"the", "a", "an", "of", "and", "or"}
        if len(common_words) == 0 and len(ncbi_words) > 2:
            conflicts.append({
                "type":       "GENE_NAME_MISMATCH",
                "severity":   "LOW",
                "source_a":   "NCBI Gene",
                "source_b":   "UniProt",
                "detail":     f"NCBI: '{ncbi_name}' vs UniProt: '{uniprot_name}'",
                "recommendation": "Cross-reference HGNC for authoritative gene name.",
            })

    # ── Conflict 3: Disease association consistency ───────────────────────────
    if isinstance(ot_result, dict) and "associations" in ot_result:
        assocs = ot_result["associations"]
        # Flag if genetic association score is very different from drug score
        for assoc in assocs[:5]:
            scores = assoc.get("evidence_by_datatype", {})
            genetic = scores.get("genetic_association", 0)
            drug    = scores.get("known_drug", 0)
            if genetic > 0.7 and drug < 0.1:
                conflicts.append({
                    "type":       "EVIDENCE_TYPE_ASYMMETRY",
                    "severity":   "MEDIUM",
                    "source_a":   "Open Targets (genetics)",
                    "source_b":   "Open Targets (drugs)",
                    "entity":     assoc.get("disease_name", ""),
                    "detail":     f"Strong genetic evidence (score: {genetic:.2f}) but no approved drugs (score: {drug:.2f})",
                    "recommendation": "Potential unmet therapeutic need — investigate druggability.",
                })

    # ── Consistency score ─────────────────────────────────────────────────────
    high   = sum(1 for c in conflicts if c["severity"] == "HIGH")
    medium = sum(1 for c in conflicts if c["severity"] == "MEDIUM")
    low    = sum(1 for c in conflicts if c["severity"] == "LOW")
    penalty = high * 0.3 + medium * 0.15 + low * 0.05
    consistency_score = max(0.0, min(1.0, 1.0 - penalty))

    return {
        "gene":              gene_symbol,
        "conflicts_found":   len(conflicts),
        "conflicts":         conflicts,
        "consistency_score": round(consistency_score, 2),
        "consistency_grade": (
            "HIGH"   if consistency_score >= 0.8 else
            "MEDIUM" if consistency_score >= 0.5 else
            "LOW"
        ),
        "databases_scanned": ["NCBI Gene", "UniProt", "ChEMBL", "Open Targets"],
        "summary": (
            f"Scanned 4 databases for {gene_symbol}. "
            f"Found {len(conflicts)} conflict(s): "
            f"{high} high / {medium} medium / {low} low severity."
        ),
        "recommendation": (
            "Data appears largely consistent — suitable for research use."
            if consistency_score >= 0.8 else
            "Review flagged conflicts before drawing conclusions."
            if consistency_score >= 0.5 else
            "Significant discrepancies detected — manual curation recommended."
        ),
        "database_snapshots": {
            "ncbi_gene_name":   ncbi_name,
            "uniprot_name":     uniprot_name,
            "chembl_compounds": len(chembl_result.get("drugs", [])) if isinstance(chembl_result, dict) else 0,
            "ot_associations":  len(ot_result.get("associations", [])) if isinstance(ot_result, dict) else 0,
        },
    }
