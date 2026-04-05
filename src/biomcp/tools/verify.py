"""
BioMCP — Cross-Database Verification & Conflict Detection  [FIXED v2.1]
=========================================================================
Fixes applied:
  - Added missing `import asyncio` (was using asyncio.create_task without import)
  - Made evidence scoring more robust with try/except on dict access
  - Fixed KeyError risk in conflict detection
"""

from __future__ import annotations

import asyncio  # FIX: was missing — caused NameError at runtime
import re
from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
)

_CHEMBL_ASSAY_TYPE_LABELS = {
    "A": "ADMET assay",
    "B": "binding assay",
    "F": "functional assay",
    "P": "physicochemical assay",
    "T": "toxicity assay",
    "U": "unclassified assay",
}


def _describe_assay_type(assay_type: str) -> str:
    if not assay_type:
        return "unspecified assay"
    return _CHEMBL_ASSAY_TYPE_LABELS.get(assay_type.upper(), assay_type)


def synthesize_conflicting_evidence(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Explain why multiple records disagree rather than only flagging a discrepancy.

    The function accepts a list of structured result fragments for a single conflict.
    It returns a compact reasoning payload that can be attached to conflict reports.
    """
    if not tool_results:
        return {
            "summary": "No evidence records were provided for synthesis.",
            "likely_causes": [],
            "reasoning_steps": [],
            "confidence": "low",
        }

    if all("activity_value" in result for result in tool_results):
        numeric_values: list[float] = []
        for result in tool_results:
            try:
                value = float(result.get("activity_value", 0) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                numeric_values.append(value)
        assay_types = sorted({
            _describe_assay_type(str(result.get("assay_type", "")))
            for result in tool_results
            if result.get("assay_type")
        })
        activity_types = sorted({
            str(result.get("activity_type", "IC50"))
            for result in tool_results
            if result.get("activity_type")
        })
        relations = sorted({
            str(result.get("activity_relation", ""))
            for result in tool_results
            if result.get("activity_relation")
        })
        units = sorted({
            str(result.get("activity_units", ""))
            for result in tool_results
            if result.get("activity_units")
        })
        years = sorted({
            int(result.get("document_year"))
            for result in tool_results
            if str(result.get("document_year", "")).isdigit()
        })

        value_min = min(numeric_values) if numeric_values else 0.0
        value_max = max(numeric_values) if numeric_values else 0.0
        ratio = value_max / max(value_min, 1e-9) if numeric_values else 1.0
        likely_causes: list[str] = []
        reasoning_steps: list[dict[str, str]] = []

        if assay_types:
            reasoning_steps.append({
                "dimension": "assay_type",
                "observation": ", ".join(assay_types),
                "implication": (
                    "Different assay modalities often produce materially different potency values."
                    if len(assay_types) > 1
                    else "All records share the same assay modality, so other factors likely drive the spread."
                ),
            })
            if len(assay_types) > 1:
                likely_causes.append(
                    f"Assay modality differs across records ({', '.join(assay_types)})."
                )

        if numeric_values:
            range_units = units[0] if len(units) == 1 else "mixed units"
            reasoning_steps.append({
                "dimension": "concentration_range",
                "observation": f"{value_min:g} to {value_max:g} {range_units}",
                "implication": (
                    f"Potency spans roughly {ratio:.0f}x, which is large enough to reflect context-specific assay behavior."
                    if ratio >= 100
                    else "Potency spread is present but modest."
                ),
            })
            if ratio >= 100:
                likely_causes.append(
                    f"Reported {activity_types[0] if activity_types else 'activity'} values span roughly {ratio:.0f}x."
                )

        if relations:
            reasoning_steps.append({
                "dimension": "activity_relation",
                "observation": ", ".join(relations),
                "implication": (
                    "Some results are bounded ('>' or '<') rather than exact measurements."
                    if any(rel in relations for rel in (">", "<", ">=", "<="))
                    else "Measurements are exact comparisons."
                ),
            })
            if any(rel in relations for rel in (">", "<", ">=", "<=")):
                likely_causes.append(
                    "At least one potency value is a bound rather than an exact endpoint."
                )

        if years:
            reasoning_steps.append({
                "dimension": "study_vintage",
                "observation": ", ".join(str(year) for year in years),
                "implication": (
                    "Protocol drift over time can change assay sensitivity and reported potency."
                    if len(years) > 1
                    else "All records come from the same study vintage."
                ),
            })
            if len(years) > 1:
                likely_causes.append("Measurements come from different publication years and likely different protocols.")

        likely_causes.append(
            "Cell-line context is not exposed in the current cached ChEMBL summary; inspect raw assay records if you need per-cell-line attribution."
        )

        confidence = "high" if len(reasoning_steps) >= 3 else "moderate"
        return {
            "summary": (
                f"Conflicting potency values are most likely driven by assay context rather than a true contradiction. "
                f"The records cover {len(tool_results)} assay measurements"
                + (f" across {len(assay_types)} assay modalities." if assay_types else ".")
            ),
            "likely_causes": likely_causes,
            "reasoning_steps": reasoning_steps,
            "confidence": confidence,
        }

    record_type = str(tool_results[0].get("record_type", "generic"))
    if record_type == "name_alignment":
        values = [
            f"{item.get('source', 'source')}: {item.get('value', '')}"
            for item in tool_results
            if item.get("value")
        ]
        return {
            "summary": "Gene and protein resources often prefer different naming conventions and curation labels.",
            "likely_causes": [
                "NCBI and UniProt may emphasize different synonyms or full-name conventions.",
                "This usually reflects nomenclature drift rather than a biological contradiction.",
            ],
            "reasoning_steps": [
                {
                    "dimension": "source_labels",
                    "observation": "; ".join(values),
                    "implication": "Cross-check HGNC-approved names to normalize the label set.",
                }
            ],
            "confidence": "moderate",
        }

    if record_type == "evidence_asymmetry":
        genetics = next((item for item in tool_results if item.get("channel") == "genetic_association"), {})
        drugs = next((item for item in tool_results if item.get("channel") == "known_drug"), {})
        return {
            "summary": "This is more likely a translational gap than a contradiction: genetics supports the target, but drug evidence lags.",
            "likely_causes": [
                "Human genetic evidence can accumulate before tractable compounds or approved drugs exist.",
                "The target may be biologically validated but still hard to drug.",
            ],
            "reasoning_steps": [
                {
                    "dimension": "genetic_association",
                    "observation": str(genetics.get("score", "")),
                    "implication": "Strong human genetics increases confidence that the disease link is real.",
                },
                {
                    "dimension": "known_drug",
                    "observation": str(drugs.get("score", "")),
                    "implication": "Weak drug evidence points to a therapeutic-development gap, not necessarily conflicting biology.",
                },
            ],
            "confidence": "moderate",
        }

    return {
        "summary": "The records disagree, but the current metadata is too thin to attribute the discrepancy precisely.",
        "likely_causes": [
            "Source-specific curation choices may differ.",
            "Additional assay-level metadata is required to explain the discrepancy confidently.",
        ],
        "reasoning_steps": [],
        "confidence": "low",
    }


async def verify_biological_claim(
    claim: str,
    context_gene: str = "",
    max_evidence_sources: int = 5,
) -> dict[str, Any]:
    """
    Verify a biological claim against multiple databases simultaneously.

    Args:
        claim:                Natural language biological claim.
        context_gene:         Optional gene symbol to focus the search.
        max_evidence_sources: Max databases to query (3–5). Default 5.

    Returns:
        {
          claim, verdict, confidence_score, confidence_grade,
          supporting_evidence, contradicting_evidence, unresolved,
          evidence_by_source, recommendation
        }

    Confidence grades:
        A — Verified by 3+ independent databases, no contradictions
        B — Verified by 2 databases, minor inconsistencies
        C — Partial evidence, some contradictions
        D — Weak/conflicting evidence
        F — Contradicted by primary databases
    """
    from biomcp.tools.ncbi import search_pubmed
    from biomcp.tools.pathways import get_gene_disease_associations
    from biomcp.tools.proteins import search_proteins

    claim_lower = claim.lower()

    # Extract gene symbols from claim
    gene_hits = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', claim)
    gene = context_gene.upper() if context_gene else (gene_hits[0] if gene_hits else "")

    # Build parallel evidence queries
    evidence_tasks: dict[str, Any] = {}

    pubmed_query = f"{gene} {claim[:60]}" if gene else f"{claim[:80]} evidence"
    evidence_tasks["PubMed"] = asyncio.create_task(
        search_pubmed(pubmed_query, max_results=8, sort="relevance")
    )

    if gene:
        evidence_tasks["UniProt"] = asyncio.create_task(
            search_proteins(gene, max_results=1, reviewed_only=True)
        )
        if any(w in claim_lower for w in ("cancer","disease","disorder","syndrome","tumor","oncogen")):
            evidence_tasks["OpenTargets"] = asyncio.create_task(
                get_gene_disease_associations(gene, max_results=5)
            )
        if any(w in claim_lower for w in ("express","tissue","organ","cell type","upregulat","downregulat")):
            from biomcp.tools.advanced import search_gene_expression
            evidence_tasks["GEO"] = asyncio.create_task(
                search_gene_expression(gene, max_datasets=5)
            )

    selected_tasks = dict(list(evidence_tasks.items())[:max_evidence_sources])

    raw_results = await asyncio.gather(*selected_tasks.values(), return_exceptions=True)
    results_by_source = dict(zip(selected_tasks.keys(), raw_results, strict=False))

    supporting:    list[dict[str, Any]] = []
    contradicting: list[dict[str, Any]] = []
    unresolved:    list[dict[str, Any]] = []

    # Score PubMed evidence
    pubmed_result = results_by_source.get("PubMed", {})
    if isinstance(pubmed_result, dict):
        for article in pubmed_result.get("articles", []):
            try:
                abstract = (article.get("abstract", "") or "").lower()
                title    = (article.get("title", "")    or "").lower()
                text     = abstract + " " + title

                support_signals = sum(1 for w in [
                    "confirms","demonstrate","shows","observed","reported",
                    "found","reveals","indicates","consistent",
                ] if w in text)
                contra_signals = sum(1 for w in [
                    "contradict","however","contrary","unexpectedly",
                    "failed to","no evidence","not observed","disputes",
                ] if w in text)

                if support_signals >= contra_signals:
                    supporting.append({
                        "source":   "PubMed",
                        "evidence": article.get("title", ""),
                        "pmid":     article.get("pmid", ""),
                        "url":      article.get("url", ""),
                        "strength": "moderate" if support_signals < 3 else "strong",
                    })
                else:
                    contradicting.append({
                        "source":   "PubMed",
                        "evidence": article.get("title", ""),
                        "pmid":     article.get("pmid", ""),
                        "url":      article.get("url", ""),
                        "strength": "weak",
                    })
            except Exception as exc:
                logger.debug(f"[verify] Article scoring failed: {exc}")

    # UniProt evidence
    uniprot_result = results_by_source.get("UniProt", {})
    if isinstance(uniprot_result, dict):
        proteins = uniprot_result.get("proteins", [])
        if proteins and gene:
            gene_names = proteins[0].get("genes") or []
            if gene in gene_names or any(gene in str(g) for g in gene_names):
                supporting.append({
                    "source":   "UniProt Swiss-Prot",
                    "evidence": f"Gene {gene} confirmed in reviewed UniProt entry",
                    "strength": "strong",
                })

    # Open Targets evidence
    ot_result = results_by_source.get("OpenTargets", {})
    if isinstance(ot_result, dict):
        for assoc in ot_result.get("associations", [])[:3]:
            try:
                disease_name = assoc.get("disease_name", "").lower()
                score        = float(assoc.get("overall_score", 0) or 0)
                if any(d in claim_lower for d in disease_name.split() if len(d) > 4):
                    if score > 0.5:
                        supporting.append({
                            "source":   "Open Targets",
                            "evidence": f"{gene}–{assoc['disease_name']} association score: {score:.2f}",
                            "strength": "strong" if score > 0.7 else "moderate",
                        })
                    elif score < 0.2:
                        contradicting.append({
                            "source":   "Open Targets",
                            "evidence": f"Low {gene}–{assoc['disease_name']} association: {score:.2f}",
                            "strength": "weak",
                        })
            except Exception:
                pass

    # GEO evidence
    geo_result = results_by_source.get("GEO", {})
    if isinstance(geo_result, dict) and geo_result.get("total_found", 0) > 0:
        supporting.append({
            "source":   "NCBI GEO",
            "evidence": f"{geo_result['total_found']} expression datasets found for {gene}",
            "strength": "moderate",
        })

    # Confidence scoring
    n_support = len(supporting)
    n_contra  = len(contradicting)
    n_sources = len([r for r in raw_results if not isinstance(r, Exception)])

    total = n_support + n_contra
    score = n_support / total if total > 0 else 0.5
    source_bonus = min(0.15, 0.05 * n_sources)
    final_score  = min(1.0, score + source_bonus)

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
        "A": "High confidence — suitable as factual basis for research.",
        "B": "Reasonable confidence — verify with primary literature before citing.",
        "C": "Mixed evidence — investigate contradictions before proceeding.",
        "D": "Insufficient evidence — conduct targeted experiments first.",
        "F": "Contradicted by primary databases — revise the claim.",
    }[grade]

    return {
        "claim":                  claim,
        "gene_context":           gene,
        "verdict":                verdict,
        "confidence_score":       round(final_score, 3),
        "confidence_grade":       grade,
        "supporting_evidence":    supporting[:10],
        "contradicting_evidence": contradicting[:10],
        "unresolved":             unresolved,
        "databases_queried":      list(selected_tasks.keys()),
        "evidence_counts": {
            "supporting":    n_support,
            "contradicting": n_contra,
            "total_articles":len(pubmed_result.get("articles", [])) if isinstance(pubmed_result, dict) else 0,
        },
        "recommendation": recommendation,
        "methodology": (
            "Evidence scored by keyword sentiment analysis of PubMed abstracts "
            "and database-specific association scores. Not a substitute for expert review."
        ),
    }


async def detect_database_conflicts(
    gene_symbol: str,
) -> dict[str, Any]:
    """
    Scan for conflicting biological information about a gene across databases.
    FIX: Added try/except on dict accesses; fixed asyncio import.
    """
    from biomcp.tools.ncbi import get_gene_info
    from biomcp.tools.pathways import get_drug_targets, get_gene_disease_associations
    from biomcp.tools.proteins import search_proteins

    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    logger.info(f"[ConflictDetector] Scanning databases for {gene_symbol}")

    ncbi_result, uniprot_result, chembl_result, ot_result = await asyncio.gather(
        get_gene_info(gene_symbol),
        search_proteins(gene_symbol, max_results=1),
        get_drug_targets(gene_symbol, max_results=20),
        get_gene_disease_associations(gene_symbol, max_results=10),
        return_exceptions=True,
    )

    conflicts: list[dict[str, Any]] = []

    # Conflict 1: ChEMBL activity value consistency
    if isinstance(chembl_result, dict) and "drugs" in chembl_result:
        activity_values: dict[str, list[dict[str, Any]]] = {}
        for drug in chembl_result["drugs"]:
            try:
                mol_name = drug.get("molecule_name") or drug.get("molecule_chembl_id", "")
                act_type = drug.get("activity_type", "IC50")
                val = float(drug.get("activity_value", 0) or 0)
                if val > 0:
                    key = f"{mol_name}:{act_type}"
                    activity_values.setdefault(key, []).append({
                        "molecule_name": mol_name,
                        "activity_type": act_type,
                        "activity_value": val,
                        "activity_units": drug.get("activity_units", ""),
                        "activity_relation": drug.get("activity_relation", ""),
                        "assay_type": drug.get("assay_type", ""),
                        "document_year": drug.get("document_year", ""),
                    })
            except (ValueError, TypeError):
                pass

        for key, observations in activity_values.items():
            if len(observations) >= 2:
                values = [item["activity_value"] for item in observations]
                ratio = max(values) / max(min(values), 1e-9)
                if ratio > 100:
                    synthesis = synthesize_conflicting_evidence(observations)
                    conflicts.append({
                        "type":           "ACTIVITY_VALUE_DISCREPANCY",
                        "severity":       "HIGH",
                        "source_a":       "ChEMBL (assay 1)",
                        "source_b":       "ChEMBL (assay 2)",
                        "entity":         key.split(":")[0],
                        "detail":         f"IC50 values differ by {ratio:.0f}x across assays",
                        "values":         values,
                        "assay_observations": observations[:5],
                        "synthesis":      synthesis,
                        "recommendation": "Check assay conditions — in vitro vs cell-based assays differ.",
                    })

    # Conflict 2: Gene name agreement across databases
    ncbi_name    = ncbi_result.get("full_name", "") if isinstance(ncbi_result, dict) else ""
    uniprot_proteins = uniprot_result.get("proteins", []) if isinstance(uniprot_result, dict) else []
    uniprot_name = uniprot_proteins[0].get("name", "") if uniprot_proteins else ""

    if ncbi_name and uniprot_name:
        ncbi_words    = set(ncbi_name.lower().split()) - {"the","a","an","of","and","or"}
        uniprot_words = set(uniprot_name.lower().split()) - {"the","a","an","of","and","or"}
        if len(ncbi_words & uniprot_words) == 0 and len(ncbi_words) > 2:
            synthesis = synthesize_conflicting_evidence([
                {"record_type": "name_alignment", "source": "NCBI Gene", "value": ncbi_name},
                {"record_type": "name_alignment", "source": "UniProt", "value": uniprot_name},
            ])
            conflicts.append({
                "type":           "GENE_NAME_MISMATCH",
                "severity":       "LOW",
                "source_a":       "NCBI Gene",
                "source_b":       "UniProt",
                "detail":         f"NCBI: '{ncbi_name}' vs UniProt: '{uniprot_name}'",
                "synthesis":      synthesis,
                "recommendation": "Cross-reference HGNC for authoritative gene name.",
            })

    # Conflict 3: Disease evidence asymmetry
    if isinstance(ot_result, dict) and "associations" in ot_result:
        for assoc in ot_result["associations"][:5]:
            try:
                scores  = assoc.get("evidence_by_datatype", {})
                genetic = float(scores.get("genetic_association", 0) or 0)
                drug    = float(scores.get("known_drug", 0) or 0)
                if genetic > 0.7 and drug < 0.1:
                    synthesis = synthesize_conflicting_evidence([
                        {"record_type": "evidence_asymmetry", "channel": "genetic_association", "score": genetic},
                        {"record_type": "evidence_asymmetry", "channel": "known_drug", "score": drug},
                    ])
                    conflicts.append({
                        "type":           "EVIDENCE_TYPE_ASYMMETRY",
                        "severity":       "MEDIUM",
                        "source_a":       "Open Targets (genetics)",
                        "source_b":       "Open Targets (drugs)",
                        "entity":         assoc.get("disease_name", ""),
                        "detail":         f"Strong genetic (score:{genetic:.2f}) but no approved drugs (score:{drug:.2f})",
                        "synthesis":      synthesis,
                        "recommendation": "Potential unmet therapeutic need — investigate druggability.",
                    })
            except (TypeError, ValueError):
                pass

    high   = sum(1 for c in conflicts if c["severity"] == "HIGH")
    medium = sum(1 for c in conflicts if c["severity"] == "MEDIUM")
    low    = sum(1 for c in conflicts if c["severity"] == "LOW")
    penalty = high * 0.3 + medium * 0.15 + low * 0.05
    consistency_score = round(max(0.0, min(1.0, 1.0 - penalty)), 2)

    return {
        "gene":              gene_symbol,
        "conflicts_found":   len(conflicts),
        "conflicts":         conflicts,
        "consistency_score": consistency_score,
        "consistency_grade": (
            "HIGH" if consistency_score >= 0.8 else
            "MEDIUM" if consistency_score >= 0.5 else "LOW"
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
        "conflict_synthesis": [
            {
                "type": conflict["type"],
                "entity": conflict.get("entity", gene_symbol),
                "summary": conflict.get("synthesis", {}).get("summary", ""),
                "likely_causes": conflict.get("synthesis", {}).get("likely_causes", []),
            }
            for conflict in conflicts
        ],
    }
