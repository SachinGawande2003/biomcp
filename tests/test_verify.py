from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_verify_biological_claim_parses_pubmed_sentiment():
    pubmed_result = {
        "articles": [
            {
                "title": "Study confirms EGFR signaling in lung cancer",
                "abstract": "This study demonstrates and confirms strong EGFR-dependent tumor growth.",
                "pmid": "1001",
                "url": "https://pubmed.ncbi.nlm.nih.gov/1001/",
            },
            {
                "title": "Unexpected EGFR finding",
                "abstract": "However, no evidence of the claimed association was observed in this cohort.",
                "pmid": "1002",
                "url": "https://pubmed.ncbi.nlm.nih.gov/1002/",
            },
        ]
    }
    protein_result = {"proteins": [{"genes": ["EGFR"]}]}
    association_result = {
        "associations": [
            {"disease_name": "lung cancer", "overall_score": 0.81},
        ]
    }

    with (
        patch("biomcp.tools.ncbi.search_pubmed", new=AsyncMock(return_value=pubmed_result)),
        patch("biomcp.tools.proteins.search_proteins", new=AsyncMock(return_value=protein_result)),
        patch(
            "biomcp.tools.pathways.get_gene_disease_associations",
            new=AsyncMock(return_value=association_result),
        ),
    ):
        from biomcp.tools.verify import verify_biological_claim

        result = await verify_biological_claim(
            "EGFR drives lung cancer progression",
            context_gene="EGFR",
        )

    assert result["gene_context"] == "EGFR"
    assert result["verdict"] == "VERIFIED"
    assert result["evidence_counts"]["supporting"] == 3
    assert result["evidence_counts"]["contradicting"] == 1
    assert result["supporting_evidence"][0]["source"] == "PubMed"
    assert any(item["source"] == "UniProt Swiss-Prot" for item in result["supporting_evidence"])
    assert any(item["source"] == "Open Targets" for item in result["supporting_evidence"])
    assert result["contradicting_evidence"][0]["source"] == "PubMed"


def test_synthesize_conflicting_evidence_explains_activity_spread():
    from biomcp.tools.verify import synthesize_conflicting_evidence

    synthesis = synthesize_conflicting_evidence([
        {
            "molecule_name": "Osimertinib",
            "activity_type": "IC50",
            "activity_value": 0.8,
            "activity_units": "nM",
            "activity_relation": "=",
            "assay_type": "B",
            "document_year": 2018,
        },
        {
            "molecule_name": "Osimertinib",
            "activity_type": "IC50",
            "activity_value": 1250.0,
            "activity_units": "nM",
            "activity_relation": ">",
            "assay_type": "F",
            "document_year": 2023,
        },
    ])

    assert "assay context" in synthesis["summary"].lower()
    assert any("assay modality differs" in cause.lower() for cause in synthesis["likely_causes"])
    assert any(step["dimension"] == "concentration_range" for step in synthesis["reasoning_steps"])
    assert synthesis["confidence"] in {"moderate", "high"}


@pytest.mark.asyncio
async def test_detect_database_conflicts_includes_synthesized_reasoning():
    ncbi_result = {"full_name": "epidermal growth factor receptor"}
    uniprot_result = {"proteins": [{"name": "epidermal growth factor receptor", "genes": ["EGFR"]}]}
    chembl_result = {
        "drugs": [
            {
                "molecule_name": "Osimertinib",
                "activity_type": "IC50",
                "activity_value": 0.8,
                "activity_units": "nM",
                "activity_relation": "=",
                "assay_type": "B",
                "document_year": 2018,
            },
            {
                "molecule_name": "Osimertinib",
                "activity_type": "IC50",
                "activity_value": 1250.0,
                "activity_units": "nM",
                "activity_relation": ">",
                "assay_type": "F",
                "document_year": 2023,
            },
        ]
    }
    association_result = {"associations": []}

    with (
        patch("biomcp.tools.ncbi.get_gene_info", new=AsyncMock(return_value=ncbi_result)),
        patch("biomcp.tools.proteins.search_proteins", new=AsyncMock(return_value=uniprot_result)),
        patch("biomcp.tools.pathways.get_drug_targets", new=AsyncMock(return_value=chembl_result)),
        patch(
            "biomcp.tools.pathways.get_gene_disease_associations",
            new=AsyncMock(return_value=association_result),
        ),
    ):
        from biomcp.tools.verify import detect_database_conflicts

        result = await detect_database_conflicts("EGFR")

    assert result["conflicts_found"] == 1
    conflict = result["conflicts"][0]
    assert conflict["type"] == "ACTIVITY_VALUE_DISCREPANCY"
    assert "synthesis" in conflict
    assert "assay context" in conflict["synthesis"]["summary"].lower()
    assert any(step["dimension"] == "activity_relation" for step in conflict["synthesis"]["reasoning_steps"])
    assert result["conflict_synthesis"][0]["type"] == "ACTIVITY_VALUE_DISCREPANCY"
