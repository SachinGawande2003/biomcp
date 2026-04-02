"""
Tests - MCP server public surface.
"""

from __future__ import annotations

import pytest

from biomcp import __version__
from biomcp.server import (
    TOOLS,
    _build_health_report,
    _build_readiness_report,
    _build_tool_health_report,
)


class TestToolRegistry:
    def test_minimum_tool_count(self):
        assert len(TOOLS) >= 20, f"Expected >=20 tools, got {len(TOOLS)}"

    def test_curated_tool_count_target(self):
        assert 25 <= len(TOOLS) <= 40, f"Expected curated surface in 25-40 range, got {len(TOOLS)}"

    def test_all_tools_have_name(self):
        for tool in TOOLS:
            assert tool.name, f"Tool missing name: {tool}"

    def test_all_tools_have_description(self):
        for tool in TOOLS:
            assert tool.description, f"Tool '{tool.name}' has empty description"

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert tool.inputSchema, f"Tool '{tool.name}' missing inputSchema"

    def test_schemas_have_required_keys(self):
        for tool in TOOLS:
            schema = tool.inputSchema
            assert "properties" in schema, f"'{tool.name}' schema missing 'properties'"
            assert "required" in schema, f"'{tool.name}' schema missing 'required'"

    def test_no_duplicate_tool_names(self):
        names = [t.name for t in TOOLS]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert len(names) == len(set(names)), f"Duplicate tool names: {duplicates}"

    def test_strategy_tools_present(self):
        names = {t.name for t in TOOLS}
        expected = {
            "search_pubmed",
            "get_gene_info",
            "run_blast",
            "get_protein_info",
            "find_protein",
            "get_alphafold_structure",
            "pathway_analysis",
            "get_drug_targets",
            "get_gene_disease_associations",
            "search_clinical_trials",
            "multi_omics_gene_report",
            "predict_structure_boltz2",
            "generate_dna_evo2",
            "crispr_analysis",
            "drug_safety",
            "variant_analysis",
            "find_repurposing_candidates",
            "verify_biological_claim",
            "search_cbio_mutations",
            "search_gwas_catalog",
            "session",
            "drug_interaction_checker",
            "protein_binding_pocket",
            "biomarker_panel_design",
            "pharmacogenomics_report",
            "protein_family_analysis",
            "network_enrichment",
            "rnaseq_deconvolution",
            "structural_similarity",
            "rare_disease_diagnosis",
            "genome_browser_snapshot",
        }
        missing = expected - names
        assert not missing, f"Missing tools: {missing}"

    def test_removed_tools_not_public(self):
        names = {t.name for t in TOOLS}
        removed = {
            "search_proteins",
            "search_pdb_structures",
            "search_pathways",
            "get_pathway_genes",
            "get_reactome_pathways",
            "design_crispr_guides",
            "score_guide_efficiency",
            "predict_off_target_sites",
            "design_base_editor_guides",
            "get_crispr_repair_outcomes",
            "query_adverse_events",
            "analyze_safety_signals",
            "get_drug_label_warnings",
            "compare_drug_safety",
            "classify_variant",
            "get_population_frequency",
            "lookup_clinvar_variant",
            "query_neuroimaging_datasets",
            "generate_research_hypothesis",
            "estimate_statistical_power",
            "suggest_cell_lines",
            "detect_database_conflicts",
            "find_research_gaps",
            "validate_reasoning_chain",
            "generate_experimental_protocol",
            "bulk_gene_analysis",
            "compute_pathway_enrichment",
            "analyze_coexpression",
            "predict_splice_impact",
            "get_cancer_hotspots",
            "get_biogrid_interactions",
            "search_orphan_diseases",
            "search_cellmarker",
            "search_metabolomics",
            "get_encode_regulatory",
            "get_disgenet_associations",
            "get_pharmgkb_variants",
            "get_ucsc_splice_variants",
        }
        assert not (removed & names), f"Unexpected public legacy tools: {removed & names}"

    def test_schema_types_are_valid(self):
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        for tool in TOOLS:
            for prop_name, prop_def in tool.inputSchema.get("properties", {}).items():
                ptype = prop_def.get("type", "")
                assert ptype in valid_types, (
                    f"Tool '{tool.name}', property '{prop_name}' has invalid type '{ptype}'"
                )


class TestOperationalHealth:
    def test_health_report_matches_server_version(self):
        report = _build_health_report("http")
        assert report["service"] == "heuris-biomcp"
        assert report["version"] == __version__
        assert report["tool_count"] == len(TOOLS)
        assert report["transport"]["sse_path"] == "/sse"
        assert report["transport"]["message_path"] == "/messages/"

    def test_readiness_report_is_ready_with_registered_tools(self):
        report = _build_readiness_report("stdio")
        assert report["ready"] is True
        assert report["tool_count"] == len(TOOLS)

    def test_tool_health_reflects_missing_optional_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NVIDIA_BOLTZ2_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_EVO2_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)

        report = _build_tool_health_report()
        gated = report["gated_capabilities"]

        assert "nvidia_boltz2" in gated
        assert "nvidia_evo2" in gated
        assert gated["nvidia_boltz2"]["status"] == "degraded"
        assert gated["nvidia_evo2"]["status"] == "degraded"
        assert "predict_structure_boltz2" in gated["nvidia_boltz2"]["tools"]
        assert "generate_dna_evo2" in gated["nvidia_evo2"]["tools"]
