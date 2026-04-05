from __future__ import annotations

from typing import Any

from biomcp.core.query_planner import AdaptiveQueryPlanner, _load_plan_registry


async def _noop_dispatch(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    return {"tool": tool_name, "args": tool_args}


def test_plan_registry_contains_expected_workflows():
    registry = _load_plan_registry()
    workflows = registry["workflows"]

    assert {
        "drug_target",
        "general_gene",
        "disease_genomics",
        "protein_structure",
        "literature_fallback",
    }.issubset(workflows)


def test_drug_target_plan_comes_from_registry_and_honors_depth():
    planner = AdaptiveQueryPlanner(dispatcher=_noop_dispatch)

    plan = planner.build_plan("Understand EGFR as a drug target in NSCLC", depth="deep")

    tool_names = [node.tool_name for node in plan.nodes]
    assert plan.strategy == "Drug target analysis pipeline for EGFR"
    assert "get_gene_info" in tool_names
    assert "get_gene_variants" in tool_names
    assert "search_gene_expression" in tool_names
    assert "multi_omics_gene_report" in tool_names

    multi_omics_node = next(node for node in plan.nodes if node.tool_name == "multi_omics_gene_report")
    assert multi_omics_node.optional is True
    assert len(multi_omics_node.depends_on) == 2


def test_protein_structure_plan_gates_optional_ligand_node():
    planner = AdaptiveQueryPlanner(dispatcher=_noop_dispatch)

    without_ligand = planner.build_plan(
        "Assess structure for P04637",
        entities={"uniprot": "P04637"},
    )
    with_ligand = planner.build_plan(
        "Assess structure for P04637 with ligand",
        entities={"uniprot": "P04637", "ligand": "CCO"},
    )

    without_tools = [node.tool_name for node in without_ligand.nodes]
    with_tools = [node.tool_name for node in with_ligand.nodes]

    assert "predict_structure_boltz2" not in without_tools
    assert "predict_structure_boltz2" in with_tools

    boltz2_node = next(node for node in with_ligand.nodes if node.tool_name == "predict_structure_boltz2")
    assert boltz2_node.optional is True
    assert boltz2_node.tool_args["ligand_smiles"] == ["CCO"]


def test_literature_fallback_plan_uses_goal_template():
    planner = AdaptiveQueryPlanner(dispatcher=_noop_dispatch)

    plan = planner.build_plan("Investigate ferroptosis in glioblastoma")

    assert plan.strategy == "Literature-driven research pipeline"
    assert [node.tool_name for node in plan.nodes] == [
        "search_pubmed",
        "generate_research_hypothesis",
    ]
    assert plan.nodes[0].tool_args["query"] == "Investigate ferroptosis in glioblastoma"
