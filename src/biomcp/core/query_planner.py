"""
BioMCP Adaptive Query Planner
=============================
Transforms a natural-language research goal into a dependency-aware DAG of
tool calls, then executes it with parallelism where possible.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised when PyYAML is unavailable
    yaml = None


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanNode:
    """A single tool call within a research plan DAG."""

    node_id: str
    tool_name: str
    tool_args: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    rationale: str = ""

    @property
    def elapsed_s(self) -> float:
        if self.started_at and self.completed_at:
            return round(self.completed_at - self.started_at, 2)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tool": self.tool_name,
            "args": self.tool_args,
            "optional": self.optional,
            "status": self.status.value,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
            "rationale": self.rationale,
        }


@dataclass
class ResearchPlan:
    """
    A directed acyclic graph of tool calls for a research goal.
    Nodes at the same dependency level execute in parallel.
    """

    plan_id: str
    goal: str
    nodes: list[PlanNode]
    strategy: str = ""
    depth: str = "standard"
    created_at: float = field(default_factory=time.monotonic)

    def execution_levels(self) -> list[list[PlanNode]]:
        """Topological sort returning groups that can execute in parallel."""

        completed: set[str] = set()
        remaining = list(self.nodes)
        levels: list[list[PlanNode]] = []

        while remaining:
            ready = [
                node
                for node in remaining
                if all(dep in completed for dep in node.depends_on)
                and node.status not in (
                    NodeStatus.COMPLETE,
                    NodeStatus.FAILED,
                    NodeStatus.SKIPPED,
                )
            ]
            if not ready:
                for node in remaining:
                    node.status = NodeStatus.SKIPPED
                break
            levels.append(ready)
            for node in ready:
                completed.add(node.node_id)
                remaining.remove(node)

        return levels

    def summary(self) -> dict[str, Any]:
        total = len(self.nodes)
        complete = sum(1 for node in self.nodes if node.status == NodeStatus.COMPLETE)
        failed = sum(1 for node in self.nodes if node.status == NodeStatus.FAILED)
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "depth": self.depth,
            "strategy": self.strategy,
            "total_steps": total,
            "completed": complete,
            "failed": failed,
            "success_rate": f"{(complete / total * 100):.0f}%" if total else "0%",
            "steps": [node.to_dict() for node in self.nodes],
        }


_PLAN_REGISTRY_PATH = Path(__file__).with_name("plan_registry.yaml")
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")
_PLAN_REGISTRY: dict[str, Any] | None = None


def _parse_plan_registry(text: str) -> dict[str, Any]:
    if yaml is not None:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ValueError("plan_registry.yaml must define a top-level mapping")

    workflows = loaded.get("workflows")
    if not isinstance(workflows, dict) or not workflows:
        raise ValueError("plan_registry.yaml must define a non-empty 'workflows' mapping")

    return loaded


def _load_plan_registry() -> dict[str, Any]:
    global _PLAN_REGISTRY
    if _PLAN_REGISTRY is None:
        _PLAN_REGISTRY = _parse_plan_registry(
            _PLAN_REGISTRY_PATH.read_text(encoding="utf-8")
        )
    return _PLAN_REGISTRY


def _render_template(value: str, context: dict[str, str]) -> str:
    rendered = _PLACEHOLDER_PATTERN.sub(
        lambda match: context.get(match.group(1), ""),
        value,
    )
    return " ".join(rendered.split())


def _resolve_template_value(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _render_template(value, context)
    if isinstance(value, list):
        return [_resolve_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_template_value(item, context)
            for key, item in value.items()
        }
    return value


def _should_include_node(
    spec: dict[str, Any],
    *,
    depth: str,
    entities: dict[str, str],
) -> bool:
    include_if = spec.get("include_if", {})
    if not isinstance(include_if, dict):
        raise ValueError("plan node 'include_if' must be a mapping")

    depth_in = include_if.get("depth_in")
    if depth_in is not None:
        if not isinstance(depth_in, list):
            raise ValueError("plan node 'include_if.depth_in' must be a list")
        if depth not in depth_in:
            return False

    entity_present = include_if.get("entity_present")
    if entity_present is not None:
        if not isinstance(entity_present, list):
            raise ValueError("plan node 'include_if.entity_present' must be a list")
        if not all(entities.get(name, "") for name in entity_present):
            return False

    return True


def _expand_workflow(
    workflow_key: str,
    *,
    goal: str,
    depth: str,
    entities: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    workflow = _load_plan_registry()["workflows"].get(workflow_key)
    if not isinstance(workflow, dict):
        raise ValueError(f"Unknown planner workflow '{workflow_key}'")

    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"Workflow '{workflow_key}' must define a non-empty node list")

    context = {
        "goal": goal,
        "depth": depth,
        **entities,
    }
    context["disease_or_goal"] = entities.get("disease", "") or goal[:30]

    selected_nodes = [
        spec
        for spec in nodes
        if isinstance(spec, dict) and _should_include_node(spec, depth=depth, entities=entities)
    ]
    id_map = {
        str(spec["key"]): uuid4().hex[:6]
        for spec in selected_nodes
    }

    expanded_nodes: list[dict[str, Any]] = []
    for spec in selected_nodes:
        spec_key = str(spec["key"])
        dependencies = [
            id_map[dependency]
            for dependency in spec.get("depends_on", [])
            if dependency in id_map
        ]
        expanded_nodes.append({
            "id": id_map[spec_key],
            "tool": str(spec["tool"]),
            "args": _resolve_template_value(spec.get("args", {}), context),
            "deps": dependencies,
            "rationale": str(spec.get("rationale", "")),
            "optional": bool(spec.get("optional", False)),
        })

    strategy_template = str(workflow.get("strategy_template", "Research workflow"))
    return _render_template(strategy_template, context), expanded_nodes


class AdaptiveQueryPlanner:
    """
    Builds and executes optimized research plans.

    Usage::
        planner = AdaptiveQueryPlanner(dispatcher_fn)
        result = await planner.plan_and_execute(
            goal="Understand KRAS G12C as a drug target in NSCLC",
            depth="standard",
        )
    """

    def __init__(self, dispatcher: Callable[[str, dict[str, Any]], Any]) -> None:
        self._dispatch = dispatcher

    def _classify_intent(self, goal: str) -> tuple[str, dict[str, str]]:
        """
        Classify the research goal and extract entities.

        Returns:
            (intent_type, entities_dict)
        """

        goal_lower = goal.lower()
        entities: dict[str, str] = {}

        gene_matches = re.findall(r"\b([A-Z][A-Z0-9]{1,9})\b", goal)
        if gene_matches:
            entities["gene"] = gene_matches[0]

        uniprot_matches = re.findall(
            r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b",
            goal,
        )
        if uniprot_matches:
            entities["uniprot"] = uniprot_matches[0]

        if any(word in goal_lower for word in ("structure", "fold", "binding", "pocket", "docking")):
            return "protein_structure", entities
        if any(
            word in goal_lower
            for word in ("drug target", "inhibit", "therapeut", "ic50", "ligand", "compound")
        ):
            return "drug_target", entities
        if any(word in goal_lower for word in ("disease", "cancer", "mutation", "variant", "genomic")):
            return "disease_genomics", entities
        if any(word in goal_lower for word in ("pathway", "signaling", "network", "interact")):
            return "pathway_analysis", entities
        if any(word in goal_lower for word in ("expression", "tissue", "rna", "transcriptom")):
            return "expression_analysis", entities

        return "general_gene", entities

    def build_plan(
        self,
        goal: str,
        depth: str = "standard",
        entities: dict[str, str] | None = None,
    ) -> ResearchPlan:
        """Build a dependency-aware execution DAG for the research goal."""

        intent, auto_entities = self._classify_intent(goal)
        resolved_entities = {**auto_entities, **(entities or {})}
        gene = resolved_entities.get("gene", "")
        uniprot = resolved_entities.get("uniprot", "")

        if intent == "drug_target" and gene:
            workflow_key = "drug_target"
        elif intent == "disease_genomics" and gene:
            workflow_key = "disease_genomics"
        elif intent == "protein_structure" and uniprot:
            workflow_key = "protein_structure"
        elif gene:
            workflow_key = "general_gene"
        else:
            workflow_key = "literature_fallback"

        strategy, node_specs = _expand_workflow(
            workflow_key,
            goal=goal,
            depth=depth,
            entities=resolved_entities,
        )

        nodes = [
            PlanNode(
                node_id=spec["id"],
                tool_name=spec["tool"],
                tool_args=spec["args"],
                depends_on=spec.get("deps", []),
                optional=bool(spec.get("optional", False)),
                rationale=spec.get("rationale", ""),
            )
            for spec in node_specs
        ]

        return ResearchPlan(
            plan_id=uuid4().hex[:8],
            goal=goal,
            nodes=nodes,
            strategy=strategy,
            depth=depth,
        )

    async def execute(
        self,
        plan: ResearchPlan,
        timeout_per_tool: float = 60.0,
        progress_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a research plan, respecting dependency order.
        Parallelizes within each dependency level.
        """

        started = time.monotonic()
        all_results: dict[str, Any] = {}
        levels = plan.execution_levels()

        logger.info(
            f"[Planner] Executing plan '{plan.goal}' - {len(plan.nodes)} steps in {len(levels)} levels"
        )

        for level_idx, level in enumerate(levels):
            logger.info(
                f"[Planner] Level {level_idx + 1}/{len(levels)}: "
                f"running {[node.tool_name for node in level]} in parallel"
            )

            if progress_callback is not None:
                await progress_callback(
                    "level_started",
                    {
                        "level": level_idx + 1,
                        "total_levels": len(levels),
                        "tools": [node.tool_name for node in level],
                    },
                )

            tasks = [
                asyncio.create_task(
                    self._execute_node(
                        node,
                        timeout_per_tool,
                        progress_callback=progress_callback,
                    )
                )
                for node in level
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            for node in level:
                if node.status == NodeStatus.COMPLETE:
                    all_results[node.node_id] = node.result

        elapsed = round(time.monotonic() - started, 2)
        execution_summary = plan.summary()
        report = {
            "goal": plan.goal,
            "strategy": plan.strategy,
            "depth": plan.depth,
            "total_elapsed_s": elapsed,
            "execution_summary": execution_summary,
            "results": all_results,
            "insights": self._synthesize_insights(plan, all_results),
        }
        if progress_callback is not None:
            await progress_callback(
                "plan_completed",
                {
                    "goal": plan.goal,
                    "completed": execution_summary["completed"],
                    "failed": execution_summary["failed"],
                    "total_steps": execution_summary["total_steps"],
                    "elapsed_s": elapsed,
                },
            )
        return report

    async def _execute_node(
        self,
        node: PlanNode,
        timeout: float,
        progress_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        """Execute a single plan node with timeout and error capture."""

        node.status = NodeStatus.RUNNING
        node.started_at = time.monotonic()
        try:
            node.result = await asyncio.wait_for(
                self._dispatch(node.tool_name, node.tool_args),
                timeout=timeout,
            )
            node.status = NodeStatus.COMPLETE
            node.completed_at = time.monotonic()
            logger.debug(f"[Planner] OK {node.tool_name} ({node.elapsed_s}s)")
        except TimeoutError:
            node.status = NodeStatus.FAILED
            node.error = f"Timed out after {timeout}s"
            node.completed_at = time.monotonic()
            logger.warning(f"[Planner] FAIL {node.tool_name} timed out")
        except Exception as exc:
            node.status = NodeStatus.FAILED
            node.error = str(exc)
            node.completed_at = time.monotonic()
            logger.warning(f"[Planner] FAIL {node.tool_name}: {exc}")
        finally:
            if progress_callback is not None:
                await progress_callback(
                    "node_finished",
                    {
                        "node": node.to_dict(),
                        "result": node.result,
                    },
                )

    def _synthesize_insights(
        self,
        plan: ResearchPlan,
        results: dict[str, Any],
    ) -> list[str]:
        """
        Generate high-level insights by cross-referencing results.
        This is where the planner adds value beyond raw tool calls.
        """

        insights: list[str] = []

        drug_nodes = [
            node
            for node in plan.nodes
            if node.tool_name == "get_drug_targets" and node.status == NodeStatus.COMPLETE
        ]
        for node in drug_nodes:
            result = node.result or {}
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {}
            drugs = result.get("data", {}).get("drugs", []) if isinstance(result, dict) else []
            if isinstance(result, dict) and "drugs" in result:
                drugs = result["drugs"]
            total = result.get("total_activities", len(drugs)) if isinstance(result, dict) else 0
            gene = node.tool_args.get("gene_symbol", "")
            if total:
                insights.append(
                    f"{gene} has {total} ChEMBL compound activities - active drug target."
                )

        variant_nodes = [
            node
            for node in plan.nodes
            if node.tool_name == "get_gene_variants" and node.status == NodeStatus.COMPLETE
        ]
        trial_nodes = [
            node
            for node in plan.nodes
            if node.tool_name == "search_clinical_trials"
            and node.status == NodeStatus.COMPLETE
        ]
        if variant_nodes and trial_nodes:
            insights.append(
                "Variant data and clinical trials retrieved - cross-reference "
                "mutations with trial eligibility criteria."
            )

        pubmed_nodes = [
            node
            for node in plan.nodes
            if node.tool_name == "search_pubmed" and node.status == NodeStatus.COMPLETE
        ]
        if pubmed_nodes and drug_nodes:
            insights.append(
                "Literature and ChEMBL data integrated - validate experimental "
                "IC50 values against published reports."
            )

        complete = sum(1 for node in plan.nodes if node.status == NodeStatus.COMPLETE)
        failed = sum(1 for node in plan.nodes if node.status == NodeStatus.FAILED)
        if failed:
            insights.append(
                f"{failed} tool call(s) failed - retry individually for complete data coverage."
            )
        insights.append(f"Research plan executed: {complete}/{len(plan.nodes)} steps successful.")

        return insights

    async def plan_and_execute(
        self,
        goal: str,
        depth: str = "standard",
        entities: dict[str, str] | None = None,
        timeout_per_tool: float = 60.0,
        progress_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        One-shot: build plan, execute it, and return an integrated report.

        Args:
            goal: Natural language research objective.
            depth: "quick" | "standard" | "deep". Default "standard".
            entities: Override auto-extracted entities.
            timeout_per_tool: Per-tool timeout in seconds.
        """

        plan = self.build_plan(goal, depth=depth, entities=entities)
        return await self.execute(
            plan,
            timeout_per_tool=timeout_per_tool,
            progress_callback=progress_callback,
        )
