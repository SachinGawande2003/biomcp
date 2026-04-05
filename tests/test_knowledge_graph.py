from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from biomcp.core import knowledge_graph as knowledge_graph_module


@pytest.mark.asyncio
async def test_saved_session_round_trip(monkeypatch: pytest.MonkeyPatch):
    temp_dir = Path(".codex_test_tmp") / f"kg-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BIOMCP_SESSION_STORE_DIR", str(temp_dir))

    try:
        knowledge_graph_module.reset_skg()

        skg = await knowledge_graph_module.get_skg()
        await skg.upsert_node(
            "EGFR",
            knowledge_graph_module.NodeType.GENE,
            properties={"summary": "growth factor receptor"},
            source="ncbi",
        )
        await skg.upsert_node("Lung cancer", knowledge_graph_module.NodeType.DISEASE, source="opentargets")
        await skg.upsert_edge(
            "EGFR",
            knowledge_graph_module.NodeType.GENE,
            knowledge_graph_module.EdgeType.ASSOCIATED_WITH,
            "Lung cancer",
            knowledge_graph_module.NodeType.DISEASE,
            source="opentargets",
        )
        skg.record_tool_call("get_gene_info", {"gene_symbol": "EGFR"}, "resolved EGFR")

        saved = await knowledge_graph_module.save_current_session(label="EGFR lung cancer")

        session_path = temp_dir / f"{saved['session_id']}.json"
        assert session_path.exists()
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        assert payload["resource_uri"] == f"biomcp://session/{saved['session_id']}"
        assert payload["graph_snapshot"]["summary"]["total_nodes"] == 2

        knowledge_graph_module.reset_skg()
        restored = await knowledge_graph_module.restore_saved_session(saved["session_id"])
        restored_skg = await knowledge_graph_module.get_skg()

        assert restored["graph_stats"]["nodes"] == 2
        assert restored_skg.find_node("EGFR") is not None
        assert restored_skg.find_node("Lung cancer") is not None

        listed = knowledge_graph_module.list_saved_sessions()
        assert listed[0]["session_id"] == saved["session_id"]
    finally:
        knowledge_graph_module.reset_skg()
        shutil.rmtree(temp_dir, ignore_errors=True)
