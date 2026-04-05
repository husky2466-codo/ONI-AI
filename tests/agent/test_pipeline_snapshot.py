# tests/agent/test_pipeline_snapshot.py
from src.agent.runner import PipelineSnapshot


def test_pipeline_snapshot_to_dict_has_required_keys():
    snap = PipelineSnapshot(tick=5, cycle=2)
    snap.add_stage("state_in", "State Received", {"cycle": 2, "dupes": 3})
    snap.add_stage("prompt", "Prompt Formatted", {"chars": 500, "preview": "Cycle: 2"})
    snap.elapsed_ms = 8200
    d = snap.to_dict()
    assert d["type"] == "pipeline"
    assert d["tick"] == 5
    assert d["cycle"] == 2
    assert d["elapsed_ms"] == 8200
    assert isinstance(d["stages"], list)
    assert len(d["stages"]) == 2


def test_pipeline_snapshot_stage_structure():
    snap = PipelineSnapshot(tick=1, cycle=1)
    snap.add_stage("validation", "Validation", {"result": "blocked", "reason": "not solid"})
    stage = snap.to_dict()["stages"][0]
    assert stage["name"] == "validation"
    assert stage["label"] == "Validation"
    assert stage["data"]["result"] == "blocked"


def test_pipeline_snapshot_empty_stages():
    snap = PipelineSnapshot(tick=1, cycle=1)
    d = snap.to_dict()
    assert d["stages"] == []
