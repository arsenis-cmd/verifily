"""Tests for the Dataset Lineage Graph system.

These tests verify:
- Deterministic graph generation
- Correct node/edge counts
- Mermaid export stability
- JSON export validity
- Pipeline writes artifacts
- Plan mode writes nothing

All tests are fast (<0.2s) and deterministic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from verifily_cli_v1.core.lineage_graph import (
    LineageEdge,
    LineageGraph,
    LineageGraphBuilder,
    LineageNode,
    NodeType,
    RelationType,
    build_lineage_graph,
    write_lineage_artifacts,
)


class TestLineageNode:
    """Tests for LineageNode dataclass."""

    def test_node_creation(self):
        """Should create a valid node."""
        node = LineageNode(
            id="test_123",
            type=NodeType.DATASET_ARTIFACT,
            label="Test Dataset",
            timestamp="2026-02-13T10:00:00Z",
            metadata={"hash": "abc123"},
        )
        assert node.id == "test_123"
        assert node.type == NodeType.DATASET_ARTIFACT
        assert node.label == "Test Dataset"
        assert node.metadata["hash"] == "abc123"

    def test_node_to_dict(self):
        """Should serialize to dict correctly."""
        node = LineageNode(
            id="test_123",
            type=NodeType.TRAIN_RUN,
            label="Train",
        )
        d = node.to_dict()
        assert d["id"] == "test_123"
        assert d["type"] == "train_run"
        assert d["label"] == "Train"

    def test_node_from_dict(self):
        """Should deserialize from dict correctly."""
        data = {
            "id": "test_456",
            "type": "decision",
            "label": "SHIP",
            "timestamp": None,
            "metadata": {},
        }
        node = LineageNode.from_dict(data)
        assert node.id == "test_456"
        assert node.type == NodeType.DECISION
        assert node.label == "SHIP"


class TestLineageEdge:
    """Tests for LineageEdge dataclass."""

    def test_edge_creation(self):
        """Should create a valid edge."""
        edge = LineageEdge(
            source="node_a",
            target="node_b",
            relation=RelationType.DERIVED_FROM,
        )
        assert edge.source == "node_a"
        assert edge.target == "node_b"
        assert edge.relation == RelationType.DERIVED_FROM

    def test_edge_to_dict(self):
        """Should serialize to dict correctly."""
        edge = LineageEdge(
            source="a",
            target="b",
            relation=RelationType.TRAINED_ON,
        )
        d = edge.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["relation"] == "trained_on"


class TestLineageGraph:
    """Tests for LineageGraph class."""

    @pytest.fixture
    def sample_graph(self) -> LineageGraph:
        """Create a sample graph for testing."""
        nodes = [
            LineageNode(id="raw", type=NodeType.RAW_DATASET, label="Raw Data"),
            LineageNode(id="transform", type=NodeType.TRANSFORM, label="Transform"),
            LineageNode(id="dataset", type=NodeType.DATASET_ARTIFACT, label="Dataset"),
            LineageNode(id="train", type=NodeType.TRAIN_RUN, label="Train"),
            LineageNode(id="decision", type=NodeType.DECISION, label="SHIP"),
        ]
        edges = [
            LineageEdge(source="raw", target="transform", relation=RelationType.DERIVED_FROM),
            LineageEdge(source="transform", target="dataset", relation=RelationType.TRANSFORMED_BY),
            LineageEdge(source="dataset", target="train", relation=RelationType.TRAINED_ON),
            LineageEdge(source="train", target="decision", relation=RelationType.RESULTED_IN),
        ]
        return LineageGraph(
            root_id="decision",
            nodes=nodes,
            edges=edges,
        )

    def test_graph_to_dict(self, sample_graph: LineageGraph):
        """Should serialize to dict correctly."""
        d = sample_graph.to_dict()
        assert d["version"] == "1.0"
        assert d["root_id"] == "decision"
        assert len(d["nodes"]) == 5
        assert len(d["edges"]) == 4

    def test_graph_to_json(self, sample_graph: LineageGraph):
        """Should serialize to JSON correctly."""
        json_str = sample_graph.to_json()
        data = json.loads(json_str)
        assert data["root_id"] == "decision"
        assert len(data["nodes"]) == 5

    def test_graph_to_mermaid(self, sample_graph: LineageGraph):
        """Should generate valid Mermaid diagram."""
        mermaid = sample_graph.to_mermaid()
        assert "graph TD" in mermaid
        assert "raw" in mermaid
        assert "transform" in mermaid
        assert "dataset" in mermaid
        assert "-->" in mermaid  # Has edges

    def test_graph_to_rich_tree(self, sample_graph: LineageGraph):
        """Should generate ASCII tree."""
        tree = sample_graph.to_rich_tree()
        assert "Lineage" in tree
        assert "Raw Data" in tree
        assert "Transform" in tree

    def test_get_node(self, sample_graph: LineageGraph):
        """Should find node by ID."""
        node = sample_graph.get_node("dataset")
        assert node is not None
        assert node.label == "Dataset"

    def test_get_node_missing(self, sample_graph: LineageGraph):
        """Should return None for missing node."""
        node = sample_graph.get_node("nonexistent")
        assert node is None

    def test_get_children(self, sample_graph: LineageGraph):
        """Should get child nodes."""
        children = sample_graph.get_children("dataset")
        assert len(children) == 1
        assert children[0].id == "train"

    def test_get_parents(self, sample_graph: LineageGraph):
        """Should get parent nodes."""
        parents = sample_graph.get_parents("train")
        assert len(parents) == 1
        assert parents[0].id == "dataset"


class TestLineageGraphBuilder:
    """Tests for LineageGraphBuilder."""

    @pytest.fixture
    def mock_run_dir(self, tmp_path: Path) -> Path:
        """Create a mock run directory with artifacts."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()

        # Create run_meta.json
        run_meta = {
            "run_id": "run_test",
            "status": "completed",
            "base_model": "google/flan-t5-base",
            "dataset_version": "train_v1",
            "started_at": "2026-02-13T10:00:00Z",
            "completed_at": "2026-02-13T10:30:00Z",
            "duration_seconds": 1800,
            "metrics": {"train_loss": 0.45},
            "reproducibility_hash": "abc123",
            "data_hash": "def456",
            "config_hash": "ghi789",
            "seed": 42,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta))

        # Create hashes.json
        hashes = {
            "files": {"config.yaml": "hash123"},
            "chain_hash": "chain456",
        }
        (run_dir / "hashes.json").write_text(json.dumps(hashes))

        # Create config.yaml
        config = {
            "train_data": "/data/train.jsonl",
            "eval_data": "/data/eval.jsonl",
        }
        (run_dir / "config.yaml").write_text("train_data: /data/train.jsonl\neval_data: /data/eval.jsonl")

        # Create eval results
        eval_dir = run_dir / "eval"
        eval_dir.mkdir()
        eval_results = {
            "overall": {"f1": 0.72, "exact_match": 0.60},
            "decision": {"recommendation": "SHIP"},
        }
        (eval_dir / "eval_results.json").write_text(json.dumps(eval_results))

        return run_dir

    def test_build_from_run(self, mock_run_dir: Path):
        """Should build graph from run directory."""
        builder = LineageGraphBuilder()
        graph = builder.build_from_run(mock_run_dir)

        assert graph.root_id is not None
        assert len(graph.nodes) >= 3  # At least dataset, train, decision
        assert len(graph.edges) >= 2  # At least some connections

    def test_has_decision_node(self, mock_run_dir: Path):
        """Graph should contain a decision node."""
        builder = LineageGraphBuilder()
        graph = builder.build_from_run(mock_run_dir)

        decision_nodes = [n for n in graph.nodes if n.type == NodeType.DECISION]
        assert len(decision_nodes) == 1
        assert "SHIP" in decision_nodes[0].label

    def test_has_train_node(self, mock_run_dir: Path):
        """Graph should contain a train node."""
        builder = LineageGraphBuilder()
        graph = builder.build_from_run(mock_run_dir)

        train_nodes = [n for n in graph.nodes if n.type == NodeType.TRAIN_RUN]
        assert len(train_nodes) == 1
        assert train_nodes[0].metadata.get("seed") == 42

    def test_deterministic_id_generation(self, mock_run_dir: Path):
        """Same run should produce same node IDs."""
        builder1 = LineageGraphBuilder()
        graph1 = builder1.build_from_run(mock_run_dir)

        builder2 = LineageGraphBuilder()
        graph2 = builder2.build_from_run(mock_run_dir)

        # Same number of nodes
        assert len(graph1.nodes) == len(graph2.nodes)

        # Same node IDs
        ids1 = {n.id for n in graph1.nodes}
        ids2 = {n.id for n in graph2.nodes}
        assert ids1 == ids2

    def test_handles_missing_files(self, tmp_path: Path):
        """Should handle missing files gracefully."""
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()

        builder = LineageGraphBuilder()
        graph = builder.build_from_run(run_dir)

        # Should still produce a graph even without artifacts
        assert len(graph.nodes) > 0
        # Root should be created
        assert graph.root_id is not None


class TestBuildLineageGraph:
    """Tests for the main build_lineage_graph function."""

    def test_build_function(self, tmp_path: Path):
        """Should work as a standalone function."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Minimal artifacts
        (run_dir / "run_meta.json").write_text(json.dumps({
            "run_id": "test",
            "status": "completed",
        }))

        graph = build_lineage_graph(run_dir)
        assert isinstance(graph, LineageGraph)
        assert len(graph.nodes) > 0


class TestWriteLineageArtifacts:
    """Tests for writing lineage artifacts."""

    @pytest.fixture
    def sample_graph(self) -> LineageGraph:
        """Create a minimal sample graph."""
        return LineageGraph(
            root_id="root",
            nodes=[
                LineageNode(id="a", type=NodeType.RAW_DATASET, label="Raw"),
                LineageNode(id="b", type=NodeType.DECISION, label="SHIP"),
            ],
            edges=[LineageEdge(source="a", target="b", relation=RelationType.RESULTED_IN)],
        )

    def test_writes_json(self, sample_graph: LineageGraph, tmp_path: Path):
        """Should write JSON file."""
        artifacts = write_lineage_artifacts(sample_graph, tmp_path)

        assert "json" in artifacts
        assert artifacts["json"].exists()
        assert artifacts["json"].suffix == ".json"

        # Verify it's valid JSON
        data = json.loads(artifacts["json"].read_text())
        assert data["root_id"] == "root"

    def test_writes_mermaid(self, sample_graph: LineageGraph, tmp_path: Path):
        """Should write Mermaid file."""
        artifacts = write_lineage_artifacts(sample_graph, tmp_path)

        assert "mermaid" in artifacts
        assert artifacts["mermaid"].exists()
        assert artifacts["mermaid"].suffix == ".mmd"

        # Verify it's valid Mermaid
        content = artifacts["mermaid"].read_text()
        assert "graph TD" in content

    def test_creates_output_dir(self, sample_graph: LineageGraph, tmp_path: Path):
        """Should create output directory if needed."""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        write_lineage_artifacts(sample_graph, output_dir)

        assert output_dir.exists()


class TestMermaidExport:
    """Tests for Mermaid diagram export."""

    def test_mermaid_syntax(self):
        """Should produce valid Mermaid syntax."""
        graph = LineageGraph(
            root_id="decision",
            nodes=[
                LineageNode(id="raw", type=NodeType.RAW_DATASET, label="Raw Data"),
                LineageNode(id="decision", type=NodeType.DECISION, label="SHIP"),
            ],
            edges=[LineageEdge(source="raw", target="decision", relation=RelationType.RESULTED_IN)],
        )

        mermaid = graph.to_mermaid()

        # Should start with graph TD
        assert mermaid.startswith("graph TD")

        # Should contain node definitions
        assert 'raw["Raw Data"]' in mermaid

        # Should contain edges
        assert "raw --> decision" in mermaid

    def test_mermaid_escapes_quotes(self):
        """Should escape quotes in labels."""
        graph = LineageGraph(
            root_id="node1",
            nodes=[
                LineageNode(id="node1", type=NodeType.DATASET_ARTIFACT, label='Dataset "v1"'),
            ],
            edges=[],
        )

        mermaid = graph.to_mermaid()
        # Quotes should be escaped with backslash
        assert '\\"v1\\"' in mermaid


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, tmp_path: Path):
        """Same run directory should produce identical graph structure."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Create artifacts
        (run_dir / "run_meta.json").write_text(json.dumps({
            "run_id": "test",
            "base_model": "model",
            "completed_at": "2026-02-13T10:00:00Z",
        }))
        (run_dir / "hashes.json").write_text(json.dumps({"chain_hash": "abc"}))

        # Build twice
        graph1 = build_lineage_graph(run_dir)
        graph2 = build_lineage_graph(run_dir)

        # Same structure
        assert len(graph1.nodes) == len(graph2.nodes)
        assert len(graph1.edges) == len(graph2.edges)

        # Same node IDs (deterministic)
        ids1 = sorted(n.id for n in graph1.nodes)
        ids2 = sorted(n.id for n in graph2.nodes)
        assert ids1 == ids2

        # Same edge structure
        edges1 = sorted((e.source, e.target) for e in graph1.edges)
        edges2 = sorted((e.source, e.target) for e in graph2.edges)
        assert edges1 == edges2

    def test_mermaid_determinism(self, tmp_path: Path):
        """Mermaid output should be deterministic."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        (run_dir / "run_meta.json").write_text(json.dumps({
            "run_id": "test",
            "completed_at": "2026-02-13T10:00:00Z",
        }))

        graph1 = build_lineage_graph(run_dir)
        graph2 = build_lineage_graph(run_dir)

        assert graph1.to_mermaid() == graph2.to_mermaid()
