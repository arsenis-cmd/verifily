"""Dataset Lineage Graph engine for Verifily.

Builds a machine-readable, deterministic graph of data transformations,
decisions, and artifacts. Supports export to JSON and Mermaid.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from verifily_cli_v1.core.io import read_json, read_yaml, write_json


class NodeType(str, Enum):
    """Types of nodes in the lineage graph."""

    RAW_DATASET = "raw_dataset"
    DATASET_ARTIFACT = "dataset_artifact"
    TRANSFORM = "transform"
    CONTAMINATION_CHECK = "contamination_check"
    TRAIN_RUN = "train_run"
    DECISION = "decision"
    EVALUATION = "evaluation"


class RelationType(str, Enum):
    """Types of edges/relations in the lineage graph."""

    DERIVED_FROM = "derived_from"
    TRANSFORMED_BY = "transformed_by"
    CHECKED_BY = "checked_by"
    PRODUCED_BY = "produced_by"
    DEPENDS_ON = "depends_on"
    TRAINED_ON = "trained_on"
    EVALUATED_BY = "evaluated_by"
    RESULTED_IN = "resulted_in"


@dataclass
class LineageNode:
    """A node in the lineage graph.

    Attributes:
        id: Unique node identifier (deterministic hash)
        type: Node type from NodeType enum
        label: Human-readable label
        timestamp: ISO timestamp of creation
        metadata: Additional node metadata (hashes, metrics, etc.)
    """

    id: str
    type: NodeType
    label: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            label=data["label"],
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LineageEdge:
    """An edge in the lineage graph representing a relationship.

    Attributes:
        source: Source node ID
        target: Target node ID
        relation: Relationship type from RelationType enum
        metadata: Additional edge metadata
    """

    source: str
    target: str
    relation: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageEdge":
        """Create from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            relation=RelationType(data["relation"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LineageGraph:
    """A complete lineage graph for a run or dataset.

    Attributes:
        root_id: ID of the root node (usually the run or decision)
        nodes: List of all nodes in the graph
        edges: List of all edges connecting nodes
        created_at: ISO timestamp of graph creation
        version: Graph schema version
    """

    root_id: str
    nodes: List[LineageNode] = field(default_factory=list)
    edges: List[LineageEdge] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph to dictionary."""
        return {
            "version": self.version,
            "root_id": self.root_id,
            "created_at": self.created_at,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Export graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_mermaid(self) -> str:
        """Export graph to Mermaid diagram format.

        Returns a string in Mermaid flowchart syntax.
        """
        lines = ["graph TD"]

        # Build node definitions with labels
        for node in self.nodes:
            # Sanitize label for Mermaid (escape quotes, replace brackets)
            safe_label = node.label.replace('"', '\\"').replace("[", "(").replace("]", ")")
            lines.append(f'  {node.id}["{safe_label}"]')

        lines.append("")  # Empty line before edges

        # Build edges
        for edge in self.edges:
            rel_arrow = "-->"
            lines.append(f'  {edge.source} {rel_arrow} {edge.target}')

        return "\n".join(lines)

    def to_rich_tree(self) -> str:
        """Export graph as ASCII tree for CLI display.

        Returns a string suitable for Rich console output.
        """
        # Build adjacency list (parents -> children)
        children: Dict[str, List[str]] = {}
        for edge in self.edges:
            children.setdefault(edge.source, []).append(edge.target)

        # Build reverse adjacency (children -> parents) for finding roots
        parents: Dict[str, List[str]] = {}
        for edge in self.edges:
            parents.setdefault(edge.target, []).append(edge.source)

        # Find all root nodes (no parents, or specified root_id)
        root_ids = []
        if self.root_id in {n.id for n in self.nodes}:
            # Check if root has parents - if so, include them too for complete picture
            root_ids.append(self.root_id)
            # Also add any nodes without parents
            for node in self.nodes:
                if node.id not in parents and node.id not in root_ids:
                    root_ids.append(node.id)
        else:
            # Find all nodes with no parents
            for node in self.nodes:
                if node.id not in parents:
                    root_ids.append(node.id)

        if not root_ids:
            root_ids = [self.nodes[0].id] if self.nodes else []

        # Build tree recursively
        node_map = {n.id: n for n in self.nodes}

        def build_tree(node_id: str, prefix: str = "", is_last: bool = True) -> List[str]:
            lines = []
            node = node_map.get(node_id)
            if not node:
                return lines

            # Build label with type indicator
            icon = {
                NodeType.RAW_DATASET: "📄",
                NodeType.DATASET_ARTIFACT: "📦",
                NodeType.TRANSFORM: "🔧",
                NodeType.CONTAMINATION_CHECK: "🔍",
                NodeType.TRAIN_RUN: "🚀",
                NodeType.DECISION: "✅",
                NodeType.EVALUATION: "📊",
            }.get(node.type, "•")

            label = f"{icon} {node.label}"
            if node.metadata.get("hash"):
                short_hash = node.metadata["hash"][:8]
                label += f" ({short_hash}...)"
            if node.metadata.get("status"):
                label += f" [{node.metadata['status']}]"

            lines.append(prefix + ("└── " if is_last else "├── ") + label)

            # Process children
            child_ids = children.get(node_id, [])
            for i, child_id in enumerate(child_ids):
                is_last_child = i == len(child_ids) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.extend(build_tree(child_id, child_prefix, is_last_child))

            return lines

        # Build tree from each root
        tree_lines = ["Dataset Lineage"]
        for i, root_id in enumerate(root_ids):
            is_last_root = i == len(root_ids) - 1
            tree_lines.extend(build_tree(root_id, "", is_last_root))

        return "\n".join(tree_lines)

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_children(self, node_id: str) -> List[LineageNode]:
        """Get all child nodes of a given node."""
        child_ids = {e.target for e in self.edges if e.source == node_id}
        return [n for n in self.nodes if n.id in child_ids]

    def get_parents(self, node_id: str) -> List[LineageNode]:
        """Get all parent nodes of a given node."""
        parent_ids = {e.source for e in self.edges if e.target == node_id}
        return [n for n in self.nodes if n.id in parent_ids]


class LineageGraphBuilder:
    """Builder for constructing lineage graphs from run artifacts."""

    def __init__(self) -> None:
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self._warnings: List[str] = []

    def _make_id(self, *parts: str) -> str:
        """Create a deterministic node ID from parts."""
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _add_node(self, node: LineageNode) -> str:
        """Add a node to the graph, returning its ID."""
        self.nodes[node.id] = node
        return node.id

    def _add_edge(self, source: str, target: str, relation: RelationType) -> None:
        """Add an edge between two nodes."""
        self.edges.append(LineageEdge(source=source, target=target, relation=relation))

    def build_from_run(self, run_dir: Union[str, Path]) -> LineageGraph:
        """Build a lineage graph from a run directory.

        Discovers and connects:
        - Raw datasets (from config)
        - Transform steps (from manifest)
        - Dataset artifacts (from hashes.json)
        - Contamination checks (from pipeline results)
        - Train runs (from run_meta.json)
        - Decisions (from eval results)

        Args:
            run_dir: Path to the run directory

        Returns:
            Complete LineageGraph
        """
        run_path = Path(run_dir)
        self.nodes.clear()
        self.edges.clear()
        self._warnings.clear()

        # Build chain from config → transform → contamination → decision
        root_id = self._build_run_node(run_path)

        # Try to discover upstream dependencies
        self._discover_upstream(run_path)

        # Try to discover downstream artifacts
        self._discover_downstream(run_path)

        return LineageGraph(
            root_id=root_id,
            nodes=list(self.nodes.values()),
            edges=self.edges,
        )

    def _build_run_node(self, run_path: Path) -> str:
        """Build the main run/decision node."""
        run_id = run_path.name

        # Try to load run_meta.json
        run_meta_path = run_path / "run_meta.json"
        run_meta: Dict[str, Any] = {}
        if run_meta_path.exists():
            try:
                run_meta = read_json(run_meta_path)
            except Exception as e:
                self._warnings.append(f"Could not read run_meta.json: {e}")

        # Determine decision status from eval results
        eval_path = run_path / "eval" / "eval_results.json"
        decision_status = "unknown"
        if eval_path.exists():
            try:
                eval_data = read_json(eval_path)
                # Look for decision recommendation
                if "decision" in eval_data:
                    decision_status = eval_data["decision"].get("recommendation", "unknown")
                elif "overall" in eval_data:
                    # Infer from metrics
                    overall = eval_data["overall"]
                    f1 = overall.get("f1", 0)
                    decision_status = "SHIP" if f1 > 0.65 else "INVESTIGATE"
            except Exception:
                pass

        # Create decision node as root
        decision_id = self._make_id("decision", run_id)
        decision_node = LineageNode(
            id=decision_id,
            type=NodeType.DECISION,
            label=f"Decision: {decision_status}",
            timestamp=run_meta.get("completed_at"),
            metadata={
                "run_id": run_id,
                "status": decision_status,
                "run_path": str(run_path),
                "reproducibility_hash": run_meta.get("reproducibility_hash"),
                "data_hash": run_meta.get("data_hash"),
                "config_hash": run_meta.get("config_hash"),
            },
        )
        self._add_node(decision_node)

        # Create train run node
        train_id = self._make_id("train", run_id)
        train_node = LineageNode(
            id=train_id,
            type=NodeType.TRAIN_RUN,
            label=f"Train: {run_meta.get('base_model', 'unknown')}",
            timestamp=run_meta.get("started_at"),
            metadata={
                "run_id": run_id,
                "base_model": run_meta.get("base_model"),
                "dataset_version": run_meta.get("dataset_version"),
                "device": run_meta.get("device"),
                "duration_seconds": run_meta.get("duration_seconds"),
                "train_loss": run_meta.get("metrics", {}).get("train_loss"),
                "seed": run_meta.get("seed"),
            },
        )
        self._add_node(train_node)
        self._add_edge(train_id, decision_id, RelationType.RESULTED_IN)

        # Create evaluation node if eval results exist
        if eval_path.exists():
            eval_id = self._make_id("eval", run_id)
            try:
                eval_data = read_json(eval_path)
                overall = eval_data.get("overall", {})
                eval_node = LineageNode(
                    id=eval_id,
                    type=NodeType.EVALUATION,
                    label="Evaluation",
                    timestamp=run_meta.get("completed_at"),
                    metadata={
                        "f1": overall.get("f1"),
                        "exact_match": overall.get("exact_match"),
                        "accuracy": overall.get("accuracy"),
                    },
                )
                self._add_node(eval_node)
                self._add_edge(train_id, eval_id, RelationType.EVALUATED_BY)
                self._add_edge(eval_id, decision_id, RelationType.RESULTED_IN)
            except Exception as e:
                self._warnings.append(f"Could not parse eval results: {e}")

        # Create dataset artifact node
        dataset_id = self._make_id("dataset", run_id)
        dataset_node = LineageNode(
            id=dataset_id,
            type=NodeType.DATASET_ARTIFACT,
            label=f"Dataset: {run_meta.get('dataset_version', 'unknown')}",
            metadata={
                "dataset_version": run_meta.get("dataset_version"),
                "data_hash": run_meta.get("data_hash"),
            },
        )
        self._add_node(dataset_node)
        self._add_edge(dataset_id, train_id, RelationType.TRAINED_ON)

        # Load hashes.json for additional metadata
        hashes_path = run_path / "hashes.json"
        if hashes_path.exists():
            try:
                hashes_data = read_json(hashes_path)
                dataset_node.metadata["chain_hash"] = hashes_data.get("chain_hash")
                dataset_node.metadata["file_hashes"] = hashes_data.get("files", {})
            except Exception:
                pass

        # Load config.yaml for transform info
        config_path = run_path / "config.yaml"
        if config_path.exists():
            try:
                config = read_yaml(config_path)
                transform_id = self._make_id("transform", run_id)
                transform_node = LineageNode(
                    id=transform_id,
                    type=NodeType.TRANSFORM,
                    label="Transform",
                    metadata={
                        "config_keys": list(config.keys()),
                        "config_hash": run_meta.get("config_hash"),
                    },
                )
                self._add_node(transform_node)
                self._add_edge(transform_id, dataset_id, RelationType.TRANSFORMED_BY)
            except Exception as e:
                self._warnings.append(f"Could not read config.yaml: {e}")

        return decision_id

    def _discover_upstream(self, run_path: Path) -> None:
        """Discover upstream dependencies (raw datasets, parent runs)."""
        # Look for ingest artifacts or source file references
        config_path = run_path / "config.yaml"
        if not config_path.exists():
            return

        try:
            config = read_yaml(config_path)
            run_id = run_path.name

            # Check for train_data reference
            train_data = config.get("train_data") or config.get("dataset_path")
            if train_data:
                raw_id = self._make_id("raw", str(train_data))
                raw_node = LineageNode(
                    id=raw_id,
                    type=NodeType.RAW_DATASET,
                    label=f"Raw: {Path(train_data).name}",
                    metadata={
                        "path": str(train_data),
                    },
                )
                self._add_node(raw_node)

                # Connect to transform if exists
                transform_id = self._make_id("transform", run_id)
                if transform_id in self.nodes:
                    self._add_edge(raw_id, transform_id, RelationType.DERIVED_FROM)

        except Exception as e:
            self._warnings.append(f"Could not discover upstream: {e}")

    def _discover_downstream(self, run_path: Path) -> None:
        """Discover downstream artifacts (contamination checks, decisions)."""
        # Look for contamination results
        contam_path = run_path / "contamination_results.json"
        if contam_path.exists():
            try:
                contam_data = read_json(contam_path)
                run_id = run_path.name
                contam_id = self._make_id("contam", run_id)

                contam_node = LineageNode(
                    id=contam_id,
                    type=NodeType.CONTAMINATION_CHECK,
                    label="Contamination Check",
                    metadata={
                        "status": contam_data.get("status"),
                        "exact_overlaps": contam_data.get("exact_overlaps"),
                        "near_duplicates": contam_data.get("near_duplicates"),
                    },
                )
                self._add_node(contam_node)

                # Connect to dataset
                dataset_id = self._make_id("dataset", run_id)
                if dataset_id in self.nodes:
                    self._add_edge(dataset_id, contam_id, RelationType.CHECKED_BY)

                # Connect to decision
                decision_id = self._make_id("decision", run_id)
                if decision_id in self.nodes:
                    self._add_edge(contam_id, decision_id, RelationType.DEPENDS_ON)

            except Exception as e:
                self._warnings.append(f"Could not parse contamination results: {e}")

    @property
    def warnings(self) -> List[str]:
        """Return list of non-fatal warnings encountered during build."""
        return list(self._warnings)


def build_lineage_graph(run_dir: Union[str, Path]) -> LineageGraph:
    """Build a lineage graph from a run directory.

    This is the main entry point for constructing lineage graphs.

    Args:
        run_dir: Path to the run directory containing artifacts

    Returns:
        Complete LineageGraph

    Example:
        >>> graph = build_lineage_graph("runs/model_v1")
        >>> print(graph.to_mermaid())
        >>> print(graph.to_json())
    """
    builder = LineageGraphBuilder()
    return builder.build_from_run(run_dir)


def write_lineage_artifacts(graph: LineageGraph, output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Write lineage graph artifacts to a directory.

    Writes:
    - lineage_graph.json: Full graph in JSON format
    - lineage_graph.mmd: Mermaid diagram

    Args:
        graph: LineageGraph to export
        output_dir: Directory to write artifacts

    Returns:
        Dict mapping artifact type to file path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {}

    # Write JSON
    json_path = out / "lineage_graph.json"
    write_json(json_path, graph.to_dict())
    artifacts["json"] = json_path

    # Write Mermaid
    mmd_path = out / "lineage_graph.mmd"
    mmd_path.write_text(graph.to_mermaid())
    artifacts["mermaid"] = mmd_path

    return artifacts
