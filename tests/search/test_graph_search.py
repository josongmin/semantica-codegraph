"""GraphSearch 테스트"""

from unittest.mock import MagicMock

from src.core.models import CodeNode
from src.search.adapters.graph.postgres_graph_adapter import PostgresGraphSearch


def test_graph_search_get_node():
    """노드 조회 테스트"""
    graph_store = MagicMock()
    graph_store.get_node.return_value = CodeNode(
        repo_id="test",
        id="node-1",
        kind="Function",
        language="python",
        file_path="test.py",
        span=(0, 0, 10, 0),
        name="test_func",
        text="def test_func(): pass"
    )

    search = PostgresGraphSearch(graph_store)

    node = search.get_node("test", "node-1")

    assert node is not None
    assert node.id == "node-1"
    graph_store.get_node.assert_called_once_with("test", "node-1")


def test_graph_search_get_node_by_location():
    """위치 기반 노드 조회 테스트"""
    graph_store = MagicMock()
    graph_store.get_node_by_location.return_value = CodeNode(
        repo_id="test",
        id="node-1",
        kind="Function",
        language="python",
        file_path="test.py",
        span=(0, 0, 10, 0),
        name="test_func",
        text="def test_func(): pass"
    )

    search = PostgresGraphSearch(graph_store)

    node = search.get_node_by_location("test", "test.py", 5, 0)

    assert node is not None
    assert node.file_path == "test.py"
    graph_store.get_node_by_location.assert_called_once_with("test", "test.py", 5, 0)


def test_graph_search_expand_neighbors():
    """이웃 확장 테스트"""
    graph_store = MagicMock()
    graph_store.neighbors.return_value = [
        CodeNode(
            repo_id="test",
            id="node-2",
            kind="Function",
            language="python",
            file_path="test.py",
            span=(10, 0, 20, 0),
            name="neighbor_func",
            text="def neighbor_func(): pass"
        )
    ]

    search = PostgresGraphSearch(graph_store)

    neighbors = search.expand_neighbors("test", "node-1", k=1)

    assert len(neighbors) == 1
    assert neighbors[0].id == "node-2"
    graph_store.neighbors.assert_called_once()


def test_graph_search_expand_neighbors_zero_k():
    """k=0 이웃 확장 테스트"""
    graph_store = MagicMock()

    search = PostgresGraphSearch(graph_store)

    neighbors = search.expand_neighbors("test", "node-1", k=0)

    assert len(neighbors) == 0
    graph_store.neighbors.assert_not_called()

