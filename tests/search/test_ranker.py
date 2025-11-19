"""Ranker 테스트"""


from src.core.models import Candidate
from src.search.adapters.ranking.ranker import Ranker


def test_ranker_basic():
    """기본 랭킹 테스트"""
    ranker = Ranker()

    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk-1",
            features={"lexical_score": 0.5, "semantic_score": 0.8},
            file_path="test.py",
            span=(0, 0, 10, 0)
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk-2",
            features={"lexical_score": 0.9, "semantic_score": 0.3},
            file_path="test.py",
            span=(10, 0, 20, 0)
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk-3",
            features={"lexical_score": 0.2, "semantic_score": 0.2},
            file_path="test.py",
            span=(20, 0, 30, 0)
        ),
    ]

    ranked = ranker.rank(candidates, max_items=2)

    assert len(ranked) == 2
    assert all("final_score" in c.features for c in ranked)

    # 점수가 높은 순서로 정렬되어야 함
    assert ranked[0].features["final_score"] >= ranked[1].features["final_score"]


def test_ranker_custom_weights():
    """커스텀 가중치 테스트"""
    ranker = Ranker(
        feature_weights={
            "lexical_score": 1.0,
            "semantic_score": 0.0,  # semantic 무시
        }
    )

    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk-1",
            features={"lexical_score": 0.5, "semantic_score": 0.9},
            file_path="test.py",
            span=(0, 0, 10, 0)
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk-2",
            features={"lexical_score": 0.8, "semantic_score": 0.1},
            file_path="test.py",
            span=(10, 0, 20, 0)
        ),
    ]

    ranked = ranker.rank(candidates, max_items=10)

    # lexical_score만 반영되므로 chunk-2가 1위
    assert ranked[0].chunk_id == "chunk-2"


def test_ranker_empty_candidates():
    """빈 후보 리스트 테스트"""
    ranker = Ranker()

    ranked = ranker.rank([], max_items=10)

    assert len(ranked) == 0


def test_ranker_update_weights():
    """가중치 업데이트 테스트"""
    ranker = Ranker()

    ranker.update_weights({"lexical_score": 0.5})

    assert ranker.feature_weights["lexical_score"] == 0.5
    assert ranker.feature_weights["semantic_score"] == 0.5  # 기존 값 유지

