"""Fusion 전략 테스트"""

import pytest

from src.core.models import Candidate
from src.search.adapters.fusion import (
    CombSumFusion,
    FusionStrategy,
    ReciprocalRankFusion,
    WeightedFusion,
)


@pytest.fixture
def sample_candidates():
    """샘플 Candidate 리스트"""
    return [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={
                "lexical_score": 0.9,
                "semantic_score": 0.7,
                "graph_score": 0.5,
                "fuzzy_score": 0.3,
            },
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk2",
            features={
                "lexical_score": 0.5,
                "semantic_score": 0.9,
                "graph_score": 0.7,
                "fuzzy_score": 0.6,
            },
            file_path="file2.py",
            span=(0, 0, 10, 0),
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk3",
            features={
                "lexical_score": 0.3,
                "semantic_score": 0.5,
                "graph_score": 0.9,
                "fuzzy_score": 0.8,
            },
            file_path="file3.py",
            span=(0, 0, 10, 0),
        ),
    ]


def test_fusion_strategy_interface():
    """FusionStrategy가 추상 클래스인지 확인"""
    from abc import ABC

    assert issubclass(FusionStrategy, ABC)

    # 인스턴스화 불가능
    with pytest.raises(TypeError):
        FusionStrategy()  # type: ignore


def test_weighted_fusion_basic(sample_candidates):
    """WeightedFusion 기본 테스트"""
    fusion = WeightedFusion()
    weights = {"lexical": 0.3, "semantic": 0.5, "graph": 0.2, "fuzzy": 0.0}

    result = fusion.fuse(sample_candidates, weights)

    assert len(result) == 3
    assert all("final_score" in c.features for c in result)
    assert all("total_score" in c.features for c in result)

    # chunk2가 가장 높은 semantic_score를 가지고 semantic 가중치가 높음
    chunk2 = next(c for c in result if c.chunk_id == "chunk2")
    expected_score = 0.5 * 0.3 + 0.9 * 0.5 + 0.7 * 0.2  # lexical, semantic, graph
    assert abs(chunk2.features["final_score"] - expected_score) < 0.001


def test_weighted_fusion_default_weights(sample_candidates):
    """WeightedFusion 기본 가중치 테스트"""
    fusion = WeightedFusion()

    # weights=None이면 기본 가중치 사용
    result = fusion.fuse(sample_candidates, weights=None)

    assert len(result) == 3
    assert all("final_score" in c.features for c in result)

    # 기본 가중치: lexical=0.25, semantic=0.45, graph=0.15, fuzzy=0.15
    chunk1 = next(c for c in result if c.chunk_id == "chunk1")
    expected_score = 0.9 * 0.25 + 0.7 * 0.45 + 0.5 * 0.15 + 0.3 * 0.15
    assert abs(chunk1.features["final_score"] - expected_score) < 0.001


def test_weighted_fusion_and_sort(sample_candidates):
    """WeightedFusion 정렬 테스트"""
    fusion = WeightedFusion()
    weights = {"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15}

    result = fusion.fuse_and_sort(sample_candidates, weights, k=None)

    # 정렬 확인
    scores = [c.features["final_score"] for c in result]
    assert scores == sorted(scores, reverse=True), "점수 기준 내림차순 정렬되어야 함"


def test_weighted_fusion_top_k(sample_candidates):
    """WeightedFusion 상위 k개 반환 테스트"""
    fusion = WeightedFusion()
    weights = {"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15}

    result = fusion.fuse_and_sort(sample_candidates, weights, k=2)

    assert len(result) == 2, "상위 2개만 반환되어야 함"

    # 가장 높은 점수 2개
    scores = [c.features["final_score"] for c in result]
    assert scores == sorted(scores, reverse=True)


def test_weighted_fusion_contributions(sample_candidates):
    """WeightedFusion 개별 기여도 테스트"""
    fusion = WeightedFusion()
    weights = {"lexical": 0.3, "semantic": 0.5, "graph": 0.2, "fuzzy": 0.0}

    result = fusion.fuse(sample_candidates, weights)

    # weighted_contributions 확인
    for candidate in result:
        assert "weighted_contributions" in candidate.features
        contributions = candidate.features["weighted_contributions"]

        # 개별 기여도 합 = final_score
        total_contribution = sum(contributions.values())
        assert abs(total_contribution - candidate.features["final_score"]) < 0.001


def test_weighted_fusion_zero_scores():
    """점수가 0인 경우 테스트"""
    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={
                "lexical_score": 0.0,
                "semantic_score": 0.0,
                "graph_score": 0.0,
                "fuzzy_score": 0.0,
            },
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
    ]

    fusion = WeightedFusion()
    weights = {"lexical": 0.25, "semantic": 0.45, "graph": 0.15, "fuzzy": 0.15}

    result = fusion.fuse(candidates, weights)

    assert len(result) == 1
    assert result[0].features["final_score"] == 0.0


def test_weighted_fusion_missing_scores():
    """일부 점수가 없는 경우 테스트"""
    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={
                "lexical_score": 0.8,
                # semantic_score 없음
                "graph_score": 0.6,
                # fuzzy_score 없음
            },
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
    ]

    fusion = WeightedFusion()
    weights = {"lexical": 0.3, "semantic": 0.5, "graph": 0.2, "fuzzy": 0.0}

    result = fusion.fuse(candidates, weights)

    # 없는 점수는 0으로 처리
    expected_score = 0.8 * 0.3 + 0.0 * 0.5 + 0.6 * 0.2 + 0.0 * 0.0
    assert abs(result[0].features["final_score"] - expected_score) < 0.001


def test_weighted_fusion_is_fusion_strategy():
    """WeightedFusion이 FusionStrategy를 구현하는지 확인"""
    fusion = WeightedFusion()
    assert isinstance(fusion, FusionStrategy)


# ===== RRF 테스트 =====


def test_rrf_basic(sample_candidates):
    """RRF 기본 테스트"""
    fusion = ReciprocalRankFusion(k=60)

    result = fusion.fuse(sample_candidates, weights=None)

    assert len(result) == 3
    assert all("final_score" in c.features for c in result)
    assert all("rrf_score" in c.features for c in result)
    assert all("rrf_rank_details" in c.features for c in result)


def test_rrf_ranking():
    """RRF 순위 계산 테스트"""
    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={
                "lexical_score": 1.0,  # lexical 1위
                "semantic_score": 0.5,  # semantic 2위
            },
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk2",
            features={
                "lexical_score": 0.5,  # lexical 2위
                "semantic_score": 1.0,  # semantic 1위
            },
            file_path="file2.py",
            span=(0, 0, 10, 0),
        ),
    ]

    fusion = ReciprocalRankFusion(k=60)
    result = fusion.fuse(candidates, weights=None)

    # 두 청크 모두 한 곳에서 1위, 다른 곳에서 2위
    # RRF 점수가 동일해야 함
    chunk1_score = result[0].features["rrf_score"]
    chunk2_score = result[1].features["rrf_score"]

    # 1/(60+0) + 1/(60+1) = 0.0167 + 0.0164 = 0.0331
    expected_score = 1 / (60 + 0) + 1 / (60 + 1)
    assert abs(chunk1_score - expected_score) < 0.001
    assert abs(chunk2_score - expected_score) < 0.001


def test_rrf_k_parameter():
    """RRF k 파라미터 테스트"""
    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={"lexical_score": 1.0},
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
    ]

    # k=1일 때
    fusion_k1 = ReciprocalRankFusion(k=1)
    result_k1 = fusion_k1.fuse(candidates.copy(), weights=None)
    score_k1 = result_k1[0].features["rrf_score"]

    # k=100일 때
    fusion_k100 = ReciprocalRankFusion(k=100)
    result_k100 = fusion_k100.fuse(candidates.copy(), weights=None)
    score_k100 = result_k100[0].features["rrf_score"]

    # k가 작을수록 점수가 높아야 함
    assert score_k1 > score_k100


def test_rrf_invalid_k():
    """잘못된 k 값 테스트"""
    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=0)

    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=-1)


def test_rrf_fuse_and_sort(sample_candidates):
    """RRF 정렬 테스트"""
    fusion = ReciprocalRankFusion(k=60)

    result = fusion.fuse_and_sort(sample_candidates, weights=None, k=None)

    # 정렬 확인
    scores = [c.features["final_score"] for c in result]
    assert scores == sorted(scores, reverse=True), "점수 기준 내림차순 정렬되어야 함"


def test_rrf_top_k(sample_candidates):
    """RRF 상위 k개 반환 테스트"""
    fusion = ReciprocalRankFusion(k=60)

    result = fusion.fuse_and_sort(sample_candidates, weights=None, k=2)

    assert len(result) == 2, "상위 2개만 반환되어야 함"


def test_rrf_missing_backend():
    """일부 백엔드 점수가 없는 경우"""
    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={
                "lexical_score": 0.8,
                # semantic_score 없음
            },
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
        Candidate(
            repo_id="test",
            chunk_id="chunk2",
            features={
                "semantic_score": 0.9,
                # lexical_score 없음
            },
            file_path="file2.py",
            span=(0, 0, 10, 0),
        ),
    ]

    fusion = ReciprocalRankFusion(k=60)
    result = fusion.fuse(candidates, weights=None)

    # 각 청크는 자신이 있는 백엔드에서만 순위를 받음
    assert len(result) == 2
    assert all("rrf_score" in c.features for c in result)


def test_rrf_rank_details(sample_candidates):
    """RRF 순위 세부정보 테스트"""
    fusion = ReciprocalRankFusion(k=60)
    result = fusion.fuse(sample_candidates, weights=None)

    for candidate in result:
        rank_details = candidate.features["rrf_rank_details"]

        # 각 백엔드별 순위와 기여도가 있어야 함
        assert isinstance(rank_details, dict)

        # rank와 contribution이 있어야 함
        for backend, details in rank_details.items():
            assert "rank" in details
            assert "contribution" in details
            assert details["rank"] >= 0
            assert details["contribution"] > 0


def test_rrf_is_fusion_strategy():
    """RRF가 FusionStrategy를 구현하는지 확인"""
    fusion = ReciprocalRankFusion(k=60)
    assert isinstance(fusion, FusionStrategy)


# ===== CombSum 테스트 =====


def test_combsum_weighted(sample_candidates):
    """CombSum 가중 합산 테스트"""
    fusion = CombSumFusion(use_weights=True)
    weights = {"lexical": 1.0, "semantic": 1.5, "graph": 0.5, "fuzzy": 0.5}

    result = fusion.fuse(sample_candidates, weights)

    assert len(result) == 3
    assert all("final_score" in c.features for c in result)
    assert all("combsum_weighted" in c.features for c in result)
    assert all(c.features["combsum_weighted"] is True for c in result)


def test_combsum_unweighted(sample_candidates):
    """CombSum 단순 합산 테스트"""
    fusion = CombSumFusion(use_weights=False)

    result = fusion.fuse(sample_candidates, weights=None)

    assert len(result) == 3
    assert all("final_score" in c.features for c in result)
    assert all(c.features["combsum_weighted"] is False for c in result)

    # 단순 합산: 모든 점수의 합
    chunk1 = next(c for c in result if c.chunk_id == "chunk1")
    expected = 0.9 + 0.7 + 0.5 + 0.3  # lexical + semantic + graph + fuzzy
    assert abs(chunk1.features["final_score"] - expected) < 0.001


def test_combsum_default_weights():
    """CombSum 기본 가중치 테스트"""
    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={
                "lexical_score": 0.8,
                "semantic_score": 0.6,
            },
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
    ]

    fusion = CombSumFusion(use_weights=True)
    result = fusion.fuse(candidates, weights=None)

    # 기본 가중치 1.0으로 합산
    expected = 0.8 * 1.0 + 0.6 * 1.0  # lexical + semantic (graph, fuzzy는 0)
    assert abs(result[0].features["final_score"] - expected) < 0.001


def test_combsum_fuse_and_sort(sample_candidates):
    """CombSum 정렬 테스트"""
    fusion = CombSumFusion(use_weights=False)

    result = fusion.fuse_and_sort(sample_candidates, weights=None, k=None)

    # 정렬 확인
    scores = [c.features["final_score"] for c in result]
    assert scores == sorted(scores, reverse=True), "점수 기준 내림차순 정렬되어야 함"


def test_combsum_top_k(sample_candidates):
    """CombSum 상위 k개 반환 테스트"""
    fusion = CombSumFusion(use_weights=True)
    weights = {"lexical": 1.0, "semantic": 1.0, "graph": 1.0, "fuzzy": 1.0}

    result = fusion.fuse_and_sort(sample_candidates, weights, k=2)

    assert len(result) == 2, "상위 2개만 반환되어야 함"


def test_combsum_vs_weighted():
    """CombSum과 WeightedFusion 비교"""
    candidates = [
        Candidate(
            repo_id="test",
            chunk_id="chunk1",
            features={
                "lexical_score": 0.8,
                "semantic_score": 0.6,
                "graph_score": 0.4,
            },
            file_path="file1.py",
            span=(0, 0, 10, 0),
        ),
    ]

    # CombSum (가중치 1.0)
    combsum = CombSumFusion(use_weights=True)
    weights = {"lexical": 1.0, "semantic": 1.0, "graph": 1.0, "fuzzy": 0.0}
    result_combsum = combsum.fuse(candidates.copy(), weights)

    # WeightedFusion (가중치 1.0)
    weighted = WeightedFusion()
    result_weighted = weighted.fuse(candidates.copy(), weights)

    # 가중치가 같으면 결과도 같아야 함
    assert (
        abs(result_combsum[0].features["final_score"] - result_weighted[0].features["final_score"])
        < 0.001
    )


def test_combsum_is_fusion_strategy():
    """CombSum이 FusionStrategy를 구현하는지 확인"""
    fusion = CombSumFusion(use_weights=True)
    assert isinstance(fusion, FusionStrategy)
