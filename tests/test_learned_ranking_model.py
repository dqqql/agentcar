from __future__ import annotations

import pytest
import torch

from backend.app.services.ranking.learned import (
    BilinearFusion,
    DistanceAwareAttention,
    LearnedRankingModel,
)


def make_inputs(batch_size: int = 3, embed_dim: int = 8):
    torch.manual_seed(7)
    return (
        torch.randn(batch_size, embed_dim),
        torch.randn(batch_size, embed_dim),
        torch.rand(batch_size, 1),
    )


def test_bilinear_fusion_returns_bounded_probabilities_and_features() -> None:
    user_vec, candidate_vec, objective_score = make_inputs()
    probability, interaction = BilinearFusion(embed_dim=8)(
        user_vec, candidate_vec, objective_score
    )

    assert probability.shape == (3, 1)
    assert interaction.shape == (3, 1)
    assert torch.all((probability >= 0) & (probability <= 1))


def test_distance_attention_uses_an_additive_monotonic_penalty() -> None:
    attention = DistanceAwareAttention(
        embed_dim=4, window_size=4, rho=2.0, distance_scale=100.0
    )
    with torch.no_grad():
        attention.relevance_projection.weight.copy_(torch.eye(4))

    query = torch.ones(4)
    identical_history = torch.ones(3, 4)
    _, weights = attention(query, identical_history, torch.tensor([0.0, 50.0, 100.0]))

    assert weights.shape == (3,)
    assert weights[0] >= weights[1] >= weights[2]
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_distance_attention_sanitizes_abnormal_distances_without_nonfinite_output() -> None:
    attention = DistanceAwareAttention(embed_dim=4, distance_scale=100.0)
    query = torch.randn(4)
    history = torch.randn(5, 4)

    context, weights = attention(
        query,
        history,
        torch.tensor([-10.0, float("nan"), float("inf"), float("-inf"), 1e30]),
    )

    assert torch.isfinite(context).all()
    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_model_supports_no_history_and_returns_expected_shapes() -> None:
    model = LearnedRankingModel(embed_dim=8)
    score, interaction, sequence_score = model(*make_inputs())

    assert score.shape == interaction.shape == sequence_score.shape == (3, 1)
    assert torch.all((score >= 0) & (score <= 1))
    assert torch.equal(sequence_score, torch.zeros_like(sequence_score))


def test_empty_history_is_equivalent_to_no_history() -> None:
    model = LearnedRankingModel(embed_dim=8).eval()
    inputs = make_inputs(batch_size=2)

    without_history = model(*inputs)
    with_empty_history = model(
        *inputs,
        history_seqs=[torch.empty(0, 8), torch.empty(0, 8)],
        history_distances=[torch.empty(0), torch.empty(0)],
    )

    for actual, expected in zip(with_empty_history, without_history):
        assert torch.equal(actual, expected)


def test_history_path_propagates_gradients_through_sequence_projection() -> None:
    model = LearnedRankingModel(embed_dim=8, max_history=3)
    inputs = make_inputs(batch_size=2)
    histories = [torch.randn(2, 8), torch.randn(3, 8)]
    distances = [torch.tensor([10.0, 20.0]), torch.tensor([5.0, 15.0, 25.0])]

    score, _, sequence_score = model(
        *inputs, history_seqs=histories, history_distances=distances
    )
    score.sum().backward()

    assert torch.all((sequence_score > 0) & (sequence_score < 1))
    gradient = model.sequence_projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_history_path_connects_every_trainable_model_parameter() -> None:
    torch.manual_seed(23)
    model = LearnedRankingModel(embed_dim=8, max_history=4)
    user_vec = torch.randn(4, 8, requires_grad=True)
    candidate_vec = torch.randn(4, 8, requires_grad=True)
    objective_score = torch.rand(4, 1, requires_grad=True)
    histories = [torch.randn(4, 8, requires_grad=True) for _ in range(4)]
    distances = [
        torch.tensor([10.0, 40.0, 90.0, 160.0]),
        torch.tensor([20.0, 50.0, 100.0, 170.0]),
        torch.tensor([30.0, 60.0, 110.0, 180.0]),
        torch.tensor([35.0, 70.0, 120.0, 200.0]),
    ]

    score, _, _ = model(
        user_vec,
        candidate_vec,
        objective_score,
        history_seqs=histories,
        history_distances=distances,
    )
    score.sum().backward()

    disconnected = []
    for name, parameter in model.named_parameters():
        if (
            parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or parameter.grad.abs().sum() == 0
        ):
            disconnected.append(name)
    assert disconnected == []
    for model_input in (user_vec, candidate_vec, objective_score, *histories):
        assert model_input.grad is not None
        assert torch.isfinite(model_input.grad).all()
        assert model_input.grad.abs().sum() > 0


def test_overlong_history_is_deterministically_truncated_to_most_recent_items() -> None:
    model = LearnedRankingModel(embed_dim=8, max_history=3).eval()
    inputs = make_inputs(batch_size=1)
    history = torch.randn(7, 8)
    distances = torch.arange(7, dtype=torch.float32)

    overlong = model(
        *inputs, history_seqs=[history], history_distances=[distances]
    )
    recent_window = model(
        *inputs, history_seqs=[history[-3:]], history_distances=[distances[-3:]]
    )

    for actual, expected in zip(overlong, recent_window):
        assert torch.allclose(actual, expected)


def test_nonempty_history_requires_aligned_distances() -> None:
    model = LearnedRankingModel(embed_dim=8)
    inputs = make_inputs(batch_size=1)

    with pytest.raises(ValueError, match="history_distances"):
        model(*inputs, history_seqs=[torch.randn(2, 8)])

    with pytest.raises(ValueError, match="same length"):
        model(
            *inputs,
            history_seqs=[torch.randn(2, 8)],
            history_distances=[torch.tensor([1.0])],
        )


@pytest.mark.parametrize(
    "invalid_history",
    [torch.tensor(1.0), torch.empty(0), torch.empty(2, 3, 8)],
)
def test_history_rejects_scalar_or_wrong_rank(invalid_history: torch.Tensor) -> None:
    model = LearnedRankingModel(embed_dim=8)
    inputs = make_inputs(batch_size=1)

    with pytest.raises(ValueError, match="history sequence must have shape"):
        model(
            *inputs,
            history_seqs=[invalid_history],
            history_distances=[torch.empty(0)],
        )


def test_empty_history_still_validates_embedding_width() -> None:
    model = LearnedRankingModel(embed_dim=8)
    inputs = make_inputs(batch_size=1)

    with pytest.raises(ValueError, match="history sequence must have shape"):
        model(
            *inputs,
            history_seqs=[torch.empty(0, 7)],
            history_distances=[torch.empty(0)],
        )


@pytest.mark.parametrize(
    "invalid_distances",
    [torch.tensor(1.0), torch.empty(0, 1), torch.empty(1, 0, 1)],
)
def test_history_rejects_scalar_or_wrong_rank_distances(
    invalid_distances: torch.Tensor,
) -> None:
    model = LearnedRankingModel(embed_dim=8)
    inputs = make_inputs(batch_size=1)

    with pytest.raises(ValueError, match="history sequence and distances"):
        model(
            *inputs,
            history_seqs=[torch.empty(0, 8)],
            history_distances=[invalid_distances],
        )


def test_empty_history_rejects_misaligned_distances() -> None:
    model = LearnedRankingModel(embed_dim=8)
    inputs = make_inputs(batch_size=1)

    with pytest.raises(ValueError, match="same length"):
        model(
            *inputs,
            history_seqs=[torch.empty(0, 8)],
            history_distances=[torch.tensor([1.0])],
        )


def test_empty_candidate_batch_returns_empty_outputs() -> None:
    model = LearnedRankingModel(embed_dim=8)
    empty_vectors = torch.empty(0, 8)
    outputs = model(empty_vectors, empty_vectors.clone(), torch.empty(0, 1))

    assert all(output.shape == (0, 1) for output in outputs)
