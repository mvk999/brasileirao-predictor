"""Testes da correção de baixo placar Dixon-Coles."""

import numpy as np
import pandas as pd

from src.model.experiment_dixon_coles import _decode, _observed_tau, outcome_probabilities, temporal_weights


def test_tau_uses_dixon_coles_adjustments_only_on_four_low_scores() -> None:
    lam = np.full(5, 1.2)
    mu = np.full(5, 0.8)
    home_goals = np.array([0, 0, 1, 1, 2])
    away_goals = np.array([0, 1, 0, 1, 2])

    tau = _observed_tau(lam, mu, home_goals, away_goals, rho=-0.1)

    np.testing.assert_allclose(tau, [1.096, 0.88, 0.92, 1.1, 1.0])


def test_dixon_coles_outcomes_remain_normalized_and_modify_draw_probability() -> None:
    lam = np.array([1.2, 0.7])
    mu = np.array([0.8, 1.1])

    independent = outcome_probabilities(lam, mu)
    corrected = outcome_probabilities(lam, mu, rho=-0.1)

    np.testing.assert_allclose(independent.sum(axis=1), 1)
    np.testing.assert_allclose(corrected.sum(axis=1), 1)
    assert np.all(corrected[:, 1] > independent[:, 1])
    assert np.all(corrected >= 0)


def test_attack_identifiability_constraint_sets_mean_log_attack_to_zero() -> None:
    # Three teams: two free attack parameters, three defenses, home effect and rho.
    parameters = np.array([0.3, -0.2, 0.1, -0.1, 0.2, 0.05, -1.0])

    log_attack, _, _, _ = _decode(parameters, team_count=3, dixon_coles=True)

    assert np.isclose(log_attack.mean(), 0)


def test_temporal_weights_halve_at_the_selected_half_life_and_reject_future_data() -> None:
    dates = pd.Series(pd.to_datetime(["2022-01-01", "2022-07-02", "2023-01-01"]))
    cutoff = pd.Timestamp("2023-01-01")

    weights = temporal_weights(dates, cutoff, half_life_days=365)

    assert np.isclose(weights[0], 0.5)
    assert 0.70 < weights[1] < 0.72
    assert np.isclose(weights[2], 1)
    with np.testing.assert_raises(ValueError):
        temporal_weights(pd.Series(pd.to_datetime(["2023-01-02"])), cutoff, 365)
