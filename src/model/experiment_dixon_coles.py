"""Walk-forward ablation: Poisson, Dixon-Coles, and Dixon-Coles with decay.

Uso: python -m src.model.experiment_dixon_coles
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import poisson, skellam
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.data.prepare_matches import load_raw_matches, prepare_matches
from src.model.train_baseline import ARTIFACTS_DIR, CLASSES, PROJECT_ROOT


YEARS = (*range(2006, 2016), *range(2017, 2025))
MIN_TRAIN_SEASONS = 4
VARIANTS = ("A", "B", "C")
HALF_LIVES_DAYS = (90, 180, 365, 730)
REPORT_PATH = ARTIFACTS_DIR / "dixon_coles_experiment.json"
def load_complete_matches() -> pd.DataFrame:
    matches = prepare_matches(load_raw_matches(), seasons=YEARS)
    counts = matches["temporada"].value_counts().sort_index().to_dict()
    if counts != {year: 380 for year in YEARS}:
        raise ValueError(f"Esperadas temporadas completas {YEARS}; encontrado {counts}.")
    if (not matches["id"].is_unique or matches.isna().any().any()
            or (matches[["gols_mandante", "gols_visitante"]] < 0).any().any()):
        raise ValueError("O histórico precisa ter IDs únicos, colunas completas e gols não negativos.")
    for year, season in matches.groupby("temporada"):
        teams = set(season["mandante"]) | set(season["visitante"])
        if (len(teams) != 20 or season["rodada"].nunique() != 38
                or not season.groupby("rodada").size().eq(10).all()):
            raise ValueError(f"Temporada {year} precisa ter 20 clubes e 38 rodadas de dez partidas.")
    return matches.sort_values(["data", "id"]).reset_index(drop=True)


def _decode(parameters: np.ndarray, team_count: int, dixon_coles: bool):
    attack_free = parameters[: team_count - 1]
    log_attack = np.concatenate([attack_free, [-attack_free.sum()]])
    log_defence = parameters[team_count - 1 : 2 * team_count - 1]
    log_home = parameters[2 * team_count - 1]
    rho_raw = parameters[2 * team_count] if dixon_coles else 0.0
    return log_attack, log_defence, log_home, rho_raw


def _rates(parameters, team_count, home_index, away_index, dixon_coles, estimate_rho=True):
    log_attack, log_defence, log_home, rho_raw = _decode(parameters, team_count, dixon_coles)
    # Unseen test clubs receive neutral attack and defense (multiplicative strength 1).
    home_attack = np.where(home_index >= 0, log_attack[np.maximum(home_index, 0)], 0.0)
    home_defence = np.where(home_index >= 0, log_defence[np.maximum(home_index, 0)], 0.0)
    away_attack = np.where(away_index >= 0, log_attack[np.maximum(away_index, 0)], 0.0)
    away_defence = np.where(away_index >= 0, log_defence[np.maximum(away_index, 0)], 0.0)
    lam = np.exp(np.clip(home_attack + away_defence + log_home, -8, 5))
    mu = np.exp(np.clip(away_attack + home_defence, -8, 5))
    rho = 0.0
    if dixon_coles and estimate_rho:
        # Parameterize rho inside bounds that keep all four low-score tau values positive.
        lower = max(-0.5, -1.0 / float(lam.max()), -1.0 / float(mu.max()))
        upper = min(0.5, 1.0 / float((lam * mu).max()))
        rho = lower + (upper - lower) * expit(rho_raw)
    return lam, mu, rho


def _observed_tau(lam, mu, home_goals, away_goals, rho):
    tau = np.ones(len(lam), dtype=float)
    score_00 = (home_goals == 0) & (away_goals == 0)
    score_01 = (home_goals == 0) & (away_goals == 1)
    score_10 = (home_goals == 1) & (away_goals == 0)
    score_11 = (home_goals == 1) & (away_goals == 1)
    tau[score_00] = 1 - lam[score_00] * mu[score_00] * rho
    tau[score_01] = 1 + lam[score_01] * rho
    tau[score_10] = 1 + mu[score_10] * rho
    tau[score_11] = 1 - rho
    return tau


def outcome_probabilities(lam: np.ndarray, mu: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """Agrega a distribuição conjunta em A/D/H; tau altera apenas placares baixos."""
    away = skellam.cdf(-1, lam, mu)
    draw = skellam.pmf(0, lam, mu)
    home = skellam.sf(0, lam, mu)
    if rho:
        p00 = np.exp(-lam - mu)
        p01 = p00 * mu
        p10 = p00 * lam
        p11 = p00 * lam * mu
        draw += -lam * mu * rho * p00 - rho * p11
        away += lam * rho * p01
        home += mu * rho * p10
    probabilities = np.column_stack([away, draw, home])
    if (not np.isfinite(probabilities).all() or (probabilities < -1e-10).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)):
        raise ValueError("Probabilidades Dixon-Coles inválidas ou não normalizadas.")
    return np.clip(probabilities, 0.0, 1.0)


def temporal_weights(dates: pd.Series, cutoff_date: pd.Timestamp, half_life_days: int) -> np.ndarray:
    if half_life_days <= 0:
        raise ValueError("A meia-vida temporal precisa ser positiva.")
    days_old = (cutoff_date - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    if np.any(days_old < 0):
        raise ValueError("O corte temporal precede alguma partida do treino.")
    xi = np.log(2.0) / half_life_days
    return np.exp(-xi * days_old)


def fit_model(
    train: pd.DataFrame,
    variant: str,
    cutoff_date: pd.Timestamp,
    half_life_days: int | None = None,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Variante desconhecida: {variant}")
    if variant == "C" and (half_life_days is None or half_life_days <= 0):
        raise ValueError("A variante C precisa de uma meia-vida positiva em dias.")
    teams = sorted(set(train["mandante"]) | set(train["visitante"]))
    team_map = {team: index for index, team in enumerate(teams)}
    home_index = train["mandante"].map(team_map).to_numpy(dtype=int)
    away_index = train["visitante"].map(team_map).to_numpy(dtype=int)
    home_goals = train["gols_mandante"].to_numpy(dtype=int)
    away_goals = train["gols_visitante"].to_numpy(dtype=int)
    dixon_coles = variant in ("B", "C")
    weights = np.ones(len(train), dtype=float)
    if variant == "C":
        weights = temporal_weights(train["data"], cutoff_date, half_life_days)

    parameter_count = (len(teams) - 1) + len(teams) + 1 + int(dixon_coles)
    initial = np.zeros(parameter_count, dtype=float)
    if dixon_coles:
        initial[-1] = -1.0
    bounds = [(-2.0, 2.0)] * (len(teams) - 1)
    bounds += [(-2.0, 2.0)] * len(teams)
    bounds += [(-1.0, 1.0)]
    if dixon_coles:
        bounds += [(-8.0, 8.0)]

    def negative_log_likelihood(parameters):
        lam, mu, rho = _rates(parameters, len(teams), home_index, away_index, dixon_coles)
        tau = _observed_tau(lam, mu, home_goals, away_goals, rho)
        if np.any(tau <= 0) or not np.isfinite(tau).all():
            return 1e100
        log_likelihood = (
            poisson.logpmf(home_goals, lam)
            + poisson.logpmf(away_goals, mu)
            + np.log(tau)
        )
        return float(-np.dot(weights, log_likelihood))

    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000, "maxfun": 60000, "ftol": 1e-9, "maxls": 50},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Ajuste da variante {variant} falhou: {result.message}")
    log_attack, log_defence, log_home, rho_raw = _decode(result.x, len(teams), dixon_coles)
    train_lam, train_mu, rho = _rates(result.x, len(teams), home_index, away_index, dixon_coles)
    if not result.success:
        raise RuntimeError(f"Ajuste da variante {variant} não convergiu: {result.message}")
    return {
        "teams": teams,
        "log_attack": log_attack,
        "log_defence": log_defence,
        "log_home": float(log_home),
        "rho": float(rho),
        "rho_raw": float(rho_raw),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "weights": weights,
        "train_lambda": train_lam,
        "train_mu": train_mu,
        "variant": variant,
    }


def predict_probabilities(model: dict, evaluation: pd.DataFrame) -> np.ndarray:
    team_map = {team: index for index, team in enumerate(model["teams"])}
    home_index = evaluation["mandante"].map(team_map).fillna(-1).to_numpy(dtype=int)
    away_index = evaluation["visitante"].map(team_map).fillna(-1).to_numpy(dtype=int)
    lam, mu, _ = _rates(
        np.r_[model["log_attack"][:-1], model["log_defence"], model["log_home"], model["rho_raw"]]
        if model["variant"] in ("B", "C")
        else np.r_[model["log_attack"][:-1], model["log_defence"], model["log_home"]],
        len(model["teams"]), home_index, away_index,
        model["variant"] in ("B", "C"), estimate_rho=False,
    )
    rho = model["rho"] if model["variant"] in ("B", "C") else 0.0
    return outcome_probabilities(lam, mu, rho)


def _brier(actual: pd.Series, probabilities: np.ndarray) -> float:
    one_hot = np.column_stack([(actual.to_numpy() == label).astype(float) for label in CLASSES])
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def summarize(actual: pd.Series, probabilities: np.ndarray) -> dict:
    predicted = np.asarray(CLASSES)[np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1_macro": float(f1_score(actual, predicted, labels=CLASSES, average="macro")),
        "log_loss": float(log_loss(actual, probabilities, labels=CLASSES)),
        "brier_score": _brier(actual, probabilities),
        "by_class": {
            label: {
                "actual": int(support[index]),
                "predicted": int((predicted == label).sum()),
                "correct": int(((actual.to_numpy() == label) & (predicted == label)).sum()),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
        "draw_correct_predicted": {
            "correct": int(((actual.to_numpy() == "D") & (predicted == "D")).sum()),
            "predicted": int((predicted == "D").sum()),
        },
    }


def confidence_bands(actual: pd.Series, probabilities: np.ndarray) -> list[dict]:
    predicted = np.asarray(CLASSES)[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predicted == actual.to_numpy()
    edges = (0.0, 0.4, 0.5, 0.6, 0.7, 1.0000001)
    bands = []
    for low, high in zip(edges, edges[1:]):
        selected = (confidence >= low) & (confidence < high)
        count = int(selected.sum())
        bands.append({
            "from": low,
            "to": min(high, 1.0),
            "games": count,
            "mean_confidence": float(confidence[selected].mean()) if count else None,
            "observed_accuracy": float(correct[selected].mean()) if count else None,
        })
    return bands


def select_decay_half_life(train: pd.DataFrame, outer_cutoff: pd.Timestamp) -> dict:
    seasons = sorted(train["temporada"].unique())
    inner_year = seasons[-1]
    inner_train = train[train["temporada"] < inner_year]
    inner_validation = train[train["temporada"] == inner_year]
    if inner_train.empty or inner_validation.empty:
        raise ValueError("Treino insuficiente para escolher ξ dentro do fold.")
    if inner_train["data"].max() > inner_validation["data"].min():
        raise ValueError(f"Vazamento temporal no corte interno de {inner_year}.")
    choices = {}
    for half_life in HALF_LIVES_DAYS:
        fitted = fit_model(inner_train, "C", inner_validation["data"].min(), half_life)
        probabilities = predict_probabilities(fitted, inner_validation)
        choices[str(half_life)] = {
            "log_loss": float(log_loss(inner_validation["resultado"], probabilities, labels=CLASSES)),
            "converged_iterations": fitted["iterations"],
        }
    selected = min(HALF_LIVES_DAYS, key=lambda value: (choices[str(value)]["log_loss"], value))
    return {"selected_half_life_days": selected, "inner_validation_year": int(inner_year), "scores": choices}


def _metric_mean_std(rows: list[dict], path: tuple[str, ...]) -> dict:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def main() -> None:
    matches = load_complete_matches()
    seasons = [int(year) for year in sorted(matches["temporada"].unique())]
    test_years = seasons[MIN_TRAIN_SEASONS:]
    report = {
        "source_years": seasons,
        "source_games": int(len(matches)),
        "test_years": test_years,
        "folds": {},
        "variants": {},
        "method": {
            "identifiability": "mean(log_attack)=0 via last attack parameter = negative sum of other attacks",
            "rho_constraints": "smooth parameterization within feasible tau-positive bounds, capped at [-0.5, 0.5]",
            "time_decay": "weight=exp(-xi*days_since_cutoff); xi=ln(2)/half_life_days",
            "xi_selection": "nested temporal validation on the latest season inside each outer fold; choose minimum log loss among 90, 180, 365, 730 days",
            "unknown_test_teams": "neutral attack=defense=1",
            "brier_definition": "mean over games of sum of squared differences across H/D/A probabilities",
            "confidence_edges": [0.0, 0.4, 0.5, 0.6, 0.7, 1.0],
        },
    }
    pooled = {variant: {"actual": [], "probabilities": []} for variant in VARIANTS}
    for year in test_years:
        train = matches[matches["temporada"] < year].copy()
        evaluation = matches[matches["temporada"] == year].copy()
        train = train.sort_values(["data", "id"]).reset_index(drop=True)
        evaluation = evaluation.sort_values(["data", "id"]).reset_index(drop=True)
        if len(evaluation) != 380 or train.empty or train["data"].max() > evaluation["data"].min():
            raise ValueError(f"Fold {year} incompleto ou com vazamento temporal.")
        cutoff = evaluation["data"].min()
        report["folds"][str(year)] = {
            "training_years": sorted(train["temporada"].unique().tolist()),
            "training_games": int(len(train)),
            "test_games": int(len(evaluation)),
            "train_last_date": train["data"].max().isoformat(),
            "test_first_date": cutoff.isoformat(),
            "metrics": {},
            "parameters": {},
        }
        decay = select_decay_half_life(train, cutoff)
        report["folds"][str(year)]["xi_selection"] = decay
        for variant in VARIANTS:
            half_life = decay["selected_half_life_days"] if variant == "C" else None
            fitted = fit_model(train, variant, cutoff, half_life)
            probabilities = predict_probabilities(fitted, evaluation)
            metrics = summarize(evaluation["resultado"], probabilities)
            report["folds"][str(year)]["metrics"][variant] = metrics
            report["folds"][str(year)]["parameters"][variant] = {
                "rho": fitted["rho"],
                "home_advantage_multiplier": float(np.exp(fitted["log_home"])),
                "selected_half_life_days": half_life,
                "objective": fitted["objective"],
                "iterations": fitted["iterations"],
            }
            pooled[variant]["actual"].extend(evaluation["resultado"].tolist())
            pooled[variant]["probabilities"].extend(probabilities.tolist())
        print(f"Fold {year}: A={report['folds'][str(year)]['metrics']['A']['f1_macro']:.3f} "
              f"B={report['folds'][str(year)]['metrics']['B']['f1_macro']:.3f} "
              f"C={report['folds'][str(year)]['metrics']['C']['f1_macro']:.3f}; "
              f"ξ meia-vida={decay['selected_half_life_days']} dias")

    summary_paths = {
        "accuracy": ("accuracy",), "f1_macro": ("f1_macro",),
        "log_loss": ("log_loss",), "brier_score": ("brier_score",),
        **{f"{metric}_{label}": ("by_class", label, metric)
           for label in CLASSES for metric in ("precision", "recall", "f1")},
        "draws_correct": ("draw_correct_predicted", "correct"),
        "draws_predicted": ("draw_correct_predicted", "predicted"),
    }
    for variant in VARIANTS:
        fold_metrics = [report["folds"][str(year)]["metrics"][variant] for year in test_years]
        report["variants"][variant] = {
            "mean_std": {name: _metric_mean_std(fold_metrics, path)
                         for name, path in summary_paths.items()},
            "pooled_draws": {
                "correct": int(sum(row["draw_correct_predicted"]["correct"] for row in fold_metrics)),
                "predicted": int(sum(row["draw_correct_predicted"]["predicted"] for row in fold_metrics)),
                "actual": int(sum(row["by_class"]["D"]["actual"] for row in fold_metrics)),
            },
            "pooled_confusion_matrix": np.sum(
                [np.asarray(row["confusion_matrix"]) for row in fold_metrics], axis=0
            ).astype(int).tolist(),
            "pooled_confidence_bands": confidence_bands(
                pd.Series(pooled[variant]["actual"]), np.asarray(pooled[variant]["probabilities"])
            ),
            "pooled_games": len(pooled[variant]["actual"]),
        }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   default=lambda value: value.item() if isinstance(value, np.generic) else value)
        + "\n",
        encoding="utf-8",
    )
    for variant in VARIANTS:
        means = report["variants"][variant]["mean_std"]
        print(f"{variant}: acurácia={means['accuracy']['mean']:.3f}±{means['accuracy']['std']:.3f} "
              f"F1-macro={means['f1_macro']['mean']:.3f}±{means['f1_macro']['std']:.3f} "
              f"log-loss={means['log_loss']['mean']:.3f}±{means['log_loss']['std']:.3f} "
              f"Brier={means['brier_score']['mean']:.3f}±{means['brier_score']['std']:.3f}")
    print(f"Folds: {len(test_years)} ({test_years[0]}–{test_years[-1]}) | jogos teste: "
          f"{sum(report['folds'][str(year)]['test_games'] for year in test_years)}")
    print(f"Relatório: {REPORT_PATH}")


if __name__ == "__main__":
    main()
