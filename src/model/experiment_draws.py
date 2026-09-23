"""Compara regras de empate e sinais de equilíbrio em validação temporal.

Uso, na raiz do projeto: python -m src.model.experiment_draws
Este experimento não altera o modelo publicado pelo treino da referência.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from src.model.train_baseline import (
    ARTIFACTS_DIR,
    CLASSES,
    FEATURE_COLUMNS,
    load_feature_table,
    make_model,
)


REPORT_PATH = ARTIFACTS_DIR / "draw_experiment.json"
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.20, 0.451, 0.01))
ILLUSTRATIVE_THRESHOLDS = (0.25, 0.30, 0.35)
EXTRA_COLUMNS = (
    "diferenca_absoluta_pontos",
    "diferenca_absoluta_gols_esperados",
    "interacao_equilibrio_gols",
)


def add_balance_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Combina médias anteriores à partida; não consulta resultado ou gols atuais."""
    result = matches.copy()
    home_points = result["mandante_pontos_por_jogo_ultimos_5"]
    away_points = result["visitante_pontos_por_jogo_ultimos_5"]
    home_goals = (
        result["mandante_media_gols_pro_ultimos_5"]
        + result["visitante_media_gols_contra_ultimos_5"]
    ) / 2
    away_goals = (
        result["visitante_media_gols_pro_ultimos_5"]
        + result["mandante_media_gols_contra_ultimos_5"]
    ) / 2
    gap = (home_goals - away_goals).abs()
    result["diferenca_absoluta_pontos"] = (home_points - away_points).abs()
    result["diferenca_absoluta_gols_esperados"] = gap
    result["interacao_equilibrio_gols"] = gap * (home_goals + away_goals)
    return result


def choose_label(probabilities: np.ndarray, classes: np.ndarray, threshold: float | None) -> np.ndarray:
    """Se D ultrapassa o limiar, escolhe D; senão escolhe entre A e H."""
    if threshold is None:
        return classes[np.argmax(probabilities, axis=1)]
    draw_index = int(np.flatnonzero(classes == "D")[0])
    other_indices = np.flatnonzero(classes != "D")
    chosen_other = classes[other_indices[np.argmax(probabilities[:, other_indices], axis=1)]]
    return np.where(probabilities[:, draw_index] >= threshold, "D", chosen_other)


def score(actual: pd.Series, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1_macro": float(f1_score(actual, predicted, labels=CLASSES, average="macro")),
        "log_loss": float(log_loss(actual, probabilities, labels=CLASSES)),
        "draw_precision": float(precision_score(actual, predicted, labels=["D"], average="macro", zero_division=0)),
        "draw_recall": float(recall_score(actual, predicted, labels=["D"], average="macro", zero_division=0)),
        "draw_f1": float(f1_score(actual, predicted, labels=["D"], average="macro", zero_division=0)),
        "draw_predicted": int((predicted == "D").sum()),
        "draw_correct": int(((actual.to_numpy() == "D") & (predicted == "D")).sum()),
        "draw_actual": int((actual == "D").sum()),
    }


def probabilities_for(train: pd.DataFrame, evaluation: pd.DataFrame, columns: tuple[str, ...]):
    model = make_model().fit(train.loc[:, columns], train["resultado"])
    return model.predict_proba(evaluation.loc[:, columns]), model.classes_


def main() -> None:
    matches = load_feature_table()
    matches = add_balance_features(matches)
    early_train = matches[matches["temporada"].isin((2020, 2021))]
    selection = matches[matches["temporada"] == 2022]
    train = matches[matches["temporada"].isin((2020, 2021, 2022))]
    validation = matches[matches["temporada"] == 2023]
    if any(len(part) != expected for part, expected in ((early_train, 760), (selection, 380), (train, 1140), (validation, 380))):
        raise ValueError("Esperadas 380 partidas completas por temporada de 2020 a 2023.")

    report = {
        "selection_season": 2022,
        "validation_season": 2023,
        "test_2024_used": False,
        "threshold_objective": "maximize macro F1 on 2022; ties prefer higher threshold",
        "threshold_grid": list(THRESHOLDS),
        "feature_sets": {},
    }
    for name, columns in (
        ("baseline", FEATURE_COLUMNS),
        ("balance_features", FEATURE_COLUMNS + EXTRA_COLUMNS),
    ):
        selection_prob, selection_classes = probabilities_for(early_train, selection, columns)
        candidates = [
            (threshold, score(selection["resultado"], choose_label(selection_prob, selection_classes, threshold), selection_prob))
            for threshold in THRESHOLDS
        ]
        threshold, selection_score = max(candidates, key=lambda item: (item[1]["f1_macro"], item[0]))

        validation_prob, validation_classes = probabilities_for(train, validation, columns)
        default = score(validation["resultado"], choose_label(validation_prob, validation_classes, None), validation_prob)
        adjusted = score(validation["resultado"], choose_label(validation_prob, validation_classes, threshold), validation_prob)
        report["feature_sets"][name] = {
            "features": list(columns),
            "threshold": threshold,
            "selection_2022": selection_score,
            "validation_2023_default": default,
            "validation_2023_adjusted": adjusted,
            "validation_2023_illustrative": {
                f"{value:.2f}": score(
                    validation["resultado"],
                    choose_label(validation_prob, validation_classes, value),
                    validation_prob,
                )
                for value in ILLUSTRATIVE_THRESHOLDS
            },
        }
        print(
            f"{name}: limiar={threshold:.2f} | 2023 padrão: D {default['draw_correct']}/{default['draw_predicted']} "
            f"F1-macro={default['f1_macro']:.3f} | ajustado: D {adjusted['draw_correct']}/{adjusted['draw_predicted']} "
            f"F1-macro={adjusted['f1_macro']:.3f} acurácia={adjusted['accuracy']:.3f}"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Relatório: {REPORT_PATH}")


if __name__ == "__main__":
    main()
