"""Verifica a separação temporal e o pré-processamento do modelo."""

import numpy as np
import pandas as pd

from src.model.train_baseline import (
    FEATURE_COLUMNS,
    make_model,
    split_by_season,
    validation_diagnostics,
)


def test_split_keeps_2024_only_for_final_evaluation() -> None:
    matches = pd.DataFrame(
        {
            "id": range(1900),
            "temporada": [year for year in range(2020, 2025) for _ in range(380)],
        }
    )

    train, validation, test = split_by_season(matches)

    assert set(train["temporada"]) == {2020, 2021, 2022}
    assert set(validation["temporada"]) == {2023}
    assert set(test["temporada"]) == {2024}
    assert (len(train), len(validation), len(test)) == (1140, 380, 380)
    assert set(train["id"]).isdisjoint(validation["id"])
    assert set(train["id"]).isdisjoint(test["id"])


def test_imputer_is_fitted_on_training_values_only() -> None:
    train = pd.DataFrame(
        {column: [1.0, 3.0, np.nan, 5.0] for column in FEATURE_COLUMNS}
    )
    target = ["A", "D", "H", "A"]
    model = make_model().fit(train, target)

    later_match = pd.DataFrame({column: [1000.0] for column in FEATURE_COLUMNS})
    model.predict(later_match)

    learned_medians = model.named_steps["simpleimputer"].statistics_
    np.testing.assert_array_equal(learned_medians, np.full(len(FEATURE_COLUMNS), 3.0))


def test_validation_diagnostics_counts_each_real_and_predicted_class() -> None:
    predictions = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "data": ["2023-01-01"] * 6,
            "mandante": ["Casa"] * 6,
            "visitante": ["Fora"] * 6,
            "resultado": ["A", "D", "H", "A", "D", "H"],
            "previsto": ["H", "D", "H", "A", "H", "A"],
            "prob_A": [0.1, 0.2, 0.1, 0.4, 0.2, 0.7],
            "prob_D": [0.2, 0.6, 0.1, 0.3, 0.2, 0.1],
            "prob_H": [0.7, 0.2, 0.8, 0.3, 0.6, 0.2],
        }
    )

    report = validation_diagnostics(predictions)

    assert report["confusion_matrix"] == [[1, 0, 1], [0, 1, 1], [1, 0, 1]]
    assert report["by_class"]["D"]["actual"] == 2
    assert report["by_class"]["D"]["predicted"] == 1
    assert report["by_class"]["D"]["correct"] == 1
    assert report["by_class"]["D"]["recall"] == 0.5
    assert sum(band["matches"] for band in report["confidence_bands"]) == 6
    assert [item["id"] for item in report["most_confident_errors"]] == [1, 6, 5]
