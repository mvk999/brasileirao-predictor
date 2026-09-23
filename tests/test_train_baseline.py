"""Verifica a separação temporal e o pré-processamento do modelo."""

import numpy as np
import pandas as pd

from src.model.train_baseline import FEATURE_COLUMNS, make_model, split_by_season


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
