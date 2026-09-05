from pathlib import Path

import pandas as pd

from src.data.prepare_matches import (
    PROCESSED_MATCHES_PATH,
    load_raw_matches,
    prepare_matches,
    validate_matches,
)


def test_prepare_matches_derives_result_from_score() -> None:
    raw_matches = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "rodata": [1, 1, 1],
            "data": ["29/05/2021", "30/05/2021", "31/05/2021"],
            "mandante": ["Casa H", "Casa D", "Casa A"],
            "visitante": ["Fora H", "Fora D", "Fora A"],
            "mandante_Placar": [2, 1, 0],
            "visitante_Placar": [1, 1, 2],
            "arena": ["A", "B", "C"],
            "mandante_Estado": ["SP", "RJ", "MG"],
            "visitante_Estado": ["RJ", "MG", "SP"],
        }
    )

    prepared = prepare_matches(raw_matches)

    assert prepared["resultado"].tolist() == ["H", "D", "A"]


def test_prepare_matches_applies_2020_season_exception() -> None:
    raw_matches = pd.DataFrame(
        {
            "ID": [1, 2],
            "rodata": [35, 1],
            "data": ["25/02/2021", "29/05/2021"],
            "mandante": ["Casa 2020", "Casa 2021"],
            "visitante": ["Fora 2020", "Fora 2021"],
            "mandante_Placar": [1, 1],
            "visitante_Placar": [0, 0],
            "arena": ["A", "B"],
            "mandante_Estado": ["SP", "RJ"],
            "visitante_Estado": ["RJ", "SP"],
        }
    )

    prepared = prepare_matches(raw_matches)

    assert prepared["temporada"].tolist() == [2020, 2021]


def test_processed_dataset_has_no_logical_duplicates_and_expected_size() -> None:
    prepared = prepare_matches(load_raw_matches())
    validate_matches(prepared)

    assert len(prepared) == 1900
    assert not prepared.duplicated(
        subset=["temporada", "rodada", "mandante", "visitante"]
    ).any()
