# tests/test_core.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import compute_children_stats

import pandas as pd

def _df():
    # Имитация фрагмента Titanic
    return pd.DataFrame([
        # S: двое детей, один погиб (Age 0.4), один выжил (Age 5)
        {"Embarked": "S", "Age": 0.4, "Survived": 0},
        {"Embarked": "S", "Age": 5.0, "Survived": 1},
        # C: двое детей, оба погибли (Age 1 и 2.9)
        {"Embarked": "C", "Age": 1.0, "Survived": 0},
        {"Embarked": "C", "Age": 2.9, "Survived": 0},
        # Q: один ребёнок (выжил), один взрослый погиб — взрослый должен быть отброшен порогом
        {"Embarked": "Q", "Age": 8.0, "Survived": 1},
        {"Embarked": "Q", "Age": 40.0, "Survived": 0},
        # неизвестный порт (проверка fallback), ребёнок погиб
        {"Embarked": "X", "Age": 3.2, "Survived": 0},
        # запись без порта — должна отфильтроваться
        {"Embarked": None, "Age": 7.0, "Survived": 0},
    ])

def test_counts_and_max_age_basic():
    df = _df()
    out = compute_children_stats(df, max_age=10)

    # Проверим базовые количества
    row_C = out[out["Код"] == "C"].iloc[0]
    row_S = out[out["Код"] == "S"].iloc[0]
    row_Q = out[out["Код"] == "Q"].iloc[0]
    row_X = out[out["Код"] == "X"].iloc[0]

    assert row_C["Погибло детей"] == 2
    assert row_S["Погибло детей"] == 1
    assert row_Q["Погибло детей"] == 0     # взрослый погибший отфильтрован
    assert row_X["Погибло детей"] == 1

    # Макс. возраст погибшего ребёнка (floor + clip)
    assert row_C["Макс. возраст погибшего ребёнка"] == 2  # floor(2.9)=2
    assert row_S["Макс. возраст погибшего ребёнка"] == 1  # floor(0.4)=0 -> clip до 1
    assert row_X["Макс. возраст погибшего ребёнка"] == 3     # floor(3.2)=3

def test_clip_respects_upper_threshold():
    df = pd.DataFrame([
        {"Embarked": "S", "Age": 17.9, "Survived": 0},
        {"Embarked": "S", "Age": 18.1, "Survived": 0},  # вне порога — должен отфильтроваться
    ])
    out = compute_children_stats(df, max_age=18)
    row = out[out["Код"] == "S"].iloc[0]
    # погиб только один в рамках порога
    assert row["Погибло детей"] == 1
    # max возраст = floor(17.9)=17, не выше 18
    assert row["Макс. возраст погибшего ребёнка"] == 17

def test_sorting_desc_by_dead_children():
    df = pd.DataFrame([
        {"Embarked": "S", "Age": 5, "Survived": 0},
        {"Embarked": "S", "Age": 6, "Survived": 0},
        {"Embarked": "C", "Age": 7, "Survived": 0},
    ])
    out = compute_children_stats(df, max_age=18)
    # S (2 погибших) должен идти раньше C (1 погибший)
    assert list(out["Код"])[:2] == ["S", "C"]

def test_port_name_mapping_and_fallback():
    df = pd.DataFrame([
        {"Embarked": "S", "Age": 5, "Survived": 0},
        {"Embarked": "Z", "Age": 6, "Survived": 0},  # неизвестный код
    ])
    out = compute_children_stats(df, max_age=18)
    # Проверяем отображение названия порта
    name_S = out[out["Код"] == "S"]["Порт"].iloc[0]
    name_Z = out[out["Код"] == "Z"]["Порт"].iloc[0]
    assert name_S == "Southampton"
    assert name_Z == "Z"  # fallback на сам код

def test_handles_missing_port_and_child_filter():
    df = pd.DataFrame([
        {"Embarked": None, "Age": 4, "Survived": 0},  # должен быть отброшен (нет порта)
        {"Embarked": "Q", "Age": 0.0, "Survived": 0}, # ребёнок возрастом 0 -> clip к 1
    ])
    out = compute_children_stats(df, max_age=3)
    row_Q = out[out["Код"] == "Q"].iloc[0]
    assert row_Q["Погибло детей"] == 1
    assert row_Q["Макс. возраст погибшего ребёнка"] == 1
