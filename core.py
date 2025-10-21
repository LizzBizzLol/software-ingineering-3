# core.py
import numpy as np
import pandas as pd

PORT_NAME = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}

def compute_children_stats(df: pd.DataFrame, max_age: int) -> pd.DataFrame:
    """Возвращает таблицу:
    ['Порт','Код','Погибло детей','Макс. возраст погибшего ребёнка'].
    Логика идентична твоему приложению.
    """
    # фильтры
    has_port = df["Embarked"].notna()
    is_child = df["Age"].between(0, max_age, inclusive="both")
    sub = df.loc[has_port & is_child, ["Embarked", "Age", "Survived"]].copy()

    # количество погибших детей по портам
    dead_cnt = sub.groupby("Embarked")["Survived"].apply(lambda s: int((s == 0).sum()))

    # максимальный возраст среди погибших детей (в выбранном пороге)
    max_dead_age = sub.loc[sub["Survived"] == 0].groupby("Embarked")["Age"].max()

    out = pd.DataFrame({"Погибло детей": dead_cnt}).join(
        max_dead_age.rename("Макс. возраст погибшего ребёнка"), how="left"
    )

    def clip_age(x):
        if pd.isna(x):
            return None
        # как в приложении: сначала floor, потом ограничение [1, max_age]
        return int(np.clip(np.floor(x), 1, max_age))

    col = "Макс. возраст погибшего ребёнка"
    out[col] = out[col].astype(object).map(clip_age)


    out = out.reset_index().rename(columns={"Embarked": "Код"})
    out.insert(0, "Порт", out["Код"].map(PORT_NAME).fillna(out["Код"].astype(str)))
    out = out.sort_values("Погибло детей", ascending=False).reset_index(drop=True)
    return out
