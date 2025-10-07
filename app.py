import numpy as np
import pandas as pd
import streamlit as st
def show_page():
    # ---------- Page config ----------
    st.set_page_config(page_title="Титаник: погибшие дети по портам", layout="wide")
    # ---------- Header image ----------
    BANNER_PATH = "banner.jpg"
    try:
        st.image(BANNER_PATH, use_container_width=True)
    except Exception:
        pass
    st.title("Погибшие дети по пунктам посадки")
    st.caption("Данные: Titanic (train). Фильтр по максимальному возрасту влияет на расчёты ниже.")
    # ---------- Data loading ----------
    @st.cache_data
    def load_csv(src):
        df = pd.read_csv(src, encoding="utf-8")
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce")
        return df
    uploaded = st.file_uploader("Загрузите data.csv", type=["csv"])
    path = st.text_input("Или укажите путь к CSV", value="data.csv")
    try:
        df = load_csv(uploaded if uploaded is not None else path)
    except Exception as e:
        st.error(f"Не удалось открыть файл: {e}")
        st.stop()
    # ---------- Age threshold control ----------
    max_age = st.slider("Максимальный возраст (включительно)", min_value=1, max_value=18, value=18, step=1)
    # ---------- Core task with dynamic threshold ----------
    has_port = df["Embarked"].notna()
    is_child = df["Age"].between(0, max_age, inclusive="both")
    sub = df.loc[has_port & is_child, ["Embarked", "Age", "Survived"]].copy()
    # погибшие дети по портам (0, если нет)
    dead_cnt = sub.groupby("Embarked")["Survived"].apply(lambda s: int((s == 0).sum()))
    # максимальный возраст среди погибших детей в рамках выбранного порога
    max_dead_age = (
        sub.loc[sub["Survived"] == 0]
          .groupby("Embarked")["Age"]
          .max()
    )
    out = pd.DataFrame({"Погибло детей": dead_cnt}).join(
        max_dead_age.rename("Макс. возраст погибшего ребёнка"), how="left"
    )
    def clip_age(x):
        if pd.isna(x):
            return None
        return int(np.clip(np.floor(x), 1, max_age))  # учитываем текущий порог
    out["Макс. возраст погибшего ребёнка"] = out["Макс. возраст погибшего ребёнка"].map(clip_age)
    port_name = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    out = out.reset_index().rename(columns={"Embarked": "Код"})
    out.insert(0, "Порт", out["Код"].map(port_name).fillna(out["Код"].astype(str)))
    out = out.sort_values("Погибло детей", ascending=False).reset_index(drop=True)
    st.dataframe(out, use_container_width=True)
    st.caption(f"Текущий порог: дети возрастом от 0 до {max_age} лет включительно.")
