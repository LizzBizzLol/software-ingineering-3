import streamlit as st
import pandas as pd

st.image(
    "https://wallpapers.com/images/hd/titanic-1680-x-1050-background-ld95nte3gk5y0pad.jpg",
    caption="Титаник",
    use_column_width=True
)

st.title("Подсчет доли выживших пассажиров Титаника по каждому пункту посадки")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

data = load_data()

st.subheader("Выбор максимального возраста")

min_age = int(df["Age"].min())
max_age = int(df["Age"].max())

selected_age = st.slider(
    "Выберите максимальный возраст пассажиров:",
    min_value=min_age,
    max_value=max_age,
    value=max_age,
    step=1,
    help="Выберите максимальный возраст — таблица обновится автоматически"
)

filtered_df = df[df["Age"] <= selected_age]

display_mode = st.radio(
    "Формат отображения доли выживших:",
    ("Доля", "Проценты"),
    horizontal=True
)

if filtered_df.empty:
    st.warning("Нет пассажиров младше указанного возраста.")
    st.stop()

survival_by_embarked = (
    filtered_df.groupby("Embarked")["Survived"]
    .mean()
    .reset_index()
    .rename(columns={
        "Embarked": "Пункт посадки",
        "Survived": "Доля выживших"
    })
)

if display_mode == "Проценты":
    styled_table = survival_by_embarked.style.format({"Доля выживших": "{:.2%}"})
else:
    styled_table = survival_by_embarked.style.format({"Доля выживших": "{:.3f}"})

st.subheader("Доля выживших по каждому пункту посадки")
st.dataframe(styled_table, use_container_width=True)

