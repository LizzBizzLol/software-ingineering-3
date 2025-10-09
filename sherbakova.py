import streamlit as st
import pandas as pd

def show_page():
    st.image(
        "https://wallpapers.com/images/hd/titanic-1680-x-1050-background-ld95nte3gk5y0pad.jpg"
    )
    
    st.title("Подсчет доли выживших пассажиров Титаника по каждому пункту посадки")
    
    @st.cache_data
    def load_data():
        df = pd.read_csv("data.csv")
        return df
    
    data = load_data()
    
    st.subheader("Выбор максимального возраста")
    
    min_age = max(1, int(data["Age"].min()))
    max_age = int(data["Age"].max())
    
    selected_age = st.slider(
        "Выберите максимальный возраст пассажиров:",
        min_value=min_age,
        max_value=max_age,
        value=max_age,
        step=1,
        help="Выберите максимальный возраст — таблица обновится автоматически"
    )
    
    filtered_df = data[data["Age"] <= selected_age]
    
    display_mode = st.radio(
        "Формат отображения доли выживших:",
        ("Доля", "Проценты"),
        horizontal=True
    )
    
    if filtered_df.empty:
        st.warning("Нет пассажиров младше указанного возраста.")
        st.stop()
    
    survival_by_embarked = (
        filtered_df.groupby("Embarked")
        .agg(
            Количество=("Survived", "count"),
            Выжило=("Survived", "sum")
        )
        .reset_index()
    )
    
    survival_by_embarked["Доля выживших"] = survival_by_embarked["Выжило"] / survival_by_embarked["Количество"]
    
    # Форматирование для отображения
    if display_mode == "Проценты":
        survival_by_embarked["Доля выживших"] = (survival_by_embarked["Доля выживших"] * 100).round(1)
        st.dataframe(survival_by_embarked.rename(columns={"Embarked":"Порт"}), use_container_width=True)
    else:
        survival_by_embarked["Доля выживших"] = survival_by_embarked["Доля выживших"].round(3)
        st.dataframe(survival_by_embarked.rename(columns={"Embarked":"Порт"}), use_container_width=True)
