import streamlit as st
import pandas as pd
import numpy as np
#@st.cache_data
def show_page():
    def load_data():
        df = pd.read_csv("data.csv")
        return df
    
    df = load_data()
    
    
    # --- Настройка страницы ---
    st.set_page_config(page_title="Анализ выживаемости по классу", layout="centered")
    
    # --- Заголовок ---
    st.title("🧮 Процент выживших по возрасту и классу билета")
    st.markdown("""
        Выберите **класс билета**, чтобы увидеть процент выживших среди:
        - **Молодых пассажиров** (возраст < 30 лет)
        - **Пожилых пассажиров** (возраст > 60 лет)
    """)
    
    # Удаляем строки без возраста
    df = df.dropna(subset=['Age'])
    
    
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
        caption="RMS Titanic",
        use_column_width=True
    )
    
    
    pclass = st.selectbox(
        "Выберите класс билета:",
        options=[1, 2, 3],
        format_func=lambda x: f"{x}-й класс"
    )
    
    
    df_class = df[df['Pclass'] == pclass]
    
    
    young = df_class[df_class['Age'] < 30]
    old = df_class[df_class['Age'] > 60]
    
    
    def survival_rate(group):
        if len(group) == 0:
            return 0.0, 0
        rate = (group['Survived'].sum() / len(group)) * 100
        return round(rate, 2), len(group)
    
    young_rate, young_count = survival_rate(young)
    old_rate, old_count = survival_rate(old)
    
    
    st.subheader(f"Результаты для {pclass}-го класса")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Молодые (<30)",
            value=f"{young_rate}%",
            delta=f"выживших из {young_count} пассажиров"
        )
    
    with col2:
        st.metric(
            label="Старые (>60)",
            value=f"{old_rate}%",
            delta=f"выживших из {old_count} пассажиров"
        )
    
    
    details = pd.DataFrame({
        'Группа': ['Молодые (<30)', 'Старые (>60)'],
        'Количество': [young_count, old_count],
        'Процент выживших (%)': [young_rate, old_rate]
    })
    
    st.dataframe(details.style.format({
        'Количество': '{:,.0f}',
        'Процент выживших (%)': '{:.2f}%'
    }).apply(lambda x: ['background-color: #f0f8ff' if x.name % 2 == 0 else 'background-color: #ffffff' for _ in x], axis=1))


