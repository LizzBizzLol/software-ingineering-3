import streamlit as st
import streamlit1
import app
st.title("Многостраничное Streamlit-приложение")
page = st.sidebar.selectbox("Выберите страницу", ["Белоглазова", "Усачёва", "Белоглазова, Нейронка"])
if page == "Белоглазова":
    streamlit1.show_page()
elif page == "Усачёва":
    app.show_page()
elif page == "Белоглазова, нейронка":
    neiroKate.show_page()
