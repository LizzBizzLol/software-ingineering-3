import streamlit as st
import streamlit1
import app
import neiroKate
st.title("Многостраничное Streamlit-приложение")
page = st.sidebar.selectbox("Выберите страницу", ["Белоглазова", "Усачёва", "Белоглазова. Нейронка"])
if page == "Белоглазова":
    streamlit1.show_page()
elif page == "Усачёва":
    app.show_page()
elif page == "Белоглазова. Нейронка":
    neiroKate.show_page()
