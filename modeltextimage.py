# modeltextimage.py
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

@st.cache_resource
def _load_summarizer():
    model_name = "IlyaGusev/ru_t5_base_summarizer"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def show_page():
    st.title("✂️ Суммаризация текста на русском")
    st.caption("Введите любой текст на русском — модель создаст краткое содержание.")

    input_text = st.text_area(
        "📝 Вставьте текст:",
        "Искусственный интеллект — это область компьютерных наук, которая занимается созданием систем, способных выполнять задачи, требующие человеческого интеллекта.",
        height=150
    )

    if st.button("Создать краткое содержание"):
        if input_text.strip():
            with st.spinner("Обработка... (первый запуск может занять до минуты)"):
                tokenizer, model = _load_summarizer()
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding="max_length"
                )
                summary_ids = model.generate(
                    inputs.input_ids,
                    max_length=256,
                    min_length=30,
                    length_penalty=1.2,
                    num_beams=4,
                    early_stopping=True
                )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.subheader("✅ Результат")
            st.write(summary)
        else:
            st.warning("Пожалуйста, введите текст.")
