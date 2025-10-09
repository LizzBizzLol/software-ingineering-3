from transformers import pipeline
import gradio as gr
# создаём пайплайн для NER
ner = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", grouped_entities=True)
def analyze(text):
    result = ner(text)
    entities = [f"{entity['entity_group']}: {entity['word']} (уверенность {entity['score']:.2f})" for entity in result]
    return "\n".join(entities)

# интерфейс
demo = gr.Interface(fn=analyze, inputs="text", outputs="text", title="NER-приложение", description="Определение сущностей (имена, организации, города) в русском тексте")

demo.launch()