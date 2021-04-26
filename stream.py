import fitz
import streamlit as st
import os

from nltk.tokenize import sent_tokenize

from transformers import BertForSequenceClassification, BertTokenizerFast


MODEL_PATH = os.getcwd() + "/bert-base.pt"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)


def main(labels_dict):
    st.title("Thesis work.")
    st.subheader("Gabdrahmanov Aydar BS17-DS-02")
    uploaded_pdf = st.file_uploader("Load pdf: ", type=['pdf'])

    if uploaded_pdf is not None:
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.getText()
        sentences = sent_tokenize(text)
        st.markdown("Labelled Sentences:")
        for sentence in sentences:
            tokenized = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = outputs.logits.argmax(dim=1)
            label = labels_dict[prediction.item()]

            st.markdown(f"**Sentence**: {sentence} **Label**: _{label}_")
        doc.close()


labels = {
 'Access control': 0,
 'Accountability': 4,
 'Availability': 5,
 'Confidentiality': 3,
 'Integrity': 1,
 'Operational': 6,
 'Other': 2}
decoded_labels = {value:key for key, value in labels.items()}
main(decoded_labels)
