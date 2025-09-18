import streamlit as st
from transformers import pipeline

st.title("📝 NLP Text Summarizer")

st.write("Enter your text below and get a concise summary.")

user_input = st.text_area("Enter Text Here", height=200)

if st.button("Summarize"):
    if user_input:
        # Load model explicitly
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        
        summary = summarizer(user_input, max_length=150, min_length=30, do_sample=False)
        
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summarize.")
