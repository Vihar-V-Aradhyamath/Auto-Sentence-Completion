import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load custom CSS for styling
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# Load pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Streamlit user interface
st.title("Auto Sentence Completion with GPT-2 and NLTK")

# Input box for the user to provide a sentence prompt
prompt = st.text_area("Enter a sentence prompt", "Once upon a time in a land far away")

# Button to trigger completion
if st.button("Complete Sentence"):
    # Get NLTK tokenization
    tokens = word_tokenize(prompt)

    # Prepare input for GPT-2
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, temperature=0.85)
    
    # Decode the generated text
    completed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display results
    st.subheader("Completed Sentence:")
    st.write(completed_text)
