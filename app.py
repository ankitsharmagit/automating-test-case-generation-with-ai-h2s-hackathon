import streamlit as st

st.set_page_config(page_title="Sample Streamlit App", page_icon="🌐", layout="centered")

st.title("🌐 My Sample Streamlit App")
st.subheader("Deployed on Google Cloud Run")

name = st.text_input("Enter your name:")

if st.button("Submit"):
    st.success(f"Hello, {name}! 🎉 Welcome to GCP Streamlit Deployment.")
