import streamlit as st

st.set_page_config(page_title="Hello Streamlit", page_icon="👋", layout="centered")

st.title("👋 Hello from Streamlit on Cloud Shell")
st.write("If you can see this, the Web Preview works ✅")

number = st.slider("Pick a number", 0, 100, 25)
st.write(f"You picked: {number}")

if st.button("Say Hello"):
    st.success("Hello there! 🎉")


