
import streamlit as st
from llm_attempt import convert_homework

st.set_page_config(
    page_title = "Homeworkhelpers",
    layout = "centered"
)

# streamlit page title
st.title("Homework Helpers - Homework Converter")

input_style = st.text_area("Enter the style you would like to convert to (i.e. 'sharks', 'pirates'):")
input_text = st.text_area("Enter the Homework Problem")

if st.button("Convert"):
    converted_homework = convert_homework(input_style, input_text)
    st.success(converted_homework)