
import streamlit as st
from ExploratoryAnalysis_LLMs import convert_homework_simple

st.set_page_config(
    page_title = "Homeworkhelpers",
    layout = "centered"
)

# streamlit page title
st.title("Homework Helpers - Textbook Translator")

input_style = st.text_area("Enter the style you would like to convert to (i.e. 'sharks', 'pirates'):")
input_text = st.text_area("Enter the text you'd like to convert")

if st.button("Convert"):
    converted_homework = convert_homework_simple(input_style, input_text)
    st.success(converted_homework)