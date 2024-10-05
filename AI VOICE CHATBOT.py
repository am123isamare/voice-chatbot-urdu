from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts.lang import tts_langs
from gtts import gTTS
import streamlit as st
import re
import os

# Constants
API_KEY = "AIzaSyD9CtjYxng6YbhcDB7_pAiafN82-e84KxA" 
STATIC_IMAGE_URL = "https://www.shaip.com/wp-content/uploads/2022/12/Blog_Difference-Between-Speech-Voice-Recognition.jpg"

# Initialize LangChain and ChatGoogleGenerativeAI
langs = tts_langs().keys()
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Please always respond to the user's query in pure Urdu language."),
        ("human", "{human_input}"),
    ]
)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
chain = chat_template | model | StrOutputParser()

# Streamlit app layout
st.title("Voice Assistant ChatBot")
st.image(STATIC_IMAGE_URL, use_column_width=True)

# Custom CSS for colorful chat bubbles
st.markdown(
    """
    <style>
    .user-bubble {
        background: linear-gradient(135deg, #a4508b, #5f0a87);
        color: white;
        padding: 10px;
        border-radius: 15px;
        text-align: right;
        margin-left: 25%;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }
    .ai-bubble {
        background: linear-gradient(135deg, #f093fb, #f5576c, #4facfe);
        color: white;
        padding: 10px;
        border-radius: 15px;
        text-align: left;
        margin-right: 25%;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Voice recording
st.subheader("Record Your Voice:")
text = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT")

if text:
    st.subheader("Text Generating")
    with st.spinner("Converting to Speech..."):
        try:
            # Get the response from the model
            response = chain.invoke({"human_input": text})
            
            # Clean the response to remove unwanted characters like '**'
            full_response = "".join(res or "" for res in response)
            cleaned_response = re.sub(r"\**\*|__", "", full_response)

            # Display the conversation in colorful chat bubbles
            st.markdown(f'<div class="user-bubble">{text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-bubble">{cleaned_response}</div>', unsafe_allow_html=True)

            # Convert cleaned text to speech
            tts = gTTS(text=cleaned_response, lang='ur')
            tts.save("output.mp3")
            st.audio("output.mp3")

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.error("Could not recognize speech. Please speak again.")
