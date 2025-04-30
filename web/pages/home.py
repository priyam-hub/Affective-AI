# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from pytz import timezone
from datetime import datetime

from config.config import Config
from src.utils.logger import LoggerSetup
from web.utils.database_manager import DatabaseManager

# LOGGER SETUP
home_logger = LoggerSetup(logger_name = "home.py", log_filename_prefix = "home").get_logger()

# Record page visit
IST = timezone(Config.TIMEZONE_IST)
database_Manager = DatabaseManager(db_path = Config.DATABASE_PATH)

database_Manager.add_page_visited_details("Home", datetime.now(IST))

def main():
    st.markdown(
        """
        <style>
            .big-font {
                font-size:36px !important;
                font-weight:700;
            }
            .tagline {
                font-size:20px;
                color: #4CAF50;
                font-style: italic;
            }
            .centered {
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # Header
    st.markdown('<p class="big-font centered">🤖 Affective-AI</p>', unsafe_allow_html = True)
    st.markdown('<p class="tagline centered">Empowering Communication Through Emotional Intelligence</p>', unsafe_allow_html = True)

    st.image("images/home_banner.jpg", use_container_width = True) 

    st.markdown("---")

    # Sections
    st.subheader("🔍 What is Affective-AI?")
    st.write("""
        Affective-AI is an intelligent platform that harnesses the power of **Natural Language Processing** and **Speech Recognition** to detect human emotions from text and voice. 
        Whether it's customer reviews, user feedback, or social media content — this tool helps you understand what truly lies beneath the words.
    """)

    st.subheader("🚀 What Can You Do Here?")
    st.markdown("""
    - 🗣️ Convert your voice into text
    - 😃 Detect emotions in text or speech
    - 📊 Visualize user behavior and predictions
    - 🧠 Analyze emotional trends in communication
    """)

    st.subheader("📦 Technologies Used")
    st.markdown("""
    - **Streamlit** for interactive UI
    - **SpeechRecognition** for voice analysis
    - **Transformers / LLMs** for emotion classification
    - **Altair & Plotly** for beautiful visualizations
    - **SQLite** for storing user metrics
    """)

    st.markdown("---")

    st.info("💡 Tip: Use the sidebar to explore features like Speech-to-Text, Emotion Detection, Monitor & More.")

    st.success("🎯 Ready to decode the language of emotions? Let's begin!")

