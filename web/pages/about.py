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

IST = timezone(Config.TIMEZONE_IST)

about_logger = LoggerSetup(logger_name = "about.py", log_filename_prefix = "about").get_logger()

def main():
    
    try:

        database_Manager = DatabaseManager(db_path = Config.DATABASE_PATH)
        database_Manager.add_page_visited_details("About", datetime.now(IST))

        st.title("📘 About - Affective-AI")
        
        st.write("""
            Welcome to the **Affective-AI** App!  
            This application utilizes the power of **Natural Language Processing** and **Machine Learning** to analyze and identify emotions in textual data.
        """)

        st.subheader("🎯 Our Mission")
        st.write("""
            At **Affective-AI**, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations 
            understand the emotional content hidden within text.  
            We believe emotions play a crucial role in communication, and by uncovering them, 
            we can gain valuable insights into sentiments and attitudes expressed in written form.
        """)

        st.subheader("⚙️ How It Works")
        st.write("""
            When you input text into the app, our system processes it and applies advanced NLP algorithms 
            to extract meaningful features.  
            These are fed into a trained model which predicts the emotions associated with the input.  
            The app then displays detected emotions along with a **confidence score** to guide your analysis.
        """)

        st.subheader("✨ Key Features")

        st.markdown("##### 1. Real-time Emotion Detection")
        st.write("""
            Instantly analyze the emotions expressed in any given text—whether it's customer feedback, 
            social media posts, or other formats—with real-time results.
        """)

        st.markdown("##### 2. Confidence Score")
        st.write("""
            Our app shows a confidence score alongside the detected emotions, helping you assess 
            the model's certainty and trustworthiness of the results.
        """)

        st.markdown("##### 3. User-friendly Interface")
        st.write("""
            Designed with simplicity in mind, our intuitive interface enables both technical and non-technical users 
            to analyze text effortlessly.
        """)

        st.subheader("📌 Applications")
        st.markdown("""
        The Emotion Detection in Text App has a wide range of applications across various domains:
        - 📱 Social media sentiment analysis  
        - 💬 Customer feedback analysis  
        - 📊 Market research & consumer insights  
        - 🛡️ Brand monitoring & reputation management  
        - 📚 Content analysis & recommendation systems  
        """)
    
        about_logger.info("About Page Loaded Successfully")

    except Exception as e:

        about_logger.error(f"Error Occurred while running the About page: {repr(e)}")
        
        st.error("An error occurred while loading the About page. Please try again later.")
