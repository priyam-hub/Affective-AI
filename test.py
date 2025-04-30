# STREAMLIT CONFIGURATION
import streamlit as st
st.set_page_config(page_title = "Affective-AI", page_icon = "🤗", layout = "centered")

# DEPENDENCIES

from librosa import ex
import pytz
import joblib
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
from datetime import datetime

from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.model_loader import ModelLoader
from web.utils.database_manager import DatabaseManager
from src.audio_recorder.record_audio import AudioRecorder
from src.audio_transcriber.transcribe_audio import AudioTranscriber

from web.pages import home
from web.pages import about
from web.pages import monitor
from web.pages import speech_to_text

app_logger = LoggerSetup(logger_name = "test.py", log_filename_prefix = "test").get_logger()

database_Manager            = DatabaseManager(db_path = Config.DATABASE_PATH)
IST                         = pytz.timezone(Config.TIMEZONE_IST)  


best_model_pipeline         = joblib.load(open(Config.BEST_MODEL_PATH, "rb"))


def predict_emotions(docx : str) -> any:
    """
        Predicts the emotion of the given text using the best model pipeline.
    
        Arguments:
            
            `docx`               {str}        : The text to be classified.
        
        Returns:
            
            any : The predicted emotion.
        
    """
    try:
        app_logger.info("Predicting Emotion...")
        results = best_model_pipeline.predict([docx])  
    
        return results[0]
    
    except Exception as e:
        app_logger.error(f"Error in predicting emotion: {repr(e)}")
        
        return None

def get_prediction_proba(docx : str ) -> any:
    """
        Returns the prediction probabilities for the given text using the best model pipeline.
    
        Arguments:
            
            `docx`             {str}       : The text to be classified.
        
        Returns:
            
            any                            : The prediction probabilities.
    
    """
    try:
        app_logger.info("Getting Predicting Probabilities...")
        results = best_model_pipeline.predict_proba([docx])
        
        return results
    
    except Exception as e:
        app_logger.error(f"Error in getting prediction probabilities: {repr(e)}")
        
        return None

emotions_emoji_dict = {"anger"     : "😠", 
                       "disgust"   : "🤮", 
                       "fear"      : "😨", 
                       "happy"     : "🤗", 
                       "joy"       : "😂", 
                       "neutral"   : "😐", 
                       "sad"       : "😔", 
                       "sadness"   : "😔", 
                       "shame"     : "😳", 
                       "surprise"  : "😮"
                       }

@st.cache_resource
def load_model():
    """Loads the model and processor and caches them."""
    
    try:

        app_logger.info("Loading model...")
        model_loader = ModelLoader(model_name = Config.MODEL_NAME)
        app_logger.info("Model loaded successfully.")
        
        return model_loader.load_model()
    
    except Exception as e:
        
        app_logger.error(f"Error loading model: {repr(e)}")
        
        return None, None
    
def main():
    
    try:

        st.sidebar.title("Navigation")
        page = st.sidebar.radio("📚 Select a Page:", ["Home", "Speech-to-Text", "Detect Your Emotion", "Monitor", "About"])

        if page == "Home":
            home.main()

        elif page == "Monitor":
            monitor.main()
        
        elif page == "About":
            about.main()

        elif page == "Speech-to-Text":
            speech_to_text.main()

        elif page == "Detect Your Emotion":

            # CUSTOM CSS STYLING
            st.markdown(
                """
                <style>
                body {
                    background-color: #f4f4f4;
                }
                .main-title {
                    font-size: 40px;
                    font-weight: bold;
                    color: #6A1B9A;
                    text-align: center;
                }
                .record-button {
                    background-color: #6200EE;
                    color: white;
                    font-size: 18px;
                    padding: 12px;
                    border-radius: 10px;
                }
                .transcription-box {
                    background-color: #ffffff;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                }
                </style>
                """,
                unsafe_allow_html = True
            )

            # Centered Container
            with st.container():
                
                st.markdown("<div class = 'container'>", unsafe_allow_html = True)

                st.markdown("<h1 class='main-title'>🎙️ Affective-AI 😮</h1>", unsafe_allow_html = True)
                st.markdown("<p style='text-align:center; font-size:18px;'><b>Bridging Emotions and Intelligence through Voice.</b></p>",
                            unsafe_allow_html = True)
                

                recording_seconds                  = st.slider("Recording Duration (in seconds):", 
                                                            min_value  = 1, 
                                                            max_value  = 60, 
                                                            value      = Config.RECORD_SECONDS, 
                                                            step       = 1)

                # Load model and processor from cache
                processor, model                   = load_model()

                # Initialize recorder and transcriber
                recorder                           = AudioRecorder(rate            = Config.SAMPLE_RATE, 
                                                                channels        = Config.CHANNELS, 
                                                                chunk           = Config.CHUNK, 
                                                                record_seconds  = recording_seconds
                                                                )
                
                transcriber                        = AudioTranscriber(processor = processor, model = model)

                # Ensure session state stores transcription
                if "transcription" not in st.session_state:
                    st.session_state.transcription = ""
                
                if "audio_path" not in st.session_state:
                    st.session_state.audio_path    = None

                if st.button("🎙️ Start Recording", key = "record", help = "Click to start recording"):

                    st.write("🎤 **Recording started...**")
                            
                    audio_path                     = recorder.record_audio()  
                    st.session_state.audio_path    = audio_path

                    transcription                  = transcriber.transcribe_audio(audio_path, sample_rate = Config.SAMPLE_RATE) 
                    st.session_state.transcription = transcription
                    
                    st.write("✅ **Recording complete. Transcribing...**")

                database_Manager.add_page_visited_details("Home", datetime.now(IST))
                st.subheader("Emotion Detection in Text")

                with st.form(key = 'emotion_clf_form'):
                    raw_text    = st.session_state.transcription

                if raw_text:
                    col1, col2  = st.columns(2)

                    prediction  = predict_emotions(docx = raw_text)
                    probability = get_prediction_proba(docx = raw_text)

                    database_Manager.add_prediction_details(rawtext      = raw_text, 
                                                            prediction   = prediction, 
                                                            probability  = np.max(probability), 
                                                            timeOfvisit  = datetime.now(IST)
                                                            )

                    with col1:
                        st.success("Original Text")
                        st.write(raw_text)

                        st.success("Prediction")
                        emoji_icon = emotions_emoji_dict[prediction]
                        st.write("{}:{}".format(prediction, emoji_icon))
                        st.write("Confidence:{}".format(np.max(probability)))

                    with col2:
                        st.success("Prediction Probability")
                        proba_df = pd.DataFrame(probability, columns = best_model_pipeline.classes_)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ["emotions", "probability"]

                        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                        st.altair_chart(fig, use_container_width=True)

        else:
            st.error("Page not found. Please select a valid page from the sidebar.")

    except Exception as e:
        app_logger.error(f"Error Occurred while running the app: {repr(e)}")

        raise

if __name__ == "__main__":
    main()