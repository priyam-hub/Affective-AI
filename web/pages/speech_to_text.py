# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import pyperclip
import streamlit as st

from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.model_loader import ModelLoader
from src.audio_recorder.record_audio import AudioRecorder
from src.audio_transcriber.transcribe_audio import AudioTranscriber

# LOGGER SETUP
speech_to_text_logger = LoggerSetup(logger_name = "speech_to_text.py", log_filename_prefix = "speech_to_text").get_logger()

@st.cache_resource
def load_model():
    """Loads the model and processor and caches them."""
    
    try:

        speech_to_text_logger.info("Loading model...")
        
        model_loader = ModelLoader(model_name = Config.MODEL_NAME)

        speech_to_text_logger.info("Model loaded successfully.")
        
        return model_loader.load_model()
    
    except Exception as e:
        
        speech_to_text_logger.error(f"Error loading model: {repr(e)}")
        
        return None, None

def main():

    try:

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

            st.markdown("<h1 class='main-title'>🎙️ Speechify</h1>", unsafe_allow_html = True)
            st.markdown("<p style='text-align:center; font-size:18px;'><b>Click the button below to record your voice and get an instant transcription.</b></p>",
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

            if st.button("🎙️ Record and Transcribe", key = "record", help = "Click to start recording"):

                st.write("🎤 **Recording started...**")
                        
                audio_path                     = recorder.record_audio()  
                st.session_state.audio_path    = audio_path

                transcription                  = transcriber.transcribe_audio(audio_path, sample_rate = Config.SAMPLE_RATE) 
                st.session_state.transcription = transcription
                
                st.write("✅ **Recording complete. Transcribing...**")

            audio_area, space, transcription_area = st.columns([4, 1, 4])

            with audio_area:
            
                st.subheader("🔊 Recorded Audio")
                
                if st.session_state.audio_path:
                    st.audio(st.session_state.audio_path, format = "audio/wav")

            with transcription_area:  

                st.subheader("Transcription:")
                
                st.write(st.session_state.transcription)

                copy_button, download_button, space = st.columns([1, 1, 6])

                with copy_button:

                    if st.session_state.transcription:
                    
                        if st.button("📋", key = "copy_text", help = "Copy your Text to Clipboard"):
                            
                            pyperclip.copy(st.session_state.transcription)  
                            
                            st.toast("✅ Text copied to clipboard!")

                with download_button:

                    if st.session_state.transcription:
                    
                        st.download_button(label      = "📥",
                                        data       = st.session_state.transcription,
                                        file_name  = "transcription.txt",
                                        mime       = "text/plain",
                                        help       = "Download your Transcription"
                                        )


            # Footer
            st.markdown("---")
            # st.markdown("<p style='text-align:center;'>Developed with ❤️ by Priyam Pal [AI and Data Science Engineer] using OpenAI-Whisper</p>", 
            #             unsafe_allow_html = True)

    except Exception as e:
        speech_to_text_logger.error(f"An error occurred: {repr(e)}")

        st.error("An error occurred. Please check the logs for more information.")