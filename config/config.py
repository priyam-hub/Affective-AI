import os
import pyaudio
from pathlib import Path

class Config:
    
    """
    Configuration class for storing credentials, file paths, and URLs.
    It also provides a method to ensure required directories exist.
    """
    
    # AUDIO CONFIGURATIONS
    CHUNK                 = 1024
    FORMAT                = pyaudio.paInt16
    CHANNELS              = 1
    SAMPLE_RATE           = 16000
    RECORD_SECONDS        = 5  

    # HUGGING FACE MODEL CONFIGURATIONS
    MODEL_NAME            = "priyammmmm/whisper_base"

    AUDIO_PATH            = "./audio/recorded_audio.wav"

    @staticmethod
    def setup_directories():
        """
        Ensures that all required directories exist.
        If a directory does not exist, it creates it.
        """
        
        directories = []
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            else:
                print(f"Directory already exists: {directory}")