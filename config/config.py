import os
from pathlib import Path

class Config:
    
    """
    Configuration class for storing credentials, file paths, and URLs.
    It also provides a method to ensure required directories exist.
    """

    AUDIO_PATH                       = "../audio/recorded_audio.wav"
    DATABASE_PATH                    = "./database/data.db"
    BEST_MODEL_PATH                  = "./models/classification_models/multinomial_logistic_regression.pkl"
    EDA_RESULTS_PATH                 = "./results/eda_results"
    ML_MODEL_SAVE_PATH               = "./models"
    EMOTION_DATASET_RAW              = "./data/emotion_dataset_raw.csv"
    EMOTION_DATASET_CLEANED          = "./data/emotion_dataset_cleaned.csv"
    MODEL_ACCURACY_PLOT_PATH         = "./results"

    # AUDIO CONFIGURATIONS
    CHUNK                            = 1024
    FORMAT                           = 16  
    CHANNELS                         = 1
    SAMPLE_RATE                      = 16000
    RECORD_SECONDS                   = 5  

    # HUGGING FACE MODEL CONFIGURATIONS
    MODEL_NAME                       = "priyammmmm/whisper_base"

    TIMEZONE_IST                     = "Asia/Kolkata"
    TIMEZONE_UTC                     = "UTC"

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