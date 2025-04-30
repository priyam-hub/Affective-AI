# DEPENDENCIES

from .logger import LoggerSetup
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

## LOGGER SETUP
model_loader_logger = LoggerSetup(logger_name = "model_loader.py", log_filename_prefix = "model_loader").get_logger()

class ModelLoader:
    """1
    A class responsible for loading the Whisper speech-to-text model and processor.
    """

    def __init__(self, model_name : str ) -> None:
        """
        Initializes the ModelLoader class with a specified Hugging Face model.

        arguments:
        ----------
            model_name                  {str}                     : The name of the pre-trained Whisper model on Hugging Face.
        
        """
        self.model_name = model_name

    def load_model(self) -> tuple:
        """
        Loads the Whisper model and processor from Hugging Face.

        returns:
        --------
            processor             {WhisperProcessor}               : The processor for tokenizing and feature extraction.
            
            model           {WhisperForConditionalGeneration}      : The Whisper model for speech-to-text generation.
        
        """
        try:
            model_loader_logger.info(f"Loading model: {self.model_name}")
           
            processor = WhisperProcessor.from_pretrained(self.model_name)
            model     = WhisperForConditionalGeneration.from_pretrained(self.model_name)

            model.eval()

            model_loader_logger.info("Model loaded successfully.")
            
            return processor, model

        except Exception as e:
            
            model_loader_logger.error(f"Error loading model: {repr(e)}")
            
            return None, None