# DEPENDENCIES

import torch
import numpy as np
from ..utils.logger import LoggerSetup
from ..utils.audio_saver import AudioSaver

## LOGGER SETUP
audio_transcriber_logger = LoggerSetup(logger_name="transcribe_audio.py", log_filename_prefix="audio_transcriber").get_logger()

class AudioTranscriber:
    """
    A class responsible for transcribing audio files using a pre-trained speech-to-text model.
    """

    def __init__(self, processor, model) -> None:
        
        """
        Initializes the AudioTranscriber class by loading the speech-to-text model and processor.
        
        """
        try:
            self.processor = processor
            self.model     = model

            if self.processor is None or self.model is None:
                raise ValueError("Processor or model is None. Transcription will not work.")

        except Exception as e:
            
            audio_transcriber_logger.error(f"Error initializing AudioTranscriber: {repr(e)}")
            
            self.processor, self.model = None, None

    def transcribe_audio(self, audio_path : str, sample_rate : int) -> str:
        """
        Transcribes the given audio file using a speech-to-text model.

        arguments:
        ----------
            audio_path          {str}        : Path to the audio file to be transcribed.
            
            sample_rate         {int}            : The sample rate of the audio file.

        returns:
        --------
            transcription       {str}            : Transcribed text from the audio file.
        """
        
        try: 

            audio, _          = AudioSaver.audio_loader(audio_path, sample_rate = sample_rate)

            if audio is None:
                return "Error loading audio."

            input_features    = self.processor(audio, sampling_rate = sample_rate, return_tensors = "pt").input_features

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)

            transcription     = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            audio_transcriber_logger.info("Audio transcribed successfully.")

            return transcription

        except Exception as e:
            
            audio_transcriber_logger.error(f"Error transcribing audio: {repr(e)}")
            
            return "Error during transcription."