# DEPENDENCIES

import os
import librosa
import soundfile as sf
from ..utils.logger import LoggerSetup

## LOGGER SETUP
audio_saver_logger = LoggerSetup(logger_name = "audio_saver.py", log_filename_prefix = "audio_saver").get_logger()

class AudioSaver:
    """
    A class responsible for saving audio files to a specified directory.
    
    """

    @staticmethod
    def audio_saver(audio_data, sample_rate: int, save_dir: str, file_name: str = "recorded_audio.wav") -> str:
        """
        Saves the processed audio data as a WAV file in the specified directory.

        arguments:
        ----------
            audio_data       {np.ndarray}     : The processed audio data to be saved.

            sample_rate          {int}        : The sample rate of the audio file.
            
            save_dir             {str}        : The directory where the audio file should be saved.
            
            file_name            {str}        : The name of the output WAV file (default: "recorded_audio.wav").

        returns:
        --------
            
            wav_file_path        {str}        : The absolute path of the saved audio file.
        
        """
        try:
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok = True)

            # Ensure filename ends with .wav
            if not file_name.endswith(".wav"):
                file_name += ".wav"

            # Construct full save path
            wav_file_path  = os.path.join(save_dir, file_name)

            # Save the audio data as a WAV file
            sf.write(wav_file_path, audio_data, sample_rate)

            audio_saver_logger.info(f"Audio saved successfully at {wav_file_path}")

            return wav_file_path
        
        except Exception as e:
            
            audio_saver_logger.error(f"Error saving audio file: {repr(e)}")
            
            return None
        
    @staticmethod
    def audio_loader(audio_path: str, sample_rate : int) -> tuple:
        
        """
        Loads an audio file from the given path and preprocesses it.

        arguments:
        ---------- 
            audio_path              {str}        : Path to the audio file.
        
        returns:
        --------
            audio                {np.ndarray}    : Loaded audio data as a NumPy array.
            
            sample_rate              {int}       : Sample rate of the loaded audio.
        
        """
        try:
        
            audio, sample_rate = librosa.load(audio_path, sr = sample_rate)  
            
            audio_saver_logger.info(f"Audio loaded successfully from {audio_path}")
        
            return audio, sample_rate
        
        except Exception as e:
        
            audio_saver_logger.error(f"Error loading audio from {audio_path}: {repr(e)}")
        
            return None, None

