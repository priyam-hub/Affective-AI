# DEPENDENCIES

import os
import numpy as np
import sounddevice as sd
import noisereduce as nr
from ..utils.logger import LoggerSetup
from ..utils.audio_saver import AudioSaver

## LOGGER SETUP
record_audio_logger           = LoggerSetup(logger_name = "record_audio.py", log_filename_prefix = "record_audio").get_logger()

os.environ["LD_LIBRARY_PATH"] = "/usr/lib"
class AudioRecorder:
    """
    A class for recording and processing audio in real-time using Sounddevice.

    attributes:
    ------------
        
        format                   {int}       : Audio format (default: 16-bit integer).
        
        channels                 {int}       : Number of recording channels (default: 1 for mono).
        
        rate                     {int}       : Sampling rate in Hz (default: 16,000 for Whisper model compatibility).
        
        chunk                    {int}       : Buffer size for each recording frame.
        
        record_seconds           {int}       : Duration of recording in seconds.
        
    """

    def __init__(self, rate : int, channels : int, chunk : int, record_seconds : int) -> None:
        
        """
        Initializes the AudioRecorder with default or user-defined parameters.

        arguments:
        ----------
            rate                  {int}      : Sampling rate in Hz.
            
            channels              {int}      : Number of audio channels.
            
            chunk                 {int}      : Number of frames per buffer.
            
            record_seconds        {int}      : Recording duration in seconds.
        
        returns:
        --------
            
            None
        
        """
        try:
            self.rate             = rate
            self.channels         = channels
            self.chunk            = chunk
            self.record_seconds   = record_seconds
            self.format           = np.int16

            # Define the save directory for audio files
            self.audio_save_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../audio/"))

            # Create directory if it doesn't exist
            os.makedirs(self.audio_save_path, exist_ok = True)

            record_audio_logger.info("AudioRecorder initialized successfully.")

        except Exception as e:
            
            record_audio_logger.error(f"Error initializing AudioRecorder: {repr(e)}")

    def record_audio(self) -> str:
        """
        Records audio for a specified duration, applies noise reduction, 
        and saves it as a WAV file.

        returns:
        --------
            wav_file_path        {str}        : The filename of the recorded and processed audio file.
        
        """
        try:
            frames        = sd.rec(int(self.rate * self.record_seconds), 
                                   samplerate = self.rate, 
                                   channels   = self.channels, 
                                   dtype      = self.format
                                   )
            sd.wait()
            
            # Noise Reduction
            reduced_noise = nr.reduce_noise(y = frames.flatten(), sr=self.rate, prop_decrease = 0.8)
            
            wav_file_path = AudioSaver.audio_saver(reduced_noise, self.rate, self.audio_save_path, "recorded_audio.wav")
            
            return wav_file_path
        
        except Exception as e:
            record_audio_logger.error(f"Error recording audio: {repr(e)}")
            return ""
