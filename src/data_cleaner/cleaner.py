# DEPENDENCIES

import pandas as pd
from librosa import ex
import neattext.functions as nfx

from ..utils.logger import LoggerSetup


# LOGGER SETUP
data_cleaner_logger = LoggerSetup(logger_name = "cleaner.py", log_filename_prefix = "cleaner").get_logger()


class TextCleaner:
    """
    A class to perform basic text preprocessing using neptune-text-cleaner.

    Methods:

        - data_cleaning()                            : Removes user handles and stopwords from the 'Text' column.
    
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the TextCleaner with a pandas DataFrame.
        
        Arguments:
            
            dataframe          {pd.DataFrame}         : The input DataFrame containing a 'Text' column.
        
        """
        try:
            if 'Text' not in dataframe.columns:
                
                data_cleaner_logger.error("DataFrame must contain a 'Text' column.")
                raise ValueError("DataFrame must contain a 'Text' column.")
            
            if not isinstance(dataframe, pd.DataFrame):
                
                data_cleaner_logger.error("Input must be a pandas DataFrame.")
                raise TypeError("Input must be a pandas DataFrame.")
        
            self.df = dataframe.copy()

            data_cleaner_logger.info("TextCleaner initialized successfully.")

        except Exception as e:
            data_cleaner_logger.error(f"Error initializing TextCleaner: {repr(e)}")
            
            raise

    def data_cleaning(self) -> pd.DataFrame:
        """
        Apply text cleaning: remove user handles and stopwords.

        Returns:
            
            pd.DataFrame: Cleaned DataFrame with a new 'Clean_Text' column.
        
        """
        try:
            if 'Text' not in self.df.columns:
                
                data_cleaner_logger.error("DataFrame must contain a 'Text' column.")
                raise ValueError("DataFrame must contain a 'Text' column.")
            
            data_cleaner_logger.info("Starting text cleaning process.")

            self.df['Clean_Text'] = self.df['Text'].apply(nfx.remove_userhandles)
            self.df['Clean_Text'] = self.df['Clean_Text'].apply(nfx.remove_stopwords)

            data_cleaner_logger.info("Text cleaning process completed successfully.")

            return self.df
        
        except Exception as e:
            data_cleaner_logger.error(f"Error during text cleaning: {repr(e)}")
            
            raise
