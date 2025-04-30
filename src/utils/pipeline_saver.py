# DEPENDENCIES  

import os
import joblib
from librosa import ex
import sklearn
import sklearn.pipeline
from .logger import LoggerSetup

# LOGGER SETUP  
pipeline_saver_logger = LoggerSetup(logger_name = "pipeline_saver.py", log_filename_prefix = "pipeline_saver").get_logger()


class PipelineSaver:
    """
    A utility class for saving trained scikit-learn pipelines to disk.

    This class uses `joblib` to serialize and store a given machine learning pipeline
    for future inference or deployment.

    Attributes:
    ----------
        `output_dir`                  {str}              : Directory where the pipeline will be saved.
        
        `pipeline`        {sklearn.pipeline.Pipeline}    : The trained scikit-learn pipeline to be saved.
    
    """

    def __init__(self, output_dir : str, pipeline : sklearn.pipeline.Pipeline) -> None:
        """
        Initialize the PipelineSaver.

        Parameters:
        ----------
            `output_dir`                   {str}                   : The directory path where the pipeline will be saved.
            
            `pipeline`          {sklearn.pipeline.Pipeline}        : The trained pipeline object to be saved.
        
        """

        try:

            self.output_dir = output_dir
            self.pipeline   = pipeline

            os.makedirs(self.output_dir, exist_ok = True)

            pipeline_saver_logger.info(f"PipelineSaver initialized with output directory: {self.output_dir}")

        except Exception as e:
            pipeline_saver_logger.error(f"Error initializing PipelineSaver: {repr(e)}")
            
            raise e

    def save_pipeline(self, filename: str = "pipeline_model.pkl") -> str:
        """
        Save the trained pipeline to a pickle file.

        Arguments:
        
            `filename`             {str, optional}        : Name of the pickle file (default is 'pipeline_model.pkl').

        Returns:

            `file_path`                 {str}             : Full path where the pipeline was saved.
        
        """

        try:

            file_path = os.path.join(self.output_dir, filename)
            
            with open(file_path, "wb") as f:
                joblib.dump(self.pipeline, f)
                pipeline_saver_logger.info(f"Pipeline saved successfully at: {file_path}")
            
            return file_path
        
        except Exception as e:
            pipeline_saver_logger.error(f"Error saving pipeline: {repr(e)}")
            
            raise e
