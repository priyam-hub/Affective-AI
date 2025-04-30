# DEPENDENCIES

from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.data_loader import DataLoader
from src.data_cleaner.cleaner import TextCleaner
from src.pipeline.model_pipeline import ModelTrainer
from sklearn.model_selection import train_test_split
from src.exploratory_data_analysis.exploratory_data_analyzer import EmotionEDA


# LOGGER SETUP
main_logger = LoggerSetup(logger_name = "main.py", log_filename_prefix = "main").get_logger()

def main():

    try:

        dataLoader            = DataLoader()   
        emotion_raw_df        = dataLoader.data_loader(file_path = Config.EMOTION_DATASET_RAW)
        main_logger.info("Data loaded successfully:")

        data_cleaner          = TextCleaner(dataframe = emotion_raw_df)
        emotion_cleaned_df    = data_cleaner.data_cleaning()
        main_logger.info("Data cleaned successfully.")

        dataLoader.data_saver(dataframe = emotion_cleaned_df, 
                              file_path = Config.EMOTION_DATASET_CLEANED
                              )

        # data_analyzer         = EmotionEDA(dataframe       = emotion_cleaned_df, 
        #                                    text_column     = "Text", 
        #                                    emotion_column  = "Emotion", 
        #                                    output_dir      = Config.EDA_RESULTS_PATH
        #                                    )     
        # data_analyzer.run_all_eda()
        # main_logger.info("All EDA completed successfully.")

        X                                = emotion_cleaned_df["Clean_Text"]
        y                                = emotion_cleaned_df["Emotion"]  

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)

        trainer                          = ModelTrainer(x_train  = x_train, 
                                                        x_test   = x_test, 
                                                        y_train  = y_train, 
                                                        y_test   = y_test
                                                        )
        
        trainer.train_and_evaluate(output_dir = Config.ML_MODEL_SAVE_PATH)
        
        main_logger.info("Model training and evaluation completed successfully.")

        trainer.plot_results(output_dir = Config.MODEL_ACCURACY_PLOT_PATH)

    except Exception as e:
        
        print(f"Error Occurred In PipeLine: {repr(e)}")
        return



if __name__ == "__main__":
    main()
