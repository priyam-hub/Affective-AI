# DEPENDENCIES

import pandas as pd
from librosa import ex
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer

from ..utils.logger import LoggerSetup
from ..utils.save_plot import PlotSaver
from ..utils.pipeline_saver import PipelineSaver

# LOGGER SETUP  
model_pipeline_logger = LoggerSetup(logger_name = "model_pipeline.py", log_filename_prefix = "model_pipeline").get_logger()

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    model_pipeline_logger.info("All Modules Imported Successfully")

except ImportError:
    XGBClassifier  = None
    LGBMClassifier = None

    model_pipeline_logger.error("Error Occurred in Importing")

class ModelTrainer:
    """
    A class for training and evaluating multiple machine learning models 
    on text classification tasks using pipelines.

    This class supports:
        - Building pipelines with CountVectorizer and classifiers.
        - Training multiple models (Logistic Regression, Naive Bayes, Random Forest, Linear SVC).
        - Evaluating models based on accuracy.
        - Displaying results in a DataFrame.
        - Visualizing results in a bar chart.
    
    """
    
    def __init__(self, x_train : list, x_test : list, y_train : list, y_test : list) -> None:
        """
        Initialize the ModelTrainer with training and testing data.

        Arguments:
        
            `x_train`            {list}        : Feature data for training (typically raw text).
            
            `x_test`             {list}        : Feature data for testing.
            
            `y_train`            {list}        : Target labels for training data.
            
            `y_test`             {list}        : Target labels for testing data.

        Returns:

            None

        """

        try:
        
            self.x_train    = x_train
            self.x_test     = x_test
            self.y_train    = y_train
            self.y_test     = y_test
            self.results    = []

            self.models     = {"Multinomial Logistic Regression" : LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000),
                            "Multinomial Naive Bayes"         : MultinomialNB(),
                            "Random Forest"                   : RandomForestClassifier(),
                            "Linear SVC"                      : LinearSVC(),
                            "RBF SVM": SVC(kernel="rbf", probability=True),
                            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
                            "Hist Gradient Boosting": HistGradientBoostingClassifier(),
                            }

            if XGBClassifier is not None:
                self.models["XGBoost"]  = XGBClassifier(use_label_encoder = False, eval_metric = 'mlogloss')
            if LGBMClassifier is not None:
                self.models["LightGBM"] = LGBMClassifier()
            
            model_pipeline_logger.info("ModelTrainer initialized successfully.")

        except Exception as e:
            model_pipeline_logger.error(f"Error initializing ModelTrainer: {repr(e)}")
            
            raise e

    def train_and_evaluate(self, output_dir : str) -> None:
        """
        Train each model using a text classification pipeline and evaluate accuracy on test data.

        The results (model name and accuracy) are stored in the `results` attribute.

        """

        try:

            model_pipeline_logger.info("Training and evaluating models...")

            for name, model in self.models.items():
                pipeline = Pipeline([('cv', CountVectorizer()),('clf', model)])

                pipeline.fit(self.x_train, self.y_train)
                preds    = pipeline.predict(self.x_test)
                acc      = accuracy_score(self.y_test, preds)

                self.results.append({"Model"    : name,
                                    "Accuracy" : round(acc, 4)
                                    })
                
                filename = f"{name.lower().replace(' ', '_')}.pkl"
                
                saver    = PipelineSaver(output_dir  = output_dir, 
                                        pipeline    = pipeline
                                        )
                
                saver.save_pipeline(filename = filename)

                model_pipeline_logger.info(f"Model: {name} | Accuracy: {round(acc, 4)}")
                model_pipeline_logger.info(f"Pipeline for {name} saved as {filename}.")

        except Exception as e:
            model_pipeline_logger.error(f"Error during training and evaluation: {repr(e)}")
            
            raise e


    def show_results(self):
        """
        Return a sorted DataFrame of model names and their accuracy scores.

        Returns:
        
            pd.DataFrame               : A DataFrame sorted by accuracy in descending order.

        """
        
        return pd.DataFrame(self.results).sort_values(by = "Accuracy", ascending = False).reset_index(drop = True)


    def plot_results(self, output_dir : str) -> None:

        """
        Plot a horizontal bar chart of model accuracy scores and save it as an image.

        Arguments:
        
            filename                 {str}                  : Directory of the Plot where it will be saved
        
        Returns:

            None
        
        """

        try:

            plt_saver        = PlotSaver(output_dir = output_dir)
            
            df               = self.show_results()
            plt.figure(figsize = (10, 6))
            
            plt.barh(df["Model"], 
                    df["Accuracy"], 
                    color = "skyblue"
                    )
            
            plt.xlabel("Accuracy")
            plt.title("Model Accuracy Comparison")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            plt_saver.save_plot(plot       = plt,
                                plot_name  = "model_accuracy_comparison.png"
                                )

            plt.close()

            model_pipeline_logger.info("Plot saved successfully.")

        except Exception as e:
            model_pipeline_logger.error(f"Error during plotting: {repr(e)}")
            
            raise e
