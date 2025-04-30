# DEPENDENCIES

import re
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
from librosa import ex
from pathlib import Path
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords

from ..utils.logger import LoggerSetup
from ..utils.save_plot import PlotSaver

# LOGGER SETUP
eda_logger = LoggerSetup(logger_name = "exploratory_data_analyzer.py", log_filename_prefix = "exploratory_data_analyzer").get_logger()


class EmotionEDA:
    
    """
    A class for comprehensive Exploratory Data Analysis (EDA) of emotion-labeled text data.
    
    Attributes:

        `df`                     {pd.DataFrame}         : DataFrame containing text and emotion data.

        `text_column`                 {str}             : Column name containing cleaned text data.

        `emotion_column`              {str}             : Column name containing emotion labels.

        `output_dir`                  {str}             : Optional directory to save generated plots.
    
    """


    def __init__(self, dataframe : pd.DataFrame, text_column : str = 'Text', emotion_column : str = 'Emotion', output_dir : str = None) -> None:
        """
        Initialize the EmotionEDA class.
        
        Arguments:

            `df`                   {pd.DataFrame}         : Input DataFrame with text and emotion data.
            
            `text_column`               {str}             : Name of the column with text.
            
            `emotion_column`            {str}             : Name of the column with emotion labels.
            
            `output_dir`                {str}             : Optional path to save plots.
        
        Raises:

            ValueError                                    : If required columns are not found in the DataFrame.
        
        Returns:

            None
        
        """
        
        try:
        
            if text_column not in dataframe.columns or emotion_column not in dataframe.columns:
                eda_logger.error(f"DataFrame must contain columns: {text_column} and {emotion_column}")
                
                raise ValueError(f"DataFrame must contain columns: {text_column} and {emotion_column}")
            
            else:
                eda_logger.info(f"All Columns are present in the Dataframe")

            self.df               = dataframe
            self.text_column      = text_column
            self.emotion_column   = emotion_column
            self.output_dir       = output_dir

            self.plt_saver        = PlotSaver(output_dir = output_dir)

            if output_dir:
                Path(output_dir).mkdir(parents = True, exist_ok = True)

                eda_logger.info(f"Output directory created: {output_dir}")

            eda_logger.info(f"EmotionEDA initialized with text column: {text_column} and emotion column: {emotion_column}")

        except Exception as e:
            eda_logger.error(f"Error initializing EmotionEDA: {repr(e)}")
            
            raise


    def plot_emotion_distribution(self) -> None:
        """
        Plot the distribution of emotions in the dataset.
        
        """

        try:

            plt.figure(figsize = (8, 5))
            
            sns.countplot(data  = self.df, 
                        x     = self.emotion_column, 
                        order = self.df[self.emotion_column].value_counts().index)
            
            plt.title("Emotion Class Distribution")
            plt.xlabel("Emotion")
            plt.ylabel("Count")
            plt.xticks(rotation = 45)
            self.plt_saver.save_plot(plot = plt, plot_name = "emotion_distribution")

            eda_logger.info("Emotion distribution plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating emotion distribution plot: {repr(e)}")
            
            raise


    def plot_text_length_distribution(self) -> None:
        """
        Plot the distribution of text lengths for each emotion.
        
        """

        try:
        
            self.df['text_length'] = self.df[self.text_column].apply(lambda x: len(str(x).split()))
            
            plt.figure(figsize = (8, 5))
            
            sns.boxplot(data   = self.df, 
                        x      = self.emotion_column, 
                        y      = 'text_length'
                        )
            
            plt.title("Text Length Distribution by Emotion")
            plt.xlabel("Emotion")
            plt.ylabel("Word Count")
            plt.xticks(rotation = 45)
            self.plt_saver.save_plot(plot = plt, plot_name = "text_length_distribution")

            eda_logger.info("Text length distribution plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating text length distribution plot: {repr(e)}")
            
            raise


    def generate_wordcloud_per_emotion(self) -> None:
        """
        Generate and display a word cloud for each emotion class.
        
        """
        
        try:
        
            stop_words    = set(stopwords.words('english'))
            
            for emotion in self.df[self.emotion_column].unique():
                
                text      = " ".join(self.df[self.df[self.emotion_column] == emotion][self.text_column].values)
                text      = re.sub(r'\W+', ' ', text.lower())
                
                wordcloud = WordCloud(width             = 800, 
                                      height            = 400, 
                                      background_color  = 'white',
                                      stopwords         = stop_words
                                      ).generate(text)
                
                plt.figure(figsize = (10, 5))
                plt.imshow(wordcloud, interpolation = 'bilinear')
                plt.axis("off")
                plt.title(f"WordCloud for Emotion: {emotion}")
                self.plt_saver.save_plot(plot = plt, plot_name = f"wordcloud_{emotion}")

                eda_logger.info(f"Word cloud generated for emotion: {emotion}")

        except Exception as e:
            eda_logger.error(f"Error generating word cloud: {repr(e)}")
            
            raise


    def plot_top_words_per_emotion(self, top_n : int = 10) -> None:
        """
        Plot the top N most common words for each emotion.
        
        Arguments:

            top_n              {int}               : Number of top words to display.
        
        """
        
        try:

            stop_words          = set(stopwords.words('english'))
            
            for emotion in self.df[self.emotion_column].unique():
            
                words           = " ".join(self.df[self.df[self.emotion_column] == emotion][self.text_column].values).lower().split()
                filtered_words  = [word for word in words if word not in stop_words and word.isalpha()]
                common_words    = Counter(filtered_words).most_common(top_n)
            
                if common_words:
                    
                    labels, values = zip(*common_words)
                    plt.figure(figsize = (8, 5))
                    
                    sns.barplot(x        = list(values), 
                                y        = list(labels), 
                                palette  = 'mako'
                                )
                    
                    plt.title(f"Top {top_n} Words for Emotion: {emotion}")
                    plt.xlabel("Frequency")
                    plt.ylabel("Word")
                    self.plt_saver.save_plot(plot = plt, plot_name = f"top_words_{emotion}")
                    plt.show()

                    eda_logger.info(f"Top words plot generated for emotion: {emotion}")

        except Exception as e:
            eda_logger.error(f"Error generating top words plot: {repr(e)}")
            
            raise


    def plot_sentiment_polarity_distribution(self) -> None:
        """
        Plot the sentiment polarity distribution using TextBlob for each emotion class.
        
        Polarity ranges from -1 (negative) to 1 (positive).
        
        """

        try:

            self.df['polarity'] = self.df[self.text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            plt.figure(figsize = (10, 5))
            
            sns.boxplot(data  = self.df, 
                        x     = self.emotion_column, 
                        y     = 'polarity'
                        )
            
            plt.title("Polarity Score Distribution by Emotion")
            plt.xlabel("Emotion")
            plt.ylabel("Polarity Score")
            plt.xticks(rotation = 45)
            self.plt_saver.save_plot(plot = plt, plot_name = "polarity_distribution")

            eda_logger.info("Polarity distribution plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating polarity distribution plot: {repr(e)}")
            
            raise


    def plot_most_common_words(self, top_n: int = 15) -> None:
        """
        Plot the top N most frequent words across the entire dataset.
        
        Arguments:

            top_n               {int}           : Number of most common words to display.
        
        """
        
        try:

            stop_words      = set(stopwords.words('english'))
            all_words       = " ".join(self.df[self.text_column].astype(str)).lower().split()
            filtered_words  = [word for word in all_words if word not in stop_words and word.isalpha()]
            common_words    = Counter(filtered_words).most_common(top_n)

            if common_words:
                labels, values = zip(*common_words)
                plt.figure(figsize = (10, 6))
                
                sns.barplot(x        = list(values), 
                            y        = list(labels), 
                            palette  = 'viridis'
                            )
                
                plt.title(f"Top {top_n} Most Common Words")
                plt.xlabel("Frequency")
                plt.ylabel("Word")
                self.plt_saver.save_plot(plot = plt, plot_name = "most_common_words")

                eda_logger.info(f"Most common words plot generated successfully")

        except Exception as e:
            eda_logger.error(f"Error generating most common words plot: {repr(e)}")
            
            raise


    def plot_avg_word_count_per_emotion(self) -> None:
        """
        Plot average word count per emotion category.
        
        """
        try:

            self.df['word_count'] = self.df[self.text_column].apply(lambda x: len(str(x).split()))
            avg_counts            = self.df.groupby(self.emotion_column)['word_count'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(8, 5))
            
            sns.barplot(x        = avg_counts.values, 
                        y        = avg_counts.index, 
                        palette  = 'cubehelix'
                        )
            
            plt.title("Average Word Count per Emotion")
            plt.xlabel("Average Word Count")
            plt.ylabel("Emotion")
            self.plt_saver.save_plot(plot = plt, plot_name = "avg_word_count_per_emotion")

            eda_logger.info("Average word count per emotion plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating average word count plot: {repr(e)}")
            
            raise


    def plot_emotion_distribution_pie(self) -> None:
        """
        Plot a pie chart of the emotion class distribution.
        
        """

        try: 

            emotion_counts = self.df[self.emotion_column].value_counts()
            plt.figure(figsize = (7, 7))
            
            plt.pie(emotion_counts.values, 
                    labels      = emotion_counts.index, 
                    autopct     = '%1.1f%%', 
                    startangle  = 140
                    )
            
            plt.title("Emotion Distribution (Pie Chart)")
            self.plt_saver.save_plot(plot = plt, plot_name = "emotion_distribution_pie")

            eda_logger.info("Emotion distribution pie chart created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating emotion distribution pie chart: {repr(e)}")
            
            raise
        

    def plot_avg_polarity_per_emotion(self) -> None:
        """
        Plot the average polarity score per emotion class.
        
        """

        try:

            self.df['polarity']  = self.df[self.text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            avg_polarity         = self.df.groupby(self.emotion_column)['polarity'].mean().sort_values()

            plt.figure(figsize = (8, 5))
            
            sns.barplot(x        = avg_polarity.values, 
                        y        = avg_polarity.index, 
                        palette  = 'coolwarm'
                        )
            
            plt.title("Average Polarity per Emotion")
            plt.xlabel("Average Polarity Score")
            plt.ylabel("Emotion")
            self.plt_saver.save_plot(plot = plt, plot_name = "avg_polarity_per_emotion")

            eda_logger.info("Average polarity per emotion plot created successfully")
        
        except Exception as e:
            eda_logger.error(f"Error creating average polarity plot: {repr(e)}")
            
            raise


    def run_all_eda(self):
        """
        Function to run all EDA functions in sequence.

        """

        try:

            self.plot_emotion_distribution()
            self.plot_text_length_distribution()
            self.plot_sentiment_polarity_distribution()
            self.plot_most_common_words()
            self.generate_wordcloud_per_emotion()
            self.plot_top_words_per_emotion()
            self.plot_avg_word_count_per_emotion()
            self.plot_emotion_distribution_pie()
            self.plot_avg_polarity_per_emotion()

            eda_logger.info("All EDA plots generated successfully")

        except Exception as e:
            eda_logger.error(f"Error running all EDA functions: {repr(e)}")
            
            raise


