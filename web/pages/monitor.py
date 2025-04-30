# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import altair as alt
import streamlit as st
from pytz import timezone
import plotly.express as px
from datetime import datetime

from config.config import Config
from src.utils.logger import LoggerSetup
from web.utils.database_manager import DatabaseManager 

IST = timezone(Config.TIMEZONE_IST)

monitor_logger               = LoggerSetup(logger_name = "monitor.py", log_filename_prefix = "monitor").get_logger()

def main():

    try:

        database_Manager         = DatabaseManager(db_path = Config.DATABASE_PATH)
        

        database_Manager.add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("📊 Monitor App")

        with st.expander("📈 Page Metrics"):
            
            page_visited_details = pd.DataFrame(database_Manager.view_all_page_visited_details(), 
                                                columns = ['Page Name', 
                                                        'Time of Visit'
                                                        ]
                                                )
            st.dataframe(page_visited_details)

            pg_count             = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name = 'Counts')

            bar_chart            = alt.Chart(pg_count).mark_bar().encode(x      = 'Page Name', 
                                                                        y      = 'Counts', 
                                                                        color  = 'Page Name'
                                                                        )
            st.altair_chart(bar_chart, use_container_width = True)

            pie_chart            = px.pie(pg_count, 
                                        values = 'Counts', 
                                        names  = 'Page Name'
                                        )
            
            st.plotly_chart(pie_chart, use_container_width = True)

        # Emotion Classifier Metrics
        with st.expander("💡 Emotion Classifier Metrics"):
            
            df_emotions          = pd.DataFrame(database_Manager.view_all_prediction_details(), 
                                                columns = ['Rawtext', 
                                                        'Prediction', 
                                                        'Probability', 
                                                        'Time_of_Visit'
                                                        ]
                                                )
            st.dataframe(df_emotions)

            prediction_count     = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name = 'Counts')

            prediction_chart     = alt.Chart(prediction_count).mark_bar().encode(x      = 'Prediction', 
                                                                                y      = 'Counts', 
                                                                                color  = 'Prediction'
                                                                                )
            
            st.altair_chart(prediction_chart, use_container_width = True)

        monitor_logger.info("Monitor Page Visited Successfully")

    except Exception as e:
        monitor_logger.error(f"Error Occurred while running the Monitor Page: {repr(e)}")

        raise