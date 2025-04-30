# DEPENDENCIES

import pytz
import sqlite3
from librosa import ex
from datetime import datetime

from ...src.utils.logger import LoggerSetup

database_manager_logger = LoggerSetup(logger_name = "database_manager.py", log_filename_prefix = "database_manager").get_logger()

class DatabaseManager:
    """
    A class for managing database operations for tracking page visits and emotion classifications.
    
    This class provides methods to create tables, add data, and retrieve data from a SQLite database.
    It handles time conversion to Indian Standard Time (IST) and manages database connections.
    """

    def __init__(self, db_path : str) -> None:
        """
        Initialize the DatabaseManager with a connection to the specified database.
        
        Arguments:
        
            db_path            {str}       : Path to the SQLite database file. Defaults to './data/data.db'.
        
        """

        try:
        
            self.conn  = sqlite3.connect(db_path, check_same_thread = False)
            self.c     = self.conn.cursor()
            self.ist   = pytz.timezone('Asia/Kolkata') 
            
            self.create_page_visited_table()
            self.create_emotionclf_table()

            database_manager_logger.info("Database Connection Successful")

        except Exception as e:
            database_manager_logger.error(f"Error Occured in Connection: {repr(e)}")

            raise

    
    def create_page_visited_table(self):
        """
        Create a table for tracking page visits if it doesn't already exist.
        
        """

        try:
        
            self.c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT, timeOfvisit TIMESTAMP)')
            self.conn.commit()

            database_manager_logger.info("Create table for Visited Page")

        except Exception as e:
            database_manager_logger.error(f"Error occured whie creating table: {repr(e)}")

            raise

    def add_page_visited_details(self, pagename : str, timeOfvisit : datetime = None) -> None:
        """
        Add details about a page visit to the database.
        
        Arguments:

            `pagename`                  {str}                  : Name of the page visited.
        
            `timeOfvisit`       {datetime, optional}           : Time when the page was visited. 
        
        """
        
        try:

            if timeOfvisit is None:
                timeOfvisit = datetime.now(self.ist).strftime("%Y-%m-%d %H:%M:%S")
            
            else:
                timeOfvisit = timeOfvisit.astimezone(self.ist).strftime("%Y-%m-%d %H:%M:%S")
                
            self.c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', 
                        (pagename, timeOfvisit))
            
            self.conn.commit()

            database_manager_logger.info("Details added Successfully")

        except Exception as e:
            database_manager_logger.error(f"Error Occurred while adding details: {repr(e)}")

            raise

    def view_all_page_visited_details(self) -> list:
        """
        Retrieve all page visit records from the database.
        
        Returns:
        
            list               : List of tuples containing page visit details.
        
        """

        try:
        
            self.c.execute('SELECT * FROM pageTrackTable')
            data = self.c.fetchall()

            database_manager_logger.info("Showing all the details")
            
            return data
        
        except Exception as e:
            database_manager_logger.error(f"Error Occurred while showing details: {repr(e)}")

            raise

    def create_emotionclf_table(self):
        """
        Create a table for storing emotion classification results if it doesn't already exist.
        
        """

        try:
        
            self.c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT, '
                        'probability NUMBER, timeOfvisit TIMESTAMP)')
            self.conn.commit()

            database_manager_logger.info("emotionclf_table created successfully")

        except Exception as e:
            database_manager_logger.error(f"Error Occurred in creating emotionclf_table: {repr(e)}")

            raise

    def add_prediction_details(self, rawtext : str, prediction : str, probability : float, timeOfvisit : datetime = None) -> None:
        """
        Add emotion classification prediction details to the database.
        
        Arguments:
            
            rawtext                    {str}             : The text that was analyzed.
            
            prediction                 {str}             : The emotion classification prediction.
            
            probability               {float}            : The probability/confidence score of the prediction.
            
            timeOfvisit        {datetime, optional}      : Time when the prediction was made.
                                            
        """

        try:

            if timeOfvisit is None:
                timeOfvisit = datetime.now(self.ist).strftime("%Y-%m-%d %H:%M:%S")
            
            else:
                timeOfvisit = timeOfvisit.astimezone(self.ist).strftime("%Y-%m-%d %H:%M:%S")
                
            self.c.execute('INSERT INTO emotionclfTable(rawtext, prediction, probability, timeOfvisit) '
                        'VALUES (?, ?, ?, ?)', (rawtext, prediction, probability, timeOfvisit))
            self.conn.commit()

            database_manager_logger.info("Prediction Details added")

        except Exception as e:
            database_manager_logger.error(f"Error Adding Prediction Details: {repr(e)}")

            raise

    def view_all_prediction_details(self) -> list:
        """
        Retrieve all emotion classification prediction records from the database.
        
        Returns:
            
            list                : List of tuples containing prediction details.
        """

        try:
        
            self.c.execute('SELECT * FROM emotionclfTable')
            data = self.c.fetchall()
            
            database_manager_logger.info("Showing all Prediction Details")

            return data
        
        except Exception as e:
            database_manager_logger.error(f"Error Showing Prediction Details: {repr(e)}")

            raise
    
    def close_connection(self) -> None:
        """
        Close the database connection.
        
        It's good practice to call this method when the database is no longer needed
        to release system resources.
        
        """
        
        try:

            if self.conn:
                self.conn.close()

            database_manager_logger.info("Connection Closed")

        except Exception as e:
            database_manager_logger.error(f"Error Closing Connection: {repr(e)}")

            raise


## EXAMPLE USAGE

# db_manager  = DatabaseManager()
# db_manager.add_page_visited_details("homepage")
# db_manager.add_prediction_details("I'm feeling great today!", "happy", 0.92)
# page_visits = db_manager.view_all_page_visited_details()
# predictions = db_manager.view_all_prediction_details()
# db_manager.close_connection()