import pandas  as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os 
from pathlib import Path
class DataProcessor():
    def __init__(self,data_path="C:\\student_analysis\\data\\Studentdata2.csv"):
        self.data_path=data_path
        self.data=None
    def load_data(self):
        if os.path.exists(self.data_path):
            self.data=pd.read_csv(self.data_path)
            print("Data Loaded Successfully......")
        else:
            raise FileNotFoundError(f"The file {self.data_path} does not exist in your system . Please check again and try again ")
    def clean_data(self):
        if self.data is not None:
            self.data.dropna(axis=0,inplace=True)
            print("Removed all the rows with missing values successfully...")
        else: 
            raise ValueError("Data is not loaded yet.")
    def save_data(self,output_path="C:\\student_analysis\\data\\Studentdata2.csv"):
        if self.data is not None:
            self.data.to_csv(output_path,index=False)
            print("Data saved successfully...")
        else:
            raise ValueError("No data to save.")
        return self.data
