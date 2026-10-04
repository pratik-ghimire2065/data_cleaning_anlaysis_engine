import os
from dotenv import load_dotenv
import pandas as pd
def envvironmental_operations():
    load_dotenv()
    os.environ['KAGGLE_USERNAME']=os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY']=os.getenv('KAGGLE_KEY')
    os.environ["KAGGLE_TOKEN"]=os.getenv("KAGGLE_KEY")
    print("Done setting up the environment")
df=pd.read_csv("C:\data_analysis_cleaning_engine\student_data1.csv")
print(df.head())
print(df.isnull().sum())
print(df["Name"].tail())
df["Gender"]=df["Gender"].str.strip()
print(df["Gender"].unique())
def replace_gender_values(a):
    if a=="M"or a=="m" or a=="Male" or a=="male" or a=="1":
        return "M"
    else:
        return "F"
df["Gender"]=df["Gender"].apply(replace_gender_values)
print(df["Gender"].unique())
df["Grade"]=df["Grade"].str.strip()
df["Grade"]=df["Grade"].str.removeprefix("Grade ")
df["Grade"]=pd.to_numeric(df["Grade"],errors="coerce").astype("int64")
print(df["Grade"].unique())
print(df["Math"].unique())
def subject_fix(a):
    print(f" {a} subject Data before cleaning\n",df[a].unique())
    df[a]=df[a].str.strip()
    df[a]=df[a].str.removesuffix(" marks")
    df[a]=pd.to_numeric(df[a],errors="coerce").astype("int64")
    print(f" {a} subject Data after cleaning\n",df[a].unique())
subject_fix("Math")
subject_fix("Science")
subject_fix("English")
df["Name"]=df["Name"].str.strip().str.title()

df["Name"]=df["Name"].str.removeprefix("'")
df["Name"]=df["Name"].str.removesuffix("'")
df["Name"]=df["Name"].str.removeprefix('"')
df["Name"]=df["Name"].str.removesuffix('"')
print(df.tail())
print(df["Name"].unique())
print(df["Name"].value_counts())
df=df.drop_duplicates(keep="first")
print(df.info())
print(df.describe())
print("Everthing clear from this dataset and ready for data analysis and data visualization")