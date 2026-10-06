import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path
def setup_environment():
    load_dotenv()
    os.environ['KAGGLE_USERNAME']=os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY']=os.getenv('KAGGLE_KEY')
    os.environ["KAGGLE_TOKEN"]=os.getenv("KAGGLE_KEY")
    print("Done setting up the environment")
    return
setup_environment()
path=Path("student_data1.csv")
try:
    df=pd.read_csv(path)
except FileNotFoundError:
    print("Dataset didn't found in the device")
print(df.head())
print(df.isnull().sum())
print(df["Name"].tail())
df["Gender"]=df["Gender"].str.strip()
print(df["Gender"].unique())
gender_map={
    "M":"M",
    "m":"M",
    "male":"M",
    "Male":"M",
    1:"M"
}
df["Gender"]=df["Gender"].map(lambda x: gender_map.get(x,"F"))
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
print("Saving the cleaned dataset to a new CSV file")
df.to_csv("cleaned_student_data.csv", index=False)
print("Cleaned dataset saved successfully!")
topper_student_marks=np.max(df["Total"],axis=0)
topper_student=df[df["Total"]==topper_student_marks]["Name"].values[0]
weakest_student_marks=np.min(df["Total"],axis=0)
subjects=["Math","Science","English"]
for subject in subjects:
    min=np.min(df[subject],axis=0)
    max=np.max(df[subject],axis=0)
    student_max=df[df[subject]==max]["Name"].values[0]
    student_min=df[df[subject]==min]["Name"].values[0]
    print("*"*30)
    print(f"{"*"*10} {subject} subject analysis{"*"*10}")
    print(f"{student_max} has scored the highest marks in {subject} with a score of {max} marks")
    print(f"{student_min} is weak  in {subject} and he/she secured  {min} marks")
    print("*"*30)

print(f"{"*"*10} Topper student analysis analysis{"*"*10}")
print(f"From the above data we can analyze that the topper student is {topper_student} with a total score of {topper_student_marks}")
print(f" {topper_student} secured marks in science is {df[df["Total"]==topper_student_marks]["Science"].values[0]} and in math is {df[df["Total"]==topper_student_marks]["Math"].values[0]} and in english is {df[df["Total"]==topper_student_marks]["English"].values[0]}")
print("*"*30)
print(f"{"*"*10} Weakest student analysis{"*"*10}")
weak_student=df[df["Total"]==weakest_student_marks]["Name"].values[0]
print(f"The weakest student from the above data is {weak_student} with marks of {weakest_student_marks}")
print(f"{weak_student}  secured marks in science is {df[df["Total"]==weakest_student_marks]["Science"].values[0]} and in math is {df[df["Total"]==weakest_student_marks]["Math"].values[0]} and in english is {df[df["Total"]==weakest_student_marks]["English"].values[0]}")
print("*"*30)


