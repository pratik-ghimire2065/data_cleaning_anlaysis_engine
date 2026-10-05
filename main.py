import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
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
print("Saving the cleaned dataset to a new CSV file")
df.to_csv("cleaned_student_data.csv", index=False)
print("Cleaned dataset saved successfully!")
highest_math_score=np.max(df["Math"],axis=0)
student_with_high_math_marks=df[df["Math"]==highest_math_score]["Name"].values[0]
print(f"The student with the highest math marks is {student_with_high_math_marks} with a score of {highest_math_score}")
highest_science_score=np.max(df["Science"],axis=0)
highest_english_score=np.max(df["English"],axis=0)
topper_student_marks=np.max(df["Total"],axis=0)
student_with_high_english=df[df["English"]==highest_english_score]["Name"].values[0]
student_with_high_science=df[df["Science"]==highest_science_score]["Name"].values[0]
topper_student=df[df["Total"]==topper_student_marks]["Name"].values[0]
lowest_math_score=np.min(df["Math"],axis=0)
lowest_science_score=np.min(df["Science"],axis=0)
lowest_english_score=np.min(df["English"],axis=0)
weakest_student_marks=np.min(df["Total"],axis=0)

print("*"*30)
print(f"{"*"*10} Science subject analysis{"*"*10}")
print(f"The student with the highest science marks is {student_with_high_science} with a score of {highest_science_score}")
print(f"The student who scored the lowest marks in science is {df[df['Science']==lowest_science_score]['Name'].values[0]} with marks of {lowest_science_score}")
print("*"*30)
print(f"{"*"*10} Math subject analysis{"*"*10}")
print(f"The student with the highest math marks is {student_with_high_math_marks} with a score of {highest_math_score}")
print(f"The student who scored the lowest marks in math is {df[df['Math']==lowest_math_score]['Name'].values[0]} with marks of {lowest_math_score}")
print("*"*30)
print(f"{"*"*10} English subject analysis{"*"*10}")
print(f"The student with the highest english marks is {student_with_high_english} with a score of {highest_english_score}")
print(f"The student who scored the lowest marks in english is {df[df['English']==lowest_english_score]['Name'].values[0]} with marks of {lowest_english_score}")
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


