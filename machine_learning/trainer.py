from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import os
import pandas as pd
class ModelTrainer():
    def __init__(self,data_path:str,target_col:str,model_dir:str):
        self.data_path=data_path
        self.models={
            "KNN":KNeighborsRegressor(),
            "Linear_regression":LinearRegression(),
            "Decision_tree":DecisionTreeRegressor(),
            "Random_forest_Regressor":RandomForestRegressor()
        }
        self.target_column=target_col
        self.model_dir=model_dir
        self.x_train,self.x_test,self.y_train,self.y_test=[None] *4
        self.preprocessor=None
    def prepare_data(self):
        if os.path.exists(self.data_path):
            df=pd.read_csv(self.data_path,index_col=False)
            x=df.drop(columns=[self.target_column],axis=1)
            y=df[self.target_column]
            x_train,x_test,y_train,y_test=train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=42
            )
            self.x_train,self.x_test,self.y_train,self.y_test=x_train,x_test,y_train,y_test
            num_col=self.x_train.select_dtypes(include=["int64","float64"]).columns.tolist()
            cat_col=self.x_train.select_dtypes(include=["object"]).columns.tolist()
            num_feature=Pipeline(steps=[
                "impute",SimpleImputer(strategy="median"),
                "scaler",StandardScaler()
            ])
            cat_feature=Pipeline(steps=[
                "impute",SimpleImputer(strategy="mean"),
                "encoder",OneHotEncoder(handle_unknown="ignore")
            ])
            self.preprocessor=ColumnTransformer(transformers=[
                ("num",num_feature,num_col),
                ("cat",cat_feature,cat_col)
            ])
        else:
            raise FileNotFoundError("File didn't found ")
    def use_data_train(self):
        self.x_train_transformed=self.preprocessor.fit_transform(self.x_train)
        self.x_test_transformed=self.preprocessor.transform(self.x_test)
        for name,model in self.models.items():
            model.fit(self.x_train_transformed,self.y_train)
            filename=name.replace(" ","_") + ".pkl"
            joblib.dump(model ,os.path.join(self.model_dir,filename))   


        
