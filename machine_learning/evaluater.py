import pandas as pd
from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
class ModelEvaluation():
    def __init__(self,models:dict,x_test,y_test):
        self.models=models
        self.x_test=x_test
        self.y_test=y_test
    def evaluation(self):
        for name,model in self.models.items():
            result=[]
            preds=model.predict(self.x_test)
            proba=model.predict_proba(self.x_test)[:,1]
            result.append({
                "Model Name":model,
                "Accuracy_score":accuracy_score(self.y_test,preds),
                "Recall_Score": recall_score(self.y_test,preds,zero_division=0),
                "Precision_score": precision_score(self.y_test,preds,zero_division=0),
                "f1_score": f1_score(self.y_test,preds,zero_division=0),
                "Roc_auc_score": roc_auc_score(self.y_test,proba)
            })
        print("Model summary ")
        print(result)
        



