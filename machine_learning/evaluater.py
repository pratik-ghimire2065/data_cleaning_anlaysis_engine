import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
class ModelEvaluation():
    def __init__(self, models, x_test, y_test):
        self.models = models
        self.x_test = x_test
        self.y_test = y_test


    def evaluation(self):
        result = []
        for name, model in self.models.items():
            preds = model.predict(self.x_test)
            result.append({
                "Model Name": name,  # Clean string name instead of the whole Pipeline object
                "Mean_Absolute_Error": mean_absolute_error(self.y_test, preds),
                "Mean_Squared_Error": mean_squared_error(self.y_test, preds),
                "R2_Score": r2_score(self.y_test, preds)
            })

        # 1. Print statement moved OUTSIDE the loop
        print("\n=== Model Final Summary ===")

        # 2. Convert to a DataFrame for a beautiful, clean table representation
        summary_df = pd.DataFrame(result)
        print(summary_df.to_string(index=False))
        best_model_index=summary_df["R2_Score"].idxmax()
        best_model_row=summary_df.iloc[best_model_index]
        best_model=best_model_row["Model Name"]
        print(f"Best model from the table is { best_model } Model")

        return summary_df ,best_model