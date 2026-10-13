import joblib
class Risk_Analysis():
  def __init__(self,trained_models,best_model_name,y_test,x_test):
    self.trained_models=trained_models
    self.best_model=trained_models[best_model_name]
    self.y_test=y_test
    self.x_test=x_test
    self.y_pred=None
  def pred(self):
    y_pred=self.best_model.predict(self.x_test)
    self.y_pred=y_pred
    return y_pred
  def analysis(self):
    errors=self.y_pred != self.y_test
    print(f"model predict total  {len(errors)} wrong prediction  ")

