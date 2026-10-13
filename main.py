import dotenv
import os
from data.process import DataProcessor
from machine_learning.trainer import ModelTrainer
from machine_learning.evaluater import ModelEvaluation
from machine_learning.risk_analysis import Risk_Analysis
from CNN.cnn_main import Imageclassifier
from tensorflow.keras.datasets import cifar10
def main():
    dotenv.load_dotenv()
    data_path=os.getenv("raw_path")
    model_dir=os.getenv("model_dir")
    target_col=os.getenv("target_col")
    if not os.path.exists(data_path):
        print("data path didn't exists")
    processor=DataProcessor(data_path)
    processor.clean_data()
    processor.save_data()
    #2
    trainer=ModelTrainer(data_path,target_col,model_dir)
    trainer.prepare_data()
    train_model,y_test,x_test=trainer.use_data_train()
    #3
    eva=ModelEvaluation(train_model,x_test,y_test)
    df,best_model_name=eva.evaluation()
    #4
    risk=Risk_Analysis(train_model,best_model_name,y_test,x_test)
    risk.pred()
    risk.analysis()
    (train_images,train_labels),(test_images,test_labels)=cifar10.load_data()
    cnn=Imageclassifier(train_labels,train_images,test_labels,test_images)
    cnn.train_model()
    
    print("Done")


if __name__ == " __main__":
    main()