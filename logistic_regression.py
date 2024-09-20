import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            return
        
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)

        # Ensure to extract features and target correctly
        self.X_train = self.training_set[['exam_score_1', 'exam_score_2']].values
        self.y_train = self.training_set['label'].values
        
        if self.perform_test:
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']].values
            self.y_test = self.test_set['label'].values

        
        
    def model_fit_linear(self):
        '''
        initialize self.model_linear here and call the fit function
        '''
        # Initialize the model
        self.model_linear = LinearRegression()
        
        # Fit the model
        self.model_linear.fit(self.X_train, self.y_train)
        
        pass
    
    def model_fit_logistic(self):
        '''
        initialize self.model_logistic here and call the fit function
        '''
        
        # Initialize the model 
        self.model_logistic = LogisticRegression()
        
        # Fit the model aka training the model with our data
        self.model_logistic.fit(self.X_train, self.y_train)

        pass
    
    # def model_predict_linear(self):
    #     '''
    #     Calculate and return the accuracy, precision, recall, f1, support of the model.
    #     '''
    #     self.model_fit_linear()
    #     accuracy = 0.0
    #     precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
    #     assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
    #     assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        
    #     if self.X_test is not None:
    #         # perform prediction here
    #         y_pred_continuous = self.model_linear.predict(self.X_test)

    #         # Convert continuous predictions to binary - Method: threshold
    #         y_pred = np.where(y_pred_continuous >= 0.5, 1, 0)

    #         # Calculate metrics
    #         accuracy = accuracy_score(self.y_test, y_pred)
    #         precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average='binary')

    #         return [accuracy, precision, recall, f1, support]
    #     else:
    #         raise ValueError("X_test or y_test not defined.")
    def model_predict_linear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_linear()
        if self.X_test is None or self.y_test is None:
            raise ValueError("X_test or y_test not defined.")
        
        y_pred_continuous = self.model_linear.predict(self.X_test)
        y_pred = np.where(y_pred_continuous >= 0.5, 1, 0)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)

        return [accuracy, precision, recall, f1, support]
            
     

    # def model_predict_logistic(self):
    #     '''
    #     Calculate and return the accuracy, precision, recall, f1, support of the model.
    #     '''
    #     self.model_fit_logistic()
    #     accuracy = 0.0
    #     precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
    #     assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
    #     assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
    #     if self.X_test is not None:
    #         # Predict class labels directly
    #         y_pred = self.model_logistic.predict(self.X_test)

    #         # Calculate metrics
    #         accuracy = accuracy_score(self.y_test, y_pred)
    #         precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average='binary')

    #         return [accuracy, precision, recall, f1, support]
    #     else:
    #         raise ValueError("X_test or y_test not defined.")
    def model_predict_logistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_logistic()
        if self.X_test is None or self.y_test is None:
            raise ValueError("X_test or y_test not defined.")
        
        y_pred = self.model_logistic.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)

        return [accuracy, precision, recall, f1, support]
    
    def plot_decision_boundary(self, model, dataset, dataset_num):
        # Set min and max values and give it some padding
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        h = 0.02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Predict the function value for the whole gid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, s=20, edgecolor='k', cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"Decision Boundary for dataset {dataset_num} - {'Logistic' if isinstance(model, LogisticRegression) else 'Linear'} Regression")
        plt.xlabel('Exam Score 1')
        plt.ylabel('Exam Score 2')
        plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic and Linear Regression Model Testing')
    parser.add_argument('-d', '--dataset_num', type=str, default="1", choices=["1", "2"], help='Dataset number. For example, 1 or 2')
    parser.add_argument('-t', '--perform_test', action='store_true', help='Flag to indicate if testing should be performed')
    args = parser.parse_args()
     # Create instances for each dataset
    for dataset_num in ['1', '2']:
        classifier = MyLogisticRegression(dataset_num, args.perform_test)

        # Predict and plot for linear model
        classifier.model_predict_linear()  # Ensure the model is fitted
        classifier.plot_decision_boundary(classifier.model_linear, 'train', dataset_num)

        # Predict and plot for logistic model
        classifier.model_predict_logistic()  # Ensure the model is fitted
        classifier.plot_decision_boundary(classifier.model_logistic, 'train', dataset_num)

    # Create an instance of the classifier
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)

    # Check if testing should be performed
    if args.perform_test:
        if classifier.X_test is not None and classifier.y_test is not None:
            linear_metrics = classifier.model_predict_linear()
            logistic_metrics = classifier.model_predict_logistic()
            print("Linear Model Metrics:", linear_metrics)
            print("Logistic Model Metrics:", logistic_metrics)
        else:
            print("Test data not defined or not loaded correctly.")
    else:
        print("Testing flag not set. No predictions will be made.")
