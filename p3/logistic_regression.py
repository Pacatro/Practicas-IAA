import math
from perceptron import Perceptron

class LogisticRegression(Perceptron):
    def __init__(self, data: list[list], data_labels: list[bool], inital_weights: list = None):
        super().__init__(data, data_labels, inital_weights)
        
    def predict_point(self, point: list) -> float:
        result: float = 0
            
        for i in range(len(point)):
            result += self.weights[i] * point[i]
        
        result += self.weights[-1] # Add bias (last element of the weights list)
    
        # sigma(x) = 1 / (1 + exp(-x))
        return 1 / (1 + math.exp(-result))
    
    def predict(self, threshold: float = 0.5, softmax: bool = False) -> list:
        if softmax:
            predictions = [self.predict_point(point) for point in self.data]
            exp_predictions = [math.exp(p) for p in predictions]
            return [p / sum(exp_predictions) for p in exp_predictions]
        
        return [self.predict_point(point) >= threshold for point in self.data]
    
    def error(self) -> float:
        error = 0
        
        for i in range(len(self.data)):
            # logloss = -ylog(ŷ) - (1-y)log(1-ŷ)
            label = self.data_labels[i] # y
            prediction = self.predict_point(self.data[i]) # ŷ
            
            logloss = -label*math.log(prediction) - (1-label)*math.log(1-prediction)
            
            error += logloss
            
        return error / len(self.data)