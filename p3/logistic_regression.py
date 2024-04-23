import math
from perceptron import Perceptron

class LogisticRegression(Perceptron):
    def __init__(self, inital_weights: list = None, grade: int = 0):
        super().__init__(inital_weights, grade)
        
    def predict_point(self, point: list) -> float:
        result: float = 0
            
        for i in range(len(point)):
            result += self.weights[i] * point[i]
        
        result += self.weights[-1] # Add bias (last element of the weights list)
    
        # sigma(x) = 1 / (1 + exp(-x))
        return 1 / (1 + math.exp(-result))
    
    def predict(self, data: list[list], threshold: float = 0.5, prob: bool = False) -> list:
        if prob:
            return [self.predict_point(point) for point in data]
        
        return [int(self.predict_point(point) >= threshold) for point in data]
    
    def error(self, data: list[list], data_labels: list[bool]) -> float:
        error = 0
        
        for i in range(len(data)):
            # logloss = -ylog(ŷ) - (1-y)log(1-ŷ)
            label = data_labels[i] # y
            prediction = self.predict_point(data[i]) # ŷ
            
            logloss = -label*math.log(prediction) - (1-label)*math.log(1-prediction)
            
            error += logloss
            
        return error / len(data)