import random

class Perceptron:
    def __init__(self, inital_weights: list = None, grade: int = 0):
        self.weights = inital_weights if inital_weights else [1 for _ in range(grade+1)]
        
    def get_weights(self) -> list:
        return self.weights.copy()
    
    def set_weights(self, weights: list):
        self.weights = weights
        
    def calc_line_ecuation(self, point: list) -> float:
        result = 0
            
        for i in range(len(point)):
            result += self.weights[i] * point[i]
        
        result += self.weights[-1] # Add bias (last element of the weights list)
    
        return result
    
    def predict_point(self, point: list) -> bool:
        return self.calc_line_ecuation(point) >= 0
    
    def predict(self, data: list[list], values: bool = False) -> list[int]: 
        if values:
            return [self.calc_line_ecuation(point) for point in data]
        
        # If x = w1 + w2 + ... + w3 >= 0, then step(x) = 1
        return [self.predict_point(point) for point in data]

    def ajust(self, epochs: int, learning_rate: float, data: list[list], data_labels: list[bool]):
        for _ in range(epochs):
            # Select a random point from the dataset and its corresponding label
            random_point_index = random.randint(0, len(data)-1)
            point = data[random_point_index]
            point_label = data_labels[random_point_index]
            
            prediction = self.predict_point(point) # Predict the label of the random point
            
            # Perceptron trick
            for i in range(len(point)):
                self.weights[i] = self.weights[i] + learning_rate * (point_label - prediction) * point[0]
            
            self.weights[-1] = self.weights[-1] + learning_rate * (point_label - prediction)
    
    def error(self, data: list[list]) -> float:
        error = 0.0
        
        for point in data:
            for i in range(len(point)):
                error += self.weights[i] * point[i]
                
            error += self.weights[-1] # error = |w1*x1 + w2*x2 + ... + wn*xn + b|
        
        # Average error of all points
        return abs(error) / len(data)
