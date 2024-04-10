import random

class Perceptron:
    def __init__(self, data: list[list], data_labels: list[bool], inital_weights: list = None):
        self.data = data
        self.data_labels = data_labels
        self.weights = inital_weights if inital_weights else [1 for _ in range(len(data[0])+1)]
        
    def get_weights(self) -> list:
        return self.weights.copy()
    
    def get_data(self) -> list[list]:
        return self.data.copy()
    
    def get_data_labels(self) -> list[bool]:
        return self.data_labels.copy()
    
    def set_weights(self, weights: list):
        self.weights = weights
        
    def set_data(self, data: list[list]):
        self.data = data
        
    def set_data_labels(self, data_labels: list[bool]):
        self.data_labels = data_labels
        
    def predict_point(self, point: list) -> bool:
        result = 0
            
        for i in range(len(point)):
            result += self.weights[i] * point[i]
        
        result += self.weights[-1] # Add bias (last element of the weights list)
    
        # If x = w1 + w2 + ... + w3 >= 0, then step(x) = 1
        return result >= 0
    
    def predict(self) -> list[bool]: 
        return [self.predict_point(point) for point in self.data]

    def ajust(self, epochs: int, learning_rate: float):
        for _ in range(epochs):
            # Select a random point from the dataset and its corresponding label
            random_point_index = random.randint(0, len(self.data)-1)
            point = self.data[random_point_index]
            point_label = self.data_labels[random_point_index]
            
            prediction = self.predict_point(point) # Predict the label of the random point
            
            # Perceptron trick
            for i in range(len(point)):
                self.weights[i] = self.weights[i] + learning_rate * (point_label - prediction) * point[0]
            
            self.weights[-1] = self.weights[-1] + learning_rate * (point_label - prediction)
    
    def error(self) -> float:
        error = 0.0
        
        for point in self.data:
            for i in range(len(point)):
                error += self.weights[i] * point[i]
                
            error += self.weights[-1] # error = |w1*x1 + w2*x2 + ... + wn*xn + b|
        
        # Average error of all points
        return abs(error) / len(self.data)
