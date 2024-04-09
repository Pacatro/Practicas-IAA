import random

class Perceptron:
    def __init__(self, num_weights: int):
        self.weights = [1 for _ in range(num_weights)]
        
    def get_weights(self): 
        return self.weights
    
    def set_weights(self, weights: list): 
        self.weights = weights
    
    def ajust(self, points: list[list], points_labels: list, epochs: int, learning_rate: float):
        for epoch in range(0, epochs):
            print(f"Epoch: {epoch}")
            
            print(f"Initial weights: {self.weights}")
            
            random_point_index = random.randint(0, len(points))
            point = points[random_point_index]
            point_label = points_labels[random_point_index]
            
            prediction = self.predict(point)
            
            # Perceptron trick
            for i in range(len(point)):
                self.weights[i] = self.weights[i] + learning_rate * (point_label - prediction) * point[0]
            
            self.weights[-1] = self.weights[-1] + learning_rate * (point_label - prediction)
            
            print(f"Ajust weights: {self.weights}")
    
    def predict(self, point: list):
        result = 0
        
        for i in range(len(point)):
            result += self.weights[i] * point[i]
        
        result += self.weights[-1] # Add bias (last element of the weights list)
        
        # If w1 + w2 + ... + w3 >= 0 -> step(x) = 1
        return result >= 0

def main():
    perceptron = Perceptron(3)
    
    print(perceptron.get_weights())
    print(perceptron.predict([1, 2]))
    
if __name__ == "__main__":
    main()