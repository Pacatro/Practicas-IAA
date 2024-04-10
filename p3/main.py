import random
import time

class Perceptron:
    def __init__(self, data: list[list], data_labels: list[bool]):
        self.data = data
        self.data_labels = data_labels
        self.weights = [1 for _ in range(len(data[0])+1)]
        
    def get_weights(self): 
        return self.weights
    
    def set_weights(self, weights: list): 
        self.weights = weights
        
    def predict_point(self, point: list):
        prediction = 0
            
        for i in range(len(point)):
            prediction += self.weights[i] * point[i]
        
        prediction += self.weights[-1] # Add bias (last element of the weights list)
    
        # Prediction is true if x = w1 + w2 + ... + w3 >= 0
        return prediction >= 0
    
    def predict(self):
        return [self.predict_point(point) for point in self.data]

    def ajust(self, epochs: int, learning_rate: float):
        for _ in range(epochs):
            # Select a random point from the dataset and its corresponding label
            random_point_index = random.randint(0, len(self.data)-1)
            point = self.data[random_point_index]
            point_label = self.data_labels[random_point_index]
            
            prediction = self.predict_point(point) # Predict the label of the random point
                
            # print(f"Epoch: {epoch}")
            # print(f"Initial weights: {self.weights}")
            
            # Perceptron trick
            for i in range(len(point)):
                self.weights[i] = self.weights[i] + learning_rate * (point_label - prediction) * point[0]
            
            self.weights[-1] = self.weights[-1] + learning_rate * (point_label - prediction)
            
            # print(f"Ajusted weights: {self.weights}")
    
    # TODO: Mirar si este es el error real
    def error(self):
        error = 0.0
        
        for point in self.data:
            for i in range(len(point)):
                error += self.weights[i] * point[i]
                
            error += self.weights[-1] # error = |w1*x1 + w2*x2 + ... + wn*xn + b|
        
	# Average error of all points
        return abs(error) / len(self.data)

def main():    
    data = [
        [1, 0],
        [0, 2],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 2],
        [2, 3],
        [3, 2],
    ]
    
    data_labels = [False, False, False, False, True, True, True, True]
    
    # data = [[1, 1]]
    # data_labels = [False]
    
    perceptron = Perceptron(data=data, data_labels=data_labels)
    # perceptron.set_weights([2, 3, -4])
    
    initial_pred = perceptron.predict()
    initial_error = perceptron.error()
    
    # Ajust the perceptron
    epochs = 1000
    learning_rate = 0.01
    
    print(f"Ajusting the perceptron for {epochs} epochs with a learning rate of {learning_rate}\n")
    init = time.time()
    perceptron.ajust(epochs=epochs, learning_rate=learning_rate)
    end = time.time()
    
    ajust_time = end - init
        
    new_pred = perceptron.predict()
    new_error = perceptron.error()
    
    print(f"Labels: {data_labels}\n")
    print(f"Initial prediction: {initial_pred}")
    print(f"Initial error: {initial_error}")
    print(f"New prediction: {new_pred}")
    print(f"New error: {new_error}\n")
    print(f"Ajust time: {ajust_time} seconds")
    
if __name__ == "__main__":
    main()
