import time
from perceptron import Perceptron
from logistic_regression import LogisticRegression

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
    
    print(f"Labels: {data_labels}\n")
    
    perceptron = Perceptron(data, data_labels)
    
    print(f"Initial weights: {perceptron.get_weights()}")
    print(f"Initial prediction: {perceptron.predict()}")
    print(f"Initial error: {perceptron.error()}\n")
    
    # Ajust the perceptron
    epochs = 1000
    learning_rate = 0.01
    
    print(f"Ajusting perceptron for {epochs} epochs with a learning rate of {learning_rate}\n")
    
    start = time.time()
    perceptron.ajust(epochs=epochs, learning_rate=learning_rate)
    end = time.time()
    
    ajust_time = end - start
    
    print(f"Ajusted weights: {perceptron.get_weights()}")
    print(f"Ajusted prediction: {perceptron.predict()}")
    print(f"Ajusted error: {perceptron.error()}\n")
    
    print(f"Perceptron time: {ajust_time} seconds\n")
    
    logistical_regression = LogisticRegression(data, data_labels)
    
    # Ajust the logistical_regression
    epochs = 10000
    learning_rate = 0.01
    threshold = 0.5
    
    print(f"Initial weights: {logistical_regression.get_weights()}")
    print(f"Initial prediction: {logistical_regression.predict(threshold=threshold, softmax=True)}")
    print(f"Initial error: {logistical_regression.error()}\n")
    
    print(f"Ajusting logistical classification for {epochs} epochs with a learning rate of {learning_rate} with a threshold of {threshold}\n")
    
    start = time.time()
    logistical_regression.ajust(epochs=epochs, learning_rate=learning_rate)
    end = time.time()
    
    ajust_time = end - start
    
    print(f"Ajusted weights: {logistical_regression.get_weights()}")
    print(f"Ajusted prediction: {logistical_regression.predict(threshold=threshold, softmax=True)}")
    print(f"Ajusted error: {logistical_regression.error()}\n")
    
    print(f"Logistic Regression time: {ajust_time} seconds")
    
if __name__ == "__main__":
    main()
