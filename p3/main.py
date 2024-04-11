import time
from perceptron import Perceptron
from logistical_classification import LogisticalClassification

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
    
    data_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    
    # data = [[1, 1]]
    # data_labels = [0]
    
    perceptron = Perceptron(data, data_labels)
    # perceptron.set_weights([2, 3, -4])
    
    initial_weights = perceptron.get_weights()
    initial_pred = perceptron.predict()
    initial_error = perceptron.error()
    
    # Ajust the perceptron
    epochs = 1000
    learning_rate = 0.01
    
    print(f"Ajusting the perceptron for {epochs} epochs with a learning rate of {learning_rate}\n")
    start = time.time()
    perceptron.ajust(epochs=epochs, learning_rate=learning_rate)
    end = time.time()
    
    ajust_time = end - start
    
    ajusted_weights = perceptron.get_weights()
    ajusted_pred = perceptron.predict()
    ajusted_error = perceptron.error()
    
    print(f"Labels: {data_labels}\n")
    
    print(f"Initial weights: {initial_weights}")
    print(f"Initial prediction: {initial_pred}")
    print(f"Initial error: {initial_error}\n")
    
    print(f"Ajusted weights: {ajusted_weights}")
    print(f"Ajusted prediction: {ajusted_pred}")
    print(f"Ajusted error: {ajusted_error}\n")
    print(f"Ajust time: {ajust_time} seconds")
    
    logistical_classification = LogisticalClassification(data, data_labels)
    
    initial_weights = logistical_classification.get_weights()
    initial_pred = logistical_classification.predict()
    initial_error = logistical_classification.error()
    
    # Ajust the logistical_classification
    epochs = 1000
    learning_rate = 0.01
    
    print(f"Ajusting the logistical_classification for {epochs} epochs with a learning rate of {learning_rate}\n")
    start = time.time()
    logistical_classification.ajust(epochs=epochs, learning_rate=learning_rate)
    end = time.time()
    
    ajust_time = end - start
    
    ajusted_weights = logistical_classification.get_weights()
    ajusted_pred = logistical_classification.predict()
    ajusted_error = logistical_classification.error()
    
    print(f"Labels: {data_labels}\n")
    
    print(f"Initial weights: {initial_weights}")
    print(f"Initial prediction: {initial_pred}")
    print(f"Initial error: {initial_error}\n")
    
    print(f"Ajusted weights: {ajusted_weights}")
    print(f"Ajusted prediction: {ajusted_pred}")
    print(f"Ajusted error: {ajusted_error}\n")
    print(f"Ajust time: {ajust_time} seconds")
    
if __name__ == "__main__":
    main()
