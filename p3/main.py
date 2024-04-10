import time
from perceptron import Perceptron

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
    
if __name__ == "__main__":
    main()
