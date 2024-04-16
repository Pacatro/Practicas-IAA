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
    
    grade = len(data[0])
    perceptron = Perceptron(grade=grade)
    
    print(f"Initial weights: {perceptron.get_weights()}")
    print(f"Initial prediction: {perceptron.predict(data=data)}")
    print(f"Initial prediction (values): {perceptron.predict(data=data, values=True)}")
    print(f"Initial error: {perceptron.error(data=data)}\n")
    
    # Ajust the perceptron
    epochs = 1000
    learning_rate = 0.01
    
    print(f"Ajusting perceptron for {epochs} epochs with a learning rate of {learning_rate}\n")
    
    start = time.time()
    perceptron.ajust(epochs=epochs, learning_rate=learning_rate, data=data, data_labels=data_labels)
    end = time.time()
    
    ajust_time = end - start
    
    print(f"Adjusted weights: {perceptron.get_weights()}")
    print(f"Adjusted prediction: {perceptron.predict(data=data)}")
    print(f"Ajusted prediction (values): {perceptron.predict(data=data, values=True)}")
    print(f"Adjusted error: {perceptron.error(data=data)}\n")
    
    print(f"Adjusted Perceptron time: {ajust_time} seconds\n")
    
    logistical_regression = LogisticRegression(grade=grade)
    
    # Ajust the logistical_regression
    epochs = 10000
    learning_rate = 0.01
    threshold = 0.5
    
    print(f"Initial weights: {logistical_regression.get_weights()}")
    print(f"Initial prediction: {logistical_regression.predict(data=data, threshold=threshold)}")
    print(f"Initial prediction (prob): {logistical_regression.predict(data=data, threshold=threshold, prob=True)}")
    print(f"Initial error: {logistical_regression.error(data=data, data_labels=data_labels)}\n")
    
    print(f"Ajusting logistical regression for {epochs} epochs with a learning rate of {learning_rate} with a threshold of {threshold}\n")
    
    start = time.time()
    logistical_regression.ajust(epochs=epochs, learning_rate=learning_rate, data=data, data_labels=data_labels)
    end = time.time()
    
    ajust_time = end - start
    
    print(f"Adjusted weights: {logistical_regression.get_weights()}")
    print(f"Adjusted prediction: {logistical_regression.predict(data=data, threshold=threshold)}")
    print(f"Adjusted prediction (prob): {logistical_regression.predict(data=data, threshold=threshold, prob=True)}")
    print(f"Adjusted error: {logistical_regression.error(data=data, data_labels=data_labels)}\n")
    
    print(f"Adjusted Logistic Regression time: {ajust_time} seconds")
    
if __name__ == "__main__":
    main()
