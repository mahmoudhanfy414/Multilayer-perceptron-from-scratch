import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import mainFUN


def normlization(target):
    target_array = np.zeros([1, 3])
    if target == 0:
        target_array[0][0] = 1
    elif target == 1:
        target_array[0][1] = 1
    elif target == 2:
        target_array[0][2] = 1
    return target_array


def mlp(input_nodes, output_nodes, num_hidden_layers, sizes_of_hiddens, learning_rate, num_of_epochs,
        activation_function, use_bias):
    data = pd.read_excel("dataset/Dry_Bean_Dataset.xlsx")
    data['MinorAxisLength'].fillna(data['MinorAxisLength'].mean(), inplace=True)
    class_splits = {}

    # Iterate over each class
    for class_label in data['Class'].unique():
        # Select samples belonging to the current class
        class_data = data[data['Class'] == class_label]

        # Split the class data into train and test sets
        train_data, test_data = train_test_split(class_data, test_size=20, stratify=class_data['Class'],
                                                 random_state=42)

        # Store the train and test sets in the dictionary
        class_splits[class_label] = {'train': train_data, 'test': test_data}

    # Combine train and test sets for each class
    train_data_combined = pd.concat([class_splits[class_label]['train'] for class_label in class_splits.keys()])
    test_data_combined = pd.concat([class_splits[class_label]['test'] for class_label in class_splits.keys()])

    # Shuffle the combined data
    train_data_combined = train_data_combined.sample(frac=1, random_state=42)
    test_data_combined = test_data_combined.sample(frac=1, random_state=42)

    x_train = train_data_combined.iloc[:, :-1]
    y_train = train_data_combined.iloc[:, -1]
    x_test = test_data_combined.iloc[:, :-1]
    y_test = test_data_combined.iloc[:, -1]

    min_max_scaler = MinMaxScaler()
    scaled_x_train = min_max_scaler.fit_transform(x_train)
    scaled_x_test = min_max_scaler.transform(x_test)

    label_encoder = LabelEncoder()
    # Fit and transform the label encoder on the training data
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Transform the test data using the same encoder
    y_test_encoded = label_encoder.transform(y_test)

    neural_network = mainFUN.NeuralNetwork(input_nodes=input_nodes, output_nodes=output_nodes,
                                           num_hidden_layer=num_hidden_layers, hidden_layer_sizes=sizes_of_hiddens,
                                           learning_rate=learning_rate, activation_function=activation_function,
                                           use_bias=use_bias)

    # Training the Neural Network
    for epoch in range(num_of_epochs):
        for inputs, targets in zip(scaled_x_train, y_train_encoded):
            neural_network.train(inputs, normlization(targets))

    correct_predictions = 0
    for train_input, true_label in zip(scaled_x_train, y_train_encoded):
        predictions = neural_network.transform(train_input)
        predicted_label = np.argmax(predictions[-1])
        correct_predictions += (predicted_label == true_label)

    accuracy = correct_predictions / len(y_train_encoded)
    train_accuracy = round(accuracy *100, 2)

    print(f"Accuracy on the train set: {accuracy * 100:.2f}%")

    # Testing the Neural Network
    correct_predictions = 0
    for test_input, true_label in zip(scaled_x_test, y_test_encoded):
        predictions = neural_network.transform(test_input)
        predicted_label = np.argmax(predictions[-1])
        correct_predictions += (predicted_label == true_label)

    accuracy = correct_predictions / len(y_test_encoded)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

    test_accuracy = round(accuracy*100, 2)

    return train_accuracy, test_accuracy
