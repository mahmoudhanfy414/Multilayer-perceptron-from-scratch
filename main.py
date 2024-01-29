import sys
import tkinter as tk
from GUI.Utils.entry_validation import validate_num
from GUI.custom_button import CustomButton
from GUI.custom_label import CustomLabel
from GUI.labeled_entry import LabeledEntry
from mainFUN import NeuralNetwork
from Test import mlp
from tkinter import messagebox

sys.path.append("../")

root = tk.Tk()
root.title("Task 2")
root.geometry("1400x800")

task_label = CustomLabel(root, text="Task 2")
task_label.pack()


def checkbox_clicked():
    if checkbox_var.get() == 1:
        print("Yes")
        return True
    else:
        print("No")
        return False


def get_values():
    try:
        hidden_layers = int(hidden_layers_entry.get())
        number_of_neurons = number_of_neurons_entry.get()
        learning_rate = float(learning_rate_entry.get())
        number_of_epochs = int(epochs_entry.get())
        print(hidden_layers, learning_rate, number_of_neurons, number_of_epochs)
    except Exception as e:
        messagebox.showerror("Error", "Make sure all the values are filled.")
        return

    return hidden_layers, learning_rate, number_of_neurons, number_of_epochs


def get_selected_option():
    selected = radio_button_option.get()
    return selected


def send_to_models():
    use_bias = checkbox_clicked()
    hidden_layers, learning_rate, number_of_neurons, number_of_epochs = get_values()

    model_type = get_selected_option()
    if model_type == 1:
        activation_function = "sigmoid"
    elif model_type == 2:
        activation_function = "tanh"

    number_of_neurons = [int(num) for num in number_of_neurons.split(",")]

    if len(number_of_neurons) != hidden_layers:
        tk.messagebox.showerror(
            "Error",
            "Number of neurons should be equal to number of hidden layers.",
        )
        return

    train_accuracy, test_accuracy = mlp(
        input_nodes=5,
        output_nodes=3,
        num_hidden_layers=hidden_layers,
        sizes_of_hiddens=number_of_neurons,
        learning_rate=learning_rate,
        num_of_epochs=number_of_epochs,
        activation_function=activation_function,
        use_bias=use_bias,
    )

    train_accuracy_label.config(text=train_accuracy)
    test_accuracy_label.config(text=test_accuracy)


radio_button_frame = tk.Frame(root)
radio_button_frame.pack(side=tk.TOP, pady=25)

radio_button_option = tk.IntVar()

sigmoid_button = tk.Radiobutton(
    radio_button_frame,
    text="Sigmoid",
    font=("Times New Roman", 18),
    variable=radio_button_option,
    value=1,
)
sigmoid_button.grid(row=0, column=0, padx=40)

hyperbolic_tangent_sigmoid_button = tk.Radiobutton(
    radio_button_frame,
    text="Hyperbolic Tangent sigmoid",
    font=("Times New Roman", 18),
    variable=radio_button_option,
    value=2,
)
hyperbolic_tangent_sigmoid_button.grid(row=0, column=1, padx=40)

checkbox_var = tk.BooleanVar()
checkbox = tk.Checkbutton(
    radio_button_frame,
    text="Add Bias?",
    font=("Times New Roman", 18),
    variable=checkbox_var,
)
checkbox.grid(row=0, column=3, padx=200)

checkbox.config(command=checkbox_clicked)

entry_frame = tk.Frame(root)
entry_frame.pack(padx=10, side=tk.TOP)

hidden_layers_entry = LabeledEntry(entry_frame, "Number of hidden layers")
hidden_layers_entry.pack(side=tk.LEFT, pady=10)

number_of_neurons_entry = LabeledEntry(entry_frame, "Number of neurons")
number_of_neurons_entry.pack(side=tk.LEFT, pady=10)

learning_rate_entry = LabeledEntry(entry_frame, "Learning rate")
learning_rate_entry.pack(side=tk.LEFT, pady=10)

epochs_entry = LabeledEntry(entry_frame, "Number of epochs")
epochs_entry.pack(side=tk.LEFT, pady=10)

train_button = CustomButton(root, command=send_to_models, text="Train")
train_button.pack(side=tk.TOP, pady=10)

train_accuracy = CustomLabel(root, text="Train Accuracy")
train_accuracy.pack(side=tk.TOP, pady=10)
train_accuracy_label = CustomLabel(root, text="")
train_accuracy_label.pack(side=tk.TOP, pady=10)

test_accuracy = CustomLabel(root, text="Test Accuracy")
test_accuracy.pack(side=tk.TOP, pady=50)
test_accuracy_label = CustomLabel(root, text="")
test_accuracy_label.pack(side=tk.TOP, pady=10)


root.mainloop()
