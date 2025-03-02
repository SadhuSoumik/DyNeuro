# DyNet: A Simple Neural Network Implementation in C

DyNet is a lightweight neural network implementation in C, designed for educational purposes and small-scale machine learning tasks. It includes basic matrix operations, activation functions, and an Adam optimizer for training. The project also demonstrates dynamic network growth by adding neurons or layers during training if the loss improvement stalls.

## Features

- **Matrix Operations**: Basic matrix operations like addition, subtraction, multiplication, and transposition.
- **Activation Functions**: Sigmoid activation function and its derivative.
- **Adam Optimizer**: Implements the Adam optimization algorithm for training neural networks.
- **Dynamic Network Growth**: Automatically adds neurons or layers during training if the loss improvement is minimal.
- **MNIST-like Data Generation**: Simulates MNIST-like data for training and testing.

## Language

- **C**: The project is written in C, making it lightweight and efficient.

## Libraries

The project uses the following standard C libraries:

- `<stdio.h>`: For input/output operations.
- `<stdlib.h>`: For memory allocation and random number generation.
- `<math.h>`: For mathematical functions like `exp`, `log`, and `sqrt`.
- `<time.h>`: For seeding the random number generator.
- `<string.h>`: For memory operations like `memcpy`.

## How to Run the Project

### Prerequisites

- A C compiler (e.g., `gcc`).
- Basic knowledge of compiling and running C programs.

### Steps to Run

1. **Clone the Repository** (if applicable):
   ```bash
   git clone https://github.com/yourusername/DyNet.git
   cd DyNet
   ```

2. **Compile the Code**:
   Use `gcc` to compile the C code:
   ```bash
   gcc -o DyNet DyNet.c -lm
   ```
   The `-lm` flag links the math library, which is required for functions like `exp` and `sqrt`.

3. **Run the Executable**:
   After compiling, run the generated executable:
   ```bash
   ./DyNet
   ```

4. **Observe the Output**:
   The program will:
   - Generate simulated MNIST-like data.
   - Train a neural network with the specified architecture.
   - Test the trained network on a few samples and print the predicted vs. actual labels.

### Expected Output

The program will print the training loss at regular intervals (every 100 epochs) and the final test results. For example:

```
Epoch 0: Loss = 2.302585
Epoch 100: Loss = 0.123456
...
Epoch 999: Loss = 0.012345

Testing the trained network:
Sample 0: Predicted = 7, Target = 7
Sample 1: Predicted = 2, Target = 2
Sample 2: Predicted = 1, Target = 1
Sample 3: Predicted = 0, Target = 0
Sample 4: Predicted = 4, Target = 4
```

## Code Structure

- **Matrix Operations**: The `DMatrix` struct and related functions handle matrix operations like creation, multiplication, addition, and subtraction.
- **Activation Functions**: The `sigmoid` and `sigmoid_derivative` functions are used for activation during forward and backward propagation.
- **Adam Optimizer**: The `Adam` struct and related functions implement the Adam optimization algorithm.
- **Neural Network**: The `NeuralNetwork` struct and related functions handle the network's forward pass, backpropagation, and dynamic growth.
- **Data Generation**: The `generate_mnist_like_data` function simulates MNIST-like data for training and testing.

## Customization

- **Network Architecture**: You can modify the network architecture by changing the `sizes_arr` array in the `main` function.
- **Dynamic Growth**: You can adjust the maximum number of neurons and layers by modifying the `max_hidden_neurons` and `max_hidden_layers` parameters in the `NeuralNetwork_new` function.
- **Training Parameters**: You can change the learning rate, number of epochs, and other training parameters in the `main` function.

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

---

Enjoy experimenting with DyNet! 🚀