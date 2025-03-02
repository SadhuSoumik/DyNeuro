#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ------------------- Matrix Data Structure and Utilities -------------------
typedef struct {
    size_t nrows;
    size_t ncols;
    double* data;
} DMatrix;

// Macro for indexing
#define IDX(m, i, j) ((m)->data[(i) * (m)->ncols + (j)])

// Creates a DMatrix with uninitialized data (allocated and zeroed)
DMatrix dmatrix_zeros(size_t rows, size_t cols) {
    DMatrix m;
    m.nrows = rows;
    m.ncols = cols;
    m.data = (double*)calloc(rows * cols, sizeof(double));
    return m;
}

// Creates a DMatrix with random values in the range [lower, upper)
DMatrix dmatrix_random(size_t rows, size_t cols, double lower, double upper) {
    DMatrix m;
    m.nrows = rows;
    m.ncols = cols;
    m.data = (double*)malloc(rows * cols * sizeof(double));
    for (size_t i = 0; i < rows * cols; i++) {
        double r = ((double)rand() / (double)RAND_MAX);
        m.data[i] = lower + r * (upper - lower);
    }
    return m;
}

// Creates a DMatrix from a given row slice array.
DMatrix dmatrix_from_row_slice(size_t rows, size_t cols, const double* slice) {
    DMatrix m;
    m.nrows = rows;
    m.ncols = cols;
    m.data = (double*)malloc(rows * cols * sizeof(double));
    memcpy(m.data, slice, rows * cols * sizeof(double));
    return m;
}

// Clones a matrix.
DMatrix dmatrix_clone(const DMatrix* m) {
    DMatrix copy;
    copy.nrows = m->nrows;
    copy.ncols = m->ncols;
    copy.data = (double*)malloc(m->nrows * m->ncols * sizeof(double));
    memcpy(copy.data, m->data, m->nrows * m->ncols * sizeof(double));
    return copy;
}

// Frees the memory associated with a matrix.
void dmatrix_free(DMatrix* m) {
    if(m->data) {
        free(m->data);
        m->data = NULL;
    }
}

// Matrix multiplication: returns a new matrix = a * b. (Assumes dimensions are compatible.)
DMatrix dmatrix_multiply(const DMatrix* a, const DMatrix* b) {
    if (a->ncols != b->nrows) {
        fprintf(stderr, "Matrix dimensions mismatch in multiplication.\n");
        exit(EXIT_FAILURE);
    }
    DMatrix result = dmatrix_zeros(a->nrows, b->ncols);
    for (size_t i = 0; i < a->nrows; i++) {
        for (size_t j = 0; j < b->ncols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < a->ncols; k++) {
                sum += IDX(a, i, k) * IDX(b, k, j);
            }
            IDX(&result, i, j) = sum;
        }
    }
    return result;
}

// Matrix addition: returns new matrix = a + b.
DMatrix dmatrix_add(const DMatrix* a, const DMatrix* b) {
    if (a->nrows != b->nrows || a->ncols != b->ncols) {
        fprintf(stderr, "Matrix dimensions mismatch in addition.\n");
        exit(EXIT_FAILURE);
    }
    DMatrix result = dmatrix_zeros(a->nrows, a->ncols);
    for (size_t i = 0; i < a->nrows * a->ncols; i++) {
        result.data[i] = a->data[i] + b->data[i];
    }
    return result;
}

// Matrix subtraction: returns new matrix = a - b.
DMatrix dmatrix_subtract(const DMatrix* a, const DMatrix* b) {
    if (a->nrows != b->nrows || a->ncols != b->ncols) {
        fprintf(stderr, "Matrix dimensions mismatch in subtraction.\n");
        exit(EXIT_FAILURE);
    }
    DMatrix result = dmatrix_zeros(a->nrows, a->ncols);
    for (size_t i = 0; i < a->nrows * a->ncols; i++) {
        result.data[i] = a->data[i] - b->data[i];
    }
    return result;
}

// In-place subtraction: m -= sub.
void dmatrix_subtract_inplace(DMatrix* m, const DMatrix* sub) {
    if (m->nrows != sub->nrows || m->ncols != sub->ncols) {
        fprintf(stderr, "Matrix dimensions mismatch in in-place subtraction.\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < m->nrows * m->ncols; i++) {
        m->data[i] -= sub->data[i];
    }
}

// Scalar multiplication: returns new matrix = m * scalar.
DMatrix dmatrix_scalar_multiply(const DMatrix* m, double scalar) {
    DMatrix result = dmatrix_zeros(m->nrows, m->ncols);
    for (size_t i = 0; i < m->nrows * m->ncols; i++) {
        result.data[i] = m->data[i] * scalar;
    }
    return result;
}

// Element-wise multiplication: returns new matrix = a o b.
DMatrix dmatrix_component_mul(const DMatrix* a, const DMatrix* b) {
    if (a->nrows != b->nrows || a->ncols != b->ncols) {
        fprintf(stderr, "Matrix dimensions mismatch in component-wise multiplication.\n");
        exit(EXIT_FAILURE);
    }
    DMatrix result = dmatrix_zeros(a->nrows, a->ncols);
    for (size_t i = 0; i < a->nrows * a->ncols; i++) {
        result.data[i] = a->data[i] * b->data[i];
    }
    return result;
}

// Transposes a matrix.
DMatrix dmatrix_transpose(const DMatrix* m) {
    DMatrix t = dmatrix_zeros(m->ncols, m->nrows);
    for (size_t i = 0; i < m->nrows; i++) {
        for (size_t j = 0; j < m->ncols; j++) {
            IDX(&t, j, i) = IDX(m, i, j);
        }
    }
    return t;
}

// Applies a function element-wise to a matrix: returns new matrix.
DMatrix dmatrix_map(const DMatrix* m, double (*func)(double)) {
    DMatrix result = dmatrix_zeros(m->nrows, m->ncols);
    for (size_t i = 0; i < m->nrows * m->ncols; i++) {
        result.data[i] = func(m->data[i]);
    }
    return result;
}

// Applies a function element-wise to two matrices: returns new matrix.
// Assumes a and b are the same dimensions.
DMatrix dmatrix_zip_map(const DMatrix* a, const DMatrix* b, double (*func)(double, double)) {
    if (a->nrows != b->nrows || a->ncols != b->ncols) {
        fprintf(stderr, "Matrix dimensions mismatch in zip map.\n");
        exit(EXIT_FAILURE);
    }
    DMatrix result = dmatrix_zeros(a->nrows, a->ncols);
    for (size_t i = 0; i < a->nrows * a->ncols; i++) {
        result.data[i] = func(a->data[i], b->data[i]);
    }
    return result;
}

// Copies data from source matrix into destination matrix slice starting at (row_offset, col_offset).
// Assumes destination matrix is already allocated and dimensions are compatible.
void dmatrix_copy_into(DMatrix* dest, size_t row_offset, size_t col_offset, const DMatrix* src) {
    if (row_offset + src->nrows > dest->nrows || col_offset + src->ncols > dest->ncols) {
        fprintf(stderr, "Submatrix copy dimensions exceed destination matrix.\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < src->nrows; i++) {
        for (size_t j = 0; j < src->ncols; j++) {
            IDX(dest, row_offset + i, col_offset + j) = IDX(src, i, j);
        }
    }
}

// ------------------- Activation Functions -------------------
// Sigmoid activation function: 1 / (1 + exp(-x))
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function.
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Helper for zip_map in Adam update: computes learning_rate * m / (sqrt(v) + epsilon)
typedef struct {
    double learning_rate;
    double epsilon;
} AdamUpdateParams;

double adam_update_func(double m_val, double v_val, void* params_void) {
    AdamUpdateParams* params = (AdamUpdateParams*)params_void;
    return params->learning_rate * m_val / (sqrt(v_val) + params->epsilon);
}

// A function suitable for dmatrix_zip_map that incorporates external parameters.
double zip_adam(double m_val, double v_val) {
    // This function should not be used directly.
    return 0.0;
}

// ------------------- Layer Structure -------------------
typedef struct {
    DMatrix weights; // dimensions: input_size x output_size
    DMatrix biases;  // dimensions: 1 x output_size
} Layer;

// Constructs a new layer with weights initialized randomly (range -0.1 to 0.1)
// and biases initialized to zero.
Layer Layer_new(size_t input_size, size_t output_size) {
    Layer l;
    l.weights = dmatrix_random(input_size, output_size, -0.1, 0.1);
    l.biases = dmatrix_zeros(1, output_size);
    return l;
}

// Frees a layer's matrices.
void Layer_free(Layer* layer) {
    dmatrix_free(&layer->weights);
    dmatrix_free(&layer->biases);
}

// ------------------- Adam Optimizer -------------------
typedef struct {
    double learning_rate;
    double beta1;   // Decay rate for the first moment (momentum)
    double beta2;   // Decay rate for the second moment (RMSProp)
    double epsilon; // Small constant to avoid division by zero
    DMatrix* m_weights; // Array of first moment estimates for weights
    DMatrix* v_weights; // Array of second moment estimates for weights
    DMatrix* m_biases;  // Array of first moment estimates for biases
    DMatrix* v_biases;  // Array of second moment estimates for biases
    double t;       // Time step (number of updates)
    size_t count;   // Number of layers (size of moment arrays)
} Adam;

// Creates a new Adam optimizer. The sizes array should have length sz_len.
Adam Adam_new(double learning_rate, const size_t* sizes, size_t sz_len) {
    Adam opt;
    opt.learning_rate = learning_rate;
    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.epsilon = 1e-8;
    opt.t = 0.0;
    opt.count = sz_len - 1;
    opt.m_weights = (DMatrix*)malloc(opt.count * sizeof(DMatrix));
    opt.v_weights = (DMatrix*)malloc(opt.count * sizeof(DMatrix));
    opt.m_biases  = (DMatrix*)malloc(opt.count * sizeof(DMatrix));
    opt.v_biases  = (DMatrix*)malloc(opt.count * sizeof(DMatrix));
    for (size_t i = 0; i < opt.count; i++) {
        opt.m_weights[i] = dmatrix_zeros(sizes[i], sizes[i+1]);
        opt.v_weights[i] = dmatrix_zeros(sizes[i], sizes[i+1]);
        opt.m_biases[i]  = dmatrix_zeros(1, sizes[i+1]);
        opt.v_biases[i]  = dmatrix_zeros(1, sizes[i+1]);
    }
    return opt;
}

// Updates a layer's parameters using Adam optimization.
// layer_idx is used to access the proper moment estimates.
void Adam_update(Adam* opt, Layer* layer, const DMatrix* weight_gradients, const DMatrix* bias_gradients, size_t layer_idx) {
    opt->t += 1.0;
    
    // Compute biased first moment estimates.
    DMatrix m_w_old = opt->m_weights[layer_idx];
    DMatrix m_w_scaled = dmatrix_scalar_multiply(&m_w_old, opt->beta1);
    DMatrix w_grad_scaled = dmatrix_scalar_multiply(weight_gradients, 1.0 - opt->beta1);
    DMatrix m_w = dmatrix_add(&m_w_scaled, &w_grad_scaled);
    dmatrix_free(&m_w_old);
    dmatrix_free(&m_w_scaled);
    dmatrix_free(&w_grad_scaled);
    
    DMatrix m_b_old = opt->m_biases[layer_idx];
    DMatrix m_b_scaled = dmatrix_scalar_multiply(&m_b_old, opt->beta1);
    DMatrix b_grad_scaled = dmatrix_scalar_multiply(bias_gradients, 1.0 - opt->beta1);
    DMatrix m_b = dmatrix_add(&m_b_scaled, &b_grad_scaled);
    dmatrix_free(&m_b_old);
    dmatrix_free(&m_b_scaled);
    dmatrix_free(&b_grad_scaled);
    
    // Compute biased second moment estimates.
    DMatrix weight_gradients_sq = dmatrix_component_mul(weight_gradients, weight_gradients);
    DMatrix v_w_old = opt->v_weights[layer_idx];
    DMatrix v_w_scaled = dmatrix_scalar_multiply(&v_w_old, opt->beta2);
    DMatrix w_grad_sq_scaled = dmatrix_scalar_multiply(&weight_gradients_sq, 1.0 - opt->beta2);
    DMatrix v_w = dmatrix_add(&v_w_scaled, &w_grad_sq_scaled);
    dmatrix_free(&v_w_old);
    dmatrix_free(&v_w_scaled);
    dmatrix_free(&w_grad_sq_scaled);
    dmatrix_free(&weight_gradients_sq);
    
    DMatrix bias_gradients_sq = dmatrix_component_mul(bias_gradients, bias_gradients);
    DMatrix v_b_old = opt->v_biases[layer_idx];
    DMatrix v_b_scaled = dmatrix_scalar_multiply(&v_b_old, opt->beta2);
    DMatrix b_grad_sq_scaled = dmatrix_scalar_multiply(&bias_gradients_sq, 1.0 - opt->beta2);
    DMatrix v_b = dmatrix_add(&v_b_scaled, &b_grad_sq_scaled);
    dmatrix_free(&v_b_old);
    dmatrix_free(&v_b_scaled);
    dmatrix_free(&b_grad_sq_scaled);
    dmatrix_free(&bias_gradients_sq);
    
    // Compute bias-corrected moment estimates.
    double beta1_t = pow(opt->beta1, opt->t);
    double beta2_t = pow(opt->beta2, opt->t);
    
    DMatrix m_w_hat = dmatrix_scalar_multiply(&m_w, 1.0 / (1.0 - beta1_t));
    DMatrix v_w_hat = dmatrix_scalar_multiply(&v_w, 1.0 / (1.0 - beta2_t));
    DMatrix m_b_hat = dmatrix_scalar_multiply(&m_b, 1.0 / (1.0 - beta1_t));
    DMatrix v_b_hat = dmatrix_scalar_multiply(&v_b, 1.0 / (1.0 - beta2_t));
    
    // Compute parameter updates element-wise.
    // For weights.
    DMatrix weight_updates = dmatrix_zeros(m_w_hat.nrows, m_w_hat.ncols);
    for (size_t i = 0; i < m_w_hat.nrows * m_w_hat.ncols; i++) {
        weight_updates.data[i] = opt->learning_rate * m_w_hat.data[i] / (sqrt(v_w_hat.data[i]) + opt->epsilon);
    }
    // Update layer weights: layer->weights = layer->weights - weight_updates.
    dmatrix_subtract_inplace(&layer->weights, &weight_updates);
    
    // For biases.
    DMatrix bias_updates = dmatrix_zeros(m_b_hat.nrows, m_b_hat.ncols);
    for (size_t i = 0; i < m_b_hat.nrows * m_b_hat.ncols; i++) {
        bias_updates.data[i] = opt->learning_rate * m_b_hat.data[i] / (sqrt(v_b_hat.data[i]) + opt->epsilon);
    }
    dmatrix_subtract_inplace(&layer->biases, &bias_updates);
    
    // Store updated moment estimates.
    opt->m_weights[layer_idx] = m_w;
    opt->v_weights[layer_idx] = v_w;
    opt->m_biases[layer_idx]  = m_b;
    opt->v_biases[layer_idx]  = v_b;
    
    // Free temporary matrices.
    dmatrix_free(&m_w_hat);
    dmatrix_free(&v_w_hat);
    dmatrix_free(&m_b_hat);
    dmatrix_free(&v_b_hat);
    dmatrix_free(&weight_updates);
    dmatrix_free(&bias_updates);
}

// Resizes the moment estimates when the network grows.
void Adam_resize(Adam* opt, const size_t* sizes, size_t sz_len) {
    // Free previous moment arrays.
    for (size_t i = 0; i < opt->count; i++) {
        dmatrix_free(&opt->m_weights[i]);
        dmatrix_free(&opt->v_weights[i]);
        dmatrix_free(&opt->m_biases[i]);
        dmatrix_free(&opt->v_biases[i]);
    }
    free(opt->m_weights);
    free(opt->v_weights);
    free(opt->m_biases);
    free(opt->v_biases);
    
    opt->count = sz_len - 1;
    opt->m_weights = (DMatrix*)malloc(opt->count * sizeof(DMatrix));
    opt->v_weights = (DMatrix*)malloc(opt->count * sizeof(DMatrix));
    opt->m_biases  = (DMatrix*)malloc(opt->count * sizeof(DMatrix));
    opt->v_biases  = (DMatrix*)malloc(opt->count * sizeof(DMatrix));
    for (size_t i = 0; i < opt->count; i++) {
        opt->m_weights[i] = dmatrix_zeros(sizes[i], sizes[i+1]);
        opt->v_weights[i] = dmatrix_zeros(sizes[i], sizes[i+1]);
        opt->m_biases[i]  = dmatrix_zeros(1, sizes[i+1]);
        opt->v_biases[i]  = dmatrix_zeros(1, sizes[i+1]);
    }
}

// Frees the memory allocated inside the Adam optimizer.
void Adam_free(Adam* opt) {
    for (size_t i = 0; i < opt->count; i++) {
        dmatrix_free(&opt->m_weights[i]);
        dmatrix_free(&opt->v_weights[i]);
        dmatrix_free(&opt->m_biases[i]);
        dmatrix_free(&opt->v_biases[i]);
    }
    free(opt->m_weights);
    free(opt->v_weights);
    free(opt->m_biases);
    free(opt->v_biases);
}

// ------------------- Neural Network -------------------
typedef struct {
    Layer* layers;
    size_t layer_count; // number of layers
    size_t max_hidden_neurons;
    size_t max_hidden_layers;
} NeuralNetwork;

// Constructs a new neural network given an array of layer sizes (length sz_len).
NeuralNetwork NeuralNetwork_new(const size_t* sizes, size_t sz_len, size_t max_hidden_neurons, size_t max_hidden_layers) {
    NeuralNetwork nn;
    nn.layer_count = sz_len - 1;
    nn.layers = (Layer*)malloc(nn.layer_count * sizeof(Layer));
    for (size_t i = 0; i < nn.layer_count; i++) {
        nn.layers[i] = Layer_new(sizes[i], sizes[i+1]);
    }
    nn.max_hidden_neurons = max_hidden_neurons;
    nn.max_hidden_layers = max_hidden_layers;
    return nn;
}

// Frees the memory associated with the neural network.
void NeuralNetwork_free(NeuralNetwork* nn) {
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer_free(&nn->layers[i]);
    }
    free(nn->layers);
}

// Performs a forward pass through the network.
// Returns an array of DMatrix activations (one per layer, including the input).
// The caller is responsible for freeing each DMatrix and the activations array.
DMatrix* NeuralNetwork_forward(const NeuralNetwork* nn, const DMatrix input, size_t* act_count) {
    size_t count = nn->layer_count + 1;
    DMatrix* activations = (DMatrix*)malloc(count * sizeof(DMatrix));
    activations[0] = dmatrix_clone(&input);
    DMatrix current = dmatrix_clone(&input);
    for (size_t i = 0; i < nn->layer_count; i++) {
        // Compute the linear transformation: current * weights + biases.
        DMatrix prod = dmatrix_multiply(&current, &nn->layers[i].weights);
        DMatrix sum = dmatrix_add(&prod, &nn->layers[i].biases);
        dmatrix_free(&prod);
        dmatrix_free(&current);
        // Apply the sigmoid activation.
        current = dmatrix_map(&sum, sigmoid);
        dmatrix_free(&sum);
        activations[i+1] = dmatrix_clone(&current);
    }
    *act_count = count;
    dmatrix_free(&current);
    return activations;
}

// Dynamically adds a neuron to a hidden layer (specified by layer index).
// The current layer’s weight matrix is expanded with one additional column,
// and the subsequent layer (if any) is updated by adding a new row.
void NeuralNetwork_add_neuron(NeuralNetwork* nn, size_t layer_idx) {
    if (layer_idx >= nn->layer_count) {
        return;
    }
    size_t current_neurons = nn->layers[layer_idx].weights.ncols;
    if (current_neurons >= nn->max_hidden_neurons) {
        printf("Max hidden neurons (%zu) reached in layer %zu, skipping.\n", nn->max_hidden_neurons, layer_idx);
        return;
    }
    // Initialize random generator will use rand()
    size_t input_size = nn->layers[layer_idx].weights.nrows;
    size_t new_neurons = current_neurons + 1;
    
    // Expand current layer's weights by adding a new column.
    DMatrix new_weights = dmatrix_zeros(input_size, new_neurons);
    // Copy existing weights.
    for (size_t i = 0; i < input_size; i++) {
        for (size_t j = 0; j < current_neurons; j++) {
            IDX(&new_weights, i, j) = IDX(&nn->layers[layer_idx].weights, i, j);
        }
    }
    // Add new column with random initialization.
    for (size_t i = 0; i < input_size; i++) {
        IDX(&new_weights, i, current_neurons) = -0.1 + ((double)rand() / RAND_MAX) * 0.2;
    }
    dmatrix_free(&nn->layers[layer_idx].weights);
    nn->layers[layer_idx].weights = new_weights;
    
    // Expand biases by adding a new column.
    DMatrix new_biases = dmatrix_zeros(1, new_neurons);
    for (size_t j = 0; j < current_neurons; j++) {
        IDX(&new_biases, 0, j) = IDX(&nn->layers[layer_idx].biases, 0, j);
    }
    dmatrix_free(&nn->layers[layer_idx].biases);
    nn->layers[layer_idx].biases = new_biases;
    
    // If there is a subsequent layer, update its weights by adding a new row.
    if (layer_idx + 1 < nn->layer_count) {
        size_t output_size = nn->layers[layer_idx + 1].weights.ncols;
        size_t old_rows = nn->layers[layer_idx + 1].weights.nrows;
        size_t new_rows = old_rows + 1;
        DMatrix new_next_weights = dmatrix_zeros(new_rows, output_size);
        // Copy existing rows.
        for (size_t i = 0; i < old_rows; i++) {
            for (size_t j = 0; j < output_size; j++) {
                IDX(&new_next_weights, i, j) = IDX(&nn->layers[layer_idx + 1].weights, i, j);
            }
        }
        // Add new row with random initialization.
        for (size_t j = 0; j < output_size; j++) {
            IDX(&new_next_weights, old_rows, j) = -0.1 + ((double)rand() / RAND_MAX) * 0.2;
        }
        dmatrix_free(&nn->layers[layer_idx + 1].weights);
        nn->layers[layer_idx + 1].weights = new_next_weights;
    }
    printf("Added neuron to layer %zu. New neuron count: %zu\n", layer_idx, new_neurons);
}

// Dynamically adds a new hidden layer with the specified number of neurons.
// The new layer is inserted before the output layer.
void NeuralNetwork_add_layer(NeuralNetwork* nn, size_t neuron_count) {
    if (nn->layer_count - 1 >= nn->max_hidden_layers) {
        printf("Max hidden layers (%zu) reached, skipping.\n", nn->max_hidden_layers);
        return;
    }
    // Identify the index of the last hidden layer.
    size_t last_hidden_idx = nn->layer_count - 2;
    size_t input_size = nn->layers[last_hidden_idx].biases.ncols;
    size_t output_size = nn->layers[last_hidden_idx + 1].weights.ncols;
    
    // Create new hidden layer and new output layer.
    Layer new_layer = Layer_new(input_size, neuron_count);
    Layer new_output_layer = Layer_new(neuron_count, output_size);
    
    // Increase layers array size by 1.
    size_t new_layer_count = nn->layer_count + 1;
    Layer* new_layers = (Layer*)malloc(new_layer_count * sizeof(Layer));
    // Copy layers up to last_hidden_idx.
    for (size_t i = 0; i <= last_hidden_idx; i++) {
        new_layers[i] = nn->layers[i];
    }
    // Insert new hidden layer.
    new_layers[last_hidden_idx + 1] = new_layer;
    // Replace output layer with new_output_layer.
    new_layers[last_hidden_idx + 2] = new_output_layer;
    // Copy any remaining layers (if any) - in this network, typically there is only one output layer.
    for (size_t i = last_hidden_idx + 2; i < nn->layer_count; i++) {
        new_layers[i+1] = nn->layers[i];
    }
    free(nn->layers);
    nn->layers = new_layers;
    nn->layer_count = new_layer_count;
    
    printf("Added hidden layer with %zu neurons. Total layers: %zu\n", neuron_count, nn->layer_count);
}

// Trains the network using the Adam optimizer and cross-entropy loss.
// Also supports dynamic network growth if loss improvement stalls.
void NeuralNetwork_train(NeuralNetwork* nn, DMatrix* inputs, DMatrix* targets, size_t sample_count, size_t epochs, Adam* optimizer) {
    double prev_loss = INFINITY;
    size_t patience = 50;
    size_t patience_counter = 0;
    double loss_threshold = 0.001;
    
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        // Iterate over each training sample.
        for (size_t s = 0; s < sample_count; s++) {
            size_t act_count = 0;
            DMatrix* activations = NeuralNetwork_forward(nn, inputs[s], &act_count);
            DMatrix output = activations[act_count - 1];
            
            // Compute cross-entropy loss and gradient.
            DMatrix error = dmatrix_zeros(output.nrows, output.ncols);
            for (size_t i = 0; i < output.nrows * output.ncols; i++) {
                double p = output.data[i];
                if (p < 1e-8) p = 1e-8;
                if (p > 1.0 - 1e-8) p = 1.0 - 1e-8;
                double t = targets[s].data[i];
                total_loss -= t * log(p) + (1.0 - t) * log(1.0 - p);
                error.data[i] = p - t; // Derivative of cross-entropy loss w.r.t. output.
            }
            // Backpropagation.
            DMatrix delta = dmatrix_clone(&error);
            // Free error since delta is its clone.
            dmatrix_free(&error);
            for (ssize_t i = nn->layer_count - 1; i >= 0; i--) {
                // Compute gradients: weight_gradients = (prev_activation)^T * delta.
                DMatrix prev_activation = activations[i];
                DMatrix prev_activation_T = dmatrix_transpose(&prev_activation);
                DMatrix weight_gradients = dmatrix_multiply(&prev_activation_T, &delta);
                dmatrix_free(&prev_activation_T);
                // bias_gradients = delta.
                DMatrix bias_gradients = dmatrix_clone(&delta);
                // Update parameters for layer i using Adam.
                Adam_update(optimizer, &nn->layers[i], &weight_gradients, &bias_gradients, i);
                
                dmatrix_free(&weight_gradients);
                dmatrix_free(&bias_gradients);
                
                if (i > 0) {
                    // Propagate delta backwards.
                    DMatrix weights_T = dmatrix_transpose(&nn->layers[i].weights);
                    DMatrix delta_new = dmatrix_multiply(&delta, &weights_T);
                    dmatrix_free(&weights_T);
                    dmatrix_free(&delta);
                    // delta = delta_new o sigmoid_derivative(activations[i])
                    DMatrix sigmoid_deriv = dmatrix_map(&activations[i], sigmoid_derivative);
                    DMatrix delta_updated = dmatrix_component_mul(&delta_new, &sigmoid_deriv);
                    dmatrix_free(&delta_new);
                    dmatrix_free(&sigmoid_deriv);
                    delta = delta_updated;
                } else {
                    dmatrix_free(&delta);
                }
            }
            // Free activations.
            for (size_t i = 0; i < act_count; i++) {
                dmatrix_free(&activations[i]);
            }
            free(activations);
            total_loss /= (double)sample_count;
        }
        
        // Dynamic growth: if improvement is minimal, add neurons or layers.
        if (fabs(prev_loss - total_loss) < loss_threshold) {
            patience_counter++;
            if (patience_counter >= patience) {
                // Check if we can add neuron to the first hidden layer.
                if (nn->layers[0].biases.ncols < nn->max_hidden_neurons) {
                    NeuralNetwork_add_neuron(nn, 0);
                    // Recompute sizes vector.
                    size_t* sizes = (size_t*)malloc((nn->layer_count + 1) * sizeof(size_t));
                    for (size_t i = 0; i < nn->layer_count; i++) {
                        sizes[i] = nn->layers[i].weights.nrows;
                    }
                    sizes[nn->layer_count] = nn->layers[nn->layer_count - 1].biases.ncols;
                    Adam_resize(optimizer, sizes, nn->layer_count + 1);
                    free(sizes);
                } else if (nn->layer_count - 1 < nn->max_hidden_layers) {
                    NeuralNetwork_add_layer(nn, 64);
                    size_t* sizes = (size_t*)malloc((nn->layer_count + 1) * sizeof(size_t));
                    for (size_t i = 0; i < nn->layer_count; i++) {
                        sizes[i] = nn->layers[i].weights.nrows;
                    }
                    sizes[nn->layer_count] = nn->layers[nn->layer_count - 1].biases.ncols;
                    Adam_resize(optimizer, sizes, nn->layer_count + 1);
                    free(sizes);
                }
                patience_counter = 0;
            }
        } else {
            patience_counter = 0;
        }
        prev_loss = total_loss;
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            printf("Epoch %zu: Loss = %lf\n", epoch, total_loss);
        }
    }
}

// ------------------- MNIST-like Data Generation -------------------
// Generates simulated MNIST-like data.
// Each input is a flattened 28×28 image (1×784) and the target is a one-hot vector of length 10.
typedef struct {
    DMatrix* inputs;
    DMatrix* targets;
    size_t sample_count;
} DataSet;

DataSet generate_mnist_like_data(size_t samples) {
    DataSet ds;
    ds.sample_count = samples;
    ds.inputs = (DMatrix*)malloc(samples * sizeof(DMatrix));
    ds.targets = (DMatrix*)malloc(samples * sizeof(DMatrix));
    for (size_t s = 0; s < samples; s++) {
        // Simulate a 28×28 image.
        double* data = (double*)malloc(784 * sizeof(double));
        for (size_t i = 0; i < 784; i++) {
            data[i] = ((double)rand() / RAND_MAX);
        }
        ds.inputs[s] = dmatrix_from_row_slice(1, 784, data);
        free(data);
        // One-hot encoded target vector for 10 classes.
        size_t label = rand() % 10;
        double* target_data = (double*)calloc(10, sizeof(double));
        target_data[label] = 1.0;
        ds.targets[s] = dmatrix_from_row_slice(1, 10, target_data);
        free(target_data);
    }
    return ds;
}

// Frees the DataSet.
void DataSet_free(DataSet* ds) {
    for (size_t i = 0; i < ds->sample_count; i++) {
        dmatrix_free(&ds->inputs[i]);
        dmatrix_free(&ds->targets[i]);
    }
    free(ds->inputs);
    free(ds->targets);
}

// ------------------- Main Function -------------------
int main() {
    srand((unsigned int)time(NULL));
    
    // Generate simulated MNIST-like data (e.g., 1000 samples).
    DataSet ds = generate_mnist_like_data(1000);
    
    // Build a network with the following architecture:
    // 784 inputs, hidden layers with 256, 128, 64 neurons, and 10 outputs.
    // Maximum allowed neurons per hidden layer: 512; maximum hidden layers: 5.
    size_t sizes_arr[] = {784, 256, 128, 64, 10};
    NeuralNetwork nn = NeuralNetwork_new(sizes_arr, 5, 512, 5);
    
    // Initialize the Adam optimizer with a learning rate of 0.001.
    Adam optimizer = Adam_new(0.001, sizes_arr, 5);
    
    // Train the network for 1000 epochs.
    NeuralNetwork_train(&nn, ds.inputs, ds.targets, ds.sample_count, 1000, &optimizer);
    
    // Test the trained network on a few samples.
    printf("\nTesting the trained network:\n");
    for (size_t i = 0; i < 5; i++) {
        size_t act_count = 0;
        DMatrix* activations = NeuralNetwork_forward(&nn, ds.inputs[i], &act_count);
        DMatrix output = activations[act_count - 1];
        // The predicted label is the index of the maximum output value.
        size_t predicted = 0;
        double max_val = output.data[0];
        for (size_t j = 1; j < output.ncols; j++) {
            if (output.data[j] > max_val) {
                max_val = output.data[j];
                predicted = j;
            }
        }
        // Determine the true label from the one-hot target vector.
        size_t target_label = 0;
        for (size_t j = 0; j < ds.targets[i].ncols; j++) {
            if (fabs(ds.targets[i].data[j] - 1.0) < 1e-6) {
                target_label = j;
                break;
            }
        }
        printf("Sample %zu: Predicted = %zu, Target = %zu\n", i, predicted, target_label);
        // Free activations.
        for (size_t j = 0; j < act_count; j++) {
            dmatrix_free(&activations[j]);
        }
        free(activations);
    }
    
    // Free allocated resources.
    DataSet_free(&ds);
    NeuralNetwork_free(&nn);
    Adam_free(&optimizer);
    
    return 0;
}
