import numpy as np # Use for matrix math
import pandas as pd # Used parsing and subsequent loading of the dataset
import math
from dataset import Dataset, Entry
import matplotlib.pyplot as plt
import matplotlib
import threading
import json
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import random
import os
import copy
from multiprocessing import Process
from PIL import Image

# Cost function is the accumulation of the loss function over a given batch of the dataset
# I average the cost function for every batch

# Mean squared error (for regression (infinite continuous values))
def loss_mse(predicted, observed):
    return math.pow(predicted - observed, 2)

# Derivative with respect to the predicted value (hypothesis/activation/output/prediction)
def mse_derivative(predicted, observed, n_training_samples):
    return 2 * (predicted - observed) / n_training_samples

# Binary cross entropy (for logistic regression (probability between 0 and 1))
def loss_binary_crossentropy(predicted, observed):
    return ((-observed * math.log(predicted + 0.0000001)) - ((1 - observed) * math.log(1 - (predicted - 0.0000001))))

def binary_crossentropy_derivative(predicted, observed, n_training_samples):
    a = predicted

    if a == 0:
        a += 0.000001
    
    b = 1 - a

    if b == 0:
        b += 0.000001

    return ((-observed / a) + ((1 - observed) / b)) / n_training_samples
    # return ((-observed / a) + ((1 - observed) / b))  / n_training_samples

def loss_categorical_crossentropy(predicted, observed):
    if observed == 1:
        result = -math.log(predicted + 0.0000001)
        return result
    else:
        return 0

def categorical_crossentropy_derivative(predicted, observed, n_training_samples):
    if observed == 0:
        return 0
    else:
        return (-observed / (predicted + 0.0000001)) / n_training_samples

def linear(z):
    return z

def linear_derivative(z):
    return 1

def sigmoid(z):
    return 1 / (1 + math.pow(math.e, -z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    # Mathematically equivalent to (z + abs(z)) / 2
    # Also max(0, z)
    if z <= 0:
        return 0
    else:
        return z
    
def relu_derivative(z):
    if z <= 0:
        return 0
    else:
        return 1

'''
z: Current node's z
z_l: All current layer's zs
'''
def softmax(z, z_l):
    numerator = math.pow(math.e, z)
    denominator = 0
    for node in range(z_l.shape[0]):
        current_node_z = z_l[node, 0]
        denominator += math.pow(math.e, current_node_z)
    
    result = numerator / denominator

    return result
'''
node: Current node number
z: Current node's z
z_l: All current layer's zs
'''
def softmax_derivative(i, j, a_i, a_j):
    # sj
    '''
    e_zi = math.pow(math.e, z)
    sfmx = softmax(z, z_l)
    result = e_zi * (1 - (e_zi * sfmx))
    '''

    result = 0
    if i == j:
        result = a_i * (1 - a_i)
    else:
        result = -a_i * a_j

    return result



# Layer numbers start at 0 (input layer)

class Layer:
    num = 0
    dim = 0
    type = 0 # 0 == input ; 1 == hidden ; 2 == output
    activation_function = None
    regularize = False
    l1_rate = 0
    l2_rate = 0

    def __init__(self, num, dim, type):
        self.num = num
        self.dim = dim
        self.type = type

    def set_activation_function(self, activation_function):
        if self.num == 0:
            print("Error: Cannot set activation function for input layer.")
            return

        self.activation_function = activation_function

class Network:
    total_batches = []
    costs = []
    averaged_costs = []
    crossvalidation_costs = []
    final_costs = []

    finished_training = False
    update_interval = 500
    batch_avg_interval = 50

    def __init__(self):
        pass



    # Returns the output matrix (result at last layer)
    # z_l = Connection weights to the current layer * Input received by current layer (extends with a 1 to account for bias)
    # a_l = g(z_l) where 'g' is the function defined as the activation function for the current layer (this is also the input to the next layer, if there is one)
    def feedforward(self, model, layer_num, input: np.matrix, cache=True):
        # Padding the beginning of the input with the value `1` for the intercept weight

        new_input = input # Includes prepended 1 to account for bias

        '''
        if not prepend_ones_prior:
            new_input = np.zeros((input.shape[0] + 1, input.shape[1]), dtype=np.float64)
            new_input[0, 0] = 1

            for row in range(1, new_input.shape[0]):
                new_input[row, 0] = input[row - 1, 0]
        '''

        # Preprends 1 for bias (layer 0 and 1 already include prepended 1)
        if layer_num == 0:
            if cache:
                new_input_without_1 = new_input[1:, 0]

                '''
                new_input_without_1 = np.zeros((new_input.shape[0] - 1, 1), dtype=np.float64)
                new_input_without_1 = np.matrix(new_input_without_1)

                for neuron in range(1, new_input.shape[0]):
                    new_input_without_1[neuron - 1, 0] = new_input[neuron, 0]

                '''

                model.layers_a.append(new_input_without_1)
                model.layers_z.append(new_input_without_1)

            # print(f"Layer: {layer_num} | Input: {new_input} | Z({layer_num}): {new_input} | A({layer_num}): {new_input}")

            # New input here has prepended 1
            return self.feedforward(model, 1, new_input, cache)
        
        elif layer_num > 1:
            new_input = np.zeros((input.shape[0] + 1, input.shape[1]), dtype=np.float64)
            new_input = np.matrix(new_input)
            new_input[0, 0] = 1

            for row in range(1, new_input.shape[0]):
                new_input[row, 0] = input[row - 1, 0]

        z_l = model.weights[-1][layer_num - 1] * new_input # i.e. assuming a linear model hypothesis = B0 + B1x_1 + B2x_2 + ...

        if cache:
            model.layers_z.append(z_l)

        layer = model.layers_in_order[layer_num]

        a_l = np.matrix(np.copy(z_l))
        for row in range(z_l.shape[0]):
            if layer.activation_function == softmax:
                a_l[row, 0] = layer.activation_function(z_l[row, 0], z_l)
            else:
                a_l[row, 0] = layer.activation_function(z_l[row, 0]) # i.e. where g is the layer's activation function = g(B0 + B1x_1 + B2_x2 + ...)

        if cache:
            model.layers_a.append(a_l)
        # Each row of output contains an output variable now
            
        # print(f"Layer: {layer_num} | Input: {new_input} | Z({layer_num}): {z_l} | A({layer_num}): {a_l}")

        if layer_num == model.sorted_layer_nums[-1]:
            return a_l

        # result = output + self.feedforward(layer_num + 1) # Is this backpropagation? Cost or loss? Hypothesis?

        # Do things after returning from the output layer to the previous layers?
        
        return self.feedforward(model, layer_num + 1, a_l, cache)
    
    def backpropagation(self, model, observed, n_training_samples):
        partials = []
        layer_errors = []

        # The loop starts at the output layer and goes back, stops at layer 1 (layer the input layer is connected to)
        # In a 3 layer arrangement, the loop goes 2 -> 1

        '''
        print(f"Input: {model.layers_a[0]} | {model.layers_a[0].shape} | {len(model.layers_a[0])}")
        print(f"W_0 (layer 0 -> layer 1): {model.weights[-1][0]} | {model.weights[-1][0].shape} | {len(model.weights[-1][0])}")
        print(f"Z_1 (layer 1 input): {model.layers_z[1]} | {model.layers_z[1].shape} | {len(model.layers_z[1])}")
        print(f"A_1 (layer 1 output): {model.layers_a[1]} | {model.layers_a[1].shape} | {len(model.layers_a[1])}")
        print(f"W_1 (layer 1 -> layer 2): {model.weights[-1][1]} | {model.weights[-1][1].shape} | {len(model.weights[-1][1])}")
        print(f"Z_2 (layer 2 input): {model.layers_z[2]} | {model.layers_z[2].shape} | {len(model.layers_z[2])}")
        print(f"A_2 (layer 2 output): {model.layers_a[2]} | {model.layers_a[2].shape} | {len(model.layers_a[2])}")
        print(f"One hot: {observed} | {observed.shape} | {len(observed)}")
        '''

        
        for layer in reversed(range(1, len(model.layers))):
            a_l = model.layers_a[layer]
            g_l = model.layers_in_order[layer].activation_function
            z_l = model.layers_z[layer]
            a_lm1 = model.layers_a[layer - 1]
            w_l = model.weights[-1][layer - 1]
            w_lp1 = None
            if layer != len(model.layers) - 1:
                w_lp1 = model.weights[-1][layer]

            layer_error = np.zeros((a_l.shape[0], 1), dtype=np.float64)
            layer_error = np.matrix(layer_error)
            partial = np.zeros(w_l.shape, dtype=np.float64)
            partial = np.matrix(partial)

            # Ignoring prepended 1 for bias
            # Bias error is not calculated
            # (Could add a placeholder node at the beginning for bias, but it's not necessary as its partial derivative is always equal to the node error)
            for node in range(0, model.layers_in_order[layer].dim):
                node_error = 0

                # dc_dak
                if layer == model.sorted_layer_nums[-1]:
                    if model.loss_function == loss_binary_crossentropy:
                        node_error += binary_crossentropy_derivative(a_l[node, 0], observed[0, node], n_training_samples)
                    elif model.loss_function == loss_categorical_crossentropy and model.layers_in_order[-1].activation_function != softmax:
                        node_error += categorical_crossentropy_derivative(a_l[node, 0], observed[0, node], n_training_samples)
                        '''
                        if node < 2 and layer == 2:
                            print(f"Node {node} categorical derivative: {node_error}")
                        '''
                    elif model.loss_function == loss_mse:
                        node_error += mse_derivative(a_l[node, 0], observed[0, node], n_training_samples)
                    elif model.loss_function != loss_categorical_crossentropy:
                        print(f"Error: Could not perform backpropagation, was unable to calculate the neuron's error at the last layer - the given cost function is not supported.")
                        exit(-1)
                
                # dzk_dak-1 / ... / dz1_da0
                else:
                    for node_next in range(model.layers_in_order[layer + 1].dim):
                        next_layer_node_error = layer_errors[-1][node_next, 0]

                        # 1 + node in order to skip the bias node
                        # Error is accumulated and weighted over the next layer nodes that this node in the current layer connects to

                        node_error += next_layer_node_error * w_lp1[node_next, 1 + node]
                    #node_error += w_l[node, layer - 1]

                # dak_dzk / dak-1_dzk-1 / ... / da1_dz1
                if g_l == linear:
                    node_error *= linear_derivative(z_l[node, 0])
                elif g_l == relu:
                    node_error *= relu_derivative(z_l[node, 0])
                elif g_l == sigmoid:
                    node_error *= sigmoid_derivative(z_l[node, 0])
                elif g_l == softmax and (not (layer == model.sorted_layer_nums[-1] and model.loss_function == loss_categorical_crossentropy)):
                    ## i == which node the softmax function is calculated over
                    ## j == which input (z_l[node, 0]) is the derivative taken with respect to

                    # Softmax depends on all z's and therefore we need to take the derivative with respective to all of them
                    
                    # print(f"Current layer: {layer}")
                    # print(f"a_l: {a_l}")
                    # print(f"z_l: {z_l}")
                    
                    tmp_node_error = 0
                    total_zs = z_l.shape[0]
                    for current_z in range(total_zs):
                        '''
                        if node < 2 and layer == 2:
                            print(f"TMMMMM current_z {current_z}: {softmax_derivative(node, current_z, a_l[node, 0], a_l[current_z, 0])}")
                        '''
                        tmp_node_error += softmax_derivative(node, current_z, a_l[node, 0], a_l[current_z, 0])
                    # print(tmp_node_error1)

                    # print(tmp_node_error1)
                    node_error *= tmp_node_error

                    '''
                    if node < 2 and layer == 2:
                        print(f"Node {node} tmp_error: {tmp_node_error}")
                    '''
                    
                    # tmp_node_error2 = 0
                    # for i in range(a_l.shape[0]):
                        # tmp_node_error2 += a_l[i, 0] - observed[0, i]
                    
                    # print(tmp_node_error2)
                    
                    # node_error *= tmp_node_error2

                    # print(node_error)

                    # print(tmp_node_error2)

                    # print(f"{tmp_node_error1},{tmp_node_error2}")

                    # exit(-1)
                elif g_l == softmax and (layer == model.sorted_layer_nums[-1] and model.loss_function == loss_categorical_crossentropy):
                    node_error = (a_l[node, 0] - observed[0, node]) / n_training_samples
                else:
                    print(f"Error: Could not perform backpropagation, the given activation function for layer {layer} is not defined. Needs to be 'linear', 'relu', 'sigmoid', or 'softmax'.")
                    exit(-1)

                layer_error[node, 0] = node_error

                # Accounting for bias (1)
                partial[node, 0] = node_error

                # print(f"Node error: {node_error}")
                # print(f"A_LM1: {a_lm1}")
                # print(f"A_L: {a_l}")
                # print(f"AS: {self.layers_a}")
                for prev_node in range(1, w_l.shape[1]):
                    partial[node, prev_node] = node_error * a_lm1[prev_node - 1, 0]
                    
                '''
                if self.loss_function == loss_mse:
                    approximation = np.zeros(partial.shape, dtype=np.float64)
                    approximation = np.matrix(approximation)
                    epsilon = 0.0001

                    tmp_input = np.matrix([[1], [a_lm1[0, 0]]])
                    for node_prev in range(a_lm1.shape[0] + 1):
                        tmp_weight_a = self.weights[-1][0]
                        tmp_weight_b = self.weights[-1][0]

                        for node_nxt in range(a_l.shape[0]):
                            tmp_weight_a[node_nxt, node_prev] = tmp_weight_a[node_nxt, node_prev] - epsilon
                            tmp_weight_b[node_nxt, node_prev] = tmp_weight_b[node_nxt, node_prev] + epsilon

                            approximation[node_nxt, node_prev] = (2 * a_l[node, 0]) * ((tmp_weight_a * tmp_input) - (tmp_weight_b * tmp_input) / (-2 * epsilon))

                    print(f"Backpropr: {partial}")
                    print(f"Approximate: {approximation}")
                '''
            
            '''
            if layer == 2:
                print(f"DELTA_ERROR_2: {layer_error} | {layer_error.shape} | {len(layer_error)}")
                print(f"DW_1: {partial} | {partial.shape} | {len(partial)}")
            elif layer == 1:
                print(f"DELTA_ERROR_1: {layer_error} | {layer_error.shape} | {len(layer_error)}")
                print(f"DW_0: {partial} | {partial.shape} | {len(partial)}")
            '''
            

            layer_errors.append(layer_error)
            partials.append(partial)

        # plot_image_from_sample(model.layers_a[0])

        # exit(-1)

        layer_errors = list(reversed(layer_errors))
        partials = list(reversed(partials))

        return partials

    def gradient_descent(self, model, partials, lr, current_batch_total_samples):
        if len(partials) != len(model.layers) - 1:
            print(f"Error: Could not perform gradient descent, the number of matrices given for the partials ({len(partials)}) does not match the number of connected layers ({len(model.layers) - 1}) in the network.")
            exit(-1)

        new_weights = []
        layer_connection_num = 1
        for weight, partial in zip(model.weights[-1], partials):
            current_layer = model.layers_in_order[layer_connection_num]
            if weight.shape != partial.shape:
                print(f"Error: Could not perform gradient descent, the partial derivative's dimension {partial.shape} does not match the weight's dimension {weight.shape} connecting layer {layer_connection_num - 1} to layer {layer_connection_num}.")
                exit(-1)

            if current_layer.regularize:

                # Apply only L2 regularization if L1 is not set or 0
                if current_layer.l1_rate == 0:
                    new_weight = (weight * (1 - (lr * current_layer.l2_rate / current_batch_total_samples))) - (lr * partial)
                    # print(f"Old weight: {weight}")
                    # print(f"New weight: {weight - (lr * partial)}")
                    # print(f"New weight (regularized): {new_weight}")
                    # exit(-1)

                # Apply only L1 regularization if L2 is not set or 0
                elif current_layer.l2_rate == 0:
                    new_weight = weight
                    for i in range(weight.shape[0]):
                        for j in range(weight.shape[1]):

                            
                            if j != 0:
                                if weight[i, j] > 0:
                                    new_weight[i, j] = (weight[i, j] - (lr * partial[i, j])) - (lr * current_layer.l1_rate / current_batch_total_samples)
                                elif weight[i, j] < 0:
                                    new_weight[i, j] = (weight[i, j] - (lr * partial[i, j])) + (lr * current_layer.l1_rate / current_batch_total_samples)

                            # Skip regularization for bias
                            else:
                                new_weight[i, j] = weight[i, j] - (lr * partial[i, j])

                # Apply both L1 and L2 regularizations
                else:
                    new_weight = (weight * (1 - (lr * current_layer.l2_rate / current_batch_total_samples))) - (lr * partial)
                    for i in range(weight.shape[0]):
                        for j in range(weight.shape[1]):

                            if j != 0:
                                if weight[i, j] > 0:
                                    new_weight[i, j] -= (lr * current_layer.l1_rate / current_batch_total_samples)
                                elif weight[i, j] < 0:
                                    new_weight[i, j] += (lr * current_layer.l1_rate / current_batch_total_samples)

                            # Skip regularization for bias
                            else:
                                new_weight[i, j] = weight[i, j] - (lr * partial[i, j])

            else:
                new_weight = weight - (lr * partial)

            new_weights.append(new_weight)
            layer_connection_num += 1

        model.weights.append(new_weights)

    def mean_partials(self, batch_partials):
        final_partials = batch_partials[0]
        total_batch_samples = len(batch_partials)
        total_layer_connections = len(model.weights[-1])

        for sample_partials_idx in range(1, total_batch_samples):
            sample_partials = batch_partials[sample_partials_idx]
            for layer_connection in range(total_layer_connections):
                final_partials[layer_connection] = final_partials[layer_connection] + sample_partials[layer_connection]

        # I was previously averaging the partials over the batch size, but, if necessary, this already happens when taking the derivative of the loss function
        # for layer_connection in range(total_layer_connections):
            # final_partials[layer_connection] = final_partials[layer_connection] / total_batch_samples

        return final_partials

    def preprocess_dataset():
        # Randomize, etc.

        '''
        if prepend_ones_prior:
            training_samples = np.zeros((1 + len(dataset.feature_list), len(dataset.filtered_entries)))
            for num, entry in enumerate(dataset.filtered_entries):
                training_samples[0, num] = 1
                #print(f"{entry.features} - {entry.dependent_variable_value}")
                pass
        '''
        return
    
    '''
    def plot(self, i):
        plt.cla()
        try:
            plt.plot(self.total_batches, self.costs)
        except:
            return
        

        if self.finished:
            self.ani.event_source.stop()
            print("Stopped animating the graph. Close it in order to predict values.")
    '''

    def sample_loss(self, model, predicted_output, observed_output):
        sample_loss = 0
        for output_neuron in range(predicted_output.shape[0]):
            
            predicted_value = predicted_output[output_neuron, 0]

            observed_value = observed_output[0, output_neuron]
            sample_loss += model.loss_function(predicted_value, observed_value)

        for layer in model.layers_in_order:
            if layer.regularize:
                layer_weights = model.weights[-1][layer.num - 1] # layer.num should never be 0 as layer.regularize should only be true for layer.num > 0

                # Apply only L2 regularization if L1 is not set or 0
                if layer.l1_rate == 0:
                    sample_loss += layer.l2_rate * np.sum(np.square(layer_weights))
                    continue
                    for i in range(layer_weights.shape[0]):
                        # Starts at 1 in order to ignore the bias weight
                        for j in range(1, layer_weights.shape[1]):
                            sample_loss += layer.l2_rate * layer_weights[i, j] * layer_weights[i, j]
                            #print(f"Difference ({layer.l2_rate}): {layer.l2_rate * layer_weights[i, j] * layer_weights[i, j]}\n Inverted difference ({1/layer.l2_rate}): {(1/layer.l2_rate) * layer_weights[i, j] * layer_weights[i, j]}")

                # Apply only L1 regularization if L2 is not set or 0
                elif layer.l2_rate == 0:
                    for i in range(layer_weights.shape[0]):
                        # Starts at 1 in order to ignore the bias weight
                        for j in range(1, layer_weights.shape[1]):
                            sample_loss += layer.l1_rate * abs(layer_weights[i, j])

                # Apply both L1 and L2 regularization
                else:
                    for i in range(layer_weights.shape[0]):
                        # Starts at 1 in order to ignore the bias weight
                        for j in range(1, layer_weights.shape[1]):
                            sample_loss += (layer.l2_rate * layer_weights[i, j] * layer_weights[i, j]) + (layer.l1_rate * abs(layer_weights[i, j]))

        # Averages all output nodes if there are more than 1 and the loss is MSE
        if model.loss_function == loss_mse and model.layers_in_order[-1].dim > 1:
            # sample_loss /= model.layers_in_order[-1].dim
            pass
    
        return sample_loss
        
    def step(self, batch_size, lr, training_samples, dependent_values, model=None):
        if use_kfolds:
            batch_partials = [[] for _ in range(crossvalidation_folds)]
        else:
            batch_partials = [[]]

        average_cost = 0 
        last_sample = 0

        ## USED DATASET NEEDS TO BE FIGURED OUT HERE
        if use_kfolds:
            
            n_training_samples = crossvalidation_training_section_size
            batches_per_step = math.ceil(n_training_samples / batch_size)
            full_batches_per_step = math.floor(n_training_samples / batch_size)
            n_last_batch_samples = n_training_samples - (full_batches_per_step * batch_size)
            last_full_batch_sample = full_batches_per_step * batch_size

            for sample in range(1, crossvalidation_training_section_size + 1):
                models_average_loss = 0

                for current_model in range(crossvalidation_folds):
                    selected_model: Model = crossvalidation_models[current_model]
                    selected_model.layers_a = []
                    selected_model.layers_z = []

                    training_samples = crossvalidation_training_samples[current_model]
                    dependent_values = crossvalidation_training_dependent_values[current_model]
                    training_features_mean = crossvalidation_training_features_mean[current_model]
                    training_dependent_variables_mean = crossvalidation_training_dependent_variables_mean[current_model]

                    test_samples = crossvalidation_test_samples[current_model]
                    test_dependent_values = crossvalidation_test_dependent_values[current_model]
                    test_features_mean = crossvalidation_test_features_mean[current_model]
                    test_dependent_variables_mean = crossvalidation_test_dependent_variables_mean[current_model]


                    observed_values = dependent_values[sample - 1, :]
                    output = self.feedforward(selected_model, 0, training_samples[:, sample - 1], True)
                    
                    sample_loss = self.sample_loss(selected_model, output, observed_values)

                    if sample <= last_full_batch_sample:
                        sample_partials = self.backpropagation(selected_model, observed_values, batch_size)
                    else:
                        sample_partials = self.backpropagation(selected_model, observed_values, n_last_batch_samples)

                    batch_partials[current_model].append(sample_partials)

                    model_sample_loss = sample_loss
                    models_average_loss += model_sample_loss / crossvalidation_folds

                    if ((sample % (batch_size)) == 0) or (sample == n_training_samples):
                        model_metrics = {}
                        micro_metrics = {}
                        macro_metrics = {}
                        class_metrics = []

                        measure_model_on_dataset(selected_model, test_samples, test_dependent_values, test_features_mean, test_dependent_variables_mean, selected_model.feature_list, selected_model.class_list, model_metrics, micro_metrics, macro_metrics, class_metrics)

                        self.models_model_metrics[current_model].append(model_metrics)
                        self.models_micro_metrics[current_model].append(micro_metrics)
                        self.models_macro_metrics[current_model].append(macro_metrics)
                        self.models_class_metrics[current_model].append(class_metrics)

                        # Average of batch partials
                        mean_partials = []
                        if len(batch_partials[current_model]) == 1:
                            mean_partials = batch_partials[current_model][0]
                        else:
                            mean_partials = self.mean_partials(batch_partials[current_model])

                        if sample <= last_full_batch_sample:
                            current_batch_total_samples = batch_size
                        else:
                            current_batch_total_samples = n_last_batch_samples

                        self.gradient_descent(selected_model, mean_partials, lr, current_batch_total_samples)

                                # print(self.weights[-1])

                average_cost += models_average_loss

                if ((sample % (batch_size)) == 0) or (sample == crossvalidation_training_section_size):
                    if len(self.total_batches) == 0:
                        self.total_batches.append(1)
                    else:
                        self.total_batches.append(self.total_batches[-1] + 1)

                    if sample <= last_full_batch_sample:
                        current_batch_total_samples = batch_size
                    else:
                        current_batch_total_samples = n_last_batch_samples

                    # MSE being a mean needs to be averaged over sample size (which in this case is the batch size)
                    if crossvalidation_models[0].loss_function == loss_mse or crossvalidation_models[0].loss_function == loss_binary_crossentropy or crossvalidation_models[0].loss_function == loss_categorical_crossentropy:
                        batch_average = average_cost / current_batch_total_samples

                    # Other losses, such as cross entropy, aren't averaged
                    else:
                        batch_average = average_cost
                        
                    if len(self.costs) == 0:
                        average_cost = batch_average

                    # Average cost is the current batch cost averaged with the previous batch cost (which is also an average with the previous last batch average and so on)
                    else:
                        # average_cost = (batch_average + self.costs[-1]) / 2
                        average_cost = batch_average

                    self.costs.append(average_cost)

                    average_cost = 0

                    if draw_graph and (len(xs) == 0 or self.total_batches[-1] - xs[-1] >= plot_update_every_n_batches):
                        # plt.cla()
                    
                        xs.append(self.total_batches[-1])

                        k_folds_avg_cost = 0

                        if crossvalidation_models[0].is_classification():
                            accuracy_avg = 0
                            precision_avg = 0
                            recall_avg = 0
                            f1score_avg = 0
                        elif crossvalidation_models[0].is_regression():
                            r_squared_avg = 0
                            # r_avg = 0
                            
                        for fold, current_model in enumerate(crossvalidation_models):
                            cost_ys[2 + fold].append(self.models_model_metrics[fold][-1]["total_cost"])
                            k_folds_avg_cost += cost_ys[2 + fold][-1] / crossvalidation_folds

                            if crossvalidation_models[0].is_classification():
                                accuracy_avg += self.models_micro_metrics[fold][-1]["accuracy"] / crossvalidation_folds
                                precision_avg += self.models_micro_metrics[fold][-1]["precision"] / crossvalidation_folds
                                recall_avg += self.models_micro_metrics[fold][-1]["recall"] / crossvalidation_folds
                                f1score_avg += self.models_micro_metrics[fold][-1]["f1score"] / crossvalidation_folds
                            elif crossvalidation_models[0].is_regression():
                                r_squared_avg += self.models_micro_metrics[fold][-1]["r_squared"] / crossvalidation_folds
                                # r_avg += self.models_micro_metrics[fold][-1]["r"] / crossvalidation_folds

                            # print(f'{self.models_micro_metrics[fold][-1]["r_squared"]} | {self.models_micro_metrics[fold][-1]["r"]}')

                        cost_ys[0].append(self.costs[-1])
                        cost_ys[1].append(k_folds_avg_cost)

                        if crossvalidation_models[0].is_classification():
                            metrics_ys[0].append(accuracy_avg)
                            metrics_ys[1].append(precision_avg)
                            metrics_ys[2].append(recall_avg)
                            metrics_ys[3].append(f1score_avg)
                        elif crossvalidation_models[0].is_regression():
                            metrics_ys[0].append(r_squared_avg)
                            # metrics_ys[1].append(r_avg)

                        # print(metrics_ys[-1])
                        
                        axes[0][0].plot(xs, cost_ys[0], color=costs_plotting_colors[0])
                        axes[0][0].plot(xs, cost_ys[1], color=costs_plotting_colors[1])

                        for fold in range(crossvalidation_folds):
                            axes[0][0].plot(xs, cost_ys[2 + fold], color=costs_plotting_colors[2 + fold])
                        
                        for idx, metric in enumerate(metrics_ys):
                            axes[0][1].plot(xs, metric, color=metrics_plotting_colors[idx])

                        if len(self.total_batches) == 1:
                            axes[0][0].legend(cost_plot_labels, loc="upper right")
                            axes[0][1].legend(metrics_plot_labels, loc="upper right")
                            axes[0][1].set_ylim(-1, 1)

                        plt.draw()
                        plt.pause(0.001)
                    
                    batch_partials = [[] for _ in range(crossvalidation_folds)]

                    last_sample = sample

            #print(f"Sample: {sample}")
        elif (not use_kfolds) and model is not None:
            # Model should only be `None` when using k-folds

            n_training_samples = training_samples.shape[1]
            
            final_cost = 0

            batches_per_step = math.ceil(n_training_samples / batch_size)
            full_batches_per_step = math.floor(n_training_samples / batch_size)
            n_last_batch_samples = n_training_samples - (full_batches_per_step * batch_size)
            last_full_batch_sample = full_batches_per_step * batch_size
            for sample in range(1, n_training_samples + 1):

                # print(self.weights[-1])

                #print(f"Sample: {sample}")
                model.layers_a = []
                model.layers_z = []
                observed_values = dependent_values[sample - 1, :]
                output = self.feedforward(model, 0, training_samples[:, sample - 1], True)

                '''
                for out in range(output.shape[0]):
                    val = output[out, 0]
                    if val > 1 or val < 0:
                        print(f"Input: {training_samples[:, sample - 1]} | Output: {output}")
                        exit(-1)
                '''
                # Allow user to give custom loss function?
                sample_loss = self.sample_loss(model, output, observed_values)

                final_cost += sample_loss          
                # sample_partials = self.backpropagation(model, observed_values, n_training_samples)
                if sample <= last_full_batch_sample: # Actual sample number, not index (starts at 1)
                    sample_partials = self.backpropagation(model, observed_values, batch_size)
                else:
                    sample_partials = self.backpropagation(model, observed_values, n_last_batch_samples)

                batch_partials[0].append(sample_partials)
                average_cost += sample_loss
                
                if ((sample % (batch_size)) == 0) or (sample == training_samples.shape[1]):
                    if len(self.total_batches) == 0:
                        self.total_batches.append(1)
                    else:
                        self.total_batches.append(self.total_batches[-1] + 1)

                    # current_batch_total_samples = sample - last_sample
                    if sample <= last_full_batch_sample:
                        current_batch_total_samples = batch_size
                    else:
                        current_batch_total_samples = n_last_batch_samples

                    # MSE being a mean needs to be averaged over sample size (which in this case is the batch size)
                    if model.loss_function == loss_mse or model.loss_function == loss_binary_crossentropy or model.loss_function == loss_categorical_crossentropy:
                        batch_average = average_cost / current_batch_total_samples

                    # Other losses, such as cross entropy, aren't averaged
                    else:
                        batch_average = average_cost
                        
                    if len(self.costs) == 0:
                        average_cost = batch_average

                    # Average cost is the current batch cost averaged with the previous batch cost (which is also an average with the previous last batch average and so on)
                    else:
                        average_cost = batch_average
                        # average_cost = (batch_average + self.costs[-1]) / 2
                        pass

                    self.costs.append(average_cost)

                    average_cost = 0

                    if use_holdout:
                        model_metrics = {}
                        micro_metrics = {}
                        macro_metrics = {}
                        class_metrics = []
                        
                        measure_model_on_dataset(model, crossvalidation_samples, crossvalidation_dependent_values, crossvalidation_features_mean, crossvalidation_dependent_variables_mean, model.feature_list, model.class_list, model_metrics, micro_metrics, macro_metrics, class_metrics)

                        self.models_model_metrics[0].append(model_metrics)
                        self.models_micro_metrics[0].append(micro_metrics)
                        self.models_macro_metrics[0].append(macro_metrics)
                        self.models_class_metrics[0].append(class_metrics)

                    if draw_graph and (len(self.total_batches) == 1 or self.total_batches[-1] - xs[-1] >= plot_update_every_n_batches):
                        xs.append(self.total_batches[-1])
                        cost_ys[0].append(self.costs[-1])
                        axes[0][0].plot(xs, cost_ys[0], color=costs_plotting_colors[0])

                        '''
                        if len(cost_ys[0]) > 1:
                            maxi = max(cost_ys[0])
                            mini = min(cost_ys[0])

                            new_ys = [Dataset.normalize_helper(cost, mini, maxi, 0, 1) for cost in cost_ys[0]]

                            for idx, cost in enumerate(new_ys):
                                print(f"{cost_ys[0][idx]} -> {cost}")

                        else:
                            new_ys = [1]

                        axes[0][0].plot(xs, new_ys, color=costs_plotting_colors[0])
                        '''

                        if use_holdout:
                            if model.is_classification():
                                accuracy = self.models_micro_metrics[0][-1]["accuracy"]
                                precision = self.models_micro_metrics[0][-1]["precision"]
                                recall = self.models_micro_metrics[0][-1]["recall"]
                                f1score = self.models_micro_metrics[0][-1]["f1score"]

                                metrics_ys[0].append(accuracy)
                                metrics_ys[1].append(precision)
                                metrics_ys[2].append(recall)
                                metrics_ys[3].append(f1score)
                            elif model.is_regression():
                                r_squared = self.models_micro_metrics[0][-1]["r_squared"]
                                # r = self.models_micro_metrics[0][-1]["r"]

                                metrics_ys[0].append(r_squared)
                                # metrics_ys[1].append(r)

                            # print(metrics_ys)
                            # print(model.weights[-1])
                            
                            cost_ys[1].append(self.models_model_metrics[0][-1]["total_cost"])
                            axes[0][0].plot(xs, cost_ys[1], color=costs_plotting_colors[1])

                            for idx, metric in enumerate(metrics_ys):
                                axes[0][1].plot(xs, metric, color=metrics_plotting_colors[idx])  

                        if len(self.total_batches) == 1:
                            axes[0][0].legend(cost_plot_labels, loc="upper right")

                            if use_holdout:
                                axes[0][1].legend(metrics_plot_labels, loc="upper right")
                                axes[0][1].set_ylim(-1, 1)
                                # axes[0][0].set_yscale("log")
                    
                        plt.draw()
                        plt.pause(0.001)
                    
                    # Average of batch partials
                    mean_partials = []
                    if len(batch_partials[0]) == 1:
                        mean_partials = batch_partials[0][0]
                    else:
                        mean_partials = self.mean_partials(batch_partials[0])

                    batch_partials = [[]]

                    self.gradient_descent(model, mean_partials, lr, current_batch_total_samples)

                    last_sample = sample

            final_cost /= n_training_samples
            self.final_costs.append(final_cost)
            # plt.plot([_ for _ in range(len(self.final_costs))], self.final_costs, color="purple")
            # print(f"Final cost: {final_cost}")
        elif not use_kfolds and model is None:
            print(f"Error: Could not start training model, `None` was passed. Model can only be `None` when using k-folds cross-validation.")
            exit(-1)
        elif use_kfolds and model is not None:
            print(f"Error: Could not start training model, Model was not `None`, but must be when using k-folds cross-validation.")
            exit(-1)

    def train(self, batch_size, steps, lr, training_samples: np.matrix, dependent_values: np.matrix, model=None):
        # Training examples will be a matrix containing all training samples, where each column is a new sample and each row is a feature
        # Dependent variable values are stored sequentially in the same sequence as the training samples, shape is (len(training_samples), 1)

        # Prepend 1s here to a temporary copy variable instead of the dataset to decrease coupling (will double the memory requirements)

        self.total_batches = []

        if use_kfolds:
            self.models_model_metrics = [[] for _ in range(crossvalidation_folds)]
            self.models_micro_metrics = [[] for _ in range(crossvalidation_folds)]
            self.models_macro_metrics = [[] for _ in range(crossvalidation_folds)]
            self.models_class_metrics = [[] for _ in range(crossvalidation_folds)]
        elif use_holdout:
            self.models_model_metrics = [[]]
            self.models_micro_metrics = [[]]
            self.models_macro_metrics = [[]]
            self.models_class_metrics = [[]]

        prev_cost_idx = 0
        for step in range(1, steps + 1):
            
            # print(f"Loss: . Precision: . Accuracy: . Cross validation loss: . F-score: . ")

            # Runs each sample in the training samples at a time through the network, accumulating the cost?
            # Note: I defined the cost to be the average of the batch with the average of the current cost so cost = mean(current_cost, mean(batch_cost))
            # Cost is average cost over samples, not individual sample/batch cost

            # Normal / hold-out
            if type(model) == Model:
                self.step(batch_size, lr, training_samples, dependent_values, model)

            # K-folds
            elif model is None:
                self.step(batch_size, lr, training_samples, dependent_values)
            else:
                print(f"Error: Could not train model. Model must be either an instance of the `Model` class, or ignored (in the case of k-folds cross-validation).")
                exit(-1)

            if not use_kfolds:
                print(f"Step: {step} | Training cost: {self.costs[-1]}", end="")
                if use_holdout:
                    print(f" | Cross-validation (hold-out) cost: {self.models_model_metrics[0][-1]['total_cost']}")
                else:
                    print()
            elif use_kfolds:
                print(f"Step: {step} | Cross-validation ({crossvalidation_folds})-folds cost: {self.costs[-1]}")

        print("Finished training.")

        # plt.plot(self.total_batches, self.costs)
        # plt.show()

        self.finished_training = True

    # If a model has been trained at least once and needs to be trained again, call this function to clear previous training info
    def reset_training_info(self):
        self.total_batches = []
        self.costs = []
        self.crossvalidation_costs = []
        self.finished_training = False
    
    def predict(self, model, input):
        return self.feedforward(model, 1, input, False)
    
    # Ignoring/skipping some iterations here because it's too slow in real-time
    def plot(self, i):
        step_size = round(0.1 * len(self.total_batches))
        print(f"Step size: {step_size}")
        print(f"XS: {xs}")
        print(f"YS: {cost_ys}")
        total_batches = []
        costs = []

        if len(xs) != 0:
            if self.total_batches[-1] - xs[-1] >= step_size:
                xs.append(self.total_batches[-1])
                cost_ys.append(self.costs[-1])
        elif len(xs) == 0 and len(self.total_batches) > 0:
            xs.append(self.total_batches[0])
            cost_ys.append(self.costs[0])
        else:
            return

        if not use_kfolds and not use_holdout:
            plot_lines[0][0].set_data(xs, cost_ys)
            plot_lines[1][0].set_data(xs, cost_ys)

            axes[0][0].autoscale_view(True, True)
            axes[0][0].relim()

            axes[1][0].autoscale_view(True, True)
            axes[1][0].relim()

        '''
        if use_holdout:
            crossvalidation_costs = [[]]
        elif use_kfolds:
            crossvalidation_costs = [[] for _ in range(crossvalidation_folds)]
        '''

        '''
        if store_micro_metrics or store_macro_metrics:
            if not use_kfolds and model.is_classification():
                accuracies = [[]]
                precisions = [[]]
                recalls = [[]]
                f1scores = [[]]
            elif not use_kfolds and model.is_regression():
                rs = [[]]
                r_squareds = [[]]
                rmses = [[]]
            elif use_kfolds and model.is_classification():
                accuracies = [[] for _ in range(crossvalidation_folds)]
                precisions = [[] for _ in range(crossvalidation_folds)]
                recalls = [[] for _ in range(crossvalidation_folds)]
                f1scores = [[] for _ in range(crossvalidation_folds)]
            elif use_kfolds and model.is_regression():
                rs = [[] for _ in range(crossvalidation_folds)]
                r_squareds = [[] for _ in range(crossvalidation_folds)]
                rmses = [[] for _ in range(crossvalidation_folds)]
        '''

        '''
        for idx in range(0, len(self.total_batches), step_size):
            total_batches.append(self.total_batches[idx])
            costs.append(self.costs[idx])

            if use_holdout:
                crossvalidation_costs[0].append(self.models_model_metrics[0][idx]["total_cost"])
            elif use_kfolds:
                for fold, current_model in enumerate(crossvalidation_models):
                    crossvalidation_costs[fold].append(self.models_model_metrics[fold][idx]["total_cost"])


        '''

        '''
        axes[0][0].cla()
        axes[1][0].cla()

        axes[0][0].set_yscale("linear")
        axes[1][0].set_yscale("log")

        if not use_kfolds and not use_holdout:
            try:
                axes[0][0].plot(total_batches, costs, color='blue')
                axes[1][0].sharex(axes[0][0])
                axes[1][0].plot(total_batches, costs, color='blue')      
            except:
                return
        
        if use_holdout:
            axes[0][1].cla()
            axes[1][1].cla()
            axes[0][1].sharex(axes[0][0])
            axes[1][1].sharex(axes[1][1])
            axes[1][1].set_xlabel("Batch")
            axes[0][1].set_ylabel(f"Cross-validation cost (hold-out) (linear scale)")
            axes[1][1].set_ylabel(f"Cross-validation cost (hold-out) (log scale)")
            axes[1][1].set_yscale("log")

            try:
                axes[0][1].plot(total_batches, crossvalidation_costs[0], color='red')
                axes[1][1].plot(total_batches, crossvalidation_costs[0], color='red')
            except:
                return
            
        elif use_kfolds:
            axes[0][1].cla()
            axes[1][1].cla()
            axes[0][1].sharex(axes[0][0])
            axes[1][1].sharex(axes[1][1])
            axes[1][1].set_xlabel("Batch")
            axes[0][1].set_ylabel(f"Cross-validation cost ({crossvalidation_folds}-folds) (linear scale)")
            axes[1][1].set_ylabel(f"Cross-validation cost ({crossvalidation_folds}-folds) (log scale)")
            axes[1][1].set_yscale("log")

            for fold, current_model in enumerate(crossvalidation_models):
                try:
                    axes[0][1].plot(total_batches, crossvalidation_costs[fold], color=costs_plotting_colors[fold], label=f"Fold {fold + 1}")
                    axes[1][1].plot(total_batches, crossvalidation_costs[fold], color=costs_plotting_colors[fold], label=f"Fold {fold + 1}")
                except:
                    return
        '''

        if self.finished_training:
            self.ani.event_source.stop()
            print("Stopped animating the graph. Close it in order to predict values.")
            return

class Model:
    ready_to_use = False

    layers_a = []
    layers_z = []
    
    activation_functions = []
    loss_function = None

                    #               HIDDEN             HIDDEN            OUTPUT              INPUT
    layers = {} # i.e. {2: Layer(2, 3, 1), 1: Layer(1, 3, 1), 3: Layer(3, 1, 2), 0: Layer(0, 5, 0)}


    sorted_layer_nums = [] # i.e. [0, 1, 2, 3]
    layers_in_order = [] # i.e. [Layer(0, 5, 0), Layer(1, 3, 1), Layer(2, 3, 1), Layer(3, 1, 2)]

    weights = [] # Each layer connection has its own weight matrix (`number of layers - 1` matrices) represented as np.matrix

    normalized = False

    feature_list = []
    feature_types = []
    class_list = []
    feature_min_maxes = []
    sample_features_mean = []
    sample_features_variance = []
    sample_features_std = []
    sample_dependent_variables_mean = []

    image_bit_depth = None

    def add_layer(self, num: int, dim: int, type: int) -> None:
        if self.ready_to_use:
            print("Cannot add a new layer: you cannot modify the model's architecture once you have called `model.setup_done()`.")
            return
        
        if num < 0:
            print(f"Could not add a new layer: layer number must be an integer >= 0, but was {num}.")
            return
        
        if type not in [0, 1, 2]:
            print(f"Could not add a new layer: invalid type {type}. Must be 0 (input), 1 (hidden), or 2 (output).")
            return

        for added_layer in self.layers.values():
            if num == added_layer.num:
                print(f"Error: Could not add layer number {num}. A layer already exists with that number.")
                return
            elif added_layer.type == 2 and num > added_layer.num:
                print(f"Error: Could not add a new layer numbered {num}. The layer number must be less than that of the output layer ({added_layer.num}), but was {num}.")
                return
            if type == added_layer.type:
                if type == 0:
                    print(f"Error: Could not add another input layer. One has already been added.")
                    return
                elif type == 2:
                    print(f"Error: Could not add another output layer. One has already been added.")
                    return
                
        # add activation function?
        
        layer = Layer(num, dim, type)

        self.layers[num] = layer

        print(f"Added layer number {num}.")

    def remove_layer(self, num) -> None:
        if self.ready_to_use:
            print("Cannot remove layer: you cannot modify the model's architecture once you have called `model.setup_done()`.")
            return

        if num in self.layers.keys():
            self.layers.pop(num)
            print(f"Removed layer number {num}.")
        else:
            print(f"Could not remove layer number {num}: it doesn't exist.")

    def replace_layer(self, old_layer_num: int, new_layer: Layer) -> None:
        if self.ready_to_use:
            print("Cannot replace layer: you cannot modify the model's architecture once you have called `model.setup_done()`.")
            return
        
        if old_layer_num not in self.layers.keys():
            print(f"Cannot replace layer {old_layer_num}: it does not exist.")
            return
        
        self.remove_layer(old_layer_num)
        self.add_layer(new_layer.num, new_layer.dim, new_layer.type)
        self.set_activation_function(new_layer.num, new_layer.activation_function)

        print(f"Replaced layer number {old_layer_num}.")

    def set_activation_function(self, layer_num, activation_function):
        ## Validate by making sure that the given activation_function pointer is one of the functions that have been defined, as well as their derivatives (i.e. linear, sigmoid, relu)
        for layer in self.layers.values():
            if layer.num == layer_num:
                layer.activation_function = activation_function

                activation_function_name = ""

                if activation_function == linear:
                    activation_function_name = "linear"
                elif activation_function == relu:
                    activation_function_name = "relu"
                elif activation_function == sigmoid:
                    activation_function_name = "sigmoid"
                elif activation_function == softmax:
                    activation_function_name = "softmax"
                else:
                    print(f"Error: Could not set activation function for layer {layer_num} - invalid activation function given.")
                    exit(-1)

                print(f"Updated activation function for layer number {layer_num} to '{activation_function_name}'.")
                return

        print(f"Error: Could not set activation function - no layer with number {layer_num} found.")

        # Should be run after the setup
    
    # Running this function on layer number 1 regularizes the weights/connections between layer 0 and layer 1
    def set_regularization(self, target_layer_num, l1_rate=0, l2_rate=0):
        if target_layer_num == 0:
            print(f"Error: Could not set regularization for layer 0, the input layer cannot be regularized, only the layers that have incoming connections.")
            exit(-1)

        if l1_rate == 0 and l2_rate == 0:
            print(f"Error: Could not set regularization for layer {target_layer_num}, both regularization rates (L2 and L1) are set to 0, but only one regularization rate can be 0 at once (otherwise don't regularize).")
            exit(-1)
        
        for layer in self.layers_in_order:
            if layer.num == target_layer_num:
                layer.regularize = True
                layer.l1_rate = l1_rate
                layer.l2_rate = l2_rate
                print(f"Regularization set for layer {target_layer_num}. L2: {l2_rate} | L1: {l1_rate}")
                return

        print(f"Error: Could not set regularization for layer {target_layer_num} - it was not found.")

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function
        print(f"Defined loss function for network.")

    '''
    Validates the model architecture.
    Checks the layer ordering (if there are missing or misnumbered layers), if input and output layers were given, and if activation functions were set.
    '''
    def setup_done(self, is_loaded_model=False):
        if not is_loaded_model:
            using_uniform_distribution = False
            using_normal_distribution = True
            while True:
                result = input("Use normal or uniform distribution for initializing weights? (0/1): ").strip()
                if result == "0":
                    using_uniform_distribution = False
                    using_normal_distribution = True
                    break
                elif result == "1":
                    using_uniform_distribution = True
                    using_normal_distribution = False
                    break

        if self.loss_function is None:
            print("Error: Could not setup network - no loss function was defined. You can define a loss function with `nn.set_activation_function(loss_function)` where `nn` is an instance of the `Network` class. A loss function takes, respectively, a predicted value and an observed value, and returns a number.`")
            exit(-1)

        self.sorted_layer_nums = sorted(self.layers.keys())

        if 0 not in self.sorted_layer_nums:
            print("Could not setup network: There is no input layer defined. (No layer with number 0 was given)")
            exit(-1)

        weights = []

        for layer_num in self.sorted_layer_nums:
            layer = self.layers[layer_num]

            type_resolved = "unknown"

            match layer.type:
                case 0:
                    type_resolved = "input"
                case 1:
                    type_resolved = "hidden"
                case 2:
                    type_resolved = "output"
                case _:
                    type_resolved = "unknown"

            if layer_num == 0:
                if layer.type != 0:
                    print(f"Could not setup network: Layer 0 (first layer) must be of type 0 (input), but is of type {layer.type} ({type_resolved}).")
                    exit(-1)
            elif layer_num == self.sorted_layer_nums[-1]:
                if layer.type != 2:
                    print(f"Could not setup network: Layer {layer_num} (last layer) must be of type 2 (output), but is of type {layer.type} ({type_resolved}).")
                    exit(-1)


            # Both should be ascending integers starting from 0 (i.e. 0, 1, 2, 3, 4...)
            if layer_num != self.sorted_layer_nums.index(layer_num):
                print(f"Invalid layer order: Missing, extra, mismatched, or invalid layers. These were the given layer numbers: {', '.join(tmp_num for tmp_num in self.sorted_layer_nums)}.")
                exit(-1)

            if layer_num != 0 and layer.activation_function is None:
                #print(f"Note: Layer number {layer_num} has no activation function set. It will not apply any functions to its inputs, but will still be multiplied by the layer connection's weights. Review your code if this is not intended behavior.")
                # print(f"No activation function for layer number {layer_num}: All layers, except the input (first) layer, MUST have an activation function set.")
                layer.activation_function = linear
                print(f"No activation function was given for layer number {layer_num}, as such, it will default to the identity linear function.")
                # exit(-1)

            self.layers_in_order.append(layer)

            if not is_loaded_model and layer_num > 0:
                previous_layer = self.layers_in_order[layer_num - 1]
                current_layer = self.layers_in_order[layer_num]

                if layer.activation_function in [linear, sigmoid, softmax]:
                    ## Xavier initialization
                    # weight_boundary = math.sqrt(6 / (input_layer.dim + output_layer.dim))
                    # weight = np.random.uniform(-weight_boundary, weight_boundary, (current_layer.dim, previous_layer.dim + 1))

                    '''
                    weight_means = [0 for _ in range(weight.shape[1] - 1)]
                    weight_stds = [0 for _ in range(weight.shape[1] - 1)]
                    for row in range(weight.shape[0]):
                        for col in range(weight.shape[1]):
                            if col == 0:
                                # Bias always starts at 0
                                weight[row, col] = 0
                            else:
                                weight_means[col - 1] += weight[row, col]

                    for row in range(weight.shape[0]):
                        for col in range(1, weight.shape[1]):
                            weight[row, col] -= weight_means[row, col - 1]

                    calc_stds(weight_stds)

                    for row in range(weight.shape[0]):
                        for col in range(1, weight.shape[1]):
                            weight[row, col] /= weight_stds[row, col - 1]
                            weight[row, col] /= 1 / sqrt(previous_layer.dim)
                    '''

                    if using_normal_distribution:
                        # Samples from normal distribution with mean = 0 and std = sqrt(2 / input_dim + output_dim)
                        # Then, if sampled weight is further than 2 standard deviations away, re-sample until it is within the range
                        std = math.sqrt(2 / (previous_layer.dim + current_layer.dim))
                        truncation_boundary = 2 * std
                        weight = np.random.normal(loc=0, scale=std, size=(current_layer.dim, previous_layer.dim + 1))

                        for row in range(weight.shape[0]):
                            for col in range(weight.shape[1]):
                                while weight[row, col] > truncation_boundary or weight[row, col] < -truncation_boundary:
                                    weight[row, col] = np.random.normal(loc=0, scale=std)

                    elif using_uniform_distribution:
                        weight_boundary = math.sqrt(6 / (previous_layer.dim + current_layer.dim))
                        weight = np.random.uniform(-weight_boundary, weight_boundary, (current_layer.dim, previous_layer.dim + 1))

                elif layer.activation_function in [relu]:
                    ## HE initialization

                    if using_normal_distribution:
                        # Samples from normal distribution with mean = 0 and std = sqrt(2 / input_dim + output_dim)
                        # Then, if sampled weight is further than 2 standard deviations away, re-sample until it is within the range
                        std = math.sqrt(2 / previous_layer.dim)
                        truncation_boundary = 2 * std

                        weight = np.random.normal(loc=0, scale=std, size=(current_layer.dim, previous_layer.dim + 1))

                        for row in range(weight.shape[0]):
                            for col in range(weight.shape[1]):
                                while weight[row, col] > truncation_boundary or weight[row, col] < -truncation_boundary:
                                    weight[row, col] = np.random.normal(loc=0, scale=std)
                            
                    elif using_uniform_distribution:
                        weight_boundary = math.sqrt(6 / previous_layer.dim)
                        weight = np.random.uniform(-weight_boundary, weight_boundary, (current_layer.dim, previous_layer.dim + 1))
                else:
                    ## Random numpy initializion (range [0, 1])
                    # As NDarray
                    weight = np.random.rand(current_layer.dim, previous_layer.dim + 1)
                    # weight = np.matrix(0, current_layer.dim, previous_layer.dim + 1)
                    weight = np.matrix(weight)

                # Re-initializing bias(es) to 0
                for row in range(weight.shape[0]):
                    weight[row, 0] = 0

                weights.append(weight)

        
        self.weights.append(weights)

        # Initialize class list to be integers in ascending order starting from 0, by default. (i.e. for a network with 4 output neurons [0, 1, 2, 3])
        # This gets overriden once a dataset is loaded or the user explicitly defines them, if he so chooses when prompted.
        if not is_loaded_model:
            for category in range(self.layers_in_order[-1].dim):
                self.class_list.append(str(category))

        self.ready_to_use = True

    # If model has been trained at least once and needs to be trained again, call this function to clear previous training info
    # `keep_weights`: If `True`, previous training's weights are kept, but removed otherwise (default behavior).
    def reset_training_data(self, keep_weights=False):
        if not self.ready_to_use:
            print(f"Error: Could not reset model's training data, it can only be done after `model.setup_done()` has been called (in order to validade layers).")
            exit(-1)

        self.layers_a = []
        self.layers_z = []

        if not keep_weights:
            self.weights = []

            weights = []
            for layer_num in self.sorted_layer_nums:
                if layer_num > 0:
                    previous_layer = self.layers_in_order[layer_num - 1]
                    current_layer = self.layers_in_order[layer_num]
                    # As NDarray
                    weight = np.random.rand(current_layer.dim, previous_layer.dim + 1)
                    # weight = np.matrix(0, current_layer.dim, previous_layer.dim + 1)
                    weight = np.matrix(weight)
                    weights.append(weight)

            self.weights.append(weights)


    def total_layers(self):
        return len(self.layers_in_order)
    
    def print_architecture(self, format=0):
        if not self.ready_to_use:
            print("Error: You need to call `model.setup_done()` first before printing the architecture, as it needs to be validated.")
            return

        if format != 0 and format != 1:
            print(f"Error: Could not print architecture, `format` must be 0 (printing each layer in a new line) or 1 (printing layers in a `->` chain), but was {format}.")
            return
        
        for layer in self.layers_in_order:
            final_layer_str = ""
            final_layer_str += f"(Layer {layer.num})"
            if layer.type == 0:
                final_layer_str += f" Input dimension: {layer.dim}"
            elif layer.type == 2:
                final_layer_str += f" Output dimension: {layer.dim}"
            else:
                final_layer_str += f" Hidden layer dimension: {layer.dim}"

            if layer.regularize:
                final_layer_str += f" | L2 regularization: {layer.l2_rate} | L1 regularization: {layer.l1_rate}"

            if layer.num > 0:
                final_layer_str += f" | Activation function: "
                if layer.activation_function == linear:
                    final_layer_str += "linear"
                elif layer.activation_function == relu:
                    final_layer_str += "relu"
                elif layer.activation_function == sigmoid:
                    final_layer_str += "sigmoid"
                elif layer.activation_function == softmax:
                    final_layer_str += "softmax"
                else:
                    final_layer_str += "unknown"
                
            if layer.type == 2:
                final_layer_str += " | Loss: "
                if self.loss_function == loss_mse:
                    final_layer_str += "Mean squared error"
                elif self.loss_function == loss_binary_crossentropy:
                    final_layer_str += "Binary cross-entropy"
                elif self.loss_function == loss_categorical_crossentropy:
                    final_layer_str += "Categorical cross-entropy"
                else:
                    final_layer_str += "UNKNOWN"

            if format == 0:
                print(final_layer_str)
            elif format == 1:
                if layer.type != 2:
                    final_layer_str += " -> "
                print(final_layer_str, end="")
        
        if format == 1:
            print()

        print(f"Normalized: {self.normalized}")

        if self.feature_list != []:
            print(f"Feature list: {', '.join(self.feature_list)}")

        if self.feature_types != []:
            print(f"Feature types: {', '.join(self.feature_types)}")

        if self.class_list != []:
            print(f"Class list: {', '.join(self.class_list)}")
        
        if self.feature_min_maxes != []:
            print(f"Feature min maxes: {self.feature_min_maxes}")

        if self.sample_features_mean != []:
            print(f"Samples' feature means: {self.sample_features_mean}")

        if self.sample_features_variance != []:
            print(f"Samples' feature variances: {self.sample_features_variance}")

        if self.sample_features_std != []:
            print(f"Samples' feature stds: {self.sample_features_std}")

    def print_weights(self):
        for layer_num, layer in enumerate(self.weights[-1]):
            print(f"(Layer {layer_num} weights): {layer}")
            print()


    # These two functions are defined like this because that's how I've been using it, this is not entirely accurate
    def is_regression(self):
        if self.layers_in_order[-1].activation_function == linear or self.layers_in_order[-1].activation_function == relu:
            return True
        else:
            return False
        
    def is_classification(self):
        if self.layers_in_order[-1].activation_function == sigmoid or self.layers_in_order[-1].activation_function == softmax:
            return True
        else:
            return False

    @staticmethod
    def fill_confusion_matrices(model, confusion_matrices, predicted_matrix, observed_matrix, positive_treshold=0.5):
        # If the last layer's activation function is a sigmoid function, treat it as a classification model and calculate the relevant metrics
        # Likely binary classification
        if model.is_classification() and model.layers_in_order[-1].dim == 1:
            observed_value = observed_matrix[0, 0]
            predicted_value = predicted_matrix[0, 0]
            
            if predicted_value >= positive_treshold:

                # False positive (predicted positive but is negative)
                if observed_value == 0:
                    confusion_matrices[0][0, 1] += 1

                # True positive (predicted positive and is positive)
                elif observed_value == 1:
                    confusion_matrices[0][0, 0] += 1
                else:
                    print(f"Error: Cannot properly calculate true/false positives/negatives. The observed values (at least in the test set) are neither 0 or 1, but should be.")

            elif predicted_value < positive_treshold:

                # True negative (predicted negative and is negative)
                if observed_value == 0:
                    confusion_matrices[0][1, 1] += 1

                # False negative (predicted negative but is positive)
                elif observed_value == 1:
                    confusion_matrices[0][1, 0] += 1
                else:
                    print(f"Error: Cannot properly calculate true/false positives/negatives. The observed values (at least in the test set) are neither 0 or 1, but should be.")

        # Likely multi-label classification
        elif model.layers_in_order[-1].activation_function == sigmoid and model.layers_in_order[-1].dim > 1:
            for label in range(model.layers_in_order[-1].dim):
                observed_value = observed_matrix[0, label]
                predicted_value = predicted_matrix[label, 0]

                if predicted_value >= positive_treshold:

                    # False positive (predicted positive but is negative)
                    if observed_value == 0:
                        confusion_matrices[label][0, 1] += 1

                    # True positive (predicted positive and is positive)
                    elif observed_value == 1:
                        confusion_matrices[label][0, 0] += 1
                    else:
                        print(f"Error: Cannot properly calculate true/false positives/negatives. The observed values (at least in the test set) are neither 0 or 1, but should be.")

                elif predicted_value < positive_treshold:

                    # True negative (predicted negative and is negative)
                    if observed_value == 0:
                        confusion_matrices[label][1, 1] += 1

                    # False negative (predicted negative but is positive)
                    elif observed_value == 1:
                        confusion_matrices[label][1, 0] += 1
                    else:
                        print(f"Error: Cannot properly calculate true/false positives/negatives. The observed values (at least in the test set) are neither 0 or 1, but should be.")

        # Likely multi-class classification
        elif model.layers_in_order[-1].activation_function == softmax and output_layer_dim > 1:        
            predicted_class = np.argmax(predicted_matrix)
            observed_class = np.argmax(observed_matrix)
            for category in range(model.layers_in_order[-1].dim):

                # True positive
                if predicted_class == category and observed_class == category:
                    confusion_matrices[category][0, 0] += 1

                # False negative
                elif predicted_class != category and observed_class == category:
                    confusion_matrices[category][1, 0] += 1

                # False positive
                elif predicted_class == category and observed_class != category:
                    confusion_matrices[category][0, 1] += 1

                # True negative
                elif predicted_class != category and observed_class != category:
                    confusion_matrices[category][1, 1] += 1
            
            # observed_value = observed_matrix[0, predicted_class]
            # predicted_value = predicted_matrix[predicted_class, 0

def calc_accuracy(tp, tn, fp, fn):
    if (tp + tn + fp + fn) != 0:
        return (tp + tn) / (tp + tn + fp + fn)
    else:
        return 1

def calc_precision(tp, fp):
    if (tp + fp) != 0:
        return tp / (tp + fp)
    else:
        return 1
    
def calc_recall(tp, fn):
    if (tp + fn) != 0:
        return tp / (tp + fn)
    else:
        return 1
    
def calc_f1score(precision, recall):
    if (precision + recall) != 0:
        return 2 * ((precision * recall) / (precision + recall))
    else:
        return 1
    
def calc_rsquared(ssr_predicted, ssr_mean):
    return 1 - (ssr_predicted / ssr_mean)

# Set the `class_metrics` argument to a list in order to fill it, ignore otherwise
# `confusion_matrices` (list) needs to be passed whenever `class_metrics` is a used (also a list)
def get_classification_micro_metrics(class_list, confusion_matrix, confusion_matrices=None, metrics=None, class_metrics=None):
    tp = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    tn = confusion_matrix[1, 1]

    micro_accuracy = calc_accuracy(tp, tn, fp, fn)
    micro_precision = calc_precision(tp, fp)
    micro_recall = calc_recall(tp, fn)
    micro_f1score = calc_f1score(micro_precision, micro_recall)

    if type(metrics) == dict:
        metrics["accuracy"] = micro_accuracy
        metrics["precision"] = micro_precision
        metrics["recall"] = micro_recall
        metrics["f1score"] = micro_f1score

    if type(class_metrics) == list and type(confusion_matrices) == list:
        for category in range(len(class_list)):
            class_micro_metrics = {}
            get_classification_micro_metrics(class_list, confusion_matrices[category], confusion_matrices, metrics=class_micro_metrics)
            class_metrics.append(class_micro_metrics)

# Set the `class_metrics` argument to a list in order to fill it, ignore otherwise
def get_classification_macro_metrics(class_list, confusion_matrices, metrics=None, class_metrics=None):
    macro_accuracy = 0
    macro_precision = 0
    macro_recall = 0
    macro_f1score = 0
    for category in range(len(class_list)):
        class_micro_metrics = {}
        get_classification_micro_metrics(class_list, confusion_matrices[category], confusion_matrices, metrics=class_micro_metrics)

        if type(class_metrics) == list:
            class_metrics.append(class_micro_metrics)

        macro_accuracy += class_micro_metrics["accuracy"]
        macro_precision += class_micro_metrics["precision"]
        macro_recall += class_micro_metrics["recall"]
        macro_f1score += class_micro_metrics["f1score"]

    macro_accuracy /= len(class_list)
    macro_precision /= len(class_list)
    macro_recall /= len(class_list)
    macro_f1score /= len(class_list)

    if type(metrics) == dict:
        metrics["accuracy"] = macro_accuracy
        metrics["precision"] = macro_precision
        metrics["recall"] = macro_recall
        metrics["f1score"] = macro_f1score

# Set the `class_metrics` argument to a list in order to fill it, ignore otherwise
def get_regression_micro_metrics(class_list, variation_in_models, variation_in_dependent_variables, metrics=None, class_metrics=None):

    # Divide by the number of dependent variables in order to get the mean variation, or leave as is?
    variation_in_model = sum(variation_in_models)
    variation_in_dependent_variable = sum(variation_in_dependent_variables)
    r_squared = calc_rsquared(variation_in_model, variation_in_dependent_variable)
    # r = math.sqrt(r_squared)

    if type(metrics) == dict:
        metrics["r_squared"] = r_squared
        # metrics["r"] = r

    if type(class_metrics) == list:
        for dependent_variable in range(len(class_list)):
            t_r_squared = calc_rsquared(variation_in_models[dependent_variable], variation_in_dependent_variables[dependent_variable])
            # t_r = math.sqrt(t_r_squared)

            class_metric = {}
            class_metric["r_squared"] = t_r_squared
            # class_metric["r"] = t_r
            class_metrics.append(class_metric)

# Set the `class_metrics` argument to a list in order to fill it, ignore otherwise
def get_regression_macro_metrics(class_list, variation_in_models, variation_in_dependent_variables, metrics=None, class_metrics=None):
    r_squared_avg = 0
    # r_avg = 0
    for dependent_variable in range(len(class_list)):
        r_squared = calc_rsquared(variation_in_models[dependent_variable], variation_in_dependent_variables[dependent_variable])
        # r = math.sqrt(r_squared)

        r_squared_avg += r_squared / len(class_list)
        # r_avg += r / len(dataset.class_list)

        if type(class_metrics) == list:     
            class_metric = {}
            class_metric["r_squared"] = r_squared
            # class_metric["r"] = r
            class_metrics.append(class_metric)

    if type(metrics) == dict:
        metrics["r_squared"] = r_squared_avg
        # metrics["r"] = r_avg


# (Given as `features_mean`) - Use training set (to avoid data leakage) or test set feature means? (Currently doing the latter)
def measure_model_on_dataset(model, samples, samples_dependent_values, features_mean, dependent_variables_mean, feature_list, class_list, model_metrics=dict, micro_metrics=dict, macro_metrics=dict, class_metrics=list, ones_padded=True):
    store_model_metrics = type(model_metrics) == dict
    store_micro_metrics = type(micro_metrics) == dict
    store_macro_metrics = type(macro_metrics) == dict
    store_class_metrics = type(class_metrics) == list

    if not store_model_metrics and not store_micro_metrics and not store_macro_metrics and not store_class_metrics:
        print("Error: Could not calculate performance metrics on given model and dataset, all of `model_metrics`, `micro_metrics`, `macro_metrics`, and `class_metrics`, were not given.")
        return
    
    dataset_cost = 0

    # Each matrix (one per class/label) is of the format
    #
    #    [ TP  FP ]
    #    [ FN  TN ]
    #
    #
    confusion_matrices = []

    # Fill out zeroed confusion matrices for each class for multilabel and multiclass classification models
    if model.is_classification():
        for category in range(len(class_list)):
            confusion_matrix = np.zeros((2, 2), dtype=np.uint64)
            confusion_matrix = np.matrix(confusion_matrix, dtype=np.uint64)
            confusion_matrices.append(confusion_matrix)

    # Aggregate results from test set into variables (true positives, true negatives, false positives, false negatives)
    # All samples are already prepended with 1 and normalized
    
    
    average_cost = 0

    if model.is_regression():
        # correlation_coefficients_per_feature = [[[0, [0, 0]] for _ in range(len(class_list))] for _ in range(len(feature_list))] # Each element is a list containing the correlation coefficient calculated with respect to the current feature, where the correlation coefficient is currently averaged over the number of classes)
        r_squareds_per_feature = [[[0, 0] for _ in range(len(class_list))] for _ in range(len(feature_list))]

    top_rsquared = 0
    bottom_rsquared = 0

    for sample_num in range(samples.shape[1]):
        sample = samples[:, sample_num]
        
        predicted_matrix = nn.predict(model, sample)
        observed_matrix = samples_dependent_values[sample_num, :]

        if model.is_regression():
            for feature in range(len(feature_list)):
                feature_mean = features_mean[feature]
                for dependent_variable in range(len(class_list)):
                    if ones_padded:
                        feature_value = sample[1 + feature, 0]
                    else:
                        feature_value = sample[feature, 0]

                    dependent_value = observed_matrix[0, dependent_variable]
                    predicted_value = predicted_matrix[dependent_variable, 0]
                    
                    dependent_variable_mean = dependent_variables_mean[dependent_variable]

                    # print(feature_value)
                    # print(feature_mean)
                    # print(dependent_value)
                    # print(dependent_variable_mean)
                    # print(predicted_value)
                    # exit(-1)

                    # top = (feature_value - feature_mean) * (dependent_value - model.sample_dependent_variables_mean[dependent_variable])
                    top = (feature_value - feature_mean) * (predicted_value - model.sample_dependent_variables_mean[dependent_variable])
                    bottom = [0, 0]

                    # bottom[0] = math.pow(feature_value - feature_mean, 2)
                    bottom[0] = math.pow(feature_value - feature_mean, 2)
                    # bottom[1] = math.pow(dependent_value - dependent_variable_mean, 2)
                    bottom[1] = math.pow(predicted_value - model.sample_dependent_variables_mean[dependent_variable], 2)
                
                    # correlation_coefficients_per_feature[feature][dependent_variable][0] += top
                    # correlation_coefficients_per_feature[feature][dependent_variable][1][0] += bottom[0]
                    # correlation_coefficients_per_feature[feature][dependent_variable][1][1] += bottom[1]

                    top_rsquared = math.pow(dependent_value - predicted_value, 2)
                    bottom_rsquared = math.pow(dependent_value - dependent_variable_mean, 2)

                    r_squareds_per_feature[feature][dependent_variable][0] += top_rsquared
                    r_squareds_per_feature[feature][dependent_variable][1] += bottom_rsquared

        elif model.is_classification():
            Model.fill_confusion_matrices(model, confusion_matrices, predicted_matrix, observed_matrix)
        
        sample_loss = nn.sample_loss(model, predicted_matrix, observed_matrix)
        average_cost += sample_loss
            
        '''
        if output_layer_dim > 1:
            for row in range(1, output_layer_dim):
                    predicted_value = max(predicted_value, predicted_matrix[row, 0])
        '''

    if model.loss_function == loss_mse or model.loss_function == loss_binary_crossentropy or model.loss_function == loss_categorical_crossentropy:
        average_cost /= samples.shape[1]

    dataset_cost = average_cost

    if store_model_metrics:
        model_metrics["total_cost"] = dataset_cost
        model_metrics["total_samples"] = samples.shape[1]
        if model.is_regression() and model.loss_function == loss_mse:
            model_metrics["rmse"] = math.sqrt(dataset_cost)

    # Calculates performance metrics for classification models
    if model.is_classification():
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for confusion_matrix in confusion_matrices:
            true_positives += confusion_matrix[0, 0]
            true_negatives += confusion_matrix[1, 1]
            false_negatives += confusion_matrix[1, 0]
            false_positives += confusion_matrix[0, 1]

        confusion_matrix = np.matrix(
            [
                [true_positives, false_positives],
                [false_negatives, true_negatives]
            ], dtype=np.float64
        )

        # Micro avgs calculation
        ##############################
        if store_micro_metrics:

            if store_class_metrics:
                get_classification_micro_metrics(class_list, confusion_matrix, confusion_matrices, metrics=micro_metrics, class_metrics=class_metrics)
            else:
                get_classification_micro_metrics(class_list, confusion_matrix, confusion_matrices, metrics=micro_metrics, class_metrics=None)

        if output_layer_dim > 1:
            ###############################

            # Macro avgs calculation
            ###############################


            if store_class_metrics and store_macro_metrics:

                if class_metrics == []:
                    get_classification_macro_metrics(class_list, confusion_matrices, metrics=macro_metrics, class_metrics=class_metrics)
                else:
                    get_classification_macro_metrics(class_list, confusion_matrices, metrics=macro_metrics, class_metrics=None)

            elif store_class_metrics and not store_macro_metrics:

                if class_metrics == []:
                    get_classification_macro_metrics(class_list, confusion_matrices, metrics=None, class_metrics=class_metrics)

            elif not store_class_metrics and store_macro_metrics:
                get_classification_macro_metrics(class_list, confusion_matrices, metrics=macro_metrics, class_metrics=None)
                

            ############################
                
        if store_model_metrics:
            model_metrics["true_positives"] = true_positives
            model_metrics["true_negatives"] = true_negatives
            model_metrics["false_positives"] = false_positives
            model_metrics["false_negatives"] = false_negatives

    ## CALCULATE REGRESSION MODEL METRICS
    elif model.is_regression():

        '''
        if store_micro_metrics:
            get_regression_micro_metrics(variation_in_models, variation_in_dependent_variables, metrics=micro_metrics)

        if output_layer_dim > 1:

            if store_class_metrics and store_macro_metrics:
                get_regression_macro_metrics(variation_in_models, variation_in_dependent_variables, metrics=macro_metrics, class_metrics=class_metrics)
            elif store_class_metrics and not store_macro_metrics:
                get_regression_macro_metrics(variation_in_models, variation_in_dependent_variables, class_metrics=class_metrics)
            elif not store_class_metrics and store_macro_metrics:
                get_regression_macro_metrics(variation_in_models, variation_in_dependent_variables, metrics=macro_metrics)
        '''

        if model.is_regression():
            correlation_coefficients_averages = [] # Correlation coefficients for each feature averaged over the dependent variables
            r_squareds_averages = []

            # print(correlation_coefficients_per_feature)

            for feature in range(len(feature_list)):
                # correlation_coefficients_per_dependent_variable = correlation_coefficients_per_feature[feature]
                r_squareds_per_dependent_variable = r_squareds_per_feature[feature]

                # feature_correlation_coefficient_average = 0
                feature_r_squared_average = 0

                for dependent_variable in range(len(class_list)):
                    # correlation_coefficient = correlation_coefficients_per_dependent_variable[dependent_variable]
                    r_squared = r_squareds_per_dependent_variable[dependent_variable]

                    # top = correlation_coefficient[0]
                    # bottom = correlation_coefficient[1]

                    # correlation_coefficient = top / math.sqrt(bottom[0] * bottom[1])
                    # feature_correlation_coefficient_average += correlation_coefficient

                    top_rsquared = r_squared[0]
                    bottom_rsquared = r_squared[1]

                    r_squared = 1 - (top_rsquared / bottom_rsquared)
                    feature_r_squared_average += r_squared

                # feature_correlation_coefficient_average /= len(class_list)
                feature_r_squared_average /= len(class_list)

                # correlation_coefficients_averages.append(feature_correlation_coefficient_average)
                r_squareds_averages.append(feature_r_squared_average)
            
            # correlation_coefficient_average = sum(correlation_coefficients_averages) / len(feature_list) # Correlation coefficient from averaging over the features, after averaging over the dependent variables for each feature
            r_squared_average = sum(r_squareds_averages) / len(feature_list)
            micro_metrics["r_squared"] = r_squared_average
            # micro_metrics["r"] = correlation_coefficient_average
            # print(f"R^2: {micro_metrics['r_squared']}")
            # exit(-1)

            # print(f"Per feature: {correlation_coefficients_per_feature}")
            # print(f"Per feature, averaged: {correlation_coefficients_averages}")
            # print(f"Final average: {correlation_coefficient_average}")
    else:
        print("Error: Could not properly identify whether the model was a regression or classification model due to an unknown activation function for the output layer. As such, performance metrics could not be calculated.")
        exit(-1)

def print_model_metrics(model, model_metrics=None, micro_metrics=None, macro_metrics=None, class_metrics=None):
    if model_metrics is None and micro_metrics is None and macro_metrics is None and class_metrics is None:
        return

    use_model_metrics = type(model_metrics) == dict
    use_micro_metrics = type(micro_metrics) == dict
    use_macro_metrics = type(macro_metrics) == dict
    use_class_metrics = type(class_metrics) == list

    if use_model_metrics:
        print(f"Total cost: {model_metrics['total_cost']} | Total samples: {model_metrics['total_samples']}")
        if model.is_classification():
            print(f"True positives: {model_metrics['true_positives']} | True negatives: {model_metrics['true_negatives']} | False positives: {model_metrics['false_positives']} | False negatives: {model_metrics['false_negatives']}")
        elif model.is_regression() and model.loss_function == loss_mse:
            print(f"MSE: {model_metrics['total_cost']} | RMSE: {model_metrics['rmse']}")

    if model.layers_in_order[-1].dim == 1:
        if model.is_classification():
            print(f"Precision: {micro_metrics['precision']} | Accuracy: {micro_metrics['accuracy']} | Recall: {micro_metrics['recall']} | F1-score: {micro_metrics['f1score']}")
        elif model.is_regression():
            print(f"R^2: {micro_metrics['r_squared']}")

        return

    if use_class_metrics:
        for class_idx, class_metric in enumerate(class_metrics):
            if model.is_classification():
                print(f"Class {class_idx} ({model.class_list[class_idx]}) - Precision: {class_metric['precision']} | Accuracy: {class_metric['accuracy']} | Recall: {class_metric['recall']} | F1-score: {class_metric['f1score']}")
            elif model.is_regression():
                print(f"Class {class_idx} ({model.class_list[class_idx]}) - R^2: {class_metric['r_squared']}")
    
    if use_micro_metrics:
        if model.is_classification():
            print(f"(MICRO AVGS) Precision: {micro_metrics['precision']} | Accuracy: {micro_metrics['accuracy']} | Recall: {micro_metrics['recall']} | F1-score: {micro_metrics['f1score']}")
        elif model.is_regression():
            print(f"(MICRO AVGS) R^2: {micro_metrics['r_squared']}")
    
    if use_macro_metrics:
        if model.is_classification():
            print(f"(MACRO AVGS) Precision: {macro_metrics['precision']} | Accuracy: {macro_metrics['accuracy']} | Recall: {macro_metrics['recall']} | F1-score: {macro_metrics['f1score']}")
        elif model.is_regression():
            print(f"(MACRO AVGS) R^2: {macro_metrics['r_squared']}")

def load_model(file):
    content = ""

    with open(MODELS_FOLDER + file + ".json") as f:
        content = f.read()

    json_parsed = json.loads(content)
    layer_nums = list(json_parsed["layers"].keys())
    layers = {}

    for layer_num in layer_nums:
        layer = json_parsed["layers"][layer_num]

        match layer["activation"]:
            case "linear":
                activation = linear
            case "relu":
                activation = relu
            case "sigmoid":
                activation = sigmoid
            case "softmax":
                activation = softmax
            case _:
                activation = None

        # Input layer
        if layer_num == "0":
            layer_type = 0

        # Hidden layer
        elif layer_num == layer_nums[-1]:
            layer_type = 2

        # Output layer
        else:
            layer_type = 1

        regularization_status = layer["regularization"]
        regularize = regularization_status["regularize"]
        l1_rate = regularization_status["l1_rate"]
        l2_rate = regularization_status["l2_rate"]

        layer = Layer(int(layer_num), int(layer["dimension"]), layer_type)
        layer.regularize = regularize
        layer.l1_rate = l1_rate
        layer.l2_rate = l2_rate

        layer.activation_function = activation

        layers[int(layer_num)] = layer

    loss = json_parsed["loss"]
    match loss:
        case "mse":
            loss = loss_mse
        case "binary_crossentropy":
            loss = loss_binary_crossentropy
        case "categorical_crossentropy":
            loss = loss_categorical_crossentropy
        case _:
            None

    model = Model()
    model.feature_list = json_parsed["feature_list"]
    model.feature_types = json_parsed["feature_types"]

    if "str" in model.feature_types:
        raise ValueError(f"Error: Could not load model {file}.json. At least one dependent variable is a string, but a way to store and load dictionaries for word<->number conversions when dealing with strings are yet to be implemented. For now you can only make predictions on a model with string as a dependent variable when training, directly after training finishes, and only during the script is execution. The model architecture and weights can still be saved however, just keep in mind that you won't be able to predict with this script if you try to load it")
    
    model.class_list = json_parsed["class_list"]
    model.feature_min_maxes = json_parsed["feature_min_maxes"]
    model.sample_features_mean = json_parsed["sample_features_mean"]
    model.sample_dependent_variables_mean = json_parsed["sample_dependent_variables_mean"]
    model.sample_features_variance = json_parsed["sample_features_variance"]
    model.sample_features_std = json_parsed["sample_features_std"]

    model.loss_function = loss
    model.layers = layers

    model.is_image_model = json_parsed["is_image_model"]
    if model.is_image_model:
        model.image_dim = json_parsed["image_dim"]
        model.image_bit_depth = json_parsed["image_bit_depth"]

    model.setup_done(is_loaded_model=True)

    json_layer_weights = json_parsed["weights"]
    layer_weights = list(json_layer_weights.keys())

    weights = []

    for layer in layer_weights:
        layer_weights = json_layer_weights[layer]
        '''
        rows = len(list(layer_weights.keys()))
        cols = len(layer_weights["0"])

        layer_weight = np.zeros((rows, cols), dtype=np.float64)
        layer_weight = np.matrix(layer_weight)

        for row in range(rows):
            for col in range(cols):
                layer_weight[row, col] = layer_weights[str(row)][col]
        '''
        layer_weight = np.matrix(layer_weights, dtype=np.float64)

        weights.append(layer_weight)

    model.weights = []
    model.weights.append(weights)

    model.normalized = json_parsed["normalized"]

    return model

def save_model(model, file):

    if not is_image_model:
        model_data_obj = {"normalized": model.normalized, "is_image_model": model.is_image_model, "layers": {}, "loss": None, "class_list": model.class_list, "feature_list": model.feature_list, "feature_types": model.feature_types, "feature_min_maxes": model.feature_min_maxes, "sample_features_mean": model.sample_features_mean, "sample_dependent_variables_mean": model.sample_dependent_variables_mean, "sample_features_variance": model.sample_features_variance, "sample_features_std": model.sample_features_std,  "weights": {}}
    else:
        model_data_obj = {"normalized": model.normalized, "is_image_model": model.is_image_model, "image_dim": model.image_dim, "image_bit_depth": model.image_bit_depth, "layers": {}, "loss": None, "class_list": model.class_list, "feature_list": model.feature_list, "feature_types": model.feature_types, "feature_min_maxes": model.feature_min_maxes, "sample_features_mean": model.sample_features_mean, "sample_dependent_variables_mean": model.sample_dependent_variables_mean, "sample_features_variance": model.sample_features_variance, "sample_features_std": model.sample_features_std, "weights": {}}

    for layer in model.layers_in_order:
        if layer.num > 0:
            if layer.activation_function == linear:
                layer_activation_function = "linear"
            elif layer.activation_function == relu:
                layer_activation_function = "relu"
            elif layer.activation_function == sigmoid:
                layer_activation_function = "sigmoid"
            elif layer.activation_function == softmax:
                layer_activation_function = "softmax"

            model_data_obj["layers"][str(layer.num)] = {
                "dimension": layer.dim,
                "activation": layer_activation_function,
                "regularization": {
                    "regularize": layer.regularize,
                    "l1_rate": layer.l1_rate,
                    "l2_rate": layer.l2_rate
                }
            }

        else:
            model_data_obj["layers"][str(layer.num)] = {
                "dimension": layer.dim,
                "activation": None,
                "regularization": {
                    "regularize": False,
                    "l1_rate": 0,
                    "l2_rate": 0
                }
            }

    loss = ""
    if model.loss_function == loss_mse:
        loss = "mse"
    elif model.loss_function == loss_binary_crossentropy:
        loss = "binary_crossentropy"
    elif model.loss_function == loss_categorical_crossentropy:
        loss = "categorical_crossentropy"

    model_data_obj["loss"] = loss

    model_weights = model.weights[-1]
    for layer_connection, layer_weight in enumerate(model_weights):
        model_data_obj["weights"][str(layer_connection)] = layer_weight.tolist()

    with open(MODELS_FOLDER + file + ".json", "w") as f:
        json.dump(model_data_obj, f)

    print(f"Model saved to file '{file}.json'.")

def predict_prompt(model, predicting_after_training=True):
    input_matrix = np.zeros((1 + model.layers_in_order[0].dim, 1), dtype=np.float64)
    input_matrix = np.matrix(input_matrix)
    input_matrix[0, 0] = 1

    predicting = True
    while predicting:
        invalid_value = False

        if model.is_image_model:
            while True:
                image = input("Name an image to predict (within the 'images/predict' folder, also include the file extension): ")
                if os.path.isfile(IMAGES_FOLDER + "predict/" + image):
                    input_matrix[1:, 0] = get_image_pixel_matrix(IMAGES_FOLDER + "predict/", image, model=model)
                    break
                else:
                    print(f"Could not find image @ `{IMAGES_FOLDER + 'predict/' + image}`")
                    continue

        if predicting_after_training:
            for feature_num, feature_name in enumerate(model.feature_list):
                feature_type = model.feature_types[feature_num]

                if not model.is_image_model:
                    feature_value = input(f"{feature_name} ({feature_type}): ")
                elif model.is_image_model:
                    feature_value = input_matrix[1 + feature_num, 0]

                if feature_type == "str":
                    print("NOTE: As it stands, the string values are random integers picked when loading the dataset. Meaning predictions will only work once, when the model is trained and finally used to predict, as the script currently doesn't store a str -> int conversion table.")
                    print("As of right now this model/prediction is only used once, since I currently don't keep track of a dictionary for storing the str -> int conversions.")
                    try:
                        feature_value = dataset.string_to_num(feature_value)
                    except:
                        print(f"Error: Could not convert string '{feature_value}' into a number. Make sure this string is present in the dataset.")
                        invalid_value = True
                        break

                    # String_to_num already retrieves the final normalized value, so there is no need to convert it as is done with numbers below
                elif feature_type == "float":
                    try:
                        feature_value = float(feature_value)

                        if model.normalized:

                            feature_value = Dataset.normalize_helper(feature_value, model.feature_min_maxes[feature_num][0], model.feature_min_maxes[feature_num][1], -1, 1)

                            # if not model.is_image_model:
                            feature_value -= model.sample_features_mean[feature_num]

                            if model.sample_features_std[feature_num] > 0.0001:
                                feature_value /= model.sample_features_std[feature_num]

                            
                            # print(f"Normalized input: {feature_value}")
                            
                    except Exception as e:
                        print(f"Error: Could not convert value '{feature_value}' into a number. Make sure this value is a valid number.")
                        invalid_value = True
                        break
                
                input_matrix[1 + feature_num, 0] = feature_value

            if invalid_value:
                continue
        else:
            for feature_num, feature_name in enumerate(model.feature_list):
                feature_type = model.feature_types[feature_num]

                if not model.is_image_model:
                    feature_value = input(f"{feature_name} ({feature_type}): ")
                elif model.is_image_model:
                    feature_value = input_matrix[1 + feature_num, 0]

                if feature_type == "str":
                    print("NOTE: As it stands, the string values are random integers picked when loading the dataset. Meaning predictions will only work once, when the model is trained and finally used to predict, as the script currently doesn't store a str -> int conversion table.")
                    print("As of right now this model/prediction is only used once, since I currently don't keep track of a dictionary for storing the str -> int conversions.")
                    try:
                        feature_value = model.string_to_num(feature_value)
                    except:
                        print(f"Error: Could not convert string '{feature_value}' into a number. Make sure this string is present in the dataset.")
                        invalid_value = True
                        break

                    # String_to_num already retrieves the final normalized value, so there is no need to convert it as is done with numbers below
                elif feature_type == "float":
                    try:
                        feature_value = float(feature_value)

                        if model.normalized:

                            feature_value = Dataset.normalize_helper(feature_value, model.feature_min_maxes[feature_num][0], model.feature_min_maxes[feature_num][1], -1, 1)

                            # if not model.is_image_model:
                            feature_value -= model.sample_features_mean[feature_num]

                            if model.sample_features_std[feature_num] > 0.00001:
                                feature_value /= model.sample_features_std[feature_num]


                        
                    except:
                        print(f"Error: Could not convert value '{feature_value}' into a number. Make sure this value is a valid number.")
                        invalid_value = True
                        break
                
                input_matrix[1 + feature_num, 0] = feature_value

            if invalid_value:
                continue

        # bias = nn.weights[-1][0][0, 0]
        # theta = nn.weights[-1][0][0, 1]
        # nn.weights[-1][0][0, 0] = theta
        # nn.weights[-1][0][0, 1] = bias

        prediction_matrix = nn.predict(model, input_matrix)
        # print(f"Input matrix: {input_matrix}")

        if model.layers_in_order[-1].dim == 1:
            predicted_value = prediction_matrix[0, 0]
            print(f"Predicted value ({model.class_list[0]}): {predicted_value}")
        else:
            for class_num, class_name in enumerate(model.class_list):
                prediction_value = prediction_matrix[class_num, 0]
                print(f"Predicted value (class {class_num} | {class_name}): {prediction_value}")

            if model.layers_in_order[-1].activation_function == softmax:
                print(f"Predicted class: {model.class_list[np.argmax(prediction_matrix)]}")
                print(f"Predict value: {prediction_matrix[np.argmax(prediction_matrix), 0]}")

            if model.is_image_model:
                plot_image_from_sample(model,input_matrix[1:, :])

        while True:
            result = input("Predict more? (y/n) ")

            if result == 'y':
                break
            elif result == 'n':
                predicting = False
                break

        ## HOW SHOULD I INTERPRET THE PREDICTION MATRIX IN EACH SPECIFIC CASES FOR CLASSIFICATION MODELS, AND REGRESSION MODELS? SHOULD I LEAVE IT UP TO THE USER TO INTERPRET THAT INFORMATION?
        ## FOR CLASSIFICATION MODELS, GO THROUGH EACH DEPENDENT VARIABLE AND PRINT THE RESPECTIVE PROBABILITY

        ## CLASSIFICATION
        ## i.e. 
        ## {dependent_var} probability: {prediction_matrix[respective_dependent_var, 0]}
        ##
        ## hypertension probability: 0.6
        ## diabetes probability: 0.7
        ## heart_disease probability: 0.8
        ## cancer probability: 0.2
        ## tooth_problems probability: 0.3

        ## REGRESSION
        ## i.e. 
        ## {dependent_var}: {prediction_matrix[respective_dependent_var, 0]}
        ##
        ## price: 440000
        ## tax: 25000
        ## hoa: 400
        ## rent: 5000

        ## ALLOW USER TO INPUT THRESHOLD FOR DEPENDENT VARIABLES FOR TRUE OR FALSE IF CLASSIFICATION MODEL
        '''
        prediction = prediction_matrix[0, 0]

        print(f"Predicted matrix: {prediction_matrix}")

        # If considered a classification model
        if output_layer_dim > 1 and nn.layers_in_order[-1].activation_function == sigmoid:
            predicted_class = 0
            for row in range(1, output_layer_dim):
                predicted_class_value = prediction_matrix[row, 0]
                if predicted_class_value > prediction:
                    prediction = predicted_class_value
                    predicted_class = row
        '''

# Only normalizes if model.normalized == True
def mean_n_variance_normalize(model, samples, update_min_maxes=True, ignore_bias=True):
    if model.is_image_model and update_min_maxes:
        for feature in range(len(model.feature_list)):
            model.feature_min_maxes[feature][0] = 0
            model.feature_min_maxes[feature][1] = 255

        return

    # if not model.is_image_model:
    for entry in range(samples.shape[1]):
        for feature_idx, feature in enumerate(model.feature_list):
            if ignore_bias:
                entry_old_value = samples[1 + feature_idx, entry]
            else:
                entry_old_value = samples[feature_idx, entry]

            if not update_min_maxes and model.normalized:
                if ignore_bias:
                    samples[1 + feature_idx, entry] -= model.sample_features_mean[feature_idx]

                    # No need to divide if it's already 0, also ignores if std (denominator) is 0
                    if samples[1 + feature_idx, entry] != 0 and model.sample_features_std[feature_idx] > 0.00001:
                        samples[1 + feature_idx, entry] /= model.sample_features_std[feature_idx]
                else:
                    samples[feature_idx, entry] -= model.sample_features_mean[feature_idx]

                    # No need to divide if it's already 0, also ignores if std (denominator) is 0 (is the case only if all samples are 0)
                    if samples[feature_idx, entry] != 0 and model.sample_features_std[feature_idx] > 0.00001:
                        samples[feature_idx, entry] /= model.sample_features_std[feature_idx]

                if model.feature_types[feature_idx] == "str":
                    for key in model.nan_values:
                        if model.nan_values[key] == entry_old_value:
                            if ignore_bias:
                                model.nan_values[key] = samples[1 + feature_idx, entry]
                            else:
                                model.nan_values[key] = samples[feature_idx, entry]

            if ignore_bias:
                entry_new_value = samples[1 + feature_idx, entry]
            else:
                entry_new_value = samples[feature_idx, entry]

            if update_min_maxes:
                model.feature_min_maxes[feature_idx][0] = min(entry_new_value, model.feature_min_maxes[feature_idx][0])
                model.feature_min_maxes[feature_idx][1] = max(entry_new_value, model.feature_min_maxes[feature_idx][1])
    
    '''
    else:
        for feature_idx in range(model.layers_in_order[0].dim):
            model.feature_min_maxes[feature_idx][0] = 0
            model.feature_min_maxes[feature_idx][1] = 255
    '''

# Puts feature in [-1, 1] range (does not mean and variance normalize)
def new_normalize(model, samples, ignore_bias=True):
    # Ignores last feature (the dependent variable y to be predicted)
    for entry in range(samples.shape[1]):
        for feature_idx in range(len(model.feature_list)):
            if ignore_bias:
                entry_old_value = samples[1 + feature_idx, entry]
            else:
                entry_old_value = samples[feature_idx, entry]

            normalized_value = Dataset.normalize_helper(
                entry_old_value,
                model.feature_min_maxes[feature_idx][0],
                model.feature_min_maxes[feature_idx][1],
                -1,
                1,
                )
            
            if model.feature_types[feature_idx] == "str":
                for key in model.nan_values.keys():
                    if model.nan_values[key] == entry_old_value:
                        model.nan_values[key] = normalized_value
            
            if ignore_bias:
                samples[1 + feature_idx, entry] = normalized_value
            else:
                samples[feature_idx, entry] = normalized_value

def color_is_similar(color_a, color_b):
    r_diff = math.pow(color_a[0] - color_b[0], 2)
    g_diff = math.pow(color_a[1] - color_b[1], 2)
    b_diff = math.pow(color_a[2] - color_b[2], 2)
    # Ignoring alpha as it's always 1
    diff = math.sqrt(r_diff + g_diff + b_diff)
    
    if diff > 0.40:
        return False
    else:
        return True

# Returns the pixel array for images (if black and white is True, return 0 or 1 for each pixel, otherwise return RGB array)
def get_image_pixel_matrix(folder, image, model=None):
    img = Image.open(folder + image)
    pixels = list(img.getdata())
    parsed_pixels = []
    for pixel in pixels:
        if type(pixel) == int or type(pixel) == float:
            parsed_pixels.append(pixel)
        else:
            if model != None:
                if model.image_bit_depth == 1 or model.image_bit_depth == 8:
                    parsed_pixels.append(pixel[0])
                else:
                    for color_channel in pixel:
                        parsed_pixels.append(color_channel)
            else:
                for color_channel in pixel:
                    parsed_pixels.append(color_channel)

        # if uses_grayscale_images:
        #     if type(pixel) == list or type(pixel) == tuple:
        #         # parsed_pixels.append(pixel[0] / 255.0)
        #         # Not dividing by 255.0 in order to let user choose whether to normalize data or not at runtime
        #         parsed_pixels.append(pixel[0])
        #     elif type(pixel) == int or type(pixel) == float:
        #         # parsed_pixels.append(pixel / 255.0)
        #         # Not dividing by 255.0 in order to let user choose whether to normalize data or not at runtime
        #         parsed_pixels.append(pixel)
        # else:
        #     for color_channel in pixel:
        #         parsed_pixels.append(color_channel)

    parsed_pixels = np.matrix(parsed_pixels).transpose()
    return parsed_pixels

# Prints the (square) pixel array (in order to check whether it properly draws a square black and white image through text)
def print_bw_image_pixels(pixel_matrix):
    pixel_total = pixel_matrix.shape[0]
    dim = int(round(math.sqrt(pixel_matrix.shape[0])))
    print(pixel_matrix.shape)
    print(dim)
    for pixel in range(pixel_total):
        print(pixel_matrix[pixel, 0], end="")
        if (pixel + 1) % dim == 0:
            print('')

        # row = "".join(str(pixel) for pixel in pixels[dim * i : (dim * i) + dim])
        # for col in range()
        # print(pixel_matrix[], end="")
        # print(row)
        # print('\n')

# It is assumed that all images are square and the number of labels exactly match the number of images in the given folder
# The labels go into the file `labels.txt` and must be on the same folder as all of the images within the file, and vice-versa for the images
# Each sample label is separated by a new line, and it is formatted as `filename,label`, where `label` is a number.
# `unique_dataset` ignores the setup for re-initializing the feature list, class list, optionally giving names to each class among other things if False. Otherwise it stores and returns it.
def load_image_dataset(folder, model=None, unique_dataset=True):
    # Line below gets files in alphanumerical order based on filename
    # files_in_folder = list(sorted(os.listdir(folder), key=len)) 

    files_in_folder = os.listdir(folder)
    images_in_folder = list(filter(lambda file: True if file[-3:] in ["png", "jpg"] else False, files_in_folder))
    total_images = len(images_in_folder)
    pixel_lists = []

    with open(folder + "labels.txt", "r") as f:
        label_data = list(map(lambda line: line.replace('\n', ''), open(folder + "labels.txt", "r").readlines()))
        f.close()

    parsed_label_data = {}
    for label in label_data:
        if label != '' and label.index(',') != -1:
            label_pair = label.split(',')
            image_filename = label_pair[0]
            image_label = label_pair[1]
            parsed_label_data[image_filename] = int(image_label)

    parsed_label_data.values()

    labels = np.matrix(np.zeros((total_images, model.layers_in_order[-1].dim)))

    for image_idx, image_filename in enumerate(images_in_folder):
        image_pixels_matrix = get_image_pixel_matrix(folder, image_filename, model)

        if image_filename == images_in_folder[0] :
            pixel_count = image_pixels_matrix.shape[0]

            if not loaded_model and unique_dataset:
                tmp_image = Image.open(folder + image_filename)
                image_dim = tmp_image.size
                match tmp_image.mode:
                    case '1':
                        image_bit_depth = 1
                    case 'L':
                        image_bit_depth = 8
                    case 'P':
                        image_bit_depth = 8
                    case 'RGB':
                        image_bit_depth = 24
                    case 'RGBA':
                        image_bit_depth = 32
                    case 'CMYK':
                        image_bit_depth = 32
                    case 'YCbCr':
                        image_bit_depth = 24
                    case 'LAB':
                        image_bit_depth = 24
                    case 'HSV':
                        image_bit_depth = 24
                    case 'I':
                        image_bit_depth = 32
                    case 'F':
                        image_bit_depth = 32
                    case _:
                        print(tmp_image.mode)
                        print("Error: Could not load image dataset, image depth is not 8, 24, or 32.")
                        exit(-1)

                feature_list = [f"p{i}" for i in range(pixel_count)]
                feature_types = ["float"] * pixel_count

                if model.layers_in_order[-1].dim == 1:
                    class_list = ["0"]
                else:
                    print(f"This image model has {model.layers_in_order[-1].dim} classes. The default behavior for displaying/storing classes' metadata is as their respective integer identifiers (i.e. [0, 1, 2]).")
                    print("Note: This is only relevant for readability/interepration (i.e. reading model configuration files and infering). It doesn't change internal behavior.")
                    while True:
                        result = input("Do you wish to give each class a name (i.e. 'dog', 'cat', 'horse')? (y/n) ")
                        if result == 'y':
                            class_list = []
                            for label in range(model.layers_in_order[-1].dim):
                                class_name = input(f"Name for label '{label}': ").strip()
                                class_list.append(class_name)

                            break
                        elif result == 'n':
                            class_list = model.class_list
                            break

            samples = np.ones((1 + pixel_count, len(images_in_folder)))
            samples = np.matrix(samples)

        samples[1:, image_idx] = image_pixels_matrix

        '''
        if image == images_in_folder[0]:
            pixel_count = len(image_pixels)
            image_dim = int(round(math.sqrt(pixel_count)))

            feature_list = [f"p{i}" for i in range(pixel_count)]
            class_list = ["label"]

            samples = np.ones((1 + pixel_count, len(images_in_folder)))
            samples = np.matrix(samples)

        for pixel_idx, pixel in enumerate(image_pixels):
            samples[1 + pixel_idx, image_idx] = pixel


        pixel_lists.append(image_pixels)
        '''

        category = parsed_label_data[image_filename]
        if model.layers_in_order[-1].dim == 1:
            labels[image_idx, 0] = category
        else:
            labels[image_idx, category] = 1

    if not loaded_model and unique_dataset:
        return (samples, labels, feature_list, feature_types, class_list, image_dim, image_bit_depth)
    else:
        return (samples, labels, None, None, None, None)
    
def plot_image_from_sample(model, sample):
    tmp_sample = sample

    # Manually re-normalizing sample from range [-1, 1] to range [0, 1], as matplotlib simply ignores values below 0.
    if model.normalized:
        # tmp_sample = (sample + 1) / 2
        for feature in range(sample.shape[0]):

            if model.sample_features_std[feature] > 0.00001:
                tmp_sample[feature, 0] = (sample[feature, 0] * model.sample_features_std[feature]) + model.sample_features_mean[feature]
            else:
                tmp_sample[feature, 0] = sample[feature, 0] + model.sample_features_mean[feature]

            tmp_sample[feature, 0] = Dataset.normalize_helper(tmp_sample[feature, 0], -1, 1, 0, 1)

    # Or simply divide by 255 if non-normalized (is in the default range of [0, 255])
    else:
        tmp_sample = sample / np.max(model.feature_min_maxes)

    

    if model.image_bit_depth == 1 or model.image_bit_depth == 8:
        parsed_sample = np.zeros((model.image_dim[0], model.image_dim[1]))
        parsed_sample = np.matrix(parsed_sample)
        
        for row in range(model.image_dim[0]):
            for col in range(model.image_dim[1]):
                parsed_sample[row, col] = tmp_sample[(row * model.image_dim[1]) + col, 0]

        plt.imshow(parsed_sample, cmap="grey")

    
    

    else:
        width, height = model.image_dim
        pixels = []
        for i in range(tmp_sample.shape[0]):
            if model.normalized:
                # pixels.append(Dataset.normalize_helper(sample[i, 0], -1, 1, 0, 255))
                pixels.append(tmp_sample[i, 0])
            else:
                pixels.append(tmp_sample[i, 0])

        if model.image_bit_depth == 24:
            pixel_array = np.array(pixels).reshape((height, width, 3))
        elif model.image_bit_depth == 32:
            pixel_array = np.array(pixels).reshape((height, width, 4))
        plt.imshow(pixel_array)

    plt.show()

def randomize_dataset(samples, dependent_values):
    for idx in range(samples.shape[1]):
        random_pos = random.randint(0, samples.shape[1] - 1)
        current_sample = samples[:, idx]
        current_dependent_values = dependent_values[idx, :]

        sample_to_swap = samples[:, random_pos]
        dependent_values_to_swap = dependent_values[random_pos, :]

        samples[:, random_pos] = current_sample
        dependent_values[random_pos, :] = current_dependent_values

        samples[:, idx] = sample_to_swap
        dependent_values[idx, :] = dependent_values_to_swap

MODELS_FOLDER = "models/"
nn = Network()
model = Model()
loaded_model = False
while True:
    result = input("Load model? (y/n) ")
    if result == 'y':
        while True:
            result = input("Load model from file (exclude extension, will be .json): ").strip()
            if not os.path.isfile(MODELS_FOLDER + result + ".json"):
                print(f"File '{result}.json', file could not be found within (relative) folder {MODELS_FOLDER[:-1]}.")
                continue

            loaded_model = True

            # Loads model from json file
            model = load_model(result)
            print(f"Model loaded from file '{result}.json'")
            model.print_architecture()
            print()
            model.print_weights()
            
            break

        if loaded_model:
            break
    elif result == 'n':
        break
    else:
        print("Invalid input. Must be `y` or `n`.")

### Define model architecture here if it hasn't been previously loaded from a file
if not loaded_model:
    '''
    model.add_layer(0, 1, 0)
    model.add_layer(1, 3, 1)
    model.set_activation_function(1, relu)
    model.add_layer(2, 3, 2)
    model.set_activation_function(2, softmax)
    model.set_loss_function(loss_categorical_crossentropy)
    '''

    '''
    model.add_layer(0, 1, 0)
    model.add_layer(1, 3, 1)
    model.set_activation_function(1, linear)
    model.add_layer(2, 1, 2)
    model.set_activation_function(2, linear)
    model.set_loss_function(loss_mse)
    '''
    # model.add_layer(0, 106*80*3, 0)
    model.add_layer(0, 106*80*3, 0)
    # model.add_layer(1, 50, 1)
    # model.add_layer(2, 1, 2)
    # model.set_activation_function(1, relu)
    model.add_layer(1, 50, 1)
    model.set_activation_function(1, relu)

    model.add_layer(2, 3, 2)
    model.set_activation_function(2, softmax)
    # model.set_activation_function(2, linear)
    # model.set_loss_function(loss_mse)
    model.set_loss_function(loss_categorical_crossentropy)

    model.setup_done(is_loaded_model=False)
    model.print_architecture()

    # model.set_regularization(1, 0, 0.1)

    # model.set_regularization(1, 0, 0)
    # model.set_regularization(2, 0, 0)
    # model.set_regularization(1, 0.01, 0.1)
    # model.set_regularization(2, 0.01, 0.1)

# 0.0001

# nn.set_regularization(1, 0, 0)
# nn.set_regularization(2, 0, 0)

# nn.set_regularization(1, l1_rate=0, l2_rate=0)


# lr = 0.001 good for 2 relu hidden layers with linear output
# lr = 0.5

# 1 dimensional input-output linear regression

# batch size 32
# lr non-normalized: 0.001 | steps 1000 (WORKING)
# lr normalized: 150 | steps 100

# batch size 11293
# lr non-normalized: 0.5 | steps: 1000000....
# lr normalized: 800 | steps: 100

#0.5, 100 - [[ 376.23059087 5532.17079501]]
#0.5, 5000 - [[16215.07718159  5413.07017895]]
#0.55, 10000 - [[34273.79647371  5277.27717278]]

# Weights: [array([[0.0737276]], dtype=float32), array([-4.80284], dtype=float32)]

'''
Step: 10000 | Training cost: 100431889464.35463
Finished training.
33
99
2 Plot showed, joined training
(Layer 0 weights): [[3361.5906031  5509.72230328]]

Training cost: 100431889464.35463 | Total samples: 11293
Predict? (You can save the model after) (y/n) y
area_m2 (float): 100
Input matrix: [[  1.]
 [100.]]
Predicted value (price_brl): 554333.8209315466


'''

# model.weights = [np.matrix([-0.7940663, 1.5971815])]
# model.weights = [np.matrix([0, 0.05])]
# model.weights = [np.matrix([-4.83674843, 0.07966317])]
# model.weights = [np.matrix([0, 0.11496624])]
# model.weights = [np.matrix([0.14220415, 0.09021131])]

print(f"Initial weights")
model.print_weights()
lr = 0.005
# batch_size = 32
# batch_size = 1
batch_size = 32
# steps = 5
# steps = 300
steps = 5
# batches_per_step = int(11293 / batch_size)

plot_update_every_n_batches = 1
# plot_update_every_n_batches = 8469
# plot_update_every_n_batches = 500
# Not normalizing
# lr = 0.000000000005
# lr = 0.1

# Normalizing
# lr = 0.000000001
# lr = 0.000000001
# lr = 0.000000001

# Normalizing doesn't seem to work properly, also drastically influences the learning rate

# Load and preprocess dataset
#dataset_file = "dataset.csv"

IMAGES_FOLDER = "images/"

if not loaded_model:
    is_image_model = False
    warned_0 = False
    while True:
        if not warned_0:
            print(f"Note: All of the training set images must be within the 'images/train' folder, relative to the folder where this script is ran from.")
            warned_0 = True

        result = input("Will the model be used for images? (y/n) ")
        if result == 'y':
            is_image_model = True
            break
        elif result == 'n':
            is_image_model = False
            break

    model.is_image_model = is_image_model
    
'''
uses_grayscale_images = False
if model.is_image_model:
    while True:
        result = input("Does the model use black and white images? (y/n) ")
        if result == 'y':
            uses_grayscale_images = True
            break
        elif result == 'n':
            uses_grayscale_images = False
            break
'''

while True:
    result = input("Train or predict? (0/1) ")
    if result == "0":
        break
    elif result == "1":
        predict_prompt(model, False)
        exit(0)

draw_graph = True
while True:
    result = input("Draw cost/performance metrics graphs? (y/n) ")
    if result == 'y':
        draw_graph = True
        break
    elif result == 'n':
        draw_graph = False
        break

use_test_set = False
use_holdout = False
use_kfolds = False
        
training_set_percentage = 0.9
crossvalidation_set_percentage = 0.1
test_set_percentage = 1 - training_set_percentage - crossvalidation_set_percentage


crossvalidation_folds = 5

while True:
    result = input("Use cross-validation? (y/n): ").strip()
    if result == "y":
        while True:
            result = input("Hold-out set or k-folds? (0/1): ").strip()
            if result == "0":
                use_holdout = True
                use_kfolds = False
                break
            elif result == "1":
                use_kfolds = True
                use_holdout = False
                break
            else:
                print("Invalid input. Must be `y` or `n`.")
        break
    elif result == "n":
        break
    else:
        print("Invalid input. Must be `y` or `n`.")

if not use_kfolds:
    while True:
        result = input("Use test set? (y/n): ").strip()
        if result == "y":
            use_test_set = True
            break
        elif result == "n":
            use_test_set = False
            break
        else:
            continue
else:
    use_test_set = False

use_separate_sample_folders = False
if model.is_image_model and (use_test_set or use_holdout):
    while True:
        result = input("Use image samples (i.e. hold-out, test) from those within the 'train' folder as percentages, or from their respective folders? (0/1) ")
        if result == '0':
            use_separate_sample_folders = False
            break
        elif result == '1':
            use_separate_sample_folders = True
            break

if not use_separate_sample_folders:
    if use_holdout:
        if use_test_set:
            print(f"NOTE: Cross-validation hold-out set will be `1 - training_set_percentage - test_set_percentage` of the total samples.")
        else:
            print(f"NOTE: Cross-validation hold-out set will be `1 - training_set_percentage` of the total samples.")

    if use_test_set and not use_holdout:
            print(f"NOTE: Test set will be `1 - training_set_percentage` of the total samples.")

    if not use_test_set and not use_holdout:
        training_set_percentage = 1
    else:
        while True:
            result = input("Input a training set percentage (> 0 and < 1): ").strip()
            try:
                result = float(result)
            except:
                print("Could not parse input into a float number.")
                continue

            if result > 0 and result < 1:
                training_set_percentage = result
                break
            else:
                print("Number must be between 0 and 1, not including.")

    # Using only training & test sets
    if use_test_set and not use_holdout:
        test_set_percentage = 1 - training_set_percentage

    # Using only training & hold-out sets
    elif not use_test_set and use_holdout:
        crossvalidation_set_percentage = 1 - training_set_percentage

    # Using training, hold-out, and test sets
    elif use_test_set and use_holdout:
        while True:
            result = input("Input a test set percentage (> 0 and < 1): ").strip()
            try:
                result = float(result)
            except:
                print("Error: Could not parse input into a float number.")
                continue

            if result > 0 and result < 1:
                if training_set_percentage + result >= 1:
                    print(f"Error: Invalid test set percentage. `training_set_percentage ({training_set_percentage}) + test_set_percentage {result}` should be less than 1, but added up to {training_set_percentage + result}.")
                    exit(-1)
                else:
                    test_set_percentage = result
                    crossvalidation_set_percentage = 1 - training_set_percentage - test_set_percentage
                    break
            else:
                print("Number must be between 0 and 1, not including.")


if use_kfolds:
    while True:
        result = input("Input the number of folds (integer greater than 0): ").strip()
        try:
            result = int(result)
        except:
            print("Error: Could not parse input into an integer.")
            continue

        if result > 1:
            crossvalidation_folds = result
            training_set_percentage = 1
            test_set_percentage = 0
            crossvalidation_set_percentage = 0
            break
        else:
            print("Integer must be positive and greater than 1.")

shuffle_dataset = True
while True:
    result = input("Shuffle dataset? (y/n) ")
    if result == 'n':
        shuffle_dataset = False
        break
    elif result == 'y':
        break
    else:
        continue

if not use_separate_sample_folders:
    print(f"Train set: {training_set_percentage * 100}%", end="")
    if use_holdout:
        print(f" | Validation (hold-out) set: {crossvalidation_set_percentage * 100}%", end="")
    if use_test_set:
        print(f" | Test set: {test_set_percentage * 100}%", end="")
elif use_separate_sample_folders:
    if model.is_image_model:
        total_images = 0
        total_training_samples = len(os.listdir(IMAGES_FOLDER + "train/")) - 1
        total_images += total_training_samples

        if use_holdout:
            total_crossvalidation_samples = len(os.listdir(IMAGES_FOLDER + "holdout/")) - 1
            total_images += total_crossvalidation_samples

        if use_test_set:
            total_test_samples = len(os.listdir(IMAGES_FOLDER + "test/")) - 1
            total_images += total_test_samples

        print(f"Train set: {(total_training_samples / total_images) * 100}%", end="")
        
        if use_holdout:
            print(f" | Validation (hold-out) set: {(total_crossvalidation_samples / total_images) * 100}%", end="")

        if use_test_set:
            print(f" | Test set: {(total_test_samples / total_images) * 100}%", end="")
        
    elif not model.is_image_model:
        print("Error: Cannot currently separate datasets by folder unless they are images.")
        exit(-1)

# Newline
print()

# The line above 


# Write a 3 line function that prints permutations of the word "yes"


if use_kfolds:
    print(f"Folds set for cross-validation: {crossvalidation_folds}")

if not model.is_image_model:
    dataset = Dataset()

    if not loaded_model:
        model.feature_list = dataset.feature_list
        model.feature_types = dataset.feature_types
        model.class_list = dataset.class_list

    total_samples = len(dataset.filtered_entries)
else:
    images, labels, feature_list, feature_types, class_list, image_dim, image_bit_depth = load_image_dataset(IMAGES_FOLDER + "train/", model=model, unique_dataset=True)

    if not loaded_model:
        model.feature_list = feature_list
        model.feature_types = feature_types
        model.class_list = class_list
        model.image_dim = image_dim
        model.image_bit_depth = image_bit_depth

    total_samples = images.shape[1]

if not use_separate_sample_folders:
    total_training_samples = round(total_samples * training_set_percentage)

    # Test set and/or hold-out set are mutually exclusive with k-folds
    if use_test_set:
        if not use_holdout:
            total_test_samples = total_samples - total_training_samples
        else:
            total_test_samples = round(total_samples * test_set_percentage)

    if use_holdout and use_test_set:    
        total_crossvalidation_samples = total_samples - total_training_samples - total_test_samples
    elif use_holdout and not use_test_set:
        total_crossvalidation_samples = total_samples - total_training_samples

    training_samples = np.zeros((len(model.feature_list) + 1, total_training_samples))
    training_dependent_values = np.zeros((total_training_samples, len(model.class_list)))
    training_samples = np.matrix(training_samples)
    training_dependent_values = np.matrix(training_dependent_values)

    if use_holdout:
        crossvalidation_samples = np.zeros((len(model.feature_list) + 1, total_crossvalidation_samples))
        crossvalidation_dependent_values = np.zeros((total_crossvalidation_samples, len(model.class_list)))
        crossvalidation_samples = np.matrix(crossvalidation_samples)
        crossvalidation_dependent_values = np.matrix(crossvalidation_dependent_values)

    if use_test_set:
        test_samples = np.zeros((len(model.feature_list) + 1, total_test_samples))
        test_dependent_values = np.zeros((total_test_samples, len(model.class_list)))
        test_samples = np.matrix(test_samples)
        test_dependent_values = np.matrix(test_dependent_values)



if not model.is_image_model:
    ## Shuffle/randomize dataset entries
    ## Include this in a data preprocessing function? i.e. dataset.randomize()?
    ## Prompt user where to shuffle


    if shuffle_dataset:
        for idx in range(len(dataset.filtered_entries)):
            random_pos = random.randint(0, len(dataset.filtered_entries) - 1)
            current_entry = dataset.filtered_entries[idx]
            entry_to_swap = dataset.filtered_entries[random_pos]
            dataset.filtered_entries[random_pos] = current_entry
            dataset.filtered_entries[idx] = entry_to_swap
            ## Calculate test/validate/train set variances for correlation coefficient if regression model

        print("Shuffled dataset.")

    for sample_num, entry in enumerate(dataset.filtered_entries):
        # Fill training set (within dataset array range [0, total_training_samples))
        if sample_num < total_training_samples:
            training_samples[0, sample_num] = 1

            for feature_num, feature in enumerate(entry.features):
                training_samples[feature_num + 1, sample_num] = entry.features[feature]
            
            for class_num, class_name in enumerate(entry.dependent_values):
                training_dependent_values[sample_num, class_num] = entry.dependent_values[class_name]

        # Fill hold-out set (within dataset array range [total_training_samples, total_crossvalidation_samples))
        elif use_holdout and sample_num < total_training_samples + total_crossvalidation_samples:
            crossvalidation_samples[0, sample_num - total_training_samples] = 1

            for feature_num, feature in enumerate(entry.features):
                crossvalidation_samples[feature_num + 1, sample_num - total_training_samples] = entry.features[feature]

            for class_num, class_name in enumerate(entry.dependent_values):
                crossvalidation_dependent_values[sample_num - total_training_samples, class_num] = entry.dependent_values[class_name]

        # Fill test set (within dataset array range [total_crossvalidation_samples, total_samples))
        elif use_test_set and use_holdout:
            test_samples[0, sample_num - total_training_samples - total_crossvalidation_samples] = 1

            for feature_num, feature in enumerate(entry.features):
                test_samples[feature_num + 1, sample_num - total_training_samples - total_crossvalidation_samples] = entry.features[feature]

            for class_num, class_name in enumerate(entry.dependent_values):
                test_dependent_values[sample_num - total_training_samples - total_crossvalidation_samples, class_num] = entry.dependent_values[class_name]
        elif use_test_set and not use_holdout:
            test_samples[0, sample_num - total_training_samples] = 1

            for feature_num, feature in enumerate(entry.features):
                test_samples[feature_num + 1, sample_num - total_training_samples] = entry.features[feature]

            for class_num, class_name in enumerate(entry.dependent_values):
                test_dependent_values[sample_num - total_training_samples, class_num] = entry.dependent_values[class_name]
elif model.is_image_model:
    if shuffle_dataset:
        randomize_dataset(images, labels)
        print("Shuffled dataset.")

    if not use_separate_sample_folders:
        training_samples = images[:, :total_training_samples]
        training_dependent_values = labels[:total_training_samples, :]

        if use_holdout:
            crossvalidation_samples = images[:, total_training_samples:total_training_samples + total_crossvalidation_samples]
            crossvalidation_dependent_values = labels[total_training_samples:total_training_samples + total_crossvalidation_samples, :]

        if use_test_set and use_holdout:
            test_samples = images[:, total_training_samples + total_crossvalidation_samples: total_samples]
            test_dependent_values = labels[total_training_samples + total_crossvalidation_samples : total_samples, :]
        elif use_test_set and not use_holdout:
            test_samples = images[:, total_training_samples:total_samples]
            test_dependent_values = labels[total_training_samples:total_samples, :]
    elif use_separate_sample_folders:
        training_samples = images
        training_dependent_values = labels

        if use_holdout:
            crossvalidation_samples, crossvalidation_dependent_values, _, _, _ = load_image_dataset(IMAGES_FOLDER + "holdout/", model=model, unique_dataset=False)

            if shuffle_dataset:
                randomize_dataset(crossvalidation_samples, crossvalidation_dependent_values)

        if use_test_set:
            test_samples, test_dependent_values, _, _, _ = load_image_dataset(IMAGES_FOLDER + "test/", model=model, unique_dataset=False)

            if shuffle_dataset:
                randomize_dataset(test_samples, test_dependent_values)


print("Loaded dataset.")

'''
from sklearn.datasets import make_circles

#making 1000 examples
total_samples = 1000

#creating circles
tmp_training_samples, training_dependent_values = make_circles(total_samples,
                    noise=0.5,
                    random_state=42)

training_dependent_values = np.transpose(np.matrix(training_dependent_values))

training_samples = np.zeros((3, total_samples))
training_samples = np.matrix(training_samples)

for ss in range(total_samples):
    training_samples[0, ss] = 1
    training_samples[1, ss] = tmp_training_samples[ss][0]
    training_samples[2, ss] = tmp_training_samples[ss][1]



model.feature_list = ['x0', 'x1']
model.feature_types = ['float', 'float']
model.class_list = ['radius']

dataset.feature_list = model.feature_list
dataset.class_list = model.class_list
'''

model.normalized = False
normalization_warned = False
while True:
    if not normalization_warned:
        print("\nNote: Normalizing the dataset for training results in weights trained on the normalized values. (Meaning different than those weights on non-normalized samples)")
        print("However, it sometimes tends to make data processing faster, among other things.")
        print("The model should perform similarly given the ideal parameters")
        print("Restricts training data to [-1, 1]")
        print("The weights lose some interpretation value compared to training on non-normalized data, if you're trying to infer things from looking at them as is.")
        print("i.e. In a model that predicts house prices with respect to its area (m²), the feature weight B1 represents the price per square meter of the model. However, if normalized, you would have to perform some additional calculations in order to get that number back, as it would need to be 'denormalized'.")
        
        normalization_warned = True

    
    should_normalize = input("Normalize dataset? (y/n) ").strip()

    if should_normalize == 'y':
        model.normalized = True
    elif should_normalize == 'n':
        model.normalized = False
    else:
        print("Invalid input. Should be 'y' or 'n'.")
        continue

    break


if use_kfolds:
    # K-fold cross-validation test sample size for given K
    # Assuming the number of folds can be divided into the training samples (i.e. (training_samples.shape[1] / crossvalidation_folds) >= 1)
    # For now, if the number of training samples is not exactly divisible by the number of folds, the spare sample will be ignored
    crossvalidation_test_section_size = math.floor(training_samples.shape[1] / crossvalidation_folds)

    if crossvalidation_test_section_size < 1:
        print(f"Error: Cannot calculate k-folds cross-validation as the number of training samples ({training_samples}) cannot be divided into the number of folds ({crossvalidation_folds}).")
        exit(-1)

    crossvalidation_test_section_size = round(crossvalidation_test_section_size)
    crossvalidation_training_section_size = training_samples.shape[1] - crossvalidation_test_section_size
    crossvalidation_training_samples = []
    crossvalidation_training_dependent_values = []
    crossvalidation_test_samples = []
    crossvalidation_test_dependent_values = []

    crossvalidation_models = []
    crossvalidation_models.append(model)
    for current_model in range(1, crossvalidation_folds):
        ## INSTEAD OF DEEPCOPYING A BASE MODEL,
        ## EACH MODEL COULD HAVE TWEAKS TO THEM
        ## IN ORDER TO BE COMPARED TO THE BASED MODEL.
        ## HERE IS AN IDEAL ENTRY POINT FOR THAT.
        ## DEFAULT BEHAVIOR IS TO MAKE K COPIES OF A BASE MODEL (WHERE K = NUMBER OF FOLDS)
        crossvalidation_models.append(copy.deepcopy(model))

    
    crossvalidation_test_sections = []
    crossvalidation_train_sections = []

    # Each element in these lists is another list, there is one list per fold
    # Inside the fold's lists is the actual list for the respective fold, representing the list for the given metric (a list because each dependent/independent variable is measured separately)
    crossvalidation_test_features_mean = []
    crossvalidation_test_features_variance = []
    crossvalidation_test_features_std = []
    crossvalidation_test_dependent_variables_mean = []
    crossvalidation_test_dependent_variables_variance = []
    crossvalidation_test_dependent_variables_std = []

    crossvalidation_training_features_mean = []
    crossvalidation_training_features_variance = []
    crossvalidation_training_features_std = []
    crossvalidation_training_dependent_variables_mean = []
    crossvalidation_training_dependent_variables_variance = []
    crossvalidation_training_dependent_variables_std = []

    for fold in range(crossvalidation_folds):  
        test_section = (None, None)
        train_section_l = (None, None)
        train_section_r = (None, None)

        # If at the first fold
        if fold == 0:
            test_section = (0, crossvalidation_test_section_size)
            # train_section_r = (test_section[1], training_samples.shape[1] - 1)
            train_section_r = (crossvalidation_test_section_size, training_samples.shape[1])

            right_training_samples = training_samples[:, train_section_r[0] : train_section_r[1]]
            right_dependent_values = training_dependent_values[train_section_r[0]:train_section_r[1], :]

            fold_training_samples = right_training_samples
            fold_dependent_values = right_dependent_values

        # If at the last fold
        elif fold == crossvalidation_folds - 1:
            # test_section = ((fold * crossvalidation_test_section_size), training_samples.shape[1] - 1)
            test_section = (crossvalidation_training_section_size, training_samples.shape[1])
            # train_section_l = (0, test_section[0])
            train_section_l = (0, crossvalidation_training_section_size)

            left_training_samples = training_samples[:, train_section_l[0] : train_section_l[1]]
            left_dependent_values = training_dependent_values[train_section_l[0]:train_section_l[1], :]

            fold_training_samples = left_training_samples
            fold_dependent_values = left_dependent_values

        # If at folds in between
        else:
            # test_section = (fold * crossvalidation_test_section_size, ((fold * crossvalidation_test_section_size) + (crossvalidation_test_section_size - 1)))
            test_section = (fold * crossvalidation_test_section_size, ((fold * crossvalidation_test_section_size) + crossvalidation_test_section_size))
            # train_section_l = (0, test_section[0])
            train_section_l = (0, fold * crossvalidation_test_section_size)
            # train_section_r = (test_section[1], training_samples.shape[1] - 1)
            train_section_r = ((fold * crossvalidation_test_section_size) + crossvalidation_test_section_size, training_samples.shape[1])

            left_training_samples = training_samples[:, train_section_l[0] : train_section_l[1]]
            right_training_samples = training_samples[:, train_section_r[0] : train_section_r[1]]          
            
            left_dependent_values = training_dependent_values[train_section_l[0]:train_section_l[1], :]
            right_dependent_values = training_dependent_values[train_section_r[0]:train_section_r[1], :]

            fold_training_samples = np.concatenate((left_training_samples, right_training_samples), axis=1)   
            fold_dependent_values = np.concatenate((left_dependent_values, right_dependent_values), axis=0)

        fold_training_samples = np.asmatrix(fold_training_samples) 
        fold_dependent_values = np.asmatrix(fold_dependent_values)
            
        fold_test_samples = training_samples[:, test_section[0] : test_section[1]]
        fold_test_dependent_values = training_dependent_values[test_section[0] : test_section[1], :]

        crossvalidation_training_samples.append(fold_training_samples)
        crossvalidation_training_dependent_values.append(fold_dependent_values)
        crossvalidation_test_samples.append(fold_test_samples)
        crossvalidation_test_dependent_values.append(fold_test_dependent_values)

        k_model = crossvalidation_models[fold]
        k_model.feature_min_maxes = []
        for feature_type in k_model.feature_types:
            if feature_type == "str":
                k_model.feature_min_maxes.append([0, -math.inf])
            elif feature_type == "float":
                k_model.feature_min_maxes.append([math.inf, -math.inf])

        # Updates features' min/max values
        mean_n_variance_normalize(k_model, crossvalidation_training_samples[fold], update_min_maxes=True, ignore_bias=True)

        # Normalizes features to range [-1, 1]
        if k_model.normalized:
            new_normalize(k_model, crossvalidation_training_samples[fold])
            new_normalize(k_model, crossvalidation_test_samples[fold])

        # Calculates means, variances, and standard deviations
        without_one_fold_training_samples = fold_training_samples[1:, :]
        result = Dataset.get_dataset_stats(without_one_fold_training_samples, fold_dependent_values, model.feature_list, model.class_list)
        crossvalidation_training_features_mean.append(result["features_mean"])
        crossvalidation_training_features_variance.append(result["features_variance"])
        crossvalidation_training_features_std.append(result["features_std"])
        crossvalidation_training_dependent_variables_mean.append(result["dependent_variables_mean"])
        crossvalidation_training_dependent_variables_variance.append(result["dependent_variables_variance"])
        crossvalidation_training_dependent_variables_std.append(result["dependent_variables_std"])

        k_model.sample_features_mean = crossvalidation_training_features_mean[fold]
        k_model.sample_features_variance = crossvalidation_training_features_variance[fold]
        k_model.sample_features_std = crossvalidation_training_features_std[fold]
        k_model.sample_dependent_variables_mean = crossvalidation_training_dependent_variables_mean[fold]
        k_model.sample_dependent_variables_variance = crossvalidation_training_dependent_variables_variance[fold]
        k_model.sample_dependent_variables_std = crossvalidation_training_dependent_variables_std[fold]

        # Mean normalizes and standardizes features
        if k_model.normalized:
            mean_n_variance_normalize(k_model, crossvalidation_training_samples[fold], update_min_maxes=False, ignore_bias=True)
            mean_n_variance_normalize(k_model, crossvalidation_test_samples[fold], update_min_maxes=False, ignore_bias=True)

elif not use_kfolds:
    model.feature_min_maxes = []
    for feature_type in model.feature_types:
        if feature_type == "str":
            model.feature_min_maxes.append([0, -math.inf])
        elif feature_type == "float":
            model.feature_min_maxes.append([math.inf, -math.inf])

    mean_n_variance_normalize(model, training_samples, update_min_maxes=True, ignore_bias=True)

    if model.normalized:
        new_normalize(model, training_samples)

        if use_holdout:
            new_normalize(model, crossvalidation_samples)
        if use_test_set:
            new_normalize(model, test_samples)

    without_one_training_samples = training_samples[1:, :]
    result = Dataset.get_dataset_stats(without_one_training_samples, training_dependent_values, model.feature_list, model.class_list)

    training_features_mean = result["features_mean"]
    training_features_variance = result["features_variance"]
    training_features_std = result["features_std"]
    training_dependent_variables_mean = result["dependent_variables_mean"]
    training_dependent_variables_variance = result["dependent_variables_variance"]
    training_dependent_variables_std = result["dependent_variables_std"]

    model.sample_features_mean = training_features_mean
    model.sample_features_variance = training_features_variance
    model.sample_features_std = training_features_std
    model.sample_dependent_variables_mean = training_dependent_variables_mean
    model.sample_dependent_variables_variance = training_dependent_variables_variance
    model.sample_dependent_variables_std = training_dependent_variables_std

    if model.normalized:
        mean_n_variance_normalize(model, training_samples, update_min_maxes=False, ignore_bias=True)

        if use_holdout:
            mean_n_variance_normalize(model, crossvalidation_samples, update_min_maxes=False, ignore_bias=True)
        if use_test_set:
            mean_n_variance_normalize(model, test_samples, update_min_maxes=False, ignore_bias=True)



    '''
    for i in range(training_samples.shape[1]):
        sample = training_samples[1:, i]
        print(f"Label: {training_dependent_values[i, :]}")
        plot_image_from_sample(sample)
    '''

if model.normalized:
    without_one_training_samples = training_samples[1:, :]
    result = Dataset.get_dataset_stats(without_one_training_samples, training_dependent_values, model.feature_list, model.class_list)

    # These are after normalizing, and kept separate from the statistics before normalization (i.e. model.sample_features_mean), in order to evaluate metrics after normalizing and also normalize values/revert normalization
    training_features_mean = result["features_mean"]
    training_features_variance = result["features_variance"]
    training_features_std = result["features_std"]
    training_dependent_variables_mean = result["dependent_variables_mean"]
    training_dependent_variables_variance = result["dependent_variables_variance"]
    training_dependent_variables_std = result["dependent_variables_std"]

    # print(f"Mean (after normalizing): {training_features_mean}")
    # print(f"Std (after normalizing): {training_features_std}")

    # print(f"Mean (b4 normalizing): {model.sample_features_mean}")
    # print(f"Std (b4 normalizing): {model.sample_features_std}")

    # exit(-1)

if use_kfolds:
    for fold in range(crossvalidation_folds):

        # These are after normalizing, and kept separate from the statistics before normalization (i.e. model.sample_features_mean), in order to evaluate metrics after normalizing and also normalize values/revert normalization
        if crossvalidation_models[fold].normalized:
            without_one_fold_training_samples = crossvalidation_training_samples[fold][1:, :]
            result = Dataset.get_dataset_stats(without_one_fold_training_samples, crossvalidation_training_dependent_values[fold], model.feature_list, model.class_list)
            crossvalidation_training_features_mean[fold] = result["features_mean"]
            crossvalidation_training_features_variance[fold] = result["features_variance"]
            crossvalidation_training_features_std[fold] = result["features_std"]
            crossvalidation_training_dependent_variables_mean[fold] = result["dependent_variables_mean"]
            crossvalidation_training_dependent_variables_variance[fold] = result["dependent_variables_variance"]
            crossvalidation_training_dependent_variables_std[fold] = result["dependent_variables_std"]

        without_one_fold_test_samples = crossvalidation_test_samples[fold][1:, :]
        result = Dataset.get_dataset_stats(without_one_fold_test_samples, crossvalidation_test_dependent_values[fold], model.feature_list, model.class_list)
        crossvalidation_test_features_mean.append(result["features_mean"])
        crossvalidation_test_features_variance.append(result["features_variance"])
        crossvalidation_test_features_std.append(result["features_std"])
        crossvalidation_test_dependent_variables_mean.append(result["dependent_variables_mean"])
        crossvalidation_test_dependent_variables_variance.append(result["dependent_variables_variance"])
        crossvalidation_test_dependent_variables_std.append(result["dependent_variables_std"])

elif not use_kfolds:
    if use_holdout:
        without_one_crossvalidation_samples = crossvalidation_samples[1:, :]
        result = Dataset.get_dataset_stats(without_one_crossvalidation_samples, crossvalidation_dependent_values, model.feature_list, model.class_list)

        crossvalidation_features_mean = result["features_mean"]
        crossvalidation_features_variance = result["features_variance"]
        crossvalidation_features_std = result["features_std"]
        crossvalidation_dependent_variables_mean = result["dependent_variables_mean"]
        crossvalidation_dependent_variables_variance = result["dependent_variables_variance"]
        crossvalidation_dependent_variables_std = result["dependent_variables_std"]

    if use_test_set:
        without_one_test_samples = test_samples[1:, :]
        result = Dataset.get_dataset_stats(without_one_test_samples, test_dependent_values, model.feature_list, model.class_list)

        test_features_mean = result["features_mean"]
        test_features_variance = result["features_variance"]
        test_features_std = result["features_std"]
        test_dependent_variables_mean = result["dependent_variables_mean"]
        test_dependent_variables_variance = result["dependent_variables_variance"]
        test_dependent_variables_std = result["dependent_variables_std"]

    # Normalization is reversed here in order to calculate the training features' mean and variance (ideally by making some changes to the scripts there should be no need for all of this, but I will leave it like this for now as I want it to be functional first)
    '''
    if dataset.should_normalize:
        prenorm_training_features_mean = [0 for _ in range(len(dataset.feature_list))]
        prenorm_training_features_variance = [0 for _ in range(len(dataset.feature_list))]

        for sample in range(total_training_samples):
            for feature_idx, feature in enumerate(dataset.feature_list):
                postnorm_feature_value = without_one_training_samples[feature_idx, sample]
                prenorm_feature_value = postnorm_feature_value * dataset.features_variance[feature_idx]
                prenorm_feature_value = prenorm_feature_value + dataset.features_mean[feature_idx]
                prenorm_feature_value = dataset.feature_min_maxes[feature_idx][0] + (((dataset.feature_min_maxes[feature_idx][1] - dataset.feature_min_maxes[feature_idx][0]) * (postnorm_feature_value + 1)) / 2)

                prenorm_training_features_mean[feature_idx] += prenorm_feature_value / total_training_samples
                prenorm_training_features_variance[feature_idx] += (prenorm_feature_value * prenorm_feature_value) / total_training_samples
    '''


    # model.dependent_variables_mean = training_dependent_variables_mean
    # model.dependent_variables_variance = training_dependent_variables_variance
    # model.dependent_variables_std = training_dependent_variables_std


## Following is done to keep track of the features, their types, the classes, and information to perform normalization (if needed), this is needed when loading models from scratch for prediction
## The values in model.feature_min_maxes, `model.samples_features_mean`, `model.sample_features_variance` are all based on the training samples, before the normalization (so the default training samples' values)

print("Processed dataset into numpy matrices to use within the network.")

# Clips batch size to the number of training samples if the given number was bigger than the number of training samples
if not use_kfolds:
    if batch_size > training_samples.shape[1]:
        batch_size = training_samples.shape[1]
else:
    if batch_size > crossvalidation_training_section_size:
        batch_size = crossvalidation_training_section_size

# print(training_dependent_values[0, :])
# plot_image_from_sample(model, training_samples[1:, 0])
# print(training_dependent_values[1, :])
# plot_image_from_sample(model, training_samples[1:, 1])
# exit(-1)


input_layer = model.layers_in_order[0]
output_layer = model.layers_in_order[-1]

input_layer_dim = model.layers_in_order[0].dim
output_layer_dim = model.layers_in_order[-1].dim

if input_layer.dim != len(model.feature_list):
    print(f"Error: Number of independent variables (features) given for the dataset was {len(model.feature_list)}, which does not match the dimension (number of nodes) of the given input layer of {model.layers_in_order[0].dim}.")
    exit(-1)

if output_layer.dim != len(model.class_list):
    print(f"Error: Number of dependent variables (classes/categories/labels) given for the dataset was {len(model.class_list)}, which does not match the dimension (number of nodes) in the given output layer of {model.layers_in_order[-1].dim}.")
    exit(-1)

if draw_graph:
    xs = []

    # First list of cost_ys is always the training cost (or training cost average for k-folds)
    cost_ys = [[]]
    metrics_ys = []
    cost_plot_labels = []
    metrics_plot_labels = []

    if use_kfolds:
        cost_plot_labels.append(f"Training cost ({crossvalidation_folds}-folds training costs average)")
        cost_plot_labels.append(f'Test cost ({crossvalidation_folds}-folds test costs average)')
        for fold in range(crossvalidation_folds):
            cost_plot_labels.append(f"Test cost (Fold {fold + 1})")
    elif use_holdout:
        cost_plot_labels = ["Training cost", "Training cost (hold-out)"]
    else:
        cost_plot_labels = ["Training cost"]

    if use_holdout:
        cost_ys.append([])
    elif use_kfolds:
        cost_ys.append([]) # Accounts for test costs average
        for _ in range(crossvalidation_folds):
            cost_ys.append([])

    if use_holdout or use_kfolds:
        if use_holdout:
            is_classification = model.is_classification()
            is_regression = model.is_regression()
        else:
            is_classification = crossvalidation_models[0].is_classification()
            is_regression = crossvalidation_models[0].is_regression()

        if is_classification:
            metrics_ys += [], [], [], [] # Accuracy, precision, recall, f1-score
            metrics_plot_labels += "Accuracy", "Precision", "Recall", "F1-Score"
        elif is_regression:
            metrics_ys.append([])
            metrics_plot_labels.append("R^2")

    if use_holdout or use_kfolds:
        figure, axes = plt.subplots(1, 2)
        axes = [[axes[0], axes[1]], [None, None]]
        axes[0][1].set_xlabel("Batch")
        axes[0][1].sharex(axes[0][0])
    else:
        figure, tmp_axes = plt.subplots(1, 1)
        axes = [[tmp_axes, None], [None, None]]
        # axes[0][0] = tmp_axes
        # axes[1][0] = tmp_axes[1]

    if not use_kfolds:
        axes[0][0].set_ylabel("Training cost (linear scale)")
    else:
        axes[0][0].set_ylabel("Training and test costs (linear scale)")
    axes[0][0].set_xlabel("Batch")

    if use_holdout:
        axes[0][1].set_ylabel("Performance metrics (hold-out micro averages)")
    elif use_kfolds:
        axes[0][1].set_ylabel(f"Performance metrics ({crossvalidation_folds}-folds test set micro averages)")

    # axes[1][0].set_xlabel("Batch")
    # axes[1][0].set_yscale("log")

    costs_plotting_colors = [[0.0, 0.0, 1.0, 1.0]] # RGBA color intensity array for each line, each color channel is in range [0, 1] instead of [0, 255]
    if use_kfolds or use_holdout:
        if is_classification:
            metrics_plotting_colors = ["blue", "red", "green", "purple"] # Accuracy, precision, recall, f1-score
        elif is_regression:
            metrics_plotting_colors = ["blue"] # R^2


        costs_plotting_colors.append([1.0, 0.0, 0.0, 1.0]) 
        if use_kfolds:
            for fold in range(crossvalidation_folds):
                color_found = False
                while not color_found:
                    random_color = random.randint(0, 0xFFFFFF)
                    r = ((random_color & 0xFF0000) >> 16) / 255.0
                    g = ((random_color & 0x00FF00) >> 8) / 255.0
                    b = (random_color & 0x0000FF) / 255.0
                    a = 1.0
                    random_color = [r, g, b, a]

                    for color in costs_plotting_colors:
                        if color_is_similar(random_color, color):
                            color_found = False
                            break

                    color_found = True
                    costs_plotting_colors.append(random_color)

        # training_thread = threading.Thread(target=nn.train, kwargs={'batch_size': batch_size, 'steps': steps, 'lr': lr, 'training_samples': training_samples, 'dependent_values': training_dependent_values})

    plt.ion()
    figure.tight_layout()
    # nn.ani = FuncAnimation(figure, nn.plot, interval=nn.update_interval, cache_frame_data=False)
    figure.show()

print(f"Steps: {steps} | Training parameters - Learning rate: {lr} | Gradient descent batch size: {batch_size}")
if use_kfolds:
    nn.train(batch_size=batch_size, steps=steps, lr=lr, training_samples=training_samples, dependent_values=training_dependent_values)
else:
    nn.train(model=model, batch_size=batch_size, steps=steps, lr=lr, training_samples=training_samples, dependent_values=training_dependent_values)

# training_thread.start()
#nn.train(batch_size=10, steps=1000, lr=0.001, training_samples=training_samples, dependent_values=dependent_values)



# print("33")

# figure.tight_layout()
# figure.show()
# print("99")
# print("2 Plot showed, joined training")

# training_thread.join()

##
##
##
##
## ------------------------------- CALCULATE TEST SET METRICS HERE AFTER TRAINING IS DONE -------------------------------
##
##
##
##

training_cost = nn.costs[-1]

if not use_kfolds:
    model.print_weights()
    print(f"Training cost: {training_cost} | Total samples: {total_training_samples}")
elif use_kfolds:
    for idx, model in enumerate(crossvalidation_models):
        print(f"Weights (model {idx}): ")
        model.print_weights()

    print(f"Training cost ({crossvalidation_folds}-folds averages): {training_cost}", end="")


# If using hold-out cross-validation, each element is a dictionary
if use_holdout:
    # Any of these variables can be `None`, in which case they're not displayed.
    model_metrics = nn.models_model_metrics[0][-1]
    micro_metrics = nn.models_micro_metrics[0][-1]
    macro_metrics = nn.models_macro_metrics[0][-1]
    class_metrics = nn.models_class_metrics[0][-1]

    print("Cross-validation (hold-out) set metrics\n--------------------------------")
    print_model_metrics(model, model_metrics, micro_metrics, macro_metrics, class_metrics)
    print("--------------------------------")

# If using k-folds cross-validation, each element is another list, where each element represents its respective fold's metrics (a dictionary or a `None` value).
elif use_kfolds:
    print(f"Cross-validation ({crossvalidation_folds}-folds) *test* sets metrics")
    k_folds_avg_cost = 0
    for k_model, model in enumerate(crossvalidation_models):
        model_metrics = nn.models_model_metrics[k_model][-1]
        micro_metrics = nn.models_micro_metrics[k_model][-1]
        macro_metrics = nn.models_macro_metrics[k_model][-1]
        class_metrics = nn.models_class_metrics[k_model][-1]
        k_folds_avg_cost += model_metrics["total_cost"] / crossvalidation_folds
        print(f"Fold {k_model + 1} (test set) model metrics\n--------------------------------")
        print_model_metrics(model, model_metrics, micro_metrics, macro_metrics, class_metrics)
        print("--------------------------------")

    print(f"Average cost over {crossvalidation_folds}-folds test sets: {k_folds_avg_cost}")                     
    exit(0)

if use_test_set:
    model_metrics = {}
    micro_metrics = {}
    macro_metrics = {}
    class_metrics = []
    measure_model_on_dataset(model, test_samples, test_dependent_values, test_features_mean, test_dependent_variables_mean, model.feature_list, model.class_list, model_metrics, micro_metrics, macro_metrics, class_metrics)
    print("Test set metrics\n--------------------------------")
    print_model_metrics(model, model_metrics, micro_metrics, macro_metrics, class_metrics)
    print("--------------------------------")

'''
x0 = []
x1 = []
x0_min = 999999
x0_max = -999999
x1_min = 999999
x1_max = -999999

for i in range(total_samples):
  x0_input = training_samples[1, i]
  x1_input = training_samples[2, i]

  x0.append(x0_input)
  x1.append(x1_input)

  if x0_input > x0_max:
    x0_max = x0_input
  elif x0_input < x0_min:
    x0_min = x0_input

  if x1_input > x1_max:
    x1_max = x1_input
  elif x1_input < x1_min:
    x1_min = x1_input

def plot_decision_boundary():
  """
  Plots the decision boundary created by a model predicting on X.
  This function has been adapted from two phenomenal resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = x0_min - 0.1, x0_max + 0.1
  y_min, y_max = x1_min - 0.1, x1_max + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))

  # Create X values (we're going to predict on all of these)
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

  # Make predictions using the trained model
  y_pred = np.zeros((x_in.shape[0], 1))
  y_pred = np.matrix(y_pred)

  for i in range(x_in.shape[0]):
    x_sample = np.matrix(np.zeros((3, 1)))
    x_sample[0, 0] = 1
    x_sample[1, 0] = x_in[i][0]
    x_sample[2, 0] = x_in[i][1]
    y_pred[i, 0] = nn.predict(model, x_sample)

  y_pred = np.array(y_pred)

  # Check for multi-class
  if model.layers_in_order[-1].dim > 1: # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
    # print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    # print("doing binary classifcation...")
    y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

  plt.cla()

  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(x0, x1, c=np.array(training_dependent_values.transpose()), s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.show()


# Check out the predictions our model is making
plot_decision_boundary()
'''

while True:
    result = input("Predict? (You can save the model after) (y/n) ")

    if result == 'y':
        predict_prompt(model, True)
        break
    elif result == 'n':
        break

while True:
    result = input("Save model? (y/n) ")
    if result == 'y':
        filename = input("Save as file (exclude extension, is saved as .json): ").strip()
        # Save weights to text file here
        save_model(model, filename)
        # print(f"Model saved to {filename}.json")
        break
    elif result == 'n':
        break
    else:
        print("Invalid input. Must be `y` or `n`.")