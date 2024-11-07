## Index
- [Summary](#summary)
- [Examples](#examples)
    - [Linear regression (`houses.csv`)](#linear-regression-housescsv)
    - [Logistic regression (`framingham.csv`)](#logistic-regression-framinghamcsv)
    - [Multi-label classification (3-folds cross-validation) (`framingham.csv`)](#multi-label-classification-3-folds-cross-validation-framinghamcsv)
    - [Multi-class classification (`mnist_png.rar`)](#multi-class-classification-mnist_pngrar)
    - [Multi-class classification (RGB) (`cars.rar`)](#multi-class-classification-rgb-carsrar)
    - [Labeling images](#labeling-images)
    - [Resizing images](#resizing-images)
- [How to use](#how-to-use)
  - [Training a model (from scratch or by retraining an already existing model):](#training-a-model-from-scratch-or-by-retraining-an-already-existing-model)
  - [Loading a model for prediction only:](#loading-a-model-for-prediction-only)
  - [Defining a model from scratch](#defining-a-model-from-scratch)
      - [Layer class](#layer-class)
  - [Activation functions](#activation-functions)
  - [Loss functions](#loss-functions)
  - [Model setup](#model-setup)
  - [Setting up the hyperparameters](#setting-up-the-hyperparameters)
  - [Dataset setup](#dataset-setup)
    - [Comma-delimited '`,`' dataset](#comma-delimited--dataset)
    - [Image dataset](#image-dataset)
      - [Label images](#label-images)
      - [**(Optional)** Resize all images to the same dimension](#optional-resize-all-images-to-the-same-dimension)
  - [Running the script](#running-the-script)
    - [Setting up a comma-delimited dataset](#setting-up-a-comma-delimited-dataset)
    - [(Optional) Filtering a comma-delimited dataset](#optional-filtering-a-comma-delimited-dataset)
- [Implementation details](#implementation-details)
    - [dataset.py](#datasetpy)
    - [main.py](#mainpy)
      - [Network class](#network-class)
      - [Model class](#model-class)
      - [Model loading and saving](#model-loading-and-saving)
      - [JSON model format](#json-model-format)
      - [Feedforward](#feedforward)
      - [Backpropagation](#backpropagation)
      - [Image loading](#image-loading)
      - [Normalization](#normalization)
      - [Dataset metrics evaluation](#dataset-metrics-evaluation)
      - [Prediction](#prediction)
      - [Training](#training)
- [Example model architectures](#example-model-architectures)
  - [Linear regression (`houses.csv`)](#linear-regression-housescsv-1)
  - [Logistic regression (`framingham.csv`)](#logistic-regression-framinghamcsv-1)
  - [Multi-label classification (`framingham.csv`)](#multi-label-classification-framinghamcsv)
  - [Multi-class classification (`mnist_labeled.rar`/`mnist.rar`)](#multi-class-classification-mnist_labeledrarmnistrar)
  - [Multi-class classification (RGB) (`cars_labeled.rar`/`cars.rar`)](#multi-class-classification-rgb-cars_labeledrarcarsrar)
- [Features](#features)
- [Notes](#notes)
- [Todo](#todo)

# Summary
A Python implementation from scratch of a neural network (using NumPy for matrix operations), made primarily for learning purposes. It is nowhere near optimized, so expect bugs, redundant and duplicate code, slow performance, high memory usage, out of place/obsolete comments, and things of the sort.

>You can find the example datasets, as well as the already labeled versions of the image datasets in the folder `datasets`.<br>
You can also find the example architectures for the models in the example videos in the section [Example model architectures](#example-model-architectures).


Tested with Python 3.10.6

Required python modules (working version)
> NumPy (2.0.1)<br>
> Pillow  (10.4.0)<br>
> Matplotlib (3.9.1)<br>

You can execute `pip install -r requirements.txt` in order to install these modules.

# Examples

  > ***The following example videos are sped-up and low quality due to GitHub's file size limit of 10MB. Also, the option "Use grayscale images" is not needed anymore***

### <p align="center">Linear regression (`houses.csv`)</p>


https://github.com/user-attachments/assets/18eecd0d-ce30-4e88-93ee-4f7696d807b0

---

### <p align="center">Logistic regression (`framingham.csv`)</p>



https://github.com/user-attachments/assets/2dc4374f-e046-46a3-8d25-04c4f4c05db3

---

### <p align="center">Multi-label classification (3-folds cross-validation) (`framingham.csv`)</p>
  > ***The plots disappear here because I minimized them***

https://github.com/user-attachments/assets/9af34dbf-9ef9-448b-8d8d-f3e5a97f0fad


---

### <p align="center">Multi-class classification (`mnist_png.rar`)</p>



https://github.com/user-attachments/assets/479b44f2-3759-4462-a155-1116f7a43019

---

### <p align="center">Multi-class classification (RGB) (`cars.rar`)</p>



https://github.com/user-attachments/assets/d2557369-eee7-42d3-ab3f-01ee4c0c41d4


---

### <p align="center">Labeling images</p>



https://github.com/user-attachments/assets/c983ead9-d4a5-4548-bea0-0289c64a371f

---

### <p align="center">Resizing images</p>


https://github.com/user-attachments/assets/5b47e22f-8db8-405c-8dc3-d6ec64956d0d


# How to use

## Training a model (from scratch or by retraining an already existing model):
1. Setup the model - [Model setup](#model-setup)
2. Adjust the hyperparameters - [Setting up the hyperparameters](#setting-up-the-hyperparameters)
3. Run the script `main.py` and adjust the settings according to the prompts - [Running the script](#running-the-script)
4. Wait for the training to finish
5. **(Optional)** Predict and/or evaluate metrics on final model - [Prediction](#prediction)
6. **(Optional)** Save the model -  [Model loading and saving](#model-loading-and-saving)

## Loading a model for prediction only:
1. Load the model (must be within the folder `models`)
2. Type `1` for prediction only when prompted

## Defining a model from scratch
You can setup the architecture right below the line `if not loaded_model:` and before `model.setup_done()`, by running the relevant setup functions of the `model` variable.

A `Layer` instance defines a layer.
#### Layer class
  > class Layer<br>
  > `num`: Layer number (starts at `0`)<br>
  > `dim`: Number of nodes<br>
  > `type`: `0` (input), `1` (hidden), `2` (output)<br>
  > `activation_function`: One of the implemented activation nfunctions<br>
  > `regularize`: Whether the layer uses regularization<br>
  > `l1_rate`: L1 Regularization rate<br>
  > `l2_rate`: L2 Regularization rate<br>

## Activation functions
  > `linear`: Linear<br>
  > `relu`: Rectified Linear Unit<br>
  > `sigmoid`: Sigmoid<br>
  > `softmax`: Softmax

## Loss functions
  > `loss_mse`: Mean squared error<br>
  > `loss_binary_crossentropy`: Binary cross-entropy<br>
  > `loss_categorical_crossentropy`: Categorical cross-entropy

## Model setup

> There needs to be exactly 1 input layer and 1 output layer, and they can only be added once.<br>
> The output layer and every hidden layer each need their own activation function<br>
> The model needs a loss function<br>
> `activation_function`: One of the activation functions at [Activation functions](#activation-functions)<br>
> `loss_function`: One of the loss functions at [Loss functions](#loss-functions)<br>

1. Add however many layers you want by calling `model.add_layer(num, dim, type)`<br><br>
2. An activation function can be set by calling `model.set_activation_function(num, activation_function)`<br><br>
3. Every model needs a loss function, which can be set by calling `model.set_loss_function(loss_function)`<br><br>
4. Lastly, you need to call `model.setup_done(is_loaded_model=False)` to ensure that the model architecture is valid (i.e. if there are no missing layers, the layer types are correct, the layers also get sorted by their `num` at this point, etc.)<br><br>

## Setting up the hyperparameters
  There are six hyperparameters:
  > Learning rate: `lr`<br><br>
  > Batch size: `batch_size`<br><br>
  > Steps: `steps`<br><br>
  > **(Optional)** Plot update interval (if enabled, and in number of batches): `plot_update_every_n_batches`<br><br>
  > **(Optional)** L1 & L2 regularization rates: `model.set_regularization(num, l1_rate, l2_rate)`<br><br>

## Dataset setup
### Comma-delimited '`,`' dataset
The dataset must be within the `datasets` folder, and be comma-separated (i.e. each variable is separated by a comma '`,`' and each sample is separated y a `newline`/`\n`).
You can also optionally filter the dataset at runtime.

The dataset loading/filtering process is further explained in the sections [Setting up a comma-delimited dataset](#setting-up-a-comma-delimited-dataset) and [(Optional) Filtering a comma-delimited dataset](#optional-filtering-a-comma-delimited-dataset).

> If enabled, the `test` and/or `hold-out` set(s) will be sampled from the dataset based on the user given percentage.
### Image dataset
>**I have only tested 3 types of images so far - RGB, RGBA, and byte sized.**

> **If you are not on windows and need to resize images and/or write labels, you can simply execute the Python files directly (i.e. `python resize.py` and/or `python write_labels.py`). This is what the `.bat` file does.**

For an image dataset you need 3 things:
> 1. Images for the training set, of the same dimension (do not need to be square)
> 2. Unique labels for each of these images in a file called `labels.txt`
> 3. Place all of the images, as well as the respective `labels.txt`, in the `images/train` folder

As for the **(optional)** test set, you can either choose a percentage of the images within the `train` folder as the `test` set, or store them separately in the `images/test` folder. If you choose the latter, the images in `images/test` also need their respective `labels.txt` file.

#### Label images

> A video example on how to automatically label images is shown in [`Labeling images`](#labeling-images)

The `labels.txt` file is in the format `filename,class`, where `filename` is the image filename, `class` is an integer in the range `[0, k]`, where `k` is the number of classes in the model, and each new line represents a separate image.

You can manually fill in the `labels.txt` file or you can also automatically generate it.

The latter requires that you are able to separate all class images into their individual folders. (i.e. if you're using MNIST, you would separate all images of a `0` in a folder called `0`, of a `1` in a folder called `1`, and so on...)

For automatic labeling:
1. Separate each image pertaining to an individual class into its own folder
2. Give each folder an unique integer as the name, where the integer ranges from `0` to the number of classes (i.e. 0-9 for MNIST)
3. Copy and paste the file(s) `utils/write_labels.py` (and `utils/write_labels.bat` if on Windows), and paste on each of these class folders
4. If on Windows, run `write_labels.bat` on each folder, otherwise directly execute `python write_labels.py`. This creates a `labels.txt` file for each folder
5. Copy all of those images into the `images/train` folder
6. Concatenate the contents of each `labels.txt` into a new `labels.txt` that also goes into `images/train`

#### **(Optional)** Resize all images to the same dimension

> A video example on how to resize images is shown in [`Resizing images`](#resizing-images)

1. Place the files `utils/resize.py` and `utils/resize.bat` in the same folder as the images you want to resize
2. Open `resize.py` with a text editor and change the variables `WIDTH` and `HEIGHT` to the width and height you're targeting for all the images
3. Execute the file `resize.bat`. (Or directly call `python resize.py` if not on Windows)
4. A new folder called `resized` will be created with all of the resized images inside of it.

## Running the script
Once everything is setup, you can run `main.py`.

Firstly, you will be prompted whether you want to load a `.json` model config file.
- If you type in `y`, you will then be prompted for the name of the model file (excluding the `.json` extension), which can be re-trained and/or used for predictions.
- If you type in `n`, the model architecture will be the one defined in the `main.py` file. If you're training a model from scratch you will always need to define a model architecture in this file beforehand.

Then you can choose in which probability distribution you want the weights to be initialized in, `0` for a normal distribution or `1` for a uniform distribution.

Now you will be prompted whether this model will be trained on images or not. Type `y` for yes or `n` for no.

Then you need choose whether to train the model or go straight into predicting. Type `0` for training or `1` for predicting.

You will be asked whether to plot any model metrics (i.e. training cost, `r^2`, `accuracy`, etc..). Type `y` for yes or `n` for no.

Similarly, you will be prompted whether to perform cross-validation (hold-out (`0`) or k-folds (`1`)), whether to use a test set, and whether to shuffle the dataset (all of it, including the subsets).

> The function `randomize_dataset` shuffles the dataset by swapping column vectors at random from the given `samples` and `dependent_values` matrices, before they are (optionally) partitioned into a `test` set or `hold-out` set.

Now, if you chose to train the model, you will then be prompted about the dataset.

If it is an image model you will be prompted whether you want to give names to the classes, as by default they're integers in the range `[0, c]`, where `c` is the number of classes in the last layer of the model.

If it is a dataset that is comma-delimited, you will need to choose the independent and dependent variables, as can be seen in the section [Setting up a comma-delimited dataset](#setting-up-a-comma-delimited-dataset) below.

---
### Setting up a comma-delimited dataset
> Each possible feature/class is a separate column, separated by a comma.
1. Input the dataset filename, including the extension (must be within the `datasets` folder)
2. Type in `0` in order to select by column/feature name, or `1` in order to select by the number (column index)
3. Type in the feature identifier by whichever mean you chose to
4. Type in `i` in order to select the variale as independent (used to predict) or `d` as dependent (to be predicted)
5. Now you need to choose whether to keep choosing independent and/or dependent variables or continue running the script. Type in `y` to choose another feature/class or `n` in order to continue the script.<br>
    - > You can keep doing this as many times as you need. Each model needs at least one independent variable and one dependent variable. 
6. Type in `y` in order to filter the dataset or `n` otherwise. More information on dataset filtering available in the below section [(Optional) Filtering a comma-delimited dataset](#optional-filtering-a-comma-delimited-dataset)

### (Optional) Filtering a comma-delimited dataset
> The following are valid comparison operators: `==`, `!=`, `<`, `<=`, `>=`, `>`

Here you can filter out samples from the dataset by making comparisons using the comparison operators listed above. This works for both independent and dependent variables. You can also do it multiple times. The filtered variables do not need to be the independent or dependent variables, as long as they are in the dataset.

1. Select which variable(s) to filter, by typing in the variable(s) name, while separated by commas if filtering more than one variable at once.
   > i.e. 'area_m2', 'area_m2,price_brl', 'lat,lon', etc...
2. Type the target number which will be compared against the given variable
3. Type in a valid comparison operator (one of `==`, `!=`, `<`, `<=`, `>=` or `>`)
4. Type in `y` to keep filtering or `n` to continue the script.
---

Then, finally and similarly, you will be prompted whether to normalize the dataset.
  > Normalizing puts all of the input features of each sample in the entire dataset within the range [-1, 1]. Then it subtracts their mean (which is calculated after normalizing to this range), then divides by the standard deviation (calculated at the same time as the mean).

Now you have to wait for the model to finish training.

Meanwhile, if you chose to plot anything, you will see the relevant graph(s), which includes the training set(s) loss(es) side by side with the performance metric(s) (i.e. r^2 for regression models, or accuracy, precision, recall and f1-score for classification models).

Once finished, you can optionally make predictions and also optionally save the model.

* If you are making predictions on a model trained on a comma-delimited '`,`' dataset, you will have to input the feature values manually for each of them.
* If it is an image model, you will have to input the filename of your desired image within the `images/predict` folder.

---

# Implementation details

### dataset.py
The `dataset.py` script has the class implementation of `Dataset`, which is meant to hold information about a raw comma-delimited dataset, though not yet ready to work with in the main script, so some further processing is done in `main.py` after loading it.

Whenever an instance of the class is created, the user will be prompted for a comma-delimited dataset file within the `datasets` folder, which gets parsed internally and has other setup related prompts for use with the main script, such as choosing independent and dependent variables for training (which ideally should be encasulapted within the `Model` class implementation, but my focus was on functionality first). This setup is further explained in the section [Setting up a comma-delimited dataset](#setting-up-a-comma-delimited-dataset).

It also contains the static function `Dataset.normalize_helper`, which is used to put features within the range `[-1, 1]` when normalizing. The function takes as arguments the `previous value`, `previous minimum feature value`, `previous maximum feature value`, `new minimum feature value`, and `new maximum feature value`.

### main.py
> The matrix multiplication order is left-to-right. (i.e. weights * input, where `input` is a row vector)

The loss functions are executed on the predicted value and the observed value (correct label/dependent variable).

Their derivatives are implemented the same way, except that they also take into account the (current batch) number of training samples for averaging.

The activation functions and their derivatives are implemented taking `z` (node input) as the input, except for softmax, which also takes `z_l`, which is the input to all of the current layer's nodes, as well as its derivative.

#### Network class
> class Network<br>
> `total_batches`: each batch, used for plotting<br>
> `costs`: each batch cost, used for plotting and printing<br>
> `crossvalidation_costs`: cost for holdout/k-folds average<br>
> `finished_training`: keeps track of whether the model has finished training<br>

#### Model class

The class that defines a model is `Model`, and you can make as many instances as you want. An instance of this class stores information relevant to the model and the setup functions. However, since throughout this entire time I have only experimented with an individual model (outside of `k-folds` cross-validation), I have made a global variable `model` which defines the model to be defined/loaded_into, so you may encounter issues doing otherwise without making changes to the code.

It stores the layers, weights, loss function, independent and dependent variables, their types, normalization status, whether the model is an image model, among other things.

You can instantiate as many models as you want. However, as it currently stands, unless you manually modify the code, the only instance of the `Model` class that will be relevant is instantiated as the variable `model`. (Copies of that model are also instantiated when using `k-folds`)

Once instantiated, you will need to setup the model. This is explained in detail in the section [Model setup](#model-setup).

#### Model loading and saving
You can load a model by calling `load_model` with the model name (excluding the extension), which must be within the `models/` folder.

You can save a model by calling `save_model` with the model instance and the model name (excluding the extension), which will be stored in the `models/` folder.

#### JSON model format
> Image model files have an extra property, right after `is_image_model`, called `image_dim` which is a 2-element list of the width and height of images like `[106, 80]`<br>
**Model configuration file format example**
```
{
    "normalized": `true`,
    "is_image_model": `false`,
    "layers": {
        "0": {
            "dimension": 1,
            "activation": null,
            "regularization": {
                "regularize": false,
                "l1_rate": 0,
                "l2_rate": 0
            }
        },
        "1": {
            "dimension": 1,
            "activation": "linear",
            "regularization": {
                "regularize": false,
                "l1_rate": 0,
                "l2_rate": 0
            }
        }
    },
    "loss": "mse",
    "class_list": [
        "price_brl"
    ],
    "feature_list": [
        "area_m2"
    ],
    "feature_types": [
        "float"
    ],
    "feature_min_maxes": [
        [
            53.0,
            252.0
        ]
    ],
    "sample_features_mean": [
        -0.39734208231239004
    ],
    "sample_dependent_variables_mean": [
        663169.2696930332
    ],
    "sample_features_variance": [
        0.22446086612788888
    ],
    "sample_features_std": [
        0.4737730111856192
    ],
    "weights": {
        "0": [
            [
                660663.8870276996,
                170975.81822938874
            ]
        ]
    }
}
```

#### Feedforward
The `feedforward` function goes through the model starting at the given layer, with the given input, then returns the last layer's output.

If `cache` is `True`, which is the case when training, each layer's outputs and inputs are stored (primarily for backpropagation) then reset for the next sample. Otherwise, which is the case when predicting, `cache` is `False`, and those values are not stored.

1. If starting at layer `0`, with `cache` being `True`, it stores the `input` in the list of inputs and outputs for the current sample.

2. Multiplies the current layer's weights by the previous layer's outputs (the `input` to the function), and stores the output in `z_l`.

3. The current layer's activation function is executed for each of the inputs in `z_l`, with the input to the node itself as the argument, and also the entire layer `z_l` in the case of `softmax`. The result is stored in the output matrix `a_l`.

4. The function is recursively called with the current layer's output as the next layer's input and `layer_num` gets incremented by 1, until it reaches the last layer then returns the final output.

> Layer weight matrices `model.weights[-1][layer_num - 1]` are of dimension `m x (1 + n)`, where `m` is the number of nodes in the current layer and `n` is the number of nodes in the previous layer (+ 1 for the bias)<br><br>
> Layer inputs (`z_l`) are a matrix of dimension `m x 1`, where `m` is the number of nodes in the given layer<br><br>
> Layer outputs (`a_l`) are a matrix of dimension `n x 1`, where `n` is the number of nodes in the given layer<br>
> `z_l` and `a_l` are supposed to always have the same dimension - that of the current layer

#### Backpropagation

The backpropagation function goes over the network backwards and calculates relevant values for the calculation of the partial derivatives for each weight variable of each layer, then returns that list.

The following steps are looped through each node, then through each layer backwards, starting at the last layer.
Most of the node indexing notation is ommitted for readability, but is implemented in the code.

First, the given node's `errors` are calculated.
> The `error` of a particular node in layer `k` and index `i` is defined as `dC/dz_k[i]` (the derivative of the cost with respect to the input of the node `i` within the layer `k`)<br><br>
> The weight indexing is 1 less than the layer indexing. (i.e. if at layer `1`, the weights that connect the input layer `0` to layer `1` are stored in `model.weights[0]`)

When at the last layer, `dC/dz_k` is calculated, by separating the steps through the chain rule.

1. `dC/da_k` is calculated (derivative of the cost wrt the layer's output)
	> The derivatives for the loss functions are individually implemented as the functions `binary_crossentropy_derivative`, `categorical_crossentropy_derivative`, and `mse_derivative`.

2. The result gets added to the `node_error` variable.

3. `da_k/dz_k` is calculated (derivative of the layer's activation function wrt to the layer's input)
  > Similarly, the activation function derivatives are also individually implemented as the functions `linear_derivative`, `relu_derivative`, `sigmoid_derivative`, and `softmax_derivative`.

4. The `node_error` variable gets multiplied by it, for the chain rule.


If at a hidden layer, the `dC/da_k` calculation is replaced by the calculation of `dz_k/da_k-1` (derivative of the current layer's input wrt the previous layer's output), as the derivative with respect to the cost only needs to be calculated once.

The value of `k` decreases for every layer in the loop (though in the script the variable that keeps track of the current layer increases).

That is it for the `error` calculation of a given node in a given layer `k`.

In order to get `dC/W_k` (partial derivatives of the cost wrt the weights in the given layer) after already calculating `dC/dz_k` we just need to multiply it by `dz_k/dW_k` (derivative of the layer's input wrt that layer's connection weights).

`dz_k/dW_k` happens to be the previous layer's output nodes (`a_lm1[prev_node - 1, 0]`, the `- 1` accounts for the bias).

Finally, the error of each node in the current layer must be multiplied by the previous layer's output to get the final partial derivatives for that layer/node.

The process repeats for all hidden layers and their nodes until the input layer is reached, then the partials are returned.

> Note that I mentioned that the implementation of the chain rule is done separately for each function and derivative, but this is not the case for the `softmax` function combined with `loss_categorical_crossentropy` (so for multi-class classification model, such as MNIST). <br>
> When working out the math, certain variables cancel out, but since in the script it's done in separate steps, one of the derivatives (`dC/da_k`) always returns `0` and as the chain rule is just multiplication, all partials and errors become `0`. So I decided to directly solve for this edge case by implementing the simplified alternative which includes both derivatives.<br>
> For this reason, when the combination is `softmax` as the last layer's activation function and `loss_categorical_crossentropy` for the loss function, the node `error` calculation is implemented in a single line: `node_error = (a_l[node, 0] - observed[0, node]) / n_training_samples`.<br>
> The error itself is the layer's output `a_l` minus the real `observed` value for the given input sample. The division by `n_training_samples` is there to average the partial derivatives over the batch size. This should be done within the loss function derivative when at the last layer, but since for this specific case it was simplified to one step, this needs to be done directly.

#### Image loading

`Pillow` is used for image loading whenever using an image model. You can get the pixel matrix of an image by calling `get_image_pixel_matrix`, and passing the absolute or relative folder path and the image filename. If the `model` argument to this function is not `None` and it doesn't have a previously set `image_bit_depth` variable, the function will set this member variable to be whatever the bit depth for this image has.

The function `load_image_dataset` loads the image dataset from the given `folder`. This folder must have a `labels.txt` file with unique filenames and correct formatting, as well as the respective images.

The images are all assumed to have the same dimension (not necessary to be square) and the same bit depth.

This function reads and parses all images and labels into matrices, retrieves other model relevant information, and also allows you to name each image class if called with a given model as an argument along with `unique_dataset = True`, and stores them in the local variables `samples`, `labels`, `feature_list`, `feature_types`, `class_list`, `image_dim`, `image_bit_depth`, which the function returns.

If you are only interested in loading the dataset irrespective of the model, which is the general case if you only want to load the samples and labels but already have a model with the relevant information (i.e. image bit depth, image dimension, pixel count and classes), you can run it with the argument `unique_dataset = False`. This is the case for the `hold-out` and `test` sets, for example.

You can plot an image from a sample matrix retrieved by calling `get_image_pixel_matrix` with the function `plot_image_from_sample`, by passing the model as argument and the sample matrix.

#### Normalization

> If normalizing, the dataset is first normalized to the range `[-1, 1]`, then the mean and standard deviation are measured<br>

When ran with `update_min_maxes` as `True`, the function `mean_n_variance_normalize` goes through the given dataset and stores the minimum and maximum values for each of the features and this is what `feature_min_maxes` is used for. `feature_min_maxes` is a 2-dimensional array where the first column is the feature index, and in the second column the minimum (index `0`) and maximum (index `1`) value for that feature.

If the model is an image model, as of right now, the minimum and maximum value for each feature is hardcoded to, respectively, `0` and `255`.

The function `new_normalize` uses the `feature_min_maxes` and the `Dataset.normalize_helper` function in order to normalize the samples to the range `[-1 ,1]`.

If ran with `update_min_maxes` as `False` (which is the case when normalization is enabled), the dataset will be mean normalized (subtracts the mean of the given feature for each feature), and standardized (divides by the standard deviation of the given feature for each feature). (Even then it is still called with `update_min_maxes = False` beforehand, in order to store the minimum and maximum values for the normalization), 

#### Dataset metrics evaluation

The function `measure_model_on_dataset` can be called with the relevant data after loading a dataset and a model, in order to measure the performance of the model on that dataset. The resulting measures are stored in the `model_metrics`, `micro_metrics`, `macro_metrics`, and `class_metrics` dictionaries and list that are passed as arguments.<br>

This function gets called `k` times for the `k` models if using `k-folds` cross-validation, for every batch.<br>
It gets called at least once when training finishes if a test set is present.<br>
The total cost is always measured.<br>
For regression models, `r^2` is measured<br>
For classification models, `accuracy`, `recall`, `precision`, and `f1-score` are measured.<br><br>

> Once a model has been measured for some dataset, you can call `print_model_metrics` to print the relevant metrics, with the `model_metrics`, `micro_metrics`, `macro_metrics`, and `class_metrics` that hold the metrics as arguments to the function.

#### Prediction

The `predict_prompt` function handles the prediction of values for a model. It initializes an input matrix of the same dimension as the input layer and prepends a `1` to account for the bias.<br>
If the model used a regular comma-delimited dataset, you will then be prompted for the inputs for each feature.<br>
If the model used an image dataset, it will load the image given as input by the user within the `images/predict` folder for prediction.<br>
Once the input is properly loaded, if the model was normalized, the input also gets normalized by normalizing the features to [-1, 1], then subtracting the mean and dividing by the standard deviation that were measured for the training set after also normalizing the training set to [-1, 1].
Finally, it performs forward propagation on the model with this matrix as the input, then prints the result to the console. If the model is an image model, the image is also plotted.

> The function `nn.predict` runs the forward propagation while ignoring first layer.

#### Training
The function `nn.train` starts the actual training process. It initializes some variables which will be filled during training, then loops through the number of training steps and calls the `nn.step` function. Whenever a step finishes, the current step, training cost, and optionally cross-validation cost are printed to the console. Once all steps are done, the variable `finished_training` is set to `True`.

It takes as arguments the training parameters, training samples, and the model to be trained (the `model` argument defaults to `None` if performing `k-folds` cross-validation), and initializes empty lists for the variables `total_batches`, `costs`, `crossvalidation_costs`, `models_model_metrics`, `models_micro_metrics`, `models_macro_metrics`, `models_class_metrics`, as those are filled during training.

The `step` function performs the training step. The behavior when `k-folds` is slightly different from a regular model, as multiple models are measured at once.

For a regular model, the following is the process for each sample, as it goes through all of the training sample:

1. The model's `layers_a` and `layers_z` member variables are initialized to empty lists, as this information is necessary for backpropagation and is different for each sample.
2. Forward propagation is performed on the model, starting from the input layer, with the current training sample.
3. The sample loss is then calculated based on the model, the output of the forward propagation, and the observed values (correct dependent values/labels).
4. The sample loss is added to the `average_cost` variable, which will later be averaged for the total batch loss.
5. Backpropagation is performed, and the resulting partial derivatives are stored in the `sample_partials` variable, which gets appended to the `batch_partials[0]` list variable (the `0` index here stands for 0th model, as for `k-folds` there is more than one model). This will also get averaged for the total batch loss.
6. Whenever the number of batch samples is reached:
	1. The `total_batches` variable which is used for plotting gets appended with the number of the current batch.
	2. The batch cost gets averaged over the number of samples in the current batch.
	3. The batch cost gets appended to the `costs` variable, which holds the cost of each batch and is used for plotting and printing the cost to the console (as of right now, the last batch's average cost is always the one printed and plotted).
	4. (Optional) If enabled, the model gets evaluated on the `hold-out` set.
	5. (Optional) If enabled and the desired number of batches set through the variable `plot_update_every_n_batches` have been iterated, the plot is drawn
	6. The batch partials get averaged over the number of samples in the current batch, and the list gets cleared for the next batch.
	7. Gradient descent is performed on the model, with the given partial derivatives, learning rate, and number of samples processed in the current batch

The behavior for `k-folds` is similar, except that for each sample the process also loops through each of the `k` models.

The function `sample_loss` calculates the sample loss for a given model, prediction, and observation. It basically goes through each output node and calls the model's loss function with the prediction and observation. The total sample loss is the accumulation of each output node's individual loss. Regularization is then accounted for if enabled.

The `gradient_descent` function updates the model's weights given the partial derivatives, learning rate, and number of samples in the current batch. It also takes into account the regularization if enabled.

Though both terms are somewhat interchangeable, I mostly reserved the `Network` class to handle the training, predicting, plotting, and sometimes performance measuring of arbitrary models. Meanwhile, the `Model` instance holds information pertaining to the model architecture, its weights, and the dataset it is trained on.

The model architecture gets defined beforehand, as explained in further detail in the section [Model setup](#model-setup).

When running the script, a few prompts will ask you for your preferences on certain settings.

The training set is always present, while the test and hold-out sets are optional. In the case of k-folds, the entire training set gets split into `k` models, where each of the `k` models have their own subsets of the training set as their training set and test set.

The training, test, and hold-out sets are all matrices of dimension `(n + 1) x m`, where `n` is the number of features the dataset has plus a prepended 1 at the beginning, which accounts for the bias weight, while `m` is the number of samples.

As for the dependent/observed values, they are matrices of dimension `m x n`, where `m` is the number of samples and `n` is the number of classes/dependent variables. Depending on the context, it may be internally transposed in certain functions or sections of the code, but generally it is `m x n` (i.e. in the case of `dependent_values`).

The dataset is loaded by the script `dataset.py` (but called in `main.py`), although some other things are done in `main.py`. It just parses through a comma-delimited file, separates the features, and stores the relevant information.

Whenever running `k-folds` cross-validation, the models are not saved and you are not able to predict anything after training, which is by design as it is mostly used as a form of evaluating the model performance. Ideally you should still be able to do these things since there's no real downside to allowing them, but combining this with the time spent implementing everything else I decided to leave it as is, for now.

The function `nn.reset_training_info` resets the member variables which keep track of the training progress. As it stands the script doesn't use this, but you could add it as you please. This could be useful for training a model after another one has already been trained.
# Example model architectures
Below are the models that were used for testing and experimenting, which are seemingly somewhat accurate
## Linear regression (`houses.csv`)
- Independent variable: `area_m2` (Area in m^2 of a house)
- Dependent variable: `price_brl` (Price in Brazilian reais)

>**Model**<br><br>
>--
>Layers
> - Input
>   - Nodes: 1
> - Output
>   - Nodes: 1
>   - Activation function: `linear`<br>
> ---
> Loss function: `loss_mse`
- Normalized
- `lr`: 0.1
- `batch_size`: 11293
- `steps`: 25

## Logistic regression (`framingham.csv`)
- Independent variable: `sysBP`  (Systolic blood pressure)
- Dependent variable: `prevalentHyp` (Whether the person has a hypertension diagnosis or not)
>**Model**<br><br>
> ---
>Layers
> - Input
>   - Nodes: 1
> - Hidden
>   - Nodes: 5
>   - Activation function: `relu`
> - Output
>   - Nodes: 1
>   - Activation function: `sigmoid`<br>
> ---
> Loss function: `loss_binary_crossentropy`
- Normalized
- `lr`: 1
- `batch_size`: 300
- `steps`: 50

## Multi-label classification (`framingham.csv`)
- Independent variables: `age`, `totChol`, `sysBP`, `glucose` (Person age, cholesterol, systolic blood pressure, and glucose)
- Dependent variables: `prevalentHyp`, `prevalentStroke`, `currentSmoker`, `diabetes` (Whether person is diagnosed with hypertension, has had a stroke, is currently a smoker, and has diabetes)
>**Model**<br><br>
>---
>Layers
> - Input
>   - Nodes: 4
> - Hidden
>   - Nodes: 50
>   - Activation function: `relu`
> - Output
>   - Nodes: 4
>   - Activation function: `sigmoid`<br>
>---
> Loss function: `loss_binary_crossentropy`
- Normalized
- `lr`: 1
- `batch_size`: 300
- `steps`: 230

## Multi-class classification (`mnist_labeled.rar`/`mnist.rar`)
- Independent variables: Image pixels (28*28, byte-sized color channel)
- Dependent variables: Integers in range [0, 9]
>**Model**<br><br>
> ---
>Layers
> - Input
>   - Nodes: 784 (28x28)
> - Hidden
>   - Nodes: 50
>   - Activation function: `relu`
> - Output
>   - Nodes: 10
>   - Activation function: `softmax`<br>
> ---
> Loss function: `loss_categorical_crossentropy`
- Normalized
- `lr`: 0.03
- `batch_size`: 32
- `steps`: 5
- Only used 50% of the entire training set by reserving the other 50% for a test set

## Multi-class classification (RGB) (`cars_labeled.rar`/`cars.rar`)
- Independent variable: Image pixels (106*80*3, 3 RGB color channels)
- Dependent variables: Integers in range [0, 2] that I renamed to the respective car colors ["black", "blue", "red"]
>**Model**<br><br>
>Layers
> - Input
>   - Nodes: 25440 (106x80x3)
> - Hidden
>   - Nodes: 50
>   - Activation function: `relu`
> - Output
>   - Nodes: 3
>   - Activation function: `softmax`<br>
>
> Loss function: `loss_categorical_crossentropy`
- Normalized
- `lr`: 0.005
- `batch_size`: 32
- `steps`: 5

--------------------------

# Features

- Model types
  - Linear regression
  - Logistic regression
  - Multi-label classification
  - Multi-class classification
- Activation functions
  - Linear: `linear`
  - Rectified Linear Unit: `relu`
  - Sigmoid: `sigmoid`
  - Softmax: `softmax`
- Loss functions
  - Mean squared error: `loss_mse`
  - Binary cross-entropy: `loss_binary_crossentropy`
  - Categorical cross-entropy: `loss_categorical_crossentropy`
- Accepts comma-delimited '`,`' files and images
- (Optional) Save and load models (.json files with model/dataset metadata)
- (Optional) Normalization (normalizes to range [-1, 1], mean normalizes, then standardizes)
- (Optional) L1 & L2 regularization
- Arbitrary batch size
- Arbitrary number of layers and nodes
- Xavier (linear, sigmoid, softmax) and HE (relu) weight initialization (optionally normally or uniformly distributed)
- Loading and filtering dataset (comma-delimited '`,`') by column names or column numbers, choose dependent or independent variable, and (optionally) filter by comparison operators (i.e. keep only the values for the given column which are `>`, `<`, `==`, or `!=` to a certain value)
- (Optional) Shuffle dataset
- (Optional) Measuring/evaluating the dataset for performance metrics on an arbitrary model (the function responsible for this is `measure_model_on_dataset` @`main.py`)
- (Optional) Hold-out, k-folds cross-validation, and test sets
- (Optional) Plotting cost graphs and test/hold-out set performance metrics (i.e. `r^2` for regression models, or `accuracy`, `precision`, `recall`, and `f1-score` for classification models)
- (Optional) Arbitrary names for classes of image multi-class classification models
  
# Notes

- Currently, the only accepted datasets are files delimited by a comma '`,`' and images

- Normalization (prior to mean normalization and standardizing) gets the features within the range [-1, 1], by design choice
  
- The dataset loader skips the first row (which should be the column/variable names) and assumes that each variable is a different column

- The class names for a model trained on a comma-delimited '`,`' dataset will be whatever column names they had. Manwhile, for images, they will be integers increasing from `0` to `k`, where `k` is the number of classes. You can optionally manually input a label for each of those classes, in the case of an image dataset
  
- Expect bugs, inaccuracies, lack of speed, high memory consumption, and general lack of optimization
  
- A practical example of the script exceeding available memory and crashing, for me with 32GB RAM, was running the MNIST example, but instead of using a batch size of 32 as shown in the example video, using a batch size of 1 caused it to crash in between the 4th and 5th steps
  
- Currently the script needs images of the same dimension (no need to be square), as the input layer's dimension is pre-defined
  
- If the batch size is set to a value that is greater than the number of training samples, it is clipped to be the number of training samples. (You can also use this to force batch gradient descent when you don't know the exact number of training samples, as any value greater than that will suffice)
  
- Do not normalize input if the model is trained on images and the images are not in the range `[0, 255]`. For example if every pixel is `0` or `1`. This is because I hardcoded all normalized images' pixels/features to assume to be in the range `[0, 255]`

- When training a model with `k-folds` cross-validation, you won't be able to make predictions afterward, nor will you be prompted whether to save a model. It is only used as a means of performance metrics evaluation. This is by design

- The script expects all images to have pixel colors in the range `[0, 255]` for normalization. This can be changed within the function `mean_n_variance_normalize`, by changing the feature min maxes from `0` and `255` to whatever minimum and maximum values you're using, respectively

- When training a model with k-folds, if the number of training samples is not exactly divisible by the number of folds, the spare sample will be ignored

- Image labels cannot have duplicate filenames in the `labels.txt` text file, as they are uniquely identified by them

- You can print a model's performance metrics (if it has been measured prior) with the function `print_model_metrics`

- Strings as inputs haven't been properly implemented nor tested
  
- The `model` class members `sample_features_mean`, `sample_features_std`, `sample_features_variance`, are measured over the training samples (if normalization was enabled, this is done after the input is in the range [-1, 1], but before mean normalizing and standardizing).<br>
Meanwhile, the non `model` class members, such as `training_features_mean` and `test_features_mean`, are measured after the input is normalized (if that is the case, otherwise `model.sample_features_mean` is the same as `training_features_mean`, for example). That is because whenever the input needs to be re-normalized, such as in the case of predicting an arbitrary user input once the model is trained, you need to know the `mean` and `std` before mean normalizing and standardizing, but after normalizing to [-1, 1], as well as the minimum and maximum values of each input feature in order to normalize it to [-1 ,1], which is what the variable `model.feature_min_maxes` is used for. These are also used for performance measurements and reversing the normalization process

- Regularization has not been extensively tested

- I could not make a model that performs well when predicting different car brands of the same color (RGB), using a similar architecture to the model that predict
s car colors. I don't know the exact reason for this, although it is evidently a more complex task than the other examples. Possibly solvable with convolutional layers?
  
# Todo

- Implement the same behavior of optionally storing images in a separate `holdout` folder as can be done with the `test` folder

- Option to convert images that are RGB/RGBA into grayscale

- Separate the bias from the normal weights and separately calculate/update it, or leave it as is?

- Estimate time remaining for training to finish based on how long it took for the last step to finish

- Refactor code and standardize variable names

- Decouple and isolate functions, settings, models, and datasets

- Separate `main.py` into multiple scripts for better isolation/readability

- Option to evalute model on a given test set without the need to train it

- Let user arbitrarily choose how often they want the cross-validation model(s) to be evaluated during training, instead of automatically doing it for every single batch

- Implement exponentially moving averages for plotting and let user choose how much to smoothen the graph(s) (if at all)

- Implement means to achieve better performance, such as vectorizing everything, using Jacobian matrices, CUDA (NVIDIA only), and SIMD instructions? I probably won't do this, as the primary goal of this project was learning, and readability is much preferred over performance

- Implement gradient checking

- Implement gradient clipping

- Implement dropout

- Implement layer batch normalization

- Implement convolutional layers

- Implement other optimizers/adaptive learning rates such as adagrad/adam/rmsprop, etc...

- Let user save/predict  any chosen model(s) with k-folds cross-validation once training ends

- Improve performance of the plots

- Automatically save or allow user to choose whether to keep checkpoints of the model at specific points/intervals during training (i.e. save model every 100 steps, or once steps are 500, etc.)

- Let user run script with "default settings" (i.e. the values already defined in the script, such as to normalize, shuffle dataset, no cross-validation, and a test set which is 25% of the training samples) mainly for avoiding repetition when testing things out

- Implement hyperparameter tweaking during cross-validation

- Properly implement strings as input and experiment with a sentiment analysis dataset

- Prompt user whether to calculate and store micro statistics, macro statistics and/or model statistics? The prompt might become too cumbersome as it's already bloated with a lot of settings. This can already be done manually, though not tested, by passing `model_metrics`, `micro_metrics`, `macro_metrics` and/or `class_metrics` as `None` to the function `measure_model_on_dataset`, instead of the former 3 being a dictionary and the latter a list. This would be particularly useful in order to increase speed and decrease memory usage for models trained with cross-validation

- Let user input the learning rate, regularization rate including the type (L1 or L2), batch size, and the frequency at which the plot (if enabled) is updated? Also worried about too many settings