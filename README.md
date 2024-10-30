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
  - [Activation functions](#activation-functions)
  - [Loss functions](#loss-functions)
  - [Model setup](#model-setup)
  - [Setting up the hyperparameters](#setting-up-the-hyperparameters)
  - [Dataset loading](#dataset-loading)
    - [Comma delimited `.csv` dataset](#comma-delimited-csv-dataset)
    - [Image dataset](#image-dataset)
      - [Label images](#label-images)
      - [**(Optional)** Resize all images to the same dimension](#optional-resize-all-images-to-the-same-dimension)
  - [Running the script](#running-the-script)
- [How it works](#how-it-works)
- [Example model architectures](#example-model-architectures)
  - [Linear regression (`houses.csv`)](#linear-regression-housescsv-1)
  - [Logistic regression (`framingham.csv`)](#logistic-regression-framinghamcsv-1)
  - [Multi-label classification (`framingham.csv`)](#multi-label-classification-framinghamcsv)
  - [Multi-class classification (`mnist.rar`)](#multi-class-classification-mnistrar)
  - [Multi-class classification (RGB) (`cars.rar`)](#multi-class-classification-rgb-carsrar-1)
- [Features](#features)
- [Notes](#notes)
- [Todo](#todo)

# Summary
A Python implementation from scratch of a neural network (using NumPy for matrix operations), made primarily for learning purposes.

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

There are two types of datasets that are currently accepted:

1. Comma '`,`' delimited `.csv` files where each variable is in a separate column and each row is an individual sample
2. Images of the same dimension, no need to be square (i.e. 28x28, 106x80, etc.)

---

## Training a model (from scratch or by retraining an already existing model):
1. Setup the model
   > *Define a model architecture in `main.py` or load a `.json` config file within the `models` folder*
2. Adjust the hyperparameters
   > - Learning rate: `lr`
   > - Batch size: `batch_size`
   > - Training steps: `steps`
   > - **(Optional)** Plot update interval: `plot_update_every_n_batches`
3. Run the script
4. Adjust the settings by following the prompts
5. Wait for the training to finish
6. **(Optional)** Predict and/or evaluate metrics on final model
7. **(Optional)** Save the model


## Loading a model for prediction only:
1. Load the model (must be within the folder `models`)
2. Type `1` for prediction only when prompted

## Defining a model from scratch
  You can setup the architecture right below the line `if not loaded_model:` and before `model.setup_done()`, by running the relevant setup functions of the `model` variable.

  A `Layer` instance defines a layer.
  A layer has:
  > `num`: Starts at `0` and identifies the position of the layer within the network. You don't need to add layers in ascending order, as they will get sorted based on this variable later when `setup_done` is called.<br><br>
  > `dim`: Number of nodes within the layer.<br>
  > `type`: `0` (input layer), `1` (hidden layer), or `2` (output layer).<br>
  > **(Optional)** `l1_rate`: L1 regularization rate<br>
  > **(Optional)** `l2_rate`: L2 regularization rate

## Activation functions
  > `linear`: Linear<br>
  > `relu`: Rectified Linear Unit<br>
  > `sigmoid`: Sigmoid<br>
  > `softmax`: Softmax

## Loss functions
  > `loss_mse`: Mean squared error<br>
  > `loss_binary_crossentropy`: Binary cross-entropy<br>
  > `loss_categorical_crossentropy`: Categorical cross-entropy

  The model setup part is pretty intuitive and self-explanatory:

## Model setup
1. Add however many layers you want by calling  `model.add_layer(num, dim, type)`<br><br>
2. Each layer that is not the input layer needs an activation function, which can be set by calling `model.set_activation_function(num, activation_function)`<br><br>
3. Every model needs a loss function, which can be set by calling `model.set_loss_function(loss_function)`<br><br>
4. Lastly, you need to call `model.setup_done(is_loaded_model=False)` to ensure that the model architecture is valid (i.e. if there are no missing layers, the layer types are correct, the layers also get sorted by their `num` at this point, etc.)<br><br>

## Setting up the hyperparameters
  There are six hyperparameters:
  > Learning rate = `lr`<br><br>
  > Batch size (if this variable is greater than the number of training samples, it defaults to the number of training samples) = `batch_size`<br><br>
  > Steps = `steps`<br><br>
  > **(Optional)** Interval (number of batches) in which the plot (if enabled) should be updated = `plot_update_every_n_batches`<br><br>
  > **(Optional)** L1 regularization & L2 regularization rates = `model.set_regularization(num, l1_rate, l2_rate)`<br><br>

## Dataset loading
### Comma delimited `.csv` dataset
> The dataset must be within the `datasets` folder.
You can optionally filter the dataset.

If enabled, the test and/or hold-out set(s) will be sampled from the dataset based on the user given percentage.
### Image dataset
>**I have only tested 3 types of images so far - RGB, RGBA, and byte sized.**

> **If you are not on windows and need to resize images and/or write labels, you can simply execute the Python files directly (i.e. `python resize.py` and/or `python write_labels.py`). This is what the `.bat` file does.**

For an image dataset you need 3 things:
>1. Images for the training set, of the same dimension (do not need to be square)
>2. Unique labels for each of these images in a file called `labels.txt`
>3. Place all of the images, as well as the respective `labels.txt` in the `images/train` folder

As for the **(optional)** test set, you can either choose a percentage of the images within the `train` folder as the test set, or store them separately in the `images/test` folder. If you choose the latter, the images in `images/test` also need their respective `labels.txt` file.

#### Label images

> A video example on how to automatically label images is shown in [`Labeling images`](#labeling-images)

The `labels.txt` file is in the format `filename,class`, where `filename` is the image filename, `class` is an integer in the range [0, k], where `k` is the number of classes in the model, and each new line represents a separate image.

You can manually fill in the `labels.txt` file or you can also automatically generate it.

The latter requires that you are able to separate all class images into their individual folders. (i.e. if you're using MNIST, you would separate all images of a 0 in a folder called `0`, of a 1 in a folder called `1`, and so on...)

For automatic labeling:
1. Separate each image pertaining to an individual class into its own folder
2. Give each folder an unique integer as the name, where the integer ranges from 0 to the number of classes (i.e. 0-9 for MNIST)
3. Copy and paste the file(s) `utils/write_labels.py` (and `utils/write_labels.bat` if on Windows) and paste on each of these class folders
4. If on Windows, run `write_labels.bat`, otherwise directly execute `python write_labels.py`, this creates a `labels.txt` file for each folder
5. Copy all of those images into the `images/train` folder
6. Concatenate the contents of each `labels.txt` into a new `labels.txt` that also goes into `images/train`

#### **(Optional)** Resize all images to the same dimension

> A video example on how to resize images is shown in [`Resizing images`](#resizing-images)

1. Place the files `utils/resize.py` and `utils/resize.bat` in the same folder as the images you want to resize
2. Open `resize.py` with a text editor and change the variables `WIDTH` and `HEIGHT` to the width and height you're targeting for all the images
3. Execute the file `resize.bat`. (Or directly call `python resize.py` if not on Windows)
4. A new folder called `resized` will be created with all of the resize images inside of it.

## Running the script
Once everything is setup, you can run `main.py`.

Firstly, you will be prompted whether you want to load a `.json` model config file.
- If you type in `y`, you will then be prompted for the name of the model file (excluding the `.json` extension), which can be re-trained used for predictions.
- If you type in `n`, the model architecture will be the one defined in the `main.py` file. If you're training a model from scratch you will always need to define a model architecture in this file beforehand.

Then you can choose in which probability distribution you want the weights to be initialized in, `0` for a normal distribution or `1` for a uniform distribution.

Now you will be prompted whether this model will be trained on images or not. Type `y` for yes or `n` for no.

Then you need choose whether to train the model or go straight into predicting. Type `0` for training or `1` for predicting.

If you chose to train the model, you will then be prompted questions about the dataset.

You will be asked whether to plot any model metrics (i.e. training cost, `r^2`, `accuracy`, etc..). Type `y` for yes or `n` for no.

Similarly, you will be prompted whether to perform cross-validation (hold-out (`0`) or k-folds (`1`)), whether to use a test set, and whether to shuffle the dataset (all of it, including the subsets).

(If it is an image model) you will be prompted whether you want to give names to the classes, as by default they're integers in the range [0, c], where `c` is the number of classes in the last layer of the model.

Then, finally and similarly, you will be prompted whether to normalize the dataset.
  > Normalizing puts all of the input features of each sample in the entire dataset within the range [-1, 1]. Then it subtracts their mean (which is calculated after normalizing to this range), then divides by the standard deviation (calculated at the same time as the mean).

Now you have to wait for the model to finish training.

Meanwhile, if you chose to plot anything, you will see the relevant graph(s), which includes the training set(s) loss(es) side by side with the performance metric(s) (i.e. r^2 for regression models, or accuracy, precision, recall and f1-score for classification models).

Once finished, you can optionally make predictions and also optionally save the model.

* If you are making predictions on a model trained on a `.csv` dataset, you will have to input the feature values manually for each of them.
* If it is an image model, you will have to input the filename of your desired image within the `images/predict` folder.

---

# How it works

The class that defines a model is `Model`, and you can make as many instances as you want. An instance of this class stores information relevant to the model. However, since throughout this entire time I have only experimented with an individual model (outside of k-folds cross-validation), I have made a global variable `model` which defines the model to be defined/loaded_into, so you may encounter issues doing otherwise without making changes to the code.

# Example model architectures
> Below are models that I used for testing and experimenting that are seemingly decently accurate
## Linear regression (`houses.csv`)
- Independent variable: `area_m2` (Area in m^2 of a house)
- Dependent variable: `price_brl` (Price in Brazilian reais)
- Normalized
- `lr`: 0.1
- `batch_size`: 11293
- `steps`: 25

## Logistic regression (`framingham.csv`)
- Independent variable: `sysBP`  (Systolic blood pressure)
- Dependent variable: `prevalentHyp` (Whether the person has a hypertension diagnosis or not)
- Normalized
- `lr`: 1
- `batch_size`: 300
- `steps`: 50

## Multi-label classification (`framingham.csv`)
- Independent variables: `age`, `totChol`, `sysBP`, `glucose` (Person age, cholesterol, systolic blood pressure and glucose)
- Dependent variables: `prevalentHyp`, `prevalentStrok`, `currentSmoker`, `diabetes` (Whether person is diagnosed with hypertension, has had a stroke, is currently a smoker, and has diabetes)
- Normalized
- `lr`: 1
- `batch_size`: 300
- `steps`: 230

## Multi-class classification (`mnist.rar`)
- Independent variable: Image pixels (28*28, byte-sized color channel)
- Dependent variables: Integers in range [0, 9]
- Normalized
- `lr`: 0.03
- `batch_size`: 32
- `steps`: 5
- Only used 50% of the entire training set by reserving the other 50% for a test set

## Multi-class classification (RGB) (`cars.rar`)
- Independent variable: Image pixels (106*80*3, 3 RGB color channels)
- Dependent variables: Integers in range [0, 2] that I renamed to the car colors (["black", "blue", "red"])
- Normalized
- `lr`: 0.005
- `batch_size`: 32
- `steps`: 5

--------------------------

# Features

- Linear regression
- Logistic regression
- Multi-label classification
- Multi-class classification
- Accepts `.csv` delimited by `,` and images (those go in the `images/train` folder alongside a `labels.txt` file, which, in each line, contains the name an image within the folder, followed by a comma, and which class it pertains to i.e. `image01.png,0`)
- (Optional) Save and load models (.json files with model metadata such as each layer's weights, model architecture, information on the dataset it was trained, etc...)
- (Optional) Normalization (normalizes to range [-1, 1], mean normalizes, then standardizes)
- (Optional) L1 & L2 regularization
- Arbitrary batch size
- Arbitrary number of layers and nodes
- Xavier (linear, sigmoid, softmax) and HE (relu) weight initialization (optionally normally or uniformly distributed)
- Loading and filtering dataset (.csv) by column names or column numbers, choose dependent or independent variable, and also (optionally) filter by comparison (i.e. keep only the values for the given column which are >, <, ==, or != to a certain value)
- (Optional) Shuffle dataset
- (Optional) Measuring/evaluating dataset for performance metrics on an arbitrary model (the function responsible for this is `measure_model_on_dataset` `@main.py`)
- (Optional) Hold-out, k-folds cross-validation, and test sets
- (Optional) Plotting cost graphs and test/hold-out set performance metrics (i.e. r^2 for regression models, or accuracy, precision, recall, and f1-score for classification models)
- (Optional) Arbitrary names for classes for image multi-class classification models
- Activation functions
  - Linear (`linear`)
  - Rectified Linear Unit (`relu`)
  - Sigmoid (`sigmoid`)
  - Softmax (`softmax`)
- Loss functions
  - Mean squared error (`loss_mse`)
  - Binary cross-entropy (`loss_binary_crossentropy`)
  - Categorical cross-entropy (`loss_categorical_crossentropy`)
# Notes

- Currently, the only accepted datasets are `.csv` files delimited by a comma `,`, and images, which must be put into the folder `images/train` for training (and testing if it was enabled and no separate folder was chosen), `images/predict` for images to predict once the training is finished or after loading a model, and `images/test` if using a test set and a separate folder was chosen in the settings

- Normalization (prior to mean normalization and standardizing) is within the range [-1, 1], by design choice.
- Dataset loader skips first row and expects each variable to be a different column
- The class names for a model trained on a `.csv` dataset will be whatever column names they had, meanwhile, for images they will be integers increasing from `0` to `k` where `k` is the number of classes, or you can optionally manually input a label for each of those classes.
- Expect bugs, inaccuracies, lack of speed, high memory consumption, and general lack of optimization.
- A practical example of the script exceeding available memory and crashing for me was running the MNIST example, but instead of using a batch size of 32 as shown in the video, using a batch size of 1 caused it to crash in between the 4th and 5th steps for me, with 32GB RAM.
- Currently the script needs images of the same dimension (no need to be square), as the input layer dimension is pre-defined.
- If the batch size is set to a value that is greater than the number of training samples, it is clipped to be the number of training samples. (You can also use this to force batch gradient descent when you don't know the exact number of training samples, as any value greater than that will suffice)
- Do not normalize input if the model is trained on images and the images are not in the range [0, 255]. This is because I hardcoded all normalized images' pixels/features to be assumed to be in the range [0, 255].
- When training a model with k-folds cross-validation, you won't be able to make predictions afterward nor will you be prompted whether to save a model. It is only used as a means of performance metrics evaluation, this is by design.
- The script expects all images to have pixel colors in the range [0, 255] for normalization. That can be changed within the function `mean_n_variance_normalize`, by changing the feature min maxes from `0` and `255` to whatever minimum and maximum values you're using, respectively.
- When training a model with k-folds, if the number of traiing samples is not exactly divisible by the number of folds, the spare sample will be ignored.
- Image labels cannot have duplicate filenames in the text file, as they are identified by them and assumed to be unique.

- You can print a model's performance metrics (if it has been measured prior) with the function `print_model_metrics`

- Strings as inputs haven't been properly implemented or tested.
- The `model` class members `sample_features_mean`, `sample_features_std`, `sample_features_variance`, are measured over the training samples (after normalizing the input to the range [-1, 1] but before mean normalizing and standardizing, if normalization was enabled). Meanwhile, the non `model` class members, such as `training_features_mean` and `test_features_mean`, are the measurements after the input is normalized (if that is the case, otherwise `model.sample_features_mean` is the same as `training_features_mean`, for example). That is because whenever the input needs to be re-normalized, such as in the case of predicting an arbitrary user input once the model is trained, you need to know the `mean` and `std` before mean normalizing and standardizing but after normalizing to [-1, 1], as well as the minimum and maximum values of each input feature in order to normalize it to [-1 ,1], which is what the variable `model.feature_min_maxes` is used for. These are also used for performance measurements and reversing the normalization.

- I could not make a model that performs well predicting different car brands of the same color (RGB), using a similar architecture to the model that predicts car colors. I don't know the exact reason for this, although it is evidently a more complex task than the other examples. Possibly solvable with convolutional layers?
# Todo

- Implement the same behavior of optionally storing images in a separate folder as the `test` folder for the `holdout` folder

- Option to convert images that are RGB/RGBA into grayscale

- Separate the bias from the normal weights and separately calculate/update it or leave it as is?

- Estimate time remaining for training to finish

- Refactor code and standardize variable names

- Decouple and isolate functions, settings, models, and datasets, 

- Separate main.py into multiple scripts for better isolation/readability

- Option to evalute model on a given test set without the need to train it

- Let user arbitrarily choose how often they want the cross-validation model(s) to be evaluated during training, instead of automatically doing it for every single batch

- Let user choose how much to smoothen the graph(s) (if at all), through exponentially moving averages

- Implement means to achieve better performance, such as vectorizing everything, using Jacobian matrices, CUDA (NVIDIA only), and SIMD instructions? I probably won't do this, as the primary goal of this project was learning, and readability is much preferred over performance

- Implement gradient checking

- Implement gradient clipping

- Implement dropout

- Implement layer batch normalization

- Implement convolutional layers

- Implement other optimizers/adaptive learning rates such as adagrad/adam/rmsprop, etc...

- Let user save/predict, once training ends, any chosen model(s) when training with k-folds cross-validation

- Improve performance of the plots

- Automatically save or allow user to choose whether to keep checkpoints of the model at specific points/intervals of the training (i.e. save model every 100 steps, or once steps are 500, etc.)

- Let user run script with "default settings" (i.e. normalize, shuffle dataset, no cross-validation, and a test set which is 25% of the training samples) in order to confuse the user less with too many settings, and also avoid repetition when testing things out?

- Implement hyperparameter tweaking during cross-validation

- Properly implement strings as input and try out a sentiment analysis dataset

- Prompt user whether to calculate and store micro statistics, macro statistics and/or model statistics? Worried that prompt would be too cumbersome as it's already bloated with settings, and this can already be done manually, though not tested, by passing `model_metrics`, `micro_metrics`, `macro_metrics` and/or `class_metrics` as `None`, instead of the 3 former being a dictionary and the latter a list. This would be particularly useful to increase speed and decrease memory usage for models trained with cross-validation.

- Let user input learning rate, regularization rate and the type (L1 or L2), batch size, and the frequency at which the plot (if enabled) is updated? Also worried about too many settings.

- Upload already labeled versions of the MNIST and cars datasets