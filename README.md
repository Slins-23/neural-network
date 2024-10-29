# Neural Network
A Python implementation from scratch of a neural network (using NumPy for matrix operations), made primarily for learning purposes. 

## Examples

* Note: The following example videos are sped-up and low quality due to GitHub's file size limit of 10MB. There's also an option that is now unnecessary, called "Use grayscale images".

### Linear regression (houses.csv)


https://github.com/user-attachments/assets/18eecd0d-ce30-4e88-93ee-4f7696d807b0



### Logistic regression (framingham.csv)



https://github.com/user-attachments/assets/2dc4374f-e046-46a3-8d25-04c4f4c05db3



### Multi-label classification (3-folds cross-validation) (framingham.csv)

https://github.com/user-attachments/assets/9af34dbf-9ef9-448b-8d8d-f3e5a97f0fad

### Multi-class classification (mnist_png.rar)



https://github.com/user-attachments/assets/479b44f2-3759-4462-a155-1116f7a43019

### Multi-class classification (RGB) (cars.rar)



https://github.com/user-attachments/assets/d2557369-eee7-42d3-ab3f-01ee4c0c41d4

### Labeling images



https://github.com/user-attachments/assets/c983ead9-d4a5-4548-bea0-0289c64a371f



### Resizing images


https://github.com/user-attachments/assets/5b47e22f-8db8-405c-8dc3-d6ec64956d0d

## How to use

There are two types of datasets that are currently accepted:
- Comma `,` delimited `.csv` files where each variable is in a separate column and each row is an individual sample
- Images of the same dimension, no need to be square (i.e. 28x28, 106x80, etc.)

Training a model (from scratch or by retraining an already existing model):
  1. Setup the model with its architecture (or optionally load a model from a `.json` config file within the `models` folder)
  2. Adjust the hyperparameters (i.e. learning rate, batch size, etc...)
  3. Run the script
  4. Adjust the settings by following the prompts
  5. Wait for the training to finish
  6. (Optional) Predict and/or evaluate metrics on final model
  7. (Optional) Save the model

Loading a model for prediction only:
  1. Load the model (must be within the folder `models`)
  2. Type `1` for prediction only when prompted

### Defining a model from scratch
  The class that defines a model is `Model`, and you can make as many instances as you want. However, since throughout this entire time I have only experimented with an individual model (outside of k-folds cross-validation), I have made a global variable `model` which defines the model to be defined/loaded_into, so you may encounter issues doing otherwise without making changes to the code.

  You can setup the architecture right below the line `if not loaded_model:`.
  A `Layer` instance defines a layer.
  A layer has:
  - `layer_num`: Starting at 0, at which position in the network is the layer at. This is so that you don't need to add layers at a specific order and they get sorted in ascending order later.
  - `layer_dimension`: Number of nodes within the layer.
  - `layer_type`: One of `0` (input layer), `1` (hidden layer), `2` (output layer).
  - (Optional) `l1_rate`: L1 regularization rate
  - (Optional) `l2_rate`: L2 regularization rate

  The model setup part is pretty intuitive and self-explanatory:

  * Activation functions
    - `linear`: Linear
    - `relu`: Rectified Linear Unit
    - `sigmoid`: Sigmoid
    - `softmax`: Softmax
  * Loss functions
    - `loss_mse`: Mean squared error
    - `loss_binary_crossentropy`: Binary cross-entropy
    - `loss_categorical_crossentropy`: Categorical cross-entropy

  - To add a layer to the model call `model.add_layer(layer_number, layer_dimension, layer_type)`
  - Each layer that is not the input layer needs an activation function, and you can set it by calling `model.set_activation_function(layer_number, activation_function)`
  - Then you need a loss function, which can be set by calling `model.set_loss_function(loss_function)`
  - Finally, you need to call `model.setup_done(is_loaded_model=False)` to ensure that the model architecture is valid (i.e. if there are no missing layers, the layer types are correct, the layers also get sorted by their `layer_num` at this point, etc.)
  - (Optional) If you want to regularize a layer, it needs to be done after calling `setup_done`. Then you need to call `model.set_regularization(layer_num, l1_rate, l2_rate)` for the layer(s) you want to regularize.

### Setting up the hyperparameters
  There are four hyperparameters (the optional regularization rates are set alongside the model architecture as explained above):
  - Learning rate = `lr`
  - Batch size (if this variable is greater than the number of training samples, it defaults to the number of training samples) = `batch_size`
  - Steps = `steps`
  - Interval (number of batches) in which the plot (if enabled) should be updated = `plot_update_every_n_batches`

### Dataset loading notes
  * If training from a `.csv` file, the (optional) `test` and `hold-out` sets will be sampled from the dataset based on a user given percentage.

  * If training an image model, the images for the training set must go inside of the folder `images/train` alongside a `labels.txt` file which contains the filenames and their respective classes, in the format `filename,class`, where each new line represents a new image sample. As for the (optional) test set, you can either choose a percentage of the images within the `train` folder as the test set, or store them separately in the `images/test` folder. If you put images in `images/test` they also need their respective `labels.txt` file.
  * I have only tested 3 types of images so far - RGB, RGBA, and byte sized.

### Running the script
Now that, if necessary, the model architecture and the hyperparemeters have been setup accordingly, you can run `main.py`.

Firstly, you will be prompted whether you want to load a `.json` model config file.
- If you type in `y`, you will then be prompted for the name of the model file (excluding the `.json` extension), which can be re-trained used for predictions.
- If you type in `n`, the model architecture will be the one in the `main.py` file, so if you're training a model from scratch you will always need to define a model architecture in that file beforehand.

Then you can choose in which probability distribution you want the weights to be initialized in, `0` for a normal distribution or `1` for a uniform distribution.

Now you will be prompted whether this model will be trained on images or not. Type `y` for yes or `n` for no.

Then you want choose whether to train the model (`0`) or go straight into predicting (`1`).

If you chose to train the model, you will be prompted questions about the dataset.

You will be asked whether to plot anything. Type `y` for yes or `n` for no.
Similarly, you will be prompted whether to perform cross-validation (hold-out (`0`) or k-folds (`1`)), whether to use a test set, and whether to shuffle the dataset (all of it, including the subsets).

(If it is an image model) you will be prompted whether you want to give names to the classes, as by default they're numbers in the range [0, c], where `c` is the number of classes in the last layer of the model.

Then, finally and similarly, you will be prompted whether to normalize the dataset. Normalizing puts all of the input features of each sample in the entire dataset within the range [-1, 1]. Then it subtracts the mean (which is calculated after normalizing to this range), and divides by the standard deviation (calculated at the same time as the mean).

Now you have to wait for the model to finish training. Meanwhile, if you chose to plot anything, you will see the relevant graph(s), which includes the training set(s) loss(es) side by side with the performance metric(s) (i.e. r^2 for regression models or accuracy, precision, recall and f1-score for classification models).

Once finished, you can optionally make predictions and then optionally save the model.

If you are making predictions on a model trained on a `.csv` dataset, you will have to input the feature values manually for each of them. If it is an image model, you will have to input the filename of your desired image within the `images/predict` folder.

### Loading/labeling images as dataset

### Labeling and setting up an image dataset

### Resizing images to a specific dimension

## How it works

## Features

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

## Notes

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

## Todo

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

- Prompt/variable `uses_grayscale_images` needs to be reworked. This is necessary for knowing when to plot images in a grayscale colormap within matplotlib, but also when the image has an 8 bit depth instead of 24/32, the latter can instead be infered programmatically, and subsequently the colormap to be used, then I can get rid of this prompt and variable, which are misleading. i.e. `get_bit_depth(img) -> for each byte: store byte as individual color channel -> if img_bit_depth == 8: plot(cmap='gray') else: plot()`