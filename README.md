# Neural Network
A Python implementation from scratch of a neural network (using NumPy for matrix operations), made primarily for learning purposes. 

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

## How to use

The parameters are `lr`, `batch_size`....

### Loading/labeling images as dataset

## Examples

* Note: The following example videos are sped-up and low quality due to GitHub's file size limit og 10MB.

### Linear regression (houses.csv)


https://github.com/user-attachments/assets/18eecd0d-ce30-4e88-93ee-4f7696d807b0



### Logistic regression (framingham.csv)



https://github.com/user-attachments/assets/2dc4374f-e046-46a3-8d25-04c4f4c05db3



### Multi-label classification (framingham.csv)



https://github.com/user-attachments/assets/998b5c6a-f42a-443a-9c3a-518d5bb7c0be



### Multi-class classification (mnist_png.rar)



https://github.com/user-attachments/assets/479b44f2-3759-4462-a155-1116f7a43019

### Multi-class classification (RGB) (cars.rar)



https://github.com/user-attachments/assets/d2557369-eee7-42d3-ab3f-01ee4c0c41d4



### Labeling and setting up an image dataset

### Resizing images to a specific dimension

## How it works

## Todo

- Prompt/variable `uses_grayscale_images` needs to be reworked. This is necessary for knowing when to plot images in a grayscale colormap within matplotlib, but also when the image has an 8 bit depth instead of 24/32, the latter can instead be infered programmatically, and subsequently the colormap to be used, then I can get rid of this prompt and variable, which are misleading. i.e. `get_bit_depth(img) -> for each byte: store byte as individual color channel -> if img_bit_depth == 8: plot(cmap='gray') else: plot()`

## Notes

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
- Strings as inputs haven't been properly implemented or tested.
- The `model` class members `sample_features_mean`, `sample_features_std`, `sample_features_variance`, are measured over the training samples (after normalizing the input to the range [-1, 1] but before mean normalizing and standardizing, if normalization was enabled). Meanwhile, the non `model` class members, such as `training_features_mean` and `test_features_mean`, are the measurements after the input is normalized (if that is the case, otherwise `model.sample_features_mean` is the same as `training_features_mean`, for example). That is because whenever the input needs to be re-normalized, such as in the case of predicting an arbitrary user input once the model is trained, you need to know the `mean` and `std` before mean normalizing and standardizing but after normalizing to [-1, 1], as well as the minimum and maximum values of each input feature in order to normalize it to [-1 ,1], which is what the variable `model.feature_min_maxes` is used for. These are also used for performance measurements and reversing the normalization.

## Issues

- I could not make a model that performs well predicting different car brands of the same color (RGB), using a similar architecture to the model that predicts car colors. I don't know the exact reason for this, although it is evidently a more complex task than the other examples. Possibly solvable with convolutional layers?
