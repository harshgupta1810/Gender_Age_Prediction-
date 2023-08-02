# Gender and Age Estimation from Facial Images

This project is designed to estimate the gender and age of individuals from facial images using a deep learning model. The model is trained on the UTKFace dataset, which is a large-scale face dataset with age, gender, and ethnicity annotations. The dataset consists of over 20,000 face images covering a wide range of ages, genders, and facial variations.

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)
9. [Documentation](#documentation)

## Project Description

This project aims to predict the gender and age of individuals from facial images using a pre-trained deep learning model. The project is divided into two parts: the backend, which includes the code for training and building the model, and the frontend, which includes the code for using the model to predict gender and age from new images.

## Dataset

The model is trained on the UTKFace dataset, which is a large-scale face dataset with long age span (ranging from 0 to 116 years old). The dataset contains over 20,000 face images with annotations of age, gender, and ethnicity. The images cover a wide variation in pose, facial expression, illumination, occlusion, resolution, etc.

For more details about the dataset, you can visit the following link: [UTKFace Dataset](https://susanqq.github.io/UTKFace/)

## Installation

To run this project, you need to download the UTKFace dataset from Kaggle using the following command:
```
kaggle datasets download -d jangedoo/utkface-new
```

## Usage

To use the pre-trained model for gender and age estimation, you can follow the steps below:

1. Load the pre-trained model using `load_model('Gender_Age_model.h5')`.
2. Preprocess the input image to the required input size of the model (160x160).
3. Use the model to predict the gender and age of the individual in the image.

Here's an example of how to use the model to predict gender and age from a single image:
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('Gender_Age_model.h5')
# Define the gender labels
gender_dict = {0: 'Male', 1: 'Female'}

# Load the image using OpenCV
img = cv2.imread('path/to/your/image.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the image to the required input size of the model (160x160)
img = cv2.resize(img, (160, 160))

# Reshape the image for model input
img = img.reshape(1, 160, 160, 1)

# Predict using the model
pred = model.predict(img)
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
```

## Model Architecture

The model architecture consists of multiple convolutional and fully connected layers. It takes an input image of size 160x160 and outputs two separate outputs: one for gender prediction and one for age prediction.

The model is trained using binary cross-entropy loss for gender prediction and mean absolute error (MAE) loss for age prediction.

## Contributing

Contributions to this project are welcome. If you have any suggestions, bug fixes, or improvements, please feel free to open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the creators of the UTKFace dataset for providing such a valuable resource for research and development.

## Documentation

For more details on how to use the code and the functionalities of the backend and frontend, please refer to the code comments and documentation within the code files. If you have any questions or need further assistance, you can contact the project creator, Harsh Gupta (Desparete Enuf).
