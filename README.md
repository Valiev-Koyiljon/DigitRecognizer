

# MNIST Digit Recognition using Logistic Regression

## Project Overview
This project implements a simple logistic regression model using PyTorch to classify handwritten digits from the MNIST dataset. The goal is to demonstrate the application of logistic regression in image classification tasks, specifically recognizing digits from 0 to 9.

## Prerequisites
Before running this project, you need to have Python and the following Python libraries installed:
- PyTorch
- torchvision
- matplotlib

You will also need to have software installed to run and execute a Jupyter Notebook.

## Installation
Clone the repository and navigate to the download location. Then install the necessary Python packages by running the following command:

```bash
pip install torch torchvision matplotlib
```

## Dataset
The MNIST dataset, which contains 70,000 images of handwritten digits, is used for training and testing the model. The dataset is split into 60,000 training images and 10,000 testing images. Each image is a 28x28 grayscale image, associated with a label from 0 to 9.

## Model
The model is a simple logistic regression model implemented using PyTorch, where the image pixels are flattened into a vector of size 784 (28x28), and it produces 10 outputs corresponding to the probabilities of the image belonging to each digit class.

## Training
The model is trained using the cross-entropy loss function and the SGD optimizer with a learning rate of 0.001. The training process is run for several epochs to minimize the loss and improve the model's accuracy on the validation set.

## Usage
After installing the required libraries and downloading the repository, run the Jupyter Notebook `DigitRecognizerWithPytorch.ipynb` to train the model and make predictions. The notebook includes:
- Data loading and preprocessing
- Model creation
- Training loop
- Evaluation on test data
- Predictions on individual images

## Example Prediction
Here's how you can use the trained model to predict the class of a new image of a handwritten digit:

```python
def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_ds[100]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
```

## Save and Load the Model
The trained model can be saved and loaded as follows:

```python
# Save the model
torch.save(model.state_dict(), 'mnist-logistic.pth')

# Load the model
model.load_state_dict(torch.load('mnist-logistic.pth'))
```

## Contributions
Contributions to this project are welcome! Open an issue to discuss improvements or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

