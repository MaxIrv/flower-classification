# Flower-102 Classification Project

This project involves building and training a Convolutional Neural Network (CNN) to classify images from the Flowers-102 dataset, which includes 102 different species of flowers. The project explores various architectures, hyperparameters, and optimizers to achieve the best possible accuracy.

**An accuracy of 69.9% was achieved.**

## Project Structure

- `final_implementation.ipynb`: Contains the final model implementation. This notebook is set up to train the model and save the best model to `best_model.pth`.
- `test_final_model.ipynb`: Set up to run the `final_model.pth` state dictionary on the test set, evaluating the model's performance.
- `old_versions/`: Contains initial models and the second model created during the project.
- `hyperparam_testing/`: Includes notebooks and scripts used for hyperparameter tuning, which contributed to the results shown in the report.
- `sched_optim_testing/`: Contains notebooks and scripts used for testing different schedulers and optimizers, contributing to the tables on Scheduler/Optimizer tuning in the report.
- `requirements.txt`: Lists all the dependencies and their versions required to run the project.

## Setting Up the Python Environment

To ensure the correct requirements and versions are used, follow these steps to set up the Python virtual environment:

1. **Ensure Python 3.12.0 is Installed**
    - Download it from [this link](https://www.python.org/downloads/release/python-3120/)
2. **Create a virtual environment**:
   ```bash 
   virtualenv -p /path/to/python3.12 pyenv
3. **Activate the virtual enbironment**:
    - On Windows:
        ```bash 
        pyenv\Scripts\activate
    - On MacOS or Linux
        ```bash
        source pyenv/bin/activate
4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
## Running the Project

Train the Model:
  - Open `final_implementation.ipynb` in Jupyter Notebook or JupyterLab and run all the cells. 
  - This will train the CNN model on the Flowers-102 dataset and save the best model to `best_model.pth.`

Test the Model:
   - Open `test_final_model.ipynb` in Jupyter Notebook or JupyterLab and run all the cells. T
   - his will load the `final_model.pth` state dictionary and evaluate the model on the test set.

## Additional Files and Directories
- `Report.pdf`: Contains the finalised IEEE Journal Report
- `graph.ipynb`: Create a graph of the Architecture of the Model
- `data_augmentation.ipynb`: Visualises images before and after a list of transformations are applied
- `architecture_graph.png`: The Architecture of the Model presented in a Graph
- `final_model.pth`: The model's state dictionary containing the 69.9% accuracy after 2000 epochs
- `requirements.txt`: The python requirements for the project
- `old_versions/`: This directory contains:
    - `initial_model.ipynb`: The initial model created for the project.
    - `second_model.ipynb`: The second iteration of the model.
- `hyperparam_testing/`: This directory includes notebooks and scripts used for hyperparameter tuning. It contains experiments that helped determine the best hyperparameters for the final model.
- `sched_optim_testing/`: This directory contains notebooks and scripts used to test different schedulers and optimizers. The results contributed to the optimization strategies used in the final model.

## Project Overview
The project aimed to build a robust CNN capable of classifying flower images with high accuracy. Initially, the goal was set to achieve over 75% accuracy, but due to the constraints of not using pre-trained models, the focus shifted to optimizing the architecture and hyperparameters. The final model achieved an accuracy of 69.9% on the test set, demonstrating the network's ability to learn from the dataset effectively.

## Future Work
Further work on this project could involve:

- Investigating the overfitting issue observed during training.
- Exploring more advanced architectures such as ResNet.
- Fine-tuning the network by further sexperimenting with different layers, activation functions, learning rates, batch sizes, optimizers, and loss functions.
