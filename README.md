# DTSA5511

```markdown
# Cancer Detection Project

This project aims to develop a machine learning algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The primary goal is to use the PatchCamelyon (PCam) benchmark dataset to train and evaluate convolutional neural networks (CNNs) for this binary classification task.

## Project Structure

The project directory is structured as follows:


Cancer Detection/
├── kaggle_submission_output/
│   └── [kaggle submission files]
├── models/
│   ├── best_simple_cnn.pth
│   ├── best_simple_cnn_baseline.pth
│   └── best_deeper_cnn.pth
├── params/
│   ├── best_simple_cnn_params.pth
│   ├── best_simple_cnn_baseline_params.pth
│   └── best_deeper_cnn_params.pth
├── .git/
│   └── [git files]
├── Cancer Detection.ipynb
└── README.md


### Directory and File Descriptions

- `kaggle_submission_output/`: Contains output files for Kaggle submission.
- `kaggle_results/`: Contains a screenshot of the Kaggle leaderboard.
  - `leaderboard_screenshot.png`: Screenshot of the Kaggle leaderboard showing the project’s ranking.
- `models/`: Contains the saved model files for different architectures.
  - `best_simple_cnn.pth`: Saved model state for the best simple CNN.
  - `best_simple_cnn_baseline.pth`: Saved model state for the baseline simple CNN.
  - `best_deeper_cnn.pth`: Saved model state for the deeper CNN.
- `params/`: Contains the saved hyperparameters for the models.
  - `best_simple_cnn_params.pth`: Hyperparameters for the best simple CNN.
  - `best_simple_cnn_baseline_params.pth`: Hyperparameters for the baseline simple CNN.
  - `best_deeper_cnn_params.pth`: Hyperparameters for the deeper CNN.
- `.git/`: Directory for Git version control.
- `Cancer Detection.ipynb`: Jupyter Notebook containing the code for data preparation, model training, evaluation, and visualization.
- `README.md`: This README file.

## Models

The project implements three different convolutional neural network (CNN) architectures:

1. **best_simple_cnn**:
    - Learning Rate: 0.001
    - Batch Size: 64
    - Dropout Rate: 0.3
    - Activation Function: LeakyReLU
    - Optimizer: Adam
    - Validation Loss: 0.1581
    - Validation Accuracy: 0.9411

2. **best_simple_cnn_baseline**:
    - Learning Rate: 0.001
    - Batch Size: 64
    - Dropout Rate: 0.3
    - Activation Function: LeakyReLU
    - Optimizer: Adam
    - Validation Loss: 0.2384
    - Validation Accuracy: 0.9022

3. **best_deeper_cnn**:
    - Learning Rate: 0.001
    - Batch Size: 64
    - Dropout Rate: 0.5
    - Activation Function: LeakyReLU
    - Optimizer: Adam
    - Validation Loss: 0.1475
    - Validation Accuracy: 0.9446

## Usage

To run the project, open the `Cancer Detection.ipynb` Jupyter Notebook and follow the instructions provided in each cell. The notebook includes steps for:

1. Data Preparation
2. Data Augmentation and Normalization
3. Model Building and Training
4. Model Evaluation
5. Result Visualization

## Results

The models are evaluated based on validation loss and validation accuracy. The best performing model is `best_deeper_cnn`, which achieves a validation accuracy of 0.9446 and a validation loss of 0.1475.

## License

This project uses the PatchCamelyon (PCam) benchmark dataset provided under the CC0 License.

Kaggle. (n.d.). Histopathologic Cancer Detection. Retrieved from https://www.kaggle.com/competitions/histopathologic-cancer-detection/data


## Acknowledgments

- Kaggle for hosting the dataset.
- The creators of the PatchCamelyon dataset for providing the data.
- PyTorch and its community for providing tools and support for deep learning research.
```










