# Feature Extraction and Classification (ionsphere dataset)

## Introduction
This project focuses on utilizing Principal Component Analysis (PCA) and Independent Component Analysis (ICA) for feature extraction in the context of ionosphere data classification. The dataset used in this model is `ionosphere_data.csv`, containing 35 features (columns) and binary classification labels.

## Data Overview
The data consists of 351 samples with two classes:
- Class 1.0: 225 samples
- Class 0.0: 126 samples

## Feature Extraction
### PCA (Principal Component Analysis)
To capture 95% of the variance within the data, 23 principal components are needed. PCA was implemented and various classification setups were explored based on these components.

### ICA (Independent Component Analysis)
Additionally, FastICA with 23 components was employed to compare its performance against PCA in the classification task.

## Classification Strategies
The classification process involved constructing different pipelines with combinations of scalers and classifiers based on PCA and FastICA-transformed data.

### PCA-based Classifiers
- PCA + StandardScaler + kNN
- PCA + MinMaxScaler + kNN
- PCA + RobustScaler + kNN
- PCA + No scaling + kNN
- PCA + StandardScaler + SVC
- PCA + MinMaxScaler + SVC
- PCA + RobustScaler + SVC
- PCA + No scaling + SVC
- PCA + No scaling + Decision Tree
- PCA + No scaling + Random Forest

### FastICA-based Classifiers
- FastICA + StandardScaler + kNN
- FastICA + MinMaxScaler + kNN
- FastICA + RobustScaler + kNN
- FastICA + No scaling + kNN
- FastICA + StandardScaler + SVC
- FastICA + MinMaxScaler + SVC
- FastICA + RobustScaler + SVC
- FastICA + No scaling + SVC
- FastICA + No scaling + Decision Tree
- FastICA + No scaling + Random Forest

## Results Summary
### Best Performing Classifiers
- PCA, MinMaxScaler, SVC: Accuracy score: 0.9577
- PCA, No scaling, SVC: Accuracy score: 0.9577
- PCA, No scaling, Random Forest: Accuracy score: 0.9577
- FastICA, MinMaxScaler, SVC: Accuracy score: 0.9577

The analysis reveals that utilizing PCA with certain configurations, especially when coupled with MinMaxScaler and SVC, produced the most consistent and accurate classification results.

## Launching Project

To run the project, you have a couple of options:

### Using Google Colab via Gist

Access the project through Google Colab using the Gist website. You can import the necessary data from the GitHub project resources. Use the following Gist link: [gist link here](https://gist.github.com/RobertNeat/ec7345ce267104feffe316b3341356ab)


### Running Locally

If you prefer to run the project on your local machine, follow these steps:

1. **Clone the Repository**: Download the repository branch from GitHub.
2. **Local Environment**:
   - **DataSpell or PyCharm**: Open the project using DataSpell or PyCharm by JetBrains.
   - **Spyder IDE**: Alternatively, you can use Spyder IDE to work with the project.
3. **Dataset Requirements**:
   - Ensure that the dataset files are available and stored inside your project directory. This step is crucial to prevent any issues related to missing data.

Running the project locally allows you to explore the code and execute it in your preferred Python environment. If you encounter any problems, make sure to check the dataset's presence in your project directory.
