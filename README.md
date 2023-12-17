# Model Analysis and Feature Extraction (digits dataset)

This repository contains code for feature extraction and classification using PCA (Principal Component Analysis) and ICA (Independent Component Analysis). The dataset used here is the handwritten digits dataset from the `sklearn.datasets` library.

## Code Overview
The code provided demonstrates the process of:
- Loading the dataset and splitting it into training and test sets.
- Utilizing PCA to extract principal components, retaining 95% variance.
- Employing kNN (k-Nearest Neighbors) classifier after feature extraction using PCA.
- Implementing a Pipeline to streamline the process of feature extraction, scaling, and classification with PCA and kNN.
- Using ICA (Independent Component Analysis) for feature extraction and kNN classification within a Pipeline.

## Feature Extraction using PCA (Principal Component Analysis)
Principal Component Analysis is performed to extract essential features while retaining 95% variance from the dataset. The cumulative explained variance plot indicates the number of principal components necessary to retain this variance.

### Results from PCA
- Number of Principal Components for 95% variance: 28
- Accuracy score achieved: 97.22%

## Utilizing Pipeline for Feature Extraction and Classification
A Pipeline is constructed to automate the process, including PCA for feature extraction, scaling, and kNN for classification. The results achieved using this pipeline mirror those obtained directly with PCA.

### Results from Pipeline
- Accuracy score achieved: 97.22%

## Feature Extraction using ICA (Independent Component Analysis)
Independent Component Analysis with 20 independent components is applied for feature extraction. This transformed dataset is then classified using kNN.

### Results from ICA
- Accuracy score achieved: 97.78%

## Conclusion
Both PCA and ICA are effective methods for feature extraction, enabling dimensionality reduction while retaining crucial information for classification. In this specific scenario, PCA and ICA, followed by kNN classification, produced comparable and high accuracy scores, showcasing their utility in preprocessing and classification tasks.


## Launching Project

To run the project, you have a couple of options:

### Using Google Colab via Gist

Access the project through Google Colab using the Gist website. You can import the necessary data from the GitHub project resources. Use the following Gist link: [gist link here](https://gist.github.com/RobertNeat/2b1a50a9cf482c8c6d5be8e89f26ac03)

### Running Locally

If you prefer to run the project on your local machine, follow these steps:

1. **Clone the Repository**: Download the repository branch from GitHub.
2. **Local Environment**:
   - **DataSpell or PyCharm**: Open the project using DataSpell or PyCharm by JetBrains.
   - **Spyder IDE**: Alternatively, you can use Spyder IDE to work with the project.
3. **Dataset Requirements**:
   - Ensure that the dataset files are available and stored inside your project directory. This step is crucial to prevent any issues related to missing data.

Running the project locally allows you to explore the code and execute it in your preferred Python environment. If you encounter any problems, make sure to check the dataset's presence in your project directory.
