# Attribute Extraction and PCA Analysis

This repository contains code snippets and analyses for attribute extraction from audio files and performing Principal Component Analysis (PCA) on voice features. The code covers different stages of data processing, visualization, and machine learning model implementation for gender classification based on voice attributes.

## Content Overview

### 1. Attribute Extraction from Audio Files
The code snippet for attribute extraction from audio files can be found in `Code Example 1`. This part utilizes the `scipy` library to read WAV files and perform a Fourier transform to extract frequency domain features. Additionally, it includes code for visualizing the waveform and magnitude spectrum of the audio file.

### 2. PCA Analysis for Gender Classification
For PCA analysis and gender classification, various code snippets are available:
- Loading the dataset from a CSV file and splitting it into training and testing sets is demonstrated in `Code Example 2`.
- Preprocessing steps, including feature standardization and PCA application, are illustrated in `Code Example 3`.
- Visualizing the reduced-dimensional data using scatter plots is provided in `Code Example 4`.
- Implementing a Random Forest Classifier and evaluating model accuracy is detailed in `Code Example 5`.

### 3. Gender Classification Analysis
An in-depth analysis of gender classification models, namely k-Nearest Neighbors (kNN), Support Vector Machine (SVM), and Decision Tree, is presented in `Code Example 6`. This part evaluates the performance of these models on voice features and showcases average confusion matrices for each classifier.

### 4. Custom Classes and Utilities
This repository includes custom classes within the pipeline for enhanced functionality:
- `OptimalPCANumber` (found in `Code Example 7`) determines the optimal number of principal components based on a specified variance percentage in PCA.
- `OutlierRemover` (available in `Code Example 8`) is a custom class to remove outliers in the dataset based on mean values.

## Usage
To utilize the provided code snippets:
- Ensure the required libraries (`numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`) are installed in your environment.
- Load and preprocess your audio data as necessary before implementing the code segments.
- Refer to the specific code examples mentioned above for detailed instructions on how to:
  - Extract attributes from audio files.
  - Perform PCA and visualize reduced dimensions.
  - Implement and evaluate gender classification models.

Please refer to the individual code examples within the repository for detailed usage instructions, customization options, and more specific information.



## Launching Project

To run the project, you have a couple of options:

### Using Google Colab via Gist

Access the project through Google Colab using the Gist website. You can import the necessary data from the GitHub project resources. Use the following Gist link: [gist link here](https://gist.github.com/RobertNeat/a9b0cb49c3205e79248dc15b26c84feb)

### Running Locally

If you prefer to run the project on your local machine, follow these steps:

1. **Clone the Repository**: Download the repository branch from GitHub.
2. **Local Environment**:
   - **DataSpell or PyCharm**: Open the project using DataSpell or PyCharm by JetBrains.
   - **Spyder IDE**: Alternatively, you can use Spyder IDE to work with the project.
3. **Dataset Requirements**:
   - Ensure that the dataset files are available and stored inside your project directory. This step is crucial to prevent any issues related to missing data.

Running the project locally allows you to explore the code and execute it in your preferred Python environment. If you encounter any problems, make sure to check the dataset's presence in your project directory.
