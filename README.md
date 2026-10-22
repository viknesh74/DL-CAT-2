Amazon Product Rating Prediction Using Deep Learning
Project Overview

This project focuses on predicting Amazon product ratings using product information and customer review text. It uses a deep learning model built with TensorFlow and Keras to learn patterns from review text, product prices, discounts, and product categories.

The project is implemented in Python using Google Colab and the Amazon Sales Dataset available on Kaggle.
Objectives
Load and preprocess the Amazon product dataset.
Handle missing values, duplicate records, and inconsistent data types.
Clean and tokenize customer review text.
Convert text into numerical sequences using TensorFlow TextVectorization.
Normalize numerical product features.
Encode categorical product features.
Develop and train a Bidirectional LSTM deep learning model.
Evaluate model performance using multiple metrics.
Compare actual and predicted product ratings.
Demonstrate rating prediction on unseen review text.
Save the trained model for future use.
Dataset

Dataset name: Amazon Sales Dataset

Source: Kaggle – Amazon Sales Dataset

Filename: amazon.csv

The dataset contains approximately 1,465 product records and includes product details, prices, ratings, and review information.

Important columns
Column	Description
product_name	Name of the product
category	Product category
about_product	Product description
discounted_price	Price after discount
actual_price	Original product price
discount_percentage	Discount percentage
review_title	Review heading
review_content	Review text
rating	Recorded product rating
rating_count	Number of recorded ratings

The rating column is used as the target variable. Product information and review text are used as model inputs.

Dataset limitation: The dataset primarily contains product-level records with aggregated review information rather than independent individual customer reviews paired with their own star ratings. Consequently, the experiment predicts recorded product ratings from the available product and review information.

Technologies and Frameworks

The following technologies are used in this project:

Python: Main programming language.
Google Colab: Cloud-based development and training environment.
Pandas: Dataset loading, inspection, and preprocessing.
NumPy: Numerical computations and array manipulation.
TensorFlow: Deep learning computation and training.
Keras: Neural network architecture and model training.
Scikit-learn: Data splitting, feature normalization, and evaluation metrics.
Matplotlib: Training plots and prediction visualizations.
Seaborn: Rating distribution plots and confusion matrix visualization.
Deep Learning Model

The project uses a Bidirectional Long Short-Term Memory (BiLSTM) neural network combined with numerical and categorical product features.

Model architecture
Text input: Receives cleaned review text.
TextVectorization: Converts text into integer token sequences.
Embedding layer: Learns dense vector representations of tokens.
Bidirectional LSTM: Learns sequential and contextual patterns in the text.
Numerical feature branch: Processes standardized prices and discount information.
Categorical feature branch: Encodes product categories using StringLookup.
Concatenation layer: Combines text, numerical, and categorical representations.
Dense layers: Learn relationships between the combined features.
Output layer: Produces a numerical rating prediction between 1 and 5 after output clipping.
Why BiLSTM?

LSTM networks can learn sequential patterns in review text. Bidirectional processing uses context from both directions of the sequence, which can help interpret expressions involving product quality, satisfaction, and negative wording.

The numerical and categorical branches allow the model to consider product information alongside review text.

Data Preprocessing

The following preprocessing operations are performed:

Remove exact duplicate records.
Remove rows with missing or invalid target ratings.
Convert price and discount fields into numeric values.
Handle missing numerical features using median imputation.
Replace missing text with empty strings.
Clean text by removing unwanted HTML tags, URLs, and extra whitespace.
Convert text to lowercase.
Tokenize text and encode it as numerical sequences.
Encode product categories.
Standardize numerical features using StandardScaler.

The vocabulary, category encoder, and scaler are fitted using training data to help prevent data leakage.

Model Training

The dataset is divided into three subsets:

Dataset	Percentage	Purpose
Training set	70%	Learn model parameters
Validation set	15%	Monitor performance and training
Test set	15%	Evaluate the trained model
Training configuration
Parameter	Value
Model	Bidirectional LSTM
Maximum vocabulary size	10,000 tokens
Maximum sequence length	150
Embedding dimension	64
LSTM units per direction	32
Optimizer	Adam
Learning rate	0.001
Loss function	Mean Squared Error
Batch size	32
Maximum epochs	30
Early stopping patience	5 epochs

The final training duration depends on validation performance. Early stopping is used to reduce overfitting.

Evaluation Metrics

The model is evaluated using the following metrics:

Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted ratings.
Root Mean Squared Error (RMSE): Penalizes larger prediction errors more strongly.
R-squared (R²): Measures how well predictions explain rating variation relative to a mean baseline.
Rounded-star accuracy: Measures the proportion of predictions matching the actual whole-star rating after rounding.

The project also includes:

Training and validation loss plots.
Actual-versus-predicted rating comparison.
Confusion matrix.
Comparison with a mean-rating baseline.
Predictions for new review text.

Actual evaluation results should be reported after running the notebook. No accuracy value is assumed in advance.

Installation and Requirements

The project is designed to run in Google Colab, where the required libraries can be imported directly.

If running locally, install the required packages:

pip install pandas numpy tensorflow scikit-learn matplotlib seaborn joblib

How to Run the Project
Download amazon.csv from the Kaggle dataset linked above.
Open Google Colab.
Create a new Python notebook.
Upload or open your project notebook.
Upload amazon.csv when prompted.
Execute the notebook cells in order.
Review the training history and evaluation metrics.
Examine the actual-versus-predicted comparison and confusion matrix.
Test the model using new review text.
Save the trained model and preprocessing files.
Project Structure
Amazon-Product-Rating-Prediction/
│
├── README.md
├── amazon_rating_prediction.ipynb
├── amazon.csv
├── amazon_rating_model.keras
├── amazon_scaler.pkl
├── amazon_numeric_columns.pkl
└── amazon_numeric_medians.pkl


The notebook and model files should be added after they have been created. The dataset and generated files do not need to be committed if they are too large or restricted by applicable licensing terms.

Example Prediction

Example input:

Review: Excellent product with great sound quality
Category: Electronics
Discounted price: 999
Actual price: 1499
Discount percentage: 33


The trained model returns a numerical rating prediction between 1 and 5 and a rounded whole-star rating.

The example illustrates the prediction process; the actual output depends on the trained model and dataset.

Limitations
The dataset contains approximately 1,465 product-level records, which limits the amount of training data.
Review information may be aggregated or associated with existing product ratings.
Including review text that reflects an existing rating can lead to target leakage for future-rating prediction.
A small dataset may cause overfitting and limited generalization.
The predicted rating is an estimate, not a guaranteed customer rating.
Further evaluation on independent customer reviews with corresponding star ratings is required for a true individual-review rating prediction system.
Future Enhancements
Train on larger datasets containing individual reviews and their star ratings.
Compare BiLSTM with GRU, CNN, and transformer-based models.
Use sentiment analysis as an additional feature.
Apply cross-validation and systematic hyperparameter tuning.
Improve generalization using more representative training data.
Develop a web application for interactive rating prediction.
Conclusion

This project demonstrates how deep learning can combine review text and product information to predict recorded Amazon product ratings. TensorFlow/Keras, Bidirectional LSTM, text vectorization, numerical standardization, categorical encoding, and Scikit-learn evaluation tools are integrated into a single workflow.

The project provides practical experience in data preprocessing, natural language processing, neural network training, model evaluation, and prediction on new input data.

References
Kaggle dataset: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
TensorFlow: https://www.tensorflow.org/
Keras: https://keras.io/
Scikit-learn: https://scikit-learn.org/
Google Colab: https://colab.research.google.com/

Author: [Your Name]

Project: Amazon Product Rating Prediction Using Deep Learning
