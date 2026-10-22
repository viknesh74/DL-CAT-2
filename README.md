# Amazon Product Rating Prediction Using Deep Learning (LSTM)

## Project Overview

This project predicts Amazon product ratings (1 to 5) from product information and customer review text. It uses an LSTM-based deep learning model built with TensorFlow and Keras. The model learns from review text, prices, discount, rating count and product category.

The project runs in Google Colab and uses the Amazon Sales Dataset from Kaggle.

## Objectives

1. Load the Amazon product dataset (about 1,465 records).
2. Handle missing values, duplicate rows and inconsistent data types.
3. Clean and preprocess product features and review text.
4. Tokenize reviews and encode them as numerical sequences.
5. Normalize numerical features.
6. Select and justify a suitable deep learning model.
7. Train the model and evaluate it with suitable metrics.
8. Compare actual and predicted ratings.
9. Demonstrate prediction on unseen review text.

## Dataset

- **Name:** Amazon Sales Dataset
- **Source:** https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
- **File:** `amazon.csv`
- **Size:** about 1,465 product records

| Column | Use in this project |
|---|---|
| `rating` | **Target** (value to predict) |
| `review_title`, `review_content`, `about_product` | Combined into one text input |
| `discounted_price`, `actual_price` | Numeric features (log-transformed) |
| `discount_percentage` | Numeric feature |
| `rating_count` | Numeric feature (log-transformed) |
| `review_id` | Counted to create `n_reviews` |
| `category` | Top-level category extracted as `main_category` |
| `product_id` | Used only to split the data without leakage |

**Dataset limitation:** the records are product-level, and the review columns hold several reviews joined together, not one review paired with its own star rating. The experiment therefore predicts the recorded product rating from the available product and review information.

## Technologies

| Tool | Purpose |
|---|---|
| Python, Google Colab | Language and environment |
| kagglehub | Download the dataset from Kaggle |
| Pandas, NumPy | Loading, cleaning and numeric work |
| `re` | Text cleaning |
| TensorFlow / Keras | Model, text vectorization, normalization, training |
| Scikit-learn | Data splitting and evaluation metrics |
| Matplotlib | Plots |

## Preprocessing

- Remove duplicate rows.
- Convert `rating` to numeric. Invalid values (for example `|`) become missing and those rows are dropped.
- Remove `₹`, `,` and `%` from price and discount columns and convert them to numbers.
- Fill missing numeric values with the median. Fill missing text with an empty string.
- Apply a log transform (`log1p`) to price and count columns, which are skewed.
- Create `n_reviews` (number of reviews per product) and `main_category`.
- Clean text: lowercase, remove URLs, keep letters only, collapse extra spaces.
- Split 70% train, 15% validation, 15% test with `GroupShuffleSplit` on `product_id`, so the same product never appears in both train and test.

The vocabulary, normalization statistics and category lookup are all learned from the training set only, which prevents data leakage.

## Tokenization and Normalization

- **`TextVectorization`** keeps the 10,000 most frequent words and converts each review to a sequence of 250 integer ids (shorter texts are padded, longer ones are cut).
- **Keras `Normalization` layer** standardizes the five numeric features using the training mean and standard deviation.
- **`StringLookup` (one-hot)** encodes the product category.

## Model

A multi-input network with three branches that are merged into one rating output.

| Branch | Layers |
|---|---|
| Text | `TextVectorization` -> `Embedding` (64) -> `LSTM` (64, dropout 0.2) -> `Dropout` (0.4) |
| Numeric | `Normalization` -> `Dense` (16, ReLU) |
| Category | `StringLookup` (one-hot) -> `Dense` (8, ReLU) |
| Merge | `Concatenate` -> `Dense` (32, ReLU) -> `Dropout` (0.3) -> `Dense` (1, sigmoid) -> `Rescaling` (x4 + 1) |

### Why this model

- Reviews are sequences, and an LSTM keeps word order and context in memory, which simple word counts lose.
- The embedding layer gives similar words similar vectors, which helps on a small dataset.
- Separate numeric and category branches let the model use product information as well as text.
- Rating is an ordered numeric value, so regression suits it better than classification. The sigmoid and rescaling keep every prediction between 1 and 5.
- Dropout, early stopping and Huber loss reduce over-fitting and the effect of outliers on a small dataset.

## Training Configuration

| Parameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Sequence length | 250 |
| Embedding dimension | 64 |
| LSTM units | 64 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Huber |
| Batch size | 32 |
| Maximum epochs | 40 |
| Early stopping | patience 6, restores best weights |
| Learning-rate reduction | factor 0.5, patience 3 |

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| MAE | Average absolute error in stars |
| MSE | Average squared error |
| RMSE | Square root of MSE, in stars |
| R-squared | Share of rating variation explained, compared with the mean |
| Within 0.5 star | Share of predictions within half a star of the actual rating |

The model is compared with a **baseline that always predicts the mean training rating**. The notebook also shows a table of actual vs predicted ratings, an actual-vs-predicted scatter plot and an error histogram.

Actual results depend on the run. Report the numbers produced by your notebook; no accuracy value is assumed here.

## How to Run

1. Open https://colab.research.google.com and create a new notebook.
2. Paste the code cells in order, or upload `amazon_rating_prediction.ipynb` (File > Upload notebook).
3. The data downloads automatically with `kagglehub`. If that fails, the notebook asks you to upload `amazon.csv`.
4. Run all cells in order.
5. Read the training curves, metrics table and actual-vs-predicted comparison.
6. Test your own text with `predict_rating(...)`.

To run locally:

```
pip install kagglehub pandas numpy tensorflow scikit-learn matplotlib
```

## Example Prediction

```python
predict_rating(
    "Excellent product with great sound quality",
    category="Electronics",
    discounted_price=999,
    actual_price=1499,
    discount_percentage=33,
)
```

The function cleans the text, encodes it, normalizes the numeric values and returns a rating between 1 and 5. Product details you leave out are filled with training medians. The output depends on the trained model.

## Optional: Save the Model

The preprocessing layers (`TextVectorization`, `Normalization`, `StringLookup`) are inside the model, so one file is enough.

```python
model.save("amazon_rating_model.keras")
```

## Project Structure

```
Amazon-Product-Rating-Prediction/
├── README.md
├── amazon_rating_prediction.ipynb
├── amazon.csv                      (optional, check the dataset license)
└── amazon_rating_model.keras       (optional, created after training)
```

## Limitations

- The dataset has only about 1,465 product-level records, so the model can over-fit and may not generalize well.
- Ratings are tightly clustered (mostly 3.8 to 4.5), so the improvement over the mean baseline is naturally small and R-squared may be low.
- Review text may already reflect the product's existing rating, which can cause target leakage if used to predict future ratings.
- Reviews are joined per product, not individual review and star pairs, so this is not a true single-review rating predictor.
- A prediction is an estimate, not a guaranteed customer rating.

## Future Enhancements

- Train on a dataset of individual reviews with their own star ratings.
- Compare LSTM with Bidirectional LSTM, GRU, CNN and transformer models.
- Use pretrained embeddings such as GloVe.
- Add cross-validation and hyperparameter tuning (for example KerasTuner).
- Build a small web app for interactive prediction.

## Conclusion

This project combines review text and product information to predict Amazon product ratings. It covers data cleaning, text tokenization, normalization, an LSTM-based multi-input network, evaluation against a baseline, and prediction on unseen reviews.

## References

- Kaggle dataset: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- Scikit-learn: https://scikit-learn.org/
- Google Colab: https://colab.research.google.com/

**Author:** [Your Name]
