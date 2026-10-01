# Transformer-Based Sentiment Analysis

A small **binary sentiment-classification** project built with
TensorFlow/Keras and the UCI Sentiment Labelled Sentences dataset. The
model predicts whether a review sentence expresses a negative (`0`) or
positive (`1`) sentiment.

## Project Overview

This project combines three labelled sentence datasets:

-   Amazon product reviews
-   IMDb movie reviews
-   Yelp restaurant reviews

The text is converted into token IDs, passed through an encoder-style
Transformer model, and classified as positive or negative.

## Dataset

**Dataset:** [UCI Sentiment Labelled
Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)

The dataset contains three tab-separated text files:

``` text
amazon_cells_labelled.txt
imdb_labelled.txt
yelp_labelled.txt
```

Each row contains a sentence and its binary sentiment label, separated
by a tab.

Place the three files in the same directory as the Python script, or
update the file paths in the loading section.

## Model Architecture

The network uses an encoder-only Transformer architecture:

``` text
Input token IDs
      ↓
Embedding (dimension 64, padding mask enabled)
      ↓
Sinusoidal Positional Encoding
      ↓
Multi-Head Self-Attention (4 heads, key_dim 16)
      ↓
Residual connection + Layer Normalization + Dropout
      ↓
Feed-Forward Network (64 → 128 → 64)
      ↓
Residual connection + Layer Normalization + Dropout
      ↓
Global Average Pooling
      ↓
Dense layer + Sigmoid
      ↓
Positive / Negative prediction
```

### Main configuration

  Setting                                                   Value
  ---------------------------------------- ----------------------
  Maximum vocabulary size (`max_tokens`)                   10,000
  Maximum sequence length                              100 tokens
  Embedding dimension (`d_model`)                              64
  Attention heads                                               4
  Attention `key_dim`                                 16 per head
  Feed-forward dimensions                            128, then 64
  Dropout rate                                                0.2
  Batch size                                                   32
  Maximum epochs                                               10
  Optimizer                                                  Adam
  Loss                                       Binary cross-entropy
  Early stopping monitor                          Validation loss
  Early stopping patience                                       2

These are project hyperparameters, not universal optimal values. They
can be tuned for other datasets.

## Data Preparation

The combined dataset is split into:

-   **Training set:** used to update model weights.
-   **Validation set:** used to monitor generalization during training
    and guide training decisions.
-   **Test set:** held out for final evaluation.

The code first reserves 20% of the data for testing, then takes 20% of
the remaining training portion for validation. This results in
approximately 64% training, 16% validation, and 20% testing.

`TextVectorization` is adapted on the training text only, avoiding
vocabulary fitting on validation or test examples. Sequences are padded
or truncated to 100 tokens.

## Requirements

Install the required packages:

``` bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

The code also uses Python's built-in `warnings` and `pickle` modules.

## How to Run

1.  Download the three dataset files from the UCI dataset page.

2.  Put them alongside the project script, or update the paths in the
    script.

3.  Install the dependencies.

4.  Run the Python script:

    ``` bash
    python sentiment_transformer.py
    ```

The script trains the model, evaluates it on the test set, prints a
confusion matrix and classification report, and saves the accuracy and
loss plots.

> Replace `sentiment_transformer.py` with the actual name of your Python
> file if it is different.

## Evaluation

The script reports:

-   Test loss
-   Test accuracy
-   Confusion matrix
-   Precision, recall, and F1-score for each class
-   Training and validation accuracy plot
-   Training and validation loss plot

No fixed performance score is claimed here because results can vary with
the exact dataset files, random split, software versions, and training
run. Add your final measured test results here after running the
finished script.

## Saved Artifacts

The script can save:

``` text
Results Accuracy.png
Results Loss.png
sentiment_transformer.keras
vocabulary.pkl
```

The model and vocabulary are saved separately so the learned model and
text vocabulary can be retained. For deployment or reuse, ensure
preprocessing is reconstructed consistently and the custom
`PositionalEncoder` layer is registered or supplied when loading the
model.

## Notes and Possible Improvements

-   Use stratified train/validation/test splitting to preserve class
    proportions.
-   Consider comparing against a simple baseline, such as TF-IDF with
    Logistic Regression.
-   Experiment with model dimensions, dropout, sequence length, and
    learning rate using validation data.
-   Inspect class-wise precision and recall instead of relying on
    accuracy alone.
-   Keep the test set separate from repeated model-selection decisions.
-   For reproducibility, set seeds for Python, NumPy, and TensorFlow and
    record package versions.
-   The positional encoding uses sinusoidal functions. The base constant
    in the exponent is set in the implementation; use `10000` if
    following the commonly presented original sinusoidal formulation.

## License and Attribution

This project uses the UCI Sentiment Labelled Sentences dataset. Check
the dataset page for its terms and attribution requirements. Add a
project license here if you choose to license your own code.

------------------------------------------------------------------------

**Project goal:** practice the end-to-end workflow of preparing text
data, implementing an encoder-only Transformer, training it, and
evaluating a binary classifier.
