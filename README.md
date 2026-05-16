# Blog Sentiment Analysis Pipeline

An end-to-end Natural Language Processing (NLP) pipeline that processes unstructured blog text, performs linguistic cleaning, and uses machine learning to classify the sentiment/topic distribution of the content.

## 🚀 Key Features

* **Data Ingestion:** Reads and parses tabular textual data from `blogs.csv` using `pandas`.
* **Text Preprocessing:** Cleans and tokenizes text using `nltk` by filtering out punctuation, handling regular expressions, and removing standard English stopwords.
* **Feature Extraction:** Converts raw textual string columns into numerical feature matrices using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
* **Supervised Classification:** Trains a **Multinomial Naive Bayes (MultinomialNB)** algorithm optimized for discrete text classification tasks.
* **Lexicon Integration:** Utilizes NLTK's `VADER Lexicon` infrastructure to supplement baseline sentiment indicators.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Natural Language Processing (NLP):** NLTK (VADER, Stopwords)
* **Machine Learning Model:** Scikit-Learn (Multinomial Naive Bayes, TfIdfVectorizer)

## 📁 Project Structure

```text
├── data/
│   └── blogs.csv            # Source text dataset
├── README.md                # Project Documentation
└── train_model.ipynb        # Jupyter Notebook with the NLP Pipeline
