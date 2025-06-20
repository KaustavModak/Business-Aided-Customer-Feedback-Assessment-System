
# 🛍️ Customer Feedback Sentiment Analysis

This project analyzes customer feedback using machine learning to classify sentiments as **Positive**, **Neutral**, or **Negative**. It helps businesses gain insights from textual reviews and make data-driven decisions.

---

## 📂 Project Structure

```
.
├── code.ipynb               # Jupyter notebook with full analysis pipeline
├── Reviews.csv              # Amazon fine food reviews dataset
├── trained_model.pkl        # Pre-trained logistic regression model (saved)
└── README.md                # Project documentation
```

---

## 📊 Dataset

- **Source:** [Kaggle – Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Records:** ~568,000 reviews
- **Fields Used:**
  - `Text` – Customer review text
  - `Score` – Review score (used to derive sentiment label)

### 🔁 Label Encoding:
- `Score >= 4` → **Positive**
- `Score == 3` → **Neutral**
- `Score <= 2` → **Negative**

---

## 🛠️ Tech Stack

| Task                         | Library/Tool                  |
|------------------------------|-------------------------------|
| Data Manipulation            | Pandas                        |
| Text Preprocessing           | NLTK, Regex                   |
| Feature Extraction (TF-IDF)  | Scikit-learn                  |
| Model Building               | Logistic Regression           |
| Evaluation                   | Accuracy, Classification Report |
| Visualization                | Matplotlib, Seaborn           |
| Deployment Ready Model       | `pickle` (saved as `.pkl`)    |

---

## 📌 Key Features

- End-to-end sentiment analysis pipeline
- Cleaned and preprocessed review texts
- Trained with TF-IDF + Logistic Regression
- Multiclass classification: **Positive**, **Neutral**, **Negative**
- Random review testing to validate predictions
- Visual performance reporting with heatmaps

---

## 📈 Example Results

| Metric        | Score  |
|---------------|--------|
| Accuracy      | ~89%   |
| Precision     | Good   |
| Recall        | Balanced |
| F1 Score      | Consistent across classes |

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-feedback.git
   cd sentiment-analysis-feedback
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook code.ipynb
   ```

---

## 📦 Dependencies

You can create a `requirements.txt` with:

```txt
pandas
scikit-learn
nltk
matplotlib
seaborn
```

---

## 🧪 Model Testing

Random review test:
```python
# Show random review from test set with actual and predicted sentiment
i = random.randint(0, len(X_test) - 1)
print("Review:", X_test[i])
print("Actual:", y_test[i])
print("Predicted:", y_pred[i])
```

---

## 📜 License

This project is under the [MIT License](LICENSE).

---

## 🤝 Acknowledgments

- Dataset: [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Libraries: scikit-learn, NLTK, Pandas, Matplotlib
