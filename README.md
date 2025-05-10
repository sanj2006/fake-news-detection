# Fake News Detection

## Overview
This project implements a fake news detection system using a Naive Bayes classifier. The model analyzes text data from news articles to classify them as **Fake (0)** or **Real (1)**. The pipeline includes text preprocessing, TF-IDF vectorization, model training, and evaluation.

## Dataset
The dataset used is `fake_news_dataset.csv`, which contains news articles with the following columns:
- `text`: The content of the news article.
- `label`: The target variable (0 for Fake, 1 for Real).

**Note**: Replace `fake_news_dataset.csv` with your dataset path. The dataset is assumed to be structured with text and label columns.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Dependencies
- Python 3.8+
- pandas
- numpy
- nltk
- scikit-learn
- regex (re)

Install dependencies using:
```bash
pip install pandas numpy nltk scikit-learn
```

## Usage
1. **Prepare the Dataset**: Ensure `fake_news_dataset.csv` is in the project directory or update the file path in the script.

2. **Run the Script**: Execute the main script to train the model and evaluate its performance:
   ```bash
   python fake_news_detection.py
   ```

3. **Output**:
   - The script prints the model's accuracy and a detailed classification report (precision, recall, F1-score).
   - It also predicts the label for a sample news headline:  
     Example: `"Breaking news! Scientists discover AI with human-level sarcasm!"`

4. **Custom Predictions**:
   - Modify the `new_text` list in the script to test new articles:
     ```python
     new_text = ["Your custom news article here"]
     ```

## Code Explanation
The script (`fake_news_detection.py`) performs the following steps:

1. **Load Dataset**:
   - Reads the CSV file using pandas.

2. **Text Preprocessing**:
   - Converts text to lowercase.
   - Removes special characters and extra spaces using regex.
   - Eliminates stopwords using NLTK's stopwords list.
   - Stores cleaned text in a new column `clean_text`.

3. **Vectorization**:
   - Uses `TfidfVectorizer` to convert text into numerical features (TF-IDF scores).
   - Limits to 5,000 features to manage computational complexity.

4. **Train-Test Split**:
   - Splits data into 80% training and 20% testing sets with a fixed random state for reproducibility.

5. **Model Training**:
   - Trains a `MultinomialNB` (Naive Bayes) classifier, suitable for text classification.

6. **Evaluation**:
   - Computes accuracy and generates a classification report with precision, recall, and F1-score.

7. **Prediction**:
   - Processes a new text sample through the same pipeline (cleaning, vectorization) and predicts its label.

## Example Output
```plaintext
Accuracy: 0.85
Classification Report:
              precision    recall  f1-score   support
           0       0.84      0.87      0.85      1000
           1       0.86      0.83      0.85      1000
    accuracy                           0.85      2000
   macro avg       0.85      0.85      0.85      2000
weighted avg       0.85      0.85      0.85      2000

Prediction (0=Fake, 1=Real): [0]
```

## Future Improvements
- Experiment with advanced models (e.g., Logistic Regression, SVM, or deep learning models like LSTM).
- Incorporate additional features (e.g., article metadata, author credibility).
- Handle imbalanced datasets using techniques like SMOTE.
- Add cross-validation for robust performance evaluation.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, please open an issue or contact [your-email@example.com].
