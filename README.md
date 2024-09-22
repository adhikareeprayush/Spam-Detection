# Spam Detection Using Machine Learning 🚀

## Overview
This project implements a spam detection system using machine learning techniques, specifically the Naive Bayes classifier. It analyzes text messages and classifies them as "spam" or "ham" (non-spam). The dataset used is a CSV file containing labeled messages.

## Table of Contents
- [Installation](#installation-)
- [Usage](#usage-)
- [Logic and Mathematics](#logic-and-mathematics-)
- [Import Statements](#import-statements-)
- [Contributions](#contributions-)
- [License](#license-)

## Installation 🛠️
To set up the project, ensure you have Python installed, then install the required libraries using:

```bash
pip install pandas scikit-learn nltk
```

### Download NLTK Resources
Run the following lines in Python to download necessary NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage 💻
1. Place the `spam.csv` dataset in the project directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. The model will train and evaluate itself, printing the accuracy and classification report.

## Logic and Mathematics 📊
### How It Works
1. **Data Loading**: The dataset is loaded using pandas.
2. **Data Preprocessing**: Text messages are cleaned and prepared for analysis:
   - Lowercasing and splitting into words.
   - Removing stopwords and non-alphanumeric characters.
3. **Feature Extraction**: The `CountVectorizer` converts the processed text into a matrix of token counts, making it suitable for machine learning algorithms.
4. **Model Training**: The Multinomial Naive Bayes model is trained on the processed data.
5. **Evaluation**: The model's performance is assessed using accuracy and a detailed classification report.

### Naive Bayes Classifier
- **Assumption**: Naive Bayes assumes that the presence of a particular feature in a class is independent of the presence of any other feature. This simplification is why it’s termed "naive."
- **Mathematics**: The classifier uses Bayes’ theorem to calculate the probability of a message being spam or ham based on its features.

## Import Statements 📦
Here's a breakdown of the important imports in the script:

- **pandas**: For data manipulation and analysis.
- **sklearn.model_selection.train_test_split**: To split the dataset into training and testing sets, ensuring model validation.
- **sklearn.feature_extraction.text.CountVectorizer**: To convert text data into numerical form (bag of words model).
- **sklearn.naive_bayes.MultinomialNB**: The classifier used for the spam detection task.
- **sklearn.metrics**: For measuring the performance of the model.
- **nltk**: A library for natural language processing.
- **nltk.corpus.stopwords**: Provides a list of common words to exclude from analysis.

## Contributions 🤝
- **Main Contributor**: Prayush Adhikari - Developed the spam detection model and organized the code.
- **Collaborators**: Contributions from the community are welcome! Feel free to suggest improvements, report bugs, or add features.

## License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for checking out this project! If you have any questions or suggestions, feel free to reach out. Happy coding! 😊