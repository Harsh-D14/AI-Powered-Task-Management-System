import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# For text preprocessing (same as training data)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class TaskClassifier:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.svm_model = None
        self.label_encoder = None
        self.categories = None

        # Initialize preprocessor components
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

        self.stemmer = PorterStemmer()

        # Add task-specific stopwords
        additional_stopwords = {'task', 'need', 'needs', 'required', 'please', 'must', 'should'}
        self.stop_words.update(additional_stopwords)

    def preprocess_text(self, text):
        """Preprocess text similar to training data"""
        if pd.isna(text):
            return ""

        # Convert to lowercase and clean
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            tokens = word_tokenize(text)

        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Stem tokens
        tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    def load_and_prepare_data(self, file_path='task_preprocessed_data.csv'):
        """Load and prepare the preprocessed data"""
        print("Loading preprocessed data...")

        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} tasks")
        except FileNotFoundError:
            print(f"Error: Could not find {file_path}")
            return None, None

        # Use the processed text column
        X = df['task_description_processed'].fillna('')
        y = df['category'].fillna('unknown')

        # Display class distribution
        print("\nClass distribution:")
        class_counts = Counter(y)
        for category, count in class_counts.items():
            print(f"  {category}: {count} ({count / len(y) * 100:.1f}%)")

        return X, y

    def setup_tfidf_vectorizer(self):
        """Setup TF-IDF vectorizer with optimal parameters"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit vocabulary size
            min_df=2,  # Ignore terms appearing in less than 2 documents
            max_df=0.8,  # Ignore terms appearing in more than 80% of documents
            ngram_range=(1, 2),  # Use unigrams and bigrams
            sublinear_tf=True,  # Apply sublinear tf scaling
            stop_words=None  # We already removed stopwords
        )

    def train_model(self, X, y):
        """Train the SVM classifier"""
        print("\nStarting model training...")

        # Setup TF-IDF vectorizer
        self.setup_tfidf_vectorizer()

        # Transform text to TF-IDF features
        print("Applying TF-IDF vectorization...")
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        print(f"TF-IDF feature shape: {X_tfidf.shape}")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.categories = self.label_encoder.classes_

        print(f"Categories: {list(self.categories)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Hyperparameter tuning with GridSearch
        print("\nPerforming hyperparameter tuning...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        svm = SVC(random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Use the best model
        self.svm_model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = self.svm_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Detailed classification report
        print("\nClassification Report:")
        target_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.categories))]
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Cross-validation scores
        print("\nCross-validation scores:")
        cv_scores = cross_val_score(self.svm_model, X_tfidf, y_encoded, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Feature importance (for linear kernel)
        if self.svm_model.kernel == 'linear':
            self.analyze_feature_importance()

        return X_test, y_test, y_pred

    def analyze_feature_importance(self):
        """Analyze most important features for each category (linear SVM only)"""
        print("\nAnalyzing feature importance...")

        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        for i, category in enumerate(self.categories):
            if len(self.categories) == 2:
                # Binary classification
                coef = self.svm_model.coef_[0].toarray().flatten()
                if i == 1:
                    coef = -coef
            else:
                # Multi-class classification
                coef = self.svm_model.coef_[i].toarray().flatten()

            # Get top positive features
            top_indices = np.argsort(coef)[-10:][::-1]
            top_features = [(feature_names[idx], coef[idx]) for idx in top_indices]

            print(f"\nTop features for '{category}':")
            for feature, weight in top_features:
                print(f"  {feature}: {weight:.4f}")

    def save_model(self, filename='task_classifier.pkl'):
        """Save the trained model and vectorizer"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'svm_model': self.svm_model,
            'label_encoder': self.label_encoder,
            'categories': self.categories
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved as {filename}")

    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.categories, yticklabels=self.categories)
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Confusion matrix saved as confusion_matrix.png")
        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")


def main():
    """Main training function"""
    print("Task Classification Model Training")
    print("=" * 50)

    # Initialize classifier
    classifier = TaskClassifier()

    # Load data
    X, y = classifier.load_and_prepare_data()
    if X is None:
        return

    # Train model
    X_test, y_test, y_pred = classifier.train_model(X, y)

    # Plot confusion matrix
    classifier.plot_confusion_matrix(y_test, y_pred)

    # Save model
    classifier.save_model()

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("Model saved as task_classifier.pkl")
    print("You can now use predict_task_category.py to make predictions")


if __name__ == "__main__":
    main()