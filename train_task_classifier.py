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

    def load_and_prepare_data(self, file_path='datasets/task_preprocessed_data.csv'):
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
        """Setup TF-IDF vectorizer with more conservative parameters"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Reduced vocabulary to prevent overfitting
            min_df=3,  # Word must appear in at least 3 documents
            max_df=0.7,  # Ignore very common words (>70%)
            ngram_range=(1, 1),  # Only unigrams to reduce complexity
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

        # Split data with more rigorous separation
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        # Add validation set for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Check for data leakage - ensure no identical samples
        print(f"Checking for data leakage...")
        train_hashes = set()
        for i in range(X_train.shape[0]):
            train_hashes.add(hash(str(X_train[i].toarray().tobytes())))

        test_duplicates = 0
        for i in range(X_test.shape[0]):
            test_hash = hash(str(X_test[i].toarray().tobytes()))
            if test_hash in train_hashes:
                test_duplicates += 1

        if test_duplicates > 0:
            print(f"⚠️  Warning: Found {test_duplicates} potential duplicates between train/test")
        else:
            print("✅ No data leakage detected")

        # Hyperparameter tuning with GridSearch (more conservative)
        print("\nPerforming hyperparameter tuning...")
        param_grid = {
            'C': [0.01, 0.1, 1],  # Lower regularization values
            'kernel': ['linear'],  # Only linear kernel to reduce complexity
            'class_weight': ['balanced']  # Handle class imbalance
        }

        svm = SVC(random_state=42, probability=True)  # Enable probability for better analysis
        grid_search = GridSearchCV(
            svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1
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

        # Cross-validation scores with different metrics
        print("\nCross-validation scores:")
        cv_accuracy = cross_val_score(self.svm_model, X_tfidf, y_encoded, cv=5, scoring='accuracy')
        cv_f1 = cross_val_score(self.svm_model, X_tfidf, y_encoded, cv=5, scoring='f1_macro')
        cv_precision = cross_val_score(self.svm_model, X_tfidf, y_encoded, cv=5, scoring='precision_macro')
        cv_recall = cross_val_score(self.svm_model, X_tfidf, y_encoded, cv=5, scoring='recall_macro')

        print(f"CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
        print(f"CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
        print(f"CV Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
        print(f"CV Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")

        # Check if model is overfitting
        train_score = self.svm_model.score(X_train, y_train)
        val_score = self.svm_model.score(X_val, y_val)
        test_score = accuracy_score(y_test, y_pred)

        print(f"\nOverfitting Analysis:")
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Validation Accuracy: {val_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")

        if train_score - test_score > 0.1:
            print("⚠️  Warning: Possible overfitting detected (train-test gap > 0.1)")
        elif train_score > 0.99:
            print("⚠️  Warning: Perfect training score suggests memorization")
        else:
            print("✅ Model appears to generalize well")

        # Feature importance (for linear kernel)
        if self.svm_model.kernel == 'linear':
            self.analyze_feature_importance()

        return X_test, y_test, y_pred, X_val, y_val

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

    def save_model(self, filename='models/task_classifier.pkl'):
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
            plt.close()  # Close the plot instead of showing it
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
    X_test, y_test, y_pred, X_val, y_val = classifier.train_model(X, y)

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