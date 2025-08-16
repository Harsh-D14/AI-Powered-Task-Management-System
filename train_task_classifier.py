import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import logging
from collections import Counter

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_nltk_dependencies() -> None:
    """Initialize required NLTK dependencies."""
    required_data = ['stopwords', 'punkt']

    for data_name in required_data:
        try:
            nltk.data.find(f'corpora/{data_name}')
        except LookupError:
            logger.info(f"Downloading NLTK {data_name}...")
            nltk.download(data_name, quiet=True)


class TextPreprocessor:
    """Text preprocessing utility for machine learning pipeline."""

    def __init__(self, custom_stopwords: Optional[set] = None):
        """
        Initialize text preprocessor.

        Args:
            custom_stopwords: Additional domain-specific stopwords
        """
        # Initialize NLTK dependencies
        initialize_nltk_dependencies()

        # Setup preprocessing components
        self.stopwords_set = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

        # Add domain-specific stopwords
        default_stopwords = {'task', 'need', 'needs', 'required', 'please', 'must', 'should'}
        if custom_stopwords:
            default_stopwords.update(custom_stopwords)
        self.stopwords_set.update(default_stopwords)

    def preprocess_text(self, text: str) -> str:
        """
        Apply complete text preprocessing pipeline.

        Args:
            text: Raw text to preprocess

        Returns:
            Processed text string
        """
        if pd.isna(text):
            return ""

        # Normalize text
        processed_text = str(text).lower()
        processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
        processed_text = ' '.join(processed_text.split())

        # Tokenize
        tokens = word_tokenize(processed_text)

        # Filter and stem
        filtered_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stopwords_set and len(token) > 2
        ]

        return ' '.join(filtered_tokens)


class ModelTrainer:
    """Handles model training and evaluation operations."""

    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()

    def prepare_features(self, text_data: pd.Series,
                         vectorizer_params: Optional[Dict] = None) -> Tuple[Any, TfidfVectorizer]:
        """
        Convert text data to TF-IDF features.

        Args:
            text_data: Series containing preprocessed text
            vectorizer_params: Parameters for TfidfVectorizer

        Returns:
            Tuple of (feature_matrix, fitted_vectorizer)
        """
        if vectorizer_params is None:
            vectorizer_params = {
                'max_features': 1000,
                'min_df': 3,
                'max_df': 0.7,
                'ngram_range': (1, 1),
                'sublinear_tf': True,
                'stop_words': None
            }

        vectorizer = TfidfVectorizer(**vectorizer_params)
        feature_matrix = vectorizer.fit_transform(text_data)

        logger.info(f"Generated TF-IDF features with shape: {feature_matrix.shape}")
        return feature_matrix, vectorizer

    def split_dataset(self, features: Any, labels: np.ndarray,
                      test_size: float = 0.3,
                      validation_size: float = 0.2) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            features: Feature matrix
            labels: Encoded labels
            test_size: Proportion for test set
            validation_size: Proportion of training set for validation

        Returns:
            Tuple of split datasets (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Initial train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size,
            random_state=self.random_state, stratify=labels
        )

        # Create validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size,
            random_state=self.random_state, stratify=y_train
        )

        logger.info(f"Dataset split - Train: {X_train.shape[0]}, "
                    f"Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def detect_data_leakage(self, X_train: Any, X_test: Any) -> int:
        """
        Check for potential data leakage between train and test sets.

        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix

        Returns:
            Number of potential duplicates found
        """
        logger.info("Checking for data leakage...")

        # Create hash set of training samples
        train_hashes = set()
        for i in range(X_train.shape[0]):
            sample_hash = hash(str(X_train[i].toarray().tobytes()))
            train_hashes.add(sample_hash)

        # Check test samples against training hashes
        duplicate_count = 0
        for i in range(X_test.shape[0]):
            test_hash = hash(str(X_test[i].toarray().tobytes()))
            if test_hash in train_hashes:
                duplicate_count += 1

        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} potential duplicates between train/test sets")
        else:
            logger.info("No data leakage detected")

        return duplicate_count

    def perform_hyperparameter_tuning(self, X_train: Any, y_train: np.ndarray,
                                      param_grid: Optional[Dict] = None) -> SVC:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for tuning

        Returns:
            Best fitted SVM model
        """
        if param_grid is None:
            param_grid = {
                'C': [0.01, 0.1, 1],
                'kernel': ['linear'],
                'class_weight': ['balanced']
            }

        logger.info("Starting hyperparameter tuning...")

        base_model = SVC(random_state=self.random_state, probability=True)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_


class ModelEvaluator:
    """Handles model evaluation and analysis."""

    def __init__(self, label_encoder: LabelEncoder):
        """
        Initialize model evaluator.

        Args:
            label_encoder: Fitted label encoder for class names
        """
        self.label_encoder = label_encoder
        self.category_names = label_encoder.classes_

    def evaluate_model_performance(self, model: SVC, X_test: Any, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing performance metrics
        """
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Generate detailed classification report
        target_names = [self.label_encoder.inverse_transform([i])[0]
                        for i in range(len(self.category_names))]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        return {
            'test_accuracy': test_accuracy,
            'predictions': y_pred
        }

    def cross_validate_model(self, model: SVC, features: Any, labels: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Perform cross-validation with multiple metrics.

        Args:
            model: Trained model
            features: Complete feature matrix
            labels: Complete label array

        Returns:
            Dictionary containing CV scores and standard deviations
        """
        logger.info("Performing cross-validation...")

        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        cv_results = {}

        for metric in metrics:
            scores = cross_val_score(model, features, labels, cv=5, scoring=metric)
            cv_results[metric] = (scores.mean(), scores.std())
            logger.info(f"CV {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_results

    def analyze_overfitting(self, model: SVC, X_train: Any, y_train: np.ndarray,
                            X_val: Any, y_val: np.ndarray, test_accuracy: float) -> Dict[str, float]:
        """
        Analyze potential overfitting in the model.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            test_accuracy: Test set accuracy

        Returns:
            Dictionary containing accuracy scores for each set
        """
        train_accuracy = model.score(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)

        logger.info("\nOverfitting Analysis:")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        # Check for overfitting indicators
        train_test_gap = train_accuracy - test_accuracy
        if train_test_gap > 0.1:
            logger.warning("Possible overfitting detected (train-test gap > 0.1)")
        elif train_accuracy > 0.99:
            logger.warning("Perfect training score suggests potential memorization")
        else:
            logger.info("Model appears to generalize well")

        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy
        }

    def analyze_feature_importance(self, model: SVC, vectorizer: TfidfVectorizer, top_n: int = 10) -> None:
        """
        Analyze feature importance for linear SVM models.

        Args:
            model: Trained SVM model
            vectorizer: Fitted TF-IDF vectorizer
            top_n: Number of top features to display per category
        """
        if model.kernel != 'linear':
            logger.warning("Feature importance analysis only available for linear kernel")
            return

        logger.info("Analyzing feature importance...")
        feature_names = vectorizer.get_feature_names_out()

        for i, category in enumerate(self.category_names):
            if len(self.category_names) == 2:
                # Binary classification
                coefficients = model.coef_[0].toarray().flatten()
                if i == 1:
                    coefficients = -coefficients
            else:
                # Multi-class classification
                coefficients = model.coef_[i].toarray().flatten()

            # Get top features
            top_indices = np.argsort(coefficients)[-top_n:][::-1]
            top_features = [(feature_names[idx], coefficients[idx]) for idx in top_indices]

            print(f"\nTop features for '{category}':")
            for feature, weight in top_features:
                print(f"  {feature}: {weight:.4f}")

    def create_confusion_matrix_plot(self, y_test: np.ndarray, y_pred: np.ndarray,
                                     output_path: str = 'confusion_matrix.png') -> None:
        """
        Create and save confusion matrix visualization.

        Args:
            y_test: True test labels
            y_pred: Predicted labels
            output_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return

        confusion_mat = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.category_names, yticklabels=self.category_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved as {output_path}")


class TaskCategoryClassifier:
    """Main classifier for task category prediction."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the task category classifier.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.trainer = ModelTrainer(random_state)

        # Model components
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[SVC] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.categories: Optional[np.ndarray] = None

    def load_training_data(self, file_path: str = 'datasets/task_preprocessed_data.csv') -> Tuple[pd.Series, pd.Series]:
        """
        Load and prepare training data from CSV file.

        Args:
            file_path: Path to preprocessed data file

        Returns:
            Tuple of (features, labels)

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        logger.info(f"Loading training data from {file_path}")

        try:
            dataframe = pd.read_csv(file_path)
            logger.info(f"Loaded {len(dataframe)} training samples")
        except FileNotFoundError:
            logger.error(f"Training data file not found: {file_path}")
            raise

        # Prepare features and labels
        features = dataframe['task_description_processed'].fillna('')
        labels = dataframe['category'].fillna('unknown')

        # Display class distribution
        class_distribution = Counter(labels)
        logger.info("Class distribution:")
        for category, count in class_distribution.items():
            percentage = count / len(labels) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")

        return features, labels

    def train_classifier(self, features: pd.Series, labels: pd.Series) -> Dict[str, Any]:
        """
        Train the complete classification pipeline.

        Args:
            features: Text features for training
            labels: Category labels for training

        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting classifier training...")

        # Prepare TF-IDF features
        feature_matrix, self.tfidf_vectorizer = self.trainer.prepare_features(features)

        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.categories = self.label_encoder.classes_

        logger.info(f"Training categories: {list(self.categories)}")

        # Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.split_dataset(
            feature_matrix, encoded_labels
        )

        # Check for data leakage
        duplicate_count = self.trainer.detect_data_leakage(X_train, X_test)

        # Train model with hyperparameter tuning
        self.model = self.trainer.perform_hyperparameter_tuning(X_train, y_train)

        # Initialize evaluator
        evaluator = ModelEvaluator(self.label_encoder)

        # Evaluate model performance
        performance_metrics = evaluator.evaluate_model_performance(self.model, X_test, y_test)

        # Cross-validation analysis
        cv_results = evaluator.cross_validate_model(self.model, feature_matrix, encoded_labels)

        # Overfitting analysis
        overfitting_analysis = evaluator.analyze_overfitting(
            self.model, X_train, y_train, X_val, y_val, performance_metrics['test_accuracy']
        )

        # Feature importance analysis
        evaluator.analyze_feature_importance(self.model, self.tfidf_vectorizer)

        # Create confusion matrix plot
        evaluator.create_confusion_matrix_plot(y_test, performance_metrics['predictions'])

        return {
            'performance_metrics': performance_metrics,
            'cv_results': cv_results,
            'overfitting_analysis': overfitting_analysis,
            'data_leakage_count': duplicate_count,
            'test_data': (X_test, y_test, performance_metrics['predictions']),
            'validation_data': (X_val, y_val)
        }

    def save_model(self, file_path: str = 'models/task_classifier.pkl') -> None:
        """
        Save the trained model and associated components.

        Args:
            file_path: Path to save the model file
        """
        # Ensure models directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        model_components = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'model': self.model,
            'label_encoder': self.label_encoder,
            'categories': self.categories,
            'preprocessor': self.preprocessor
        }

        try:
            with open(file_path, 'wb') as file:
                pickle.dump(model_components, file)
            logger.info(f"Model saved successfully to {file_path}")
        except Exception as error:
            logger.error(f"Error saving model: {error}")
            raise

    def load_model(self, file_path: str = 'models/task_classifier.pkl') -> None:
        """
        Load a previously trained model.

        Args:
            file_path: Path to the saved model file
        """
        try:
            with open(file_path, 'rb') as file:
                model_components = pickle.load(file)

            self.tfidf_vectorizer = model_components['tfidf_vectorizer']
            self.model = model_components['model']
            self.label_encoder = model_components['label_encoder']
            self.categories = model_components['categories']

            if 'preprocessor' in model_components:
                self.preprocessor = model_components['preprocessor']

            logger.info(f"Model loaded successfully from {file_path}")
        except Exception as error:
            logger.error(f"Error loading model: {error}")
            raise

    def predict_category(self, text: str) -> Tuple[str, float]:
        """
        Predict category for a single text input.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if not all([self.model, self.tfidf_vectorizer, self.label_encoder]):
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)

        # Transform to TF-IDF features
        features = self.tfidf_vectorizer.transform([processed_text])

        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # Get category name and confidence
        category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities.max()

        return category, confidence


def main(input_file: str = 'datasets/task_preprocessed_data.csv',
         model_output: str = 'models/task_classifier.pkl') -> None:
    """
    Main training pipeline execution.

    Args:
        input_file: Path to preprocessed training data
        model_output: Path to save the trained model
    """
    logger.info("=" * 60)
    logger.info("TASK CATEGORY CLASSIFIER TRAINING")
    logger.info("=" * 60)

    try:
        # Initialize classifier
        classifier = TaskCategoryClassifier()

        # Load training data
        features, labels = classifier.load_training_data(input_file)

        # Train the classifier
        training_results = classifier.train_classifier(features, labels)

        # Save the trained model
        classifier.save_model(model_output)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {model_output}")
        logger.info("Confusion matrix saved as: confusion_matrix.png")
        logger.info("The model is ready for making predictions")

    except Exception as error:
        logger.error(f"Training failed: {error}")
        raise


if __name__ == "__main__":
    main()