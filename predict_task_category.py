import pickle
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

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
    """Text preprocessing utility matching training pipeline."""

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
        if not text or not text.strip():
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

    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate input text for processing.

        Args:
            text: Input text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not isinstance(text, str):
            return False, "Invalid input: Text must be a non-empty string"

        text = text.strip()
        if not text:
            return False, "Invalid input: Text cannot be empty or whitespace only"

        # Check for code artifacts
        if text.startswith('if __name__'):
            return False, "Invalid input: Code detected, not a task description"

        # Check if meaningful text remains after preprocessing
        processed = self.preprocess_text(text)
        if not processed:
            return False, "Invalid input: No meaningful text found after preprocessing"

        return True, ""


class ModelLoader:
    """Handles loading and validation of trained models."""

    @staticmethod
    def load_model_components(model_path: str) -> Dict[str, Any]:
        """
        Load trained model components from file.

        Args:
            model_path: Path to the saved model file

        Returns:
            Dictionary containing model components

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: For other loading errors
        """
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_file, 'rb') as file:
                model_data = pickle.load(file)

            return model_data

        except Exception as error:
            logger.error(f"Error loading model: {error}")
            raise

    @staticmethod
    def validate_model_components(model_data: Dict[str, Any]) -> None:
        """
        Validate that all required model components are present.

        Args:
            model_data: Dictionary containing model components

        Raises:
            ValueError: If required components are missing
        """
        required_components = ['tfidf_vectorizer', 'label_encoder', 'categories']

        # Check for different possible model keys (handle both old and new formats)
        model_keys = ['svm_model', 'model']
        has_model = any(key in model_data for key in model_keys)

        if not has_model:
            raise ValueError("Model component not found in saved data")

        for component in required_components:
            if component not in model_data:
                raise ValueError(f"Required component '{component}' not found in model data")


class PredictionEngine:
    """Core prediction engine for task categorization."""

    def __init__(self, model_components: Dict[str, Any]):
        """
        Initialize prediction engine with loaded model components.

        Args:
            model_components: Dictionary containing trained model components
        """
        self.tfidf_vectorizer = model_components['tfidf_vectorizer']
        self.label_encoder = model_components['label_encoder']
        self.categories = model_components['categories']

        # Handle both old and new model key formats
        if 'model' in model_components:
            self.model = model_components['model']
        else:
            self.model = model_components['svm_model']

        # Initialize preprocessor (try to use saved one, fallback to new instance)
        if 'preprocessor' in model_components:
            self.preprocessor = model_components['preprocessor']
        else:
            self.preprocessor = TextPreprocessor()

    def predict_category(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Predict category for input text.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (predicted_category, confidence_scores)
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)

        # Transform to TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([processed_text])

        # Make prediction
        prediction = self.model.predict(tfidf_features)[0]
        predicted_category = self.label_encoder.inverse_transform([prediction])[0]

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(tfidf_features)

        return predicted_category, confidence_scores

    def _calculate_confidence_scores(self, features: Any) -> Dict[str, float]:
        """
        Calculate confidence scores for all categories.

        Args:
            features: TF-IDF features for the input text

        Returns:
            Dictionary mapping categories to confidence scores
        """
        try:
            decision_scores = self.model.decision_function(features)[0]
            confidence_scores = {}

            if len(self.categories) == 2:
                # Binary classification
                confidence = abs(decision_scores)
                predicted_idx = self.model.predict(features)[0]
                predicted_category = self.label_encoder.inverse_transform([predicted_idx])[0]
                confidence_scores[predicted_category] = float(confidence)
            else:
                # Multi-class classification
                for i, category in enumerate(self.categories):
                    score = decision_scores[i] if hasattr(decision_scores, '__len__') else decision_scores
                    confidence_scores[category] = float(score)

            return confidence_scores

        except Exception as error:
            logger.warning(f"Could not calculate confidence scores: {error}")
            return {}


class ResultFormatter:
    """Handles formatting and display of prediction results."""

    @staticmethod
    def format_prediction_result(text: str, predicted_category: str,
                                 confidence_scores: Dict[str, float],
                                 show_preprocessing: bool = True) -> str:
        """
        Format prediction results for display.

        Args:
            text: Original input text
            predicted_category: Predicted category
            confidence_scores: Category confidence scores
            show_preprocessing: Whether to show preprocessing details

        Returns:
            Formatted result string
        """
        result_lines = [
            "=" * 60,
            "TASK CATEGORY PREDICTION RESULT",
            "=" * 60,
            f"Input: '{text}'",
            "-" * 60,
            f"Predicted Category: {predicted_category}"
        ]

        if confidence_scores:
            result_lines.append("\nConfidence Scores:")
            sorted_scores = sorted(confidence_scores.items(),
                                   key=lambda x: abs(x[1]), reverse=True)

            for category, score in sorted_scores:
                indicator = ">> " if category == predicted_category else "   "
                result_lines.append(f"{indicator}{category}: {score:.4f}")

        result_lines.append("=" * 60)
        return "\n".join(result_lines)

    @staticmethod
    def format_batch_results(results: List[Dict[str, Any]]) -> str:
        """
        Format batch prediction results.

        Args:
            results: List of prediction result dictionaries

        Returns:
            Formatted batch results string
        """
        output_lines = [
            f"\nBATCH PREDICTION RESULTS ({len(results)} tasks)",
            "=" * 80
        ]

        for i, result in enumerate(results, 1):
            status = "SUCCESS" if result['success'] else "ERROR"
            output_lines.extend([
                f"{i}. Task: {result['task']}",
                f"   Status: {status}",
                f"   Result: {result['predicted_category']}",
                "-" * 80
            ])

        return "\n".join(output_lines)


class TaskCategoryPredictor:
    """Main interface for task category prediction."""

    def __init__(self, model_path: str = 'models/task_classifier.pkl'):
        """
        Initialize the task category predictor.

        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.prediction_engine: Optional[PredictionEngine] = None
        self.preprocessor: Optional[TextPreprocessor] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and initialize components."""
        try:
            # Load model components (suppress detailed logging during init)
            model_data = ModelLoader.load_model_components(self.model_path)
            ModelLoader.validate_model_components(model_data)

            # Initialize prediction engine
            self.prediction_engine = PredictionEngine(model_data)

            # Get categories for display
            self.categories = model_data['categories']

        except Exception as error:
            logger.error(f"Failed to load model: {error}")
            raise

    def predict(self, text: str, detailed: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Predict category for a single text input.

        Args:
            text: Input text to classify
            detailed: Whether to return detailed results

        Returns:
            Predicted category string or detailed result dictionary
        """
        # Validate input
        preprocessor = TextPreprocessor()
        is_valid, error_message = preprocessor.validate_input(text)

        if not is_valid:
            if detailed:
                return {'success': False, 'error': error_message, 'predicted_category': None}
            else:
                logger.error(error_message)
                return "PREDICTION_ERROR"

        try:
            # Make prediction
            predicted_category, confidence_scores = self.prediction_engine.predict_category(text)

            if detailed:
                return {
                    'success': True,
                    'predicted_category': predicted_category,
                    'confidence_scores': confidence_scores,
                    'original_text': text
                }
            else:
                return predicted_category

        except Exception as error:
            logger.error(f"Prediction failed: {error}")
            if detailed:
                return {'success': False, 'error': str(error), 'predicted_category': None}
            else:
                return "PREDICTION_ERROR"

    def predict_with_details(self, text: str) -> Optional[str]:
        """
        Predict category with detailed formatted output.

        Args:
            text: Input text to classify

        Returns:
            Predicted category or None if prediction failed
        """
        result = self.predict(text, detailed=True)

        if not result['success']:
            print(f"Error: {result['error']}")
            return None

        # Format and display results
        formatted_result = ResultFormatter.format_prediction_result(
            text, result['predicted_category'], result['confidence_scores']
        )
        print(formatted_result)

        return result['predicted_category']

    def batch_predict(self, task_list: List[str]) -> List[Dict[str, Any]]:
        """
        Predict categories for multiple tasks.

        Args:
            task_list: List of task descriptions

        Returns:
            List of prediction result dictionaries
        """
        logger.info(f"Processing batch of {len(task_list)} tasks...")
        results = []

        for i, task in enumerate(task_list, 1):
            logger.info(f"Processing task {i}: {task}")

            result = self.predict(task, detailed=True)
            result['task'] = task
            results.append(result)

            if result['success']:
                logger.info(f"Predicted: {result['predicted_category']}")
            else:
                logger.warning(f"Failed: {result['error']}")

        return results

    def get_available_categories(self) -> List[str]:
        """
        Get list of available prediction categories.

        Returns:
            List of category names
        """
        return list(self.categories)

    def show_startup_info(self) -> None:
        """Display startup information about the loaded model."""
        print("Task Category Predictor - Initialization Complete")
        print("=" * 60)
        print(f"Model loaded from: {self.model_path}")
        print(f"Available categories: {', '.join(self.categories)}")
        print(f"Number of categories: {len(self.categories)}")
        print("=" * 60)

    def show_model_info(self) -> None:
        """Display information about the loaded model."""
        print("\nMODEL INFORMATION")
        print("-" * 50)
        print(f"Model path: {self.model_path}")
        print(f"Available categories: {', '.join(self.categories)}")
        print(f"Number of categories: {len(self.categories)}")
        print("-" * 50)


class InteractiveInterface:
    """Interactive command-line interface for predictions."""

    def __init__(self, predictor: TaskCategoryPredictor):
        """
        Initialize interactive interface.

        Args:
            predictor: TaskCategoryPredictor instance
        """
        self.predictor = predictor

    def run_interactive_mode(self) -> None:
        """Run interactive prediction mode."""
        print("\nInteractive Mode")
        print("-" * 60)
        print("Enter task descriptions to predict their categories.")
        print("Available commands:")
        print("  'quit', 'exit', 'q' - Exit the program")
        print("  'help' - Show help information")
        print("  'info' - Show model information")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nEnter task description: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.lower() == 'info':
                    self.predictor.show_model_info()
                    continue

                if not user_input:
                    print("Please enter a task description.")
                    continue

                # Make prediction
                self.predictor.predict_with_details(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as error:
                logger.error(f"Error during prediction: {error}")

    def _show_help(self) -> None:
        """Display help information."""
        print("\nHELP - How to use the Task Category Predictor")
        print("-" * 50)
        print("• Enter any task description in natural language")
        print("• The system will predict the most likely category")
        print("• Examples of good task descriptions:")
        print("  - 'Fix the login bug in the authentication system'")
        print("  - 'Update the marketing website with new content'")
        print("  - 'Review the quarterly financial report'")
        print("  - 'Set up meeting room for client presentation'")
        print("• Available commands:")
        print("  - 'help': Show this help")
        print("  - 'info': Show model information")
        print("  - 'quit' or 'exit': Exit the program")
        print(f"• Available categories: {', '.join(self.predictor.get_available_categories())}")


def run_example_predictions() -> None:
    """Run example predictions for demonstration."""
    try:
        predictor = TaskCategoryPredictor()
    except Exception as error:
        logger.error(f"Failed to initialize predictor: {error}")
        return

    example_tasks = [
        "Fix the database connection error in the user authentication module",
        "Create a marketing campaign for the new product launch",
        "Schedule team meeting for project planning next week",
        "Update the company website with latest news and announcements",
        "Review and approve the budget proposal for Q4",
        "Install new software on employee computers",
        "Conduct user research for mobile app improvements"
    ]

    logger.info("Running example predictions...")
    results = predictor.batch_predict(example_tasks)

    # Display formatted results
    formatted_results = ResultFormatter.format_batch_results(results)
    print(formatted_results)


def main() -> None:
    """Main execution function."""
    try:
        # Initialize predictor
        predictor = TaskCategoryPredictor()

        # Check command line arguments
        if len(sys.argv) > 1:
            # Single prediction mode
            task_description = ' '.join(sys.argv[1:])
            logger.info(f"Command line input: '{task_description}'")
            predictor.predict_with_details(task_description)
        else:
            # Interactive mode
            interface = InteractiveInterface(predictor)
            interface.run_interactive_mode()

    except Exception as error:
        logger.error(f"Application failed to start: {error}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)