import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class TaskCategoryPredictor:
    def __init__(self, model_path='models/task_classifier.pkl'):
        self.model_path = model_path
        self.tfidf_vectorizer = None
        self.svm_model = None
        self.label_encoder = None
        self.categories = None

        # Initialize text preprocessing components
        self.stemmer = PorterStemmer()

        # Download NLTK data if needed
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))

        # Add task-specific stopwords
        additional_stopwords = {'task', 'need', 'needs', 'required', 'please', 'must', 'should'}
        self.stop_words.update(additional_stopwords)

        # Load the trained model
        self.load_model()

    def load_model(self):
        """Load the trained model and components"""
        try:
            print(f"Loading model from {self.model_path}...")
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            self.categories = model_data['categories']

            print("Model loaded successfully!")
            print(f"Available categories: {list(self.categories)}")

        except FileNotFoundError:
            print(f"Error: Model file '{self.model_path}' not found.")
            print("Please run the training script first to create the model.")
            exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    def preprocess_text(self, text):
        """Preprocess text using the same method as training data"""
        if not text or text.strip() == "":
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

    def predict_category(self, task_description, show_confidence=True):
        """Predict the category of a task description"""
        # Clean and validate input
        if not task_description or not isinstance(task_description, str):
            return "Unable to process: Invalid input", None

        # Remove common artifacts that might be in the input
        task_description = task_description.strip()

        # Check for common code artifacts and remove them
        if task_description.startswith('if __name__'):
            return "Unable to process: Code detected, not a task description", None

        # Preprocess the input
        processed_text = self.preprocess_text(task_description)

        if not processed_text:
            return "Unable to process: No meaningful text found after preprocessing", None

        if show_confidence:
            print(f"Original input: '{task_description}'")
            print(f"Processed text: '{processed_text}'")

        # Transform to TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([processed_text])

        # Make prediction
        prediction = self.svm_model.predict(tfidf_features)[0]
        predicted_category = self.label_encoder.inverse_transform([prediction])[0]

        # Validate the prediction is in known categories
        if predicted_category not in self.categories:
            return f"Error: Predicted category '{predicted_category}' not in known categories", None

        # Get prediction probabilities (if available)
        confidence_scores = None
        if hasattr(self.svm_model, 'predict_proba'):
            try:
                # For SVM, we need to use decision_function for confidence
                decision_scores = self.svm_model.decision_function(tfidf_features)[0]

                if len(self.categories) == 2:
                    # Binary classification
                    confidence = abs(decision_scores)
                    confidence_scores = {predicted_category: confidence}
                else:
                    # Multi-class classification
                    confidence_scores = {}
                    for i, category in enumerate(self.categories):
                        score = decision_scores[i] if len(decision_scores) > 1 else decision_scores
                        confidence_scores[category] = float(score)

            except Exception as e:
                print(f"Could not calculate confidence scores: {e}")

        return predicted_category, confidence_scores

    def predict_with_details(self, task_description):
        """Predict category with detailed output"""
        print("\n" + "=" * 60)
        print("TASK CATEGORY PREDICTION")
        print("=" * 60)
        print(f"Input: '{task_description}'")
        print("-" * 60)

        predicted_category, confidence_scores = self.predict_category(task_description)

        # Check if prediction is valid
        if "Unable to process" in predicted_category:
            print(f"Error: {predicted_category}")
            return None

        print(f"Predicted Category: {predicted_category}")

        if confidence_scores:
            print("\nConfidence Scores:")
            # Sort by confidence (descending)
            sorted_scores = sorted(confidence_scores.items(),
                                   key=lambda x: abs(x[1]), reverse=True)  # Use abs for proper sorting

            for category, score in sorted_scores:
                indicator = "👉" if category == predicted_category else "  "
                print(f"{indicator} {category}: {score:.4f}")

        print("=" * 60)
        return predicted_category

    def interactive_mode(self):
        """Run in interactive mode for continuous predictions"""
        print("\n🔍 Task Category Predictor - Interactive Mode")
        print("=" * 60)
        print("Enter task descriptions to predict their categories.")
        print("Type 'quit', 'exit', or 'q' to stop.")
        print("Type 'help' for more information.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nEnter task description: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                if not user_input:
                    print("Please enter a task description.")
                    continue

                # Make prediction
                self.predict_with_details(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during prediction: {e}")

    def show_help(self):
        """Show help information"""
        print("\n📋 HELP - How to use the Task Category Predictor")
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
        print("  - 'quit' or 'exit': Exit the program")
        print(f"• Available categories: {', '.join(self.categories)}")

    def batch_predict(self, task_list):
        """Predict categories for a list of tasks"""
        results = []
        print(f"\nProcessing {len(task_list)} tasks...")

        for i, task in enumerate(task_list, 1):
            print(f"\nTask {i}: {task}")
            category, confidence = self.predict_category(task, show_confidence=False)
            results.append({
                'task': task,
                'predicted_category': category,
                'confidence': confidence
            })
            print(f"Predicted: {category}")

        return results


def main():
    """Main function"""
    import sys

    # Initialize predictor
    try:
        predictor = TaskCategoryPredictor()
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return

    # Check command line arguments
    if len(sys.argv) > 1:
        # Single prediction mode - join all arguments as the task description
        task_description = ' '.join(sys.argv[1:])
        print(f"Command line input: '{task_description}'")  # Debug output
        predictor.predict_with_details(task_description)
    else:
        # Interactive mode
        predictor.interactive_mode()


# Example usage function
def example_predictions():
    """Run some example predictions"""
    predictor = TaskCategoryPredictor()

    example_tasks = [
        "Fix the database connection error in the user authentication module",
        "Create a marketing campaign for the new product launch",
        "Schedule team meeting for project planning next week",
        "Update the company website with latest news and announcements",
        "Review and approve the budget proposal for Q4",
        "Install new software on employee computers",
        "Conduct user research for mobile app improvements"
    ]

    print("🧪 Running example predictions...")
    results = predictor.batch_predict(example_tasks)

    print("\n📊 BATCH PREDICTION RESULTS")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. Task: {result['task']}")
        print(f"   Predicted Category: {result['predicted_category']}")
        print("-" * 80)


if __name__ == "__main__":
    main()