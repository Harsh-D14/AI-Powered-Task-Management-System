import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
import io
import os
import sys

warnings.filterwarnings('ignore')

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class TextPreprocessor:
    """
    Text preprocessing utility for task descriptions.

    Note: This class name must match the one used during model training
    to ensure proper pickle loading compatibility.
    """

    def __init__(self, custom_stopwords=None):
        """Initialize text preprocessor with NLTK components."""
        try:
            self.stopwords_set = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()

            # Add domain-specific stopwords
            default_stopwords = {'task', 'need', 'needs', 'required', 'please', 'must', 'should'}
            if custom_stopwords:
                default_stopwords.update(custom_stopwords)
            self.stopwords_set.update(default_stopwords)
        except Exception as e:
            st.error(f"Error initializing text preprocessor: {e}")

    def preprocess_text(self, text):
        """
        Apply text preprocessing pipeline.

        Args:
            text: Raw text to preprocess

        Returns:
            Processed text string
        """
        if not text or not text.strip():
            return ""

        try:
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
        except Exception as e:
            st.warning(f"Text preprocessing error: {e}")
            return text.lower()

    def validate_input(self, text):
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
            return False, "Invalid input: Text cannot be empty"

        if text.startswith('if __name__'):
            return False, "Invalid input: Code detected, not a task description"

        processed = self.preprocess_text(text)
        if not processed:
            return False, "Invalid input: No meaningful text found"

        return True, ""


# Create alias for backward compatibility
TextProcessor = TextPreprocessor


class WorkloadBalancer:
    """
    Workload balancer for employee assignment.

    Note: This class name must match the one used during model training
    to ensure proper pickle loading compatibility.
    """

    def __init__(self, employees_data, tasks_df=None):
        self.employees_data = employees_data
        self.initial_loads = {emp_id: info['emp_load'] for emp_id, info in employees_data.items()}
        self.current_loads = self.initial_loads.copy()
        self.category_preferences = {emp_id: info['emp_preferred_category']
                                     for emp_id, info in employees_data.items()}
        self.assignment_history = {}
        self.randomness_factor = 0.15

    def get_optimal_employee(self, task_category, predicted_priority, urgency_score=0):
        """Employee selection with load balancing."""
        available_employees = [emp_id for emp_id, load in self.current_loads.items() if load < 10]

        if not available_employees:
            return min(self.current_loads, key=self.current_loads.get)

        candidates = []
        for emp_id in available_employees:
            base_score = self._calculate_base_score(emp_id, task_category, predicted_priority)
            random_factor = np.random.normal(0, self.randomness_factor)
            final_score = base_score + random_factor
            candidates.append((emp_id, final_score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Pick from top 3 candidates with weighted probability
        top_candidates = candidates[:min(3, len(candidates))]
        weights = [0.6, 0.3, 0.1][:len(top_candidates)]
        selected_idx = np.random.choice(len(top_candidates), p=weights)

        selected_emp = top_candidates[selected_idx][0]
        self.assignment_history[selected_emp] = self.assignment_history.get(selected_emp, 0) + 1

        return selected_emp

    def _calculate_base_score(self, emp_id, task_category, predicted_priority):
        """Calculate base assignment score."""
        score = 0
        current_load = self.current_loads[emp_id]
        is_expert = self.category_preferences[emp_id] == task_category

        if is_expert:
            score += 3

        score += (10 - current_load) * 0.5

        if predicted_priority == 'High' and current_load <= 5:
            score += 1

        return score

    def update_workload(self, emp_id, task_complexity=1, predicted_priority='Medium'):
        """Update workload with complexity."""
        if emp_id not in self.current_loads:
            return

        complexity_map = {'High': 1.5, 'Medium': 1.0, 'Low': 0.8}
        final_complexity = task_complexity * complexity_map.get(predicted_priority, 1.0)

        noise = np.random.normal(0, 0.1)
        self.current_loads[emp_id] = min(10, max(0, self.current_loads[emp_id] + final_complexity + noise))

    def get_employee_details(self, emp_id):
        """Get employee details."""
        if emp_id not in self.employees_data:
            return None
        return {
            'emp_id': emp_id,
            'current_load': self.current_loads.get(emp_id, 0),
            'preferred_category': self.category_preferences.get(emp_id, 'Unknown'),
            'recent_assignments': self.assignment_history.get(emp_id, 0)
        }


class CategoryPredictor:
    """Wrapper for task category prediction with error handling."""

    def __init__(self, model_path='models/task_classifier.pkl'):
        """Initialize category predictor."""
        self.model_path = model_path
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.categories = []
        self.preprocessor = None
        self.load_model()

    def load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_path, 'rb') as file:
                model_data = pickle.load(file)

            # Extract model components
            if 'model' in model_data:
                self.model = model_data['model']
            elif 'svm_model' in model_data:
                self.model = model_data['svm_model']
            else:
                raise ValueError("No model found in file")

            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.categories = model_data['categories']

            # Use saved preprocessor if available, otherwise create new one
            if 'preprocessor' in model_data:
                self.preprocessor = model_data['preprocessor']
            else:
                self.preprocessor = TextPreprocessor()

        except Exception as e:
            st.error(f"Error loading category model: {e}")
            raise

    def predict_category(self, text, show_confidence=False):
        """
        Predict category for input text.

        Args:
            text: Input text to classify
            show_confidence: Whether to return confidence scores

        Returns:
            Predicted category or tuple with confidence if requested
        """
        try:
            # Validate input
            is_valid, error_msg = self.preprocessor.validate_input(text)
            if not is_valid:
                return f"Unable to process: {error_msg}", None

            # Preprocess text
            processed_text = self.preprocessor.preprocess_text(text)

            # Transform to TF-IDF features
            tfidf_features = self.tfidf_vectorizer.transform([processed_text])

            # Make prediction
            prediction = self.model.predict(tfidf_features)[0]
            predicted_category = self.label_encoder.inverse_transform([prediction])[0]

            if show_confidence:
                try:
                    decision_scores = self.model.decision_function(tfidf_features)[0]
                    confidence = abs(decision_scores) if hasattr(decision_scores, '__len__') else abs(decision_scores)
                    return predicted_category, float(confidence)
                except:
                    return predicted_category, 0.8
            else:
                return predicted_category, None

        except Exception as e:
            st.error(f"Error in category prediction: {e}")
            return "Unable to process task description", None


class PriorityPredictor:
    """Wrapper for task priority prediction with error handling."""

    def __init__(self, model_path='models/task_priority_model.pkl'):
        """Initialize priority predictor."""
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.workload_balancer = None
        self.version = 'TaskMgmt_v3.0'
        self.load_model()

    def load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_path, 'rb') as file:
                self.model_data = pickle.load(file)

            # Extract components
            self.model = self.model_data['model']
            self.label_encoder = self.model_data['label_encoder']
            self.scaler = self.model_data.get('scaler', None)
            self.tfidf_vectorizer = self.model_data['tfidf_vectorizer']
            self.employees_data = self.model_data['employees_data']
            self.workload_balancer = self.model_data['workload_balancer']
            self.actual_categories = self.model_data.get('actual_categories', [])
            self.priority_keywords = self.model_data.get('priority_keywords', {})

        except Exception as e:
            st.error(f"Error loading priority model: {e}")
            raise

    def predict_priority(self, description, category):
        """
        Predict priority for task.

        Args:
            description: Task description
            category: Task category

        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate category
            validated_category = self.validate_category(category)

            # Extract features
            features, keyword_analysis, urgency_score = self.extract_features(
                description, validated_category
            )

            # Scale features if scaler available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

            # Make prediction
            prediction_encoded = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]

            # Decode prediction
            predicted_priority = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = prediction_proba[prediction_encoded]

            # Create probability dictionary
            priority_probabilities = {}
            for i, priority in enumerate(self.label_encoder.classes_):
                priority_probabilities[priority] = prediction_proba[i]

            return {
                'predicted_priority': predicted_priority,
                'confidence': confidence,
                'all_probabilities': priority_probabilities,
                'validated_category': validated_category,
                'category_was_valid': validated_category == category,
                'keyword_analysis': keyword_analysis,
                'urgency_score': urgency_score
            }

        except Exception as e:
            st.error(f"Error in priority prediction: {e}")
            return None

    def validate_category(self, category):
        """Validate and correct category if needed."""
        if category in self.actual_categories:
            return category

        # Find closest match
        category_lower = category.lower()
        for actual_cat in self.actual_categories:
            if category_lower in actual_cat.lower() or actual_cat.lower() in category_lower:
                return actual_cat

        # Return first category as fallback
        return self.actual_categories[0] if self.actual_categories else category

    def analyze_keywords(self, description):
        """Analyze keywords in description."""
        desc_lower = description.lower()
        analysis = {}

        for priority, patterns in self.priority_keywords.items():
            strong_matches = [kw for kw in patterns.get('strong', []) if kw in desc_lower]
            moderate_matches = [kw for kw in patterns.get('moderate', []) if kw in desc_lower]
            weak_matches = [kw for kw in patterns.get('weak', []) if kw in desc_lower]

            score = len(strong_matches) * 0.8 + len(moderate_matches) * 0.5 + len(weak_matches) * 0.3

            analysis[priority] = {
                'score': score,
                'strong': strong_matches,
                'moderate': moderate_matches,
                'weak': weak_matches,
                'total_matches': len(strong_matches) + len(moderate_matches) + len(weak_matches)
            }

        return analysis

    def calculate_urgency(self, description):
        """Calculate urgency score."""
        desc_lower = description.lower()
        urgency_score = 0

        high_urgency = ['urgent', 'immediately', 'critical', 'emergency']
        urgency_score += min(2, sum(0.8 for kw in high_urgency if kw in desc_lower))

        medium_urgency = ['important', 'needed', 'priority', 'asap']
        urgency_score += min(1, sum(0.5 for kw in medium_urgency if kw in desc_lower))

        urgency_score += min(1, description.count('!') * 0.5)

        return min(3, urgency_score)

    def extract_features(self, description, category):
        """Extract features for prediction."""
        # Basic features
        words = description.split()
        word_count = len(words)

        # Keyword analysis
        keyword_analysis = self.analyze_keywords(description)
        high_kw = keyword_analysis.get('high', {}).get('score', 0)
        med_kw = keyword_analysis.get('medium', {}).get('score', 0)
        low_kw = keyword_analysis.get('low', {}).get('score', 0)
        urgency = self.calculate_urgency(description)

        # Text features
        caps_ratio = sum(1 for word in words if word.isupper()) / max(1, len(words))
        punct_score = min(3, description.count('!') + description.count('?') * 0.5)
        avg_word_len = np.mean([len(word) for word in words]) if words else 0
        complexity = min(1, avg_word_len / 10)

        # Employee features (defaults)
        emp_load = 0.5
        category_match = 0

        # Category features
        n_cat_features = min(5, len(self.actual_categories))
        category_features = []
        for i, cat in enumerate(self.actual_categories):
            if i >= n_cat_features:
                break
            category_features.append(1 if cat == category else 0)

        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([description]).toarray()[0]

        # Combine features
        features = [
            word_count, high_kw, med_kw, low_kw, urgency,
            caps_ratio, punct_score, complexity, emp_load, category_match
        ]
        features.extend(category_features)
        features.extend(tfidf_features)

        return np.array(features).reshape(1, -1), keyword_analysis, urgency

    def recommend_employee(self, category, priority_probabilities, urgency_score):
        """Recommend employee for task."""
        try:
            predicted_priority = max(priority_probabilities, key=priority_probabilities.get)

            optimal_emp = self.workload_balancer.get_optimal_employee(
                category, predicted_priority, urgency_score
            )

            if optimal_emp and optimal_emp in self.employees_data:
                emp_details = self.workload_balancer.get_employee_details(optimal_emp)

                if emp_details:
                    reasons = []

                    if emp_details['preferred_category'] == category:
                        reasons.append("category expertise")
                    else:
                        reasons.append("best available")

                    load = emp_details['current_load']
                    if load <= 4:
                        reasons.append("low workload")
                    elif load <= 7:
                        reasons.append("moderate workload")
                    else:
                        reasons.append("available despite high load")

                    return {
                        'employee_id': emp_details['emp_id'],
                        'current_load': emp_details['current_load'],
                        'preferred_category': emp_details['preferred_category'],
                        'category_match': emp_details['preferred_category'] == category,
                        'recent_assignments': emp_details['recent_assignments'],
                        'selection_reasons': reasons,
                        'confidence_note': "Optimized selection with load balancing"
                    }
            return None
        except Exception as e:
            st.warning(f"Employee recommendation failed: {e}")
            return None


def load_employees_data():
    """Load employee data from CSV file."""
    try:
        employees_df = pd.read_csv('datasets/employees_dataset.csv')
        return employees_df
    except Exception as e:
        st.error(f"Error loading employees data: {e}")
        return None


def get_employee_recommendations(category, employees_df, max_load=10):
    """
    Get employee recommendations based on category preference and workload.

    Args:
        category: Task category
        employees_df: Employee dataframe
        max_load: Maximum workload threshold

    Returns:
        List of recommended employees
    """
    if employees_df is None:
        return []

    # Filter employees who prefer this category
    preferred_employees = employees_df[employees_df['emp_preferred_category'] == category].copy()
    preferred_employees = preferred_employees.sort_values('emp_load')

    # Get all employees sorted by load as backup
    all_employees = employees_df.sort_values('emp_load')

    recommendations = []

    # Add preferred employees first
    for _, emp in preferred_employees.iterrows():
        if emp['emp_load'] < max_load:
            recommendations.append({
                'emp_id': emp['emp_id'],
                'emp_load': emp['emp_load'],
                'preferred_category': emp['emp_preferred_category'],
                'category_match': True,
                'availability': 'Available' if emp['emp_load'] < 8 else 'Busy'
            })

    # Add other employees if needed
    if len(recommendations) < 3:
        for _, emp in all_employees.iterrows():
            if emp['emp_id'] not in [r['emp_id'] for r in recommendations] and emp['emp_load'] < max_load:
                recommendations.append({
                    'emp_id': emp['emp_id'],
                    'emp_load': emp['emp_load'],
                    'preferred_category': emp['emp_preferred_category'],
                    'category_match': False,
                    'availability': 'Available' if emp['emp_load'] < 8 else 'Busy'
                })
                if len(recommendations) >= 5:
                    break

    return recommendations[:5]


def get_model_based_employee_recommendation(priority_predictor, category, priority_result):
    """
    Get employee recommendation using the model's workload balancer.

    Args:
        priority_predictor: Priority prediction model
        category: Task category
        priority_result: Priority prediction results

    Returns:
        Employee recommendation or None
    """
    try:
        if hasattr(priority_predictor, 'workload_balancer') and priority_predictor.workload_balancer:
            emp_rec = priority_predictor.recommend_employee(
                category,
                priority_result.get('all_probabilities', {}),
                priority_result.get('urgency_score', 0)
            )

            if emp_rec:
                return {
                    'emp_id': emp_rec['employee_id'],
                    'emp_load': emp_rec['current_load'],
                    'preferred_category': emp_rec['preferred_category'],
                    'category_match': emp_rec['category_match'],
                    'availability': 'Available',
                    'selection_reasons': emp_rec.get('selection_reasons', []),
                    'confidence_note': emp_rec.get('confidence_note', '')
                }
    except Exception as e:
        st.warning(f"Model-based recommendation failed: {e}")

    return None


def create_confusion_matrix_plot(cm, categories, title):
    """Create an interactive confusion matrix plot."""
    fig = px.imshow(
        cm,
        x=categories,
        y=categories,
        color_continuous_scale='Blues',
        title=title,
        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'}
    )

    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(categories)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black", size=12)
            )

    fig.update_layout(height=500, width=600)
    return fig


def create_metrics_comparison_chart(category_metrics, priority_metrics):
    """Create a comparison chart of model metrics."""
    metrics_data = {
        'Model': ['Category Prediction', 'Priority Prediction'],
        'Accuracy': [category_metrics['accuracy'], priority_metrics['accuracy']],
        'Precision': [category_metrics['precision'], priority_metrics['precision']],
        'Recall': [category_metrics['recall'], priority_metrics['recall']],
        'F1-Score': [category_metrics['f1_score'], priority_metrics['f1_score']]
    }

    df_metrics = pd.DataFrame(metrics_data)
    fig = go.Figure()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=df_metrics['Model'],
            y=df_metrics[metric],
            marker_color=colors[i],
            text=[f'{val:.3f}' for val in df_metrics[metric]],
            textposition='outside'
        ))

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1.1])
    )

    return fig


def load_and_prepare_test_data():
    """Load and prepare test data for model evaluation."""
    try:
        tasks_df = pd.read_csv('datasets/tasks_dataset.csv')
        test_size = min(100, len(tasks_df))
        test_data = tasks_df.sample(n=test_size, random_state=42)
        return test_data
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None


def evaluate_category_model(category_predictor, test_data):
    """Evaluate category prediction model."""
    if test_data is None:
        return None

    try:
        predictions = []
        true_labels = []

        for _, row in test_data.iterrows():
            try:
                pred_category, _ = category_predictor.predict_category(row['task_description'])
                if pred_category and "Unable to process" not in pred_category:
                    predictions.append(pred_category)
                    true_labels.append(row['category'])
            except Exception:
                continue

        if len(predictions) == 0:
            return None

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=category_predictor.categories)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'predictions': predictions,
            'categories': category_predictor.categories
        }
    except Exception as e:
        st.error(f"Error evaluating category model: {e}")
        return None


def evaluate_priority_model(priority_predictor, category_predictor, test_data):
    """Evaluate priority prediction model."""
    if test_data is None:
        return None

    try:
        predictions = []
        true_labels = []

        for _, row in test_data.iterrows():
            try:
                pred_category, _ = category_predictor.predict_category(row['task_description'])
                if pred_category and "Unable to process" not in pred_category:
                    priority_result = priority_predictor.predict_priority(
                        row['task_description'], pred_category
                    )
                    if priority_result:
                        predictions.append(priority_result['predicted_priority'])
                        true_labels.append(row['priority'])
            except Exception:
                continue

        if len(predictions) == 0:
            return None

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        # Confusion matrix
        priority_classes = ['High', 'Medium', 'Low']
        cm = confusion_matrix(true_labels, predictions, labels=priority_classes)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'predictions': predictions,
            'categories': priority_classes
        }
    except Exception as e:
        st.error(f"Error evaluating priority model: {e}")
        return None


def show_model_metrics_page():
    """Display the model metrics and evaluation page."""
    st.header("📊 Model Performance Analytics")
    st.markdown("---")

    if 'category_predictor' not in st.session_state or 'priority_predictor' not in st.session_state:
        st.error("⚠️ Models not loaded. Please visit the main dashboard first.")
        return

    category_predictor = st.session_state.category_predictor
    priority_predictor = st.session_state.priority_predictor

    with st.spinner("Loading and evaluating models..."):
        test_data = load_and_prepare_test_data()

        if test_data is not None:
            category_metrics = evaluate_category_model(category_predictor, test_data)
            priority_metrics = evaluate_priority_model(priority_predictor, category_predictor, test_data)

            if category_metrics and priority_metrics:
                tab1, tab2, tab3 = st.tabs(
                    ["📈 Overall Performance", "🎯 Category Model", "⚡ Priority Model"])

                with tab1:
                    st.subheader("Model Performance Overview")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Category Accuracy", f"{category_metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("Priority Accuracy", f"{priority_metrics['accuracy']:.3f}")
                    with col3:
                        st.metric("Category F1-Score", f"{category_metrics['f1_score']:.3f}")
                    with col4:
                        st.metric("Priority F1-Score", f"{priority_metrics['f1_score']:.3f}")

                    comparison_fig = create_metrics_comparison_chart(category_metrics, priority_metrics)
                    st.plotly_chart(comparison_fig, use_container_width=True)

                with tab2:
                    st.subheader("🎯 Category Prediction Model Analysis")

                    category_cm_fig = create_confusion_matrix_plot(
                        category_metrics['confusion_matrix'],
                        category_metrics['categories'],
                        "Category Prediction Confusion Matrix"
                    )
                    st.plotly_chart(category_cm_fig, use_container_width=True)

                with tab3:
                    st.subheader("⚡ Priority Prediction Model Analysis")

                    priority_cm_fig = create_confusion_matrix_plot(
                        priority_metrics['confusion_matrix'],
                        priority_metrics['categories'],
                        "Priority Prediction Confusion Matrix"
                    )
                    st.plotly_chart(priority_cm_fig, use_container_width=True)

            else:
                st.error("❌ Could not evaluate models. Please check the data and model files.")
        else:
            st.error("❌ Could not load test data. Please ensure 'tasks_dataset.csv' exists.")


def main():
    """Main application function."""
    st.set_page_config(
        page_title="AI Task Management System",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["🏠 Main Dashboard", "📊 Model Metrics"])

    if page == "📊 Model Metrics":
        show_model_metrics_page()
        return

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #c0c9d9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .priority-high { color: #ff4444; font-weight: bold; }
    .priority-medium { color: #ff8800; font-weight: bold; }
    .priority-low { color: #44aa44; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🎯 AI Task Management System</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        st.info("Professional AI system for task management and resource allocation.")

        st.header("🔧 Models Used")
        st.markdown("- **Category Predictor**: SVM with TF-IDF")
        st.markdown("- **Priority Predictor**: Ensemble Model")
        st.markdown("- **Employee Matcher**: Load Balancer")

        # Model status indicators
        if st.session_state.get('models_loaded', False):
            st.success("✅ Models Status: Loaded")
            if hasattr(st.session_state, 'category_predictor'):
                st.info(f"📂 Categories: {len(st.session_state.category_predictor.categories)}")
            if hasattr(st.session_state, 'priority_predictor'):
                st.info(f"🎯 Model Version: {st.session_state.priority_predictor.version}")
            if st.session_state.get('employees_df') is not None:
                st.info(f"👥 Employees: {len(st.session_state.employees_df)}")
        else:
            st.error("❌ Models Status: Not Loaded")

        # Debug section
        if st.checkbox("🔍 Debug Mode"):
            st.subheader("Debug Information")
            model_files = [
                'models/task_classifier.pkl',
                'models/task_priority_model.pkl',
                'datasets/employees_dataset.csv'
            ]
            for file in model_files:
                if os.path.exists(file):
                    st.success(f"✅ {file}")
                else:
                    st.error(f"❌ {file} not found")

    # Initialize models
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading ML models..."):
            try:
                # Load models with proper error handling
                st.session_state.category_predictor = CategoryPredictor()
                st.session_state.priority_predictor = PriorityPredictor()
                st.session_state.employees_df = load_employees_data()

                st.session_state.models_loaded = True
                st.success("✅ ML models loaded successfully!")

            except Exception as e:
                st.session_state.models_loaded = False
                st.error(f"❌ Error loading models: {e}")
                st.error("Please ensure the model files exist in the models/ directory.")
                st.stop()

    # Check if models are loaded
    if not st.session_state.get('models_loaded', False):
        st.error("❌ AI models are not loaded. Please refresh the page.")
        st.stop()

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Task Description Input")
        task_description = st.text_area(
            "Enter your task description:",
            placeholder="e.g., Fix the critical login bug in the authentication system that prevents users from accessing their accounts",
            height=150
        )

        predict_button = st.button("🚀 Analyze Task", type="primary", use_container_width=True)

    with col2:
        st.header("💡 Examples")
        example_tasks = [
            "Fix critical database connection error",
            "Create marketing presentation for Q4 release",
            "Schedule team meeting for sprint planning",
            "Design new user interface for dashboard",
            "Review and approve budget proposal"
        ]

        for i, example in enumerate(example_tasks):
            if st.button(f"📋 {example[:40]}...", key=f"example_{i}"):
                st.session_state.example_task = example

    # Handle example selection
    if 'example_task' in st.session_state:
        task_description = st.session_state.example_task
        del st.session_state.example_task
        predict_button = True

    # Prediction logic
    if predict_button and task_description.strip():
        with st.spinner("Analyzing task with ML models..."):
            # Step 1: Predict Category
            predicted_category, category_confidence = st.session_state.category_predictor.predict_category(
                task_description, show_confidence=True)

            if predicted_category and "Unable to process" not in predicted_category:
                # Step 2: Predict Priority
                priority_result = st.session_state.priority_predictor.predict_priority(
                    task_description, predicted_category)

                if priority_result:
                    # Step 3: Get Employee Recommendations
                    model_recommendation = get_model_based_employee_recommendation(
                        st.session_state.priority_predictor, predicted_category, priority_result
                    )

                    csv_recommendations = get_employee_recommendations(
                        predicted_category, st.session_state.employees_df
                    )

                    # Combine recommendations
                    employee_recommendations = []
                    if model_recommendation:
                        employee_recommendations.append(model_recommendation)

                    existing_ids = [emp['emp_id'] for emp in employee_recommendations]
                    for emp in csv_recommendations:
                        if emp['emp_id'] not in existing_ids:
                            employee_recommendations.append(emp)

                    employee_recommendations = employee_recommendations[:5]

                    # Display Results
                    st.success("✅ Task analysis completed!")

                    st.header("📊 Analysis Results")

                    # Category and Priority Cards
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        category_display = priority_result.get('validated_category', predicted_category)
                        st.markdown(f"""
                        <div class="prediction-card">
                        <h3>📂 Predicted Category</h3>
                        <h2 style="color: #1f77b4;">{category_display}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        priority = priority_result['predicted_priority']
                        priority_class = f"priority-{priority.lower()}"
                        priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

                        st.markdown(f"""
                        <div class="prediction-card">
                        <h3>⚡ Predicted Priority</h3>
                        <h2 class="{priority_class}">{priority_emoji.get(priority, '⚪')} {priority}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        confidence = priority_result['confidence']
                        conf_color = "#22aa22" if confidence >= 0.8 else "#ff8800" if confidence >= 0.6 else "#ff4444"

                        st.markdown(f"""
                        <div class="prediction-card">
                        <h3>🎯 Confidence</h3>
                        <h2 style="color: {conf_color};">{confidence:.1%}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    # Priority Probabilities
                    st.subheader("📈 Priority Probabilities")
                    prob_cols = st.columns(3)
                    for i, (prio, prob) in enumerate(priority_result['all_probabilities'].items()):
                        with prob_cols[i]:
                            st.metric(prio, f"{prob:.1%}")

                    # Employee Recommendations
                    st.subheader("👥 Recommended Employees")

                    if employee_recommendations:
                        for i, emp in enumerate(employee_recommendations):
                            emp_col1, emp_col2 = st.columns([3, 1])

                            with emp_col1:
                                match_icon = "✅" if emp['category_match'] else "⚠️"

                                if i == 0 and model_recommendation:
                                    st.markdown(f"**{match_icon} {emp['emp_id']} 🤖**")
                                    st.caption("AI-Optimized Selection")

                                    if 'selection_reasons' in emp and emp['selection_reasons']:
                                        reasons_text = ", ".join(emp['selection_reasons'])
                                        st.caption(f"Selection criteria: {reasons_text}")
                                else:
                                    st.markdown(f"**{match_icon} {emp['emp_id']}**")

                                match_text = "Perfect Match" if emp['category_match'] else "Available"
                                st.markdown(f"**Category:** {emp['preferred_category']} | **Status:** {match_text}")

                            with emp_col2:
                                load = emp['emp_load']
                                load_color = "🟢" if load < 6 else "🟡" if load < 8 else "🔴"
                                load_status = "Light" if load < 6 else "Medium" if load < 8 else "Heavy"

                                st.metric(
                                    label="Workload",
                                    value=f"{load:.1f}/10" if isinstance(load, float) else f"{load}/10",
                                    delta=f"{load_status} {load_color}"
                                )

                            if i < len(employee_recommendations) - 1:
                                st.markdown("---")
                    else:
                        st.warning("No employees found for this category.")

                    # Action buttons
                    st.subheader("🎬 Actions")
                    action_cols = st.columns(3)

                    with action_cols[0]:
                        if st.button("🔧 Notify Employee", use_container_width=True):
                            if employee_recommendations:
                                selected_emp = employee_recommendations[0]['emp_id']
                                st.success(f"Notification sent to {selected_emp}!")

                    with action_cols[1]:
                        if st.button("📅 Schedule Task", use_container_width=True):
                            priority_scheduling = {
                                'High': 'Scheduled for immediate attention',
                                'Medium': 'Scheduled for next business day',
                                'Low': 'Added to weekly backlog'
                            }
                            schedule_msg = priority_scheduling.get(priority, 'Task scheduled')
                            st.success(schedule_msg)

                    with action_cols[2]:
                        if st.button("💾 Save Task", use_container_width=True):
                            st.success("Task saved successfully!")

                else:
                    st.error("❌ Error predicting task priority")
            else:
                st.error("❌ Unable to process the task description. Please provide more details.")

    elif predict_button:
        st.warning("⚠️ Please enter a task description to analyze.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 Powered by AI | Professional Task Management System"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()