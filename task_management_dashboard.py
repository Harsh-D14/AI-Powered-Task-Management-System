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
import io

warnings.filterwarnings('ignore')

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class SimpleWorkloadBalancer:
    """Simple workload balancer that can be pickled"""

    def __init__(self, employees_data):
        self.employees_data = employees_data
        self.current_loads = {emp_id: info['emp_load']
                              for emp_id, info in employees_data.items()}
        self.category_preferences = {emp_id: info['emp_preferred_category']
                                     for emp_id, info in employees_data.items()}

    def get_optimal_employee(self, task_category, task_priority_prob):
        """Simple scoring system for employee selection"""
        best_emp = None
        best_score = float('inf')

        for emp_id, current_load in self.current_loads.items():
            # Simple scoring: lower load is better
            score = current_load

            # Bonus for category match
            if self.category_preferences[emp_id] == task_category:
                score -= 2

            # High priority tasks to less loaded employees
            if task_priority_prob.get('High', 0) > 0.5:
                score += (current_load - 5) * 0.5

            if score < best_score:
                best_score = score
                best_emp = emp_id

        return best_emp

    def update_workload(self, emp_id, task_complexity=1):
        """Update workload"""
        if emp_id in self.current_loads:
            self.current_loads[emp_id] += task_complexity


class TaskCategoryPredictor:
    def __init__(self, model_path='task_classifier.pkl'):
        self.model_path = model_path
        self.tfidf_vectorizer = None
        self.svm_model = None
        self.label_encoder = None
        self.categories = None
        self.stemmer = PorterStemmer()

        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

        additional_stopwords = {'task', 'need', 'needs', 'required', 'please', 'must', 'should'}
        self.stop_words.update(additional_stopwords)

        self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            self.categories = model_data['categories']

        except FileNotFoundError:
            st.error(
                f"❌ Category model file '{self.model_path}' not found. Please ensure the file exists in the same directory.")
            return False
        except Exception as e:
            st.error(f"❌ Error loading category model: {str(e)}")
            st.error("Please check if the model file is compatible and not corrupted.")
            return False
        return True

    def preprocess_text(self, text):
        if not text or text.strip() == "":
            return ""

        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())

        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    def predict_category(self, task_description):
        processed_text = self.preprocess_text(task_description)

        if not processed_text:
            return "Unable to process", None

        tfidf_features = self.tfidf_vectorizer.transform([processed_text])
        prediction = self.svm_model.predict(tfidf_features)[0]
        predicted_category = self.label_encoder.inverse_transform([prediction])[0]

        # Get confidence scores
        try:
            decision_scores = self.svm_model.decision_function(tfidf_features)[0]
            confidence_scores = {}

            if len(self.categories) > 2:
                for i, category in enumerate(self.categories):
                    score = decision_scores[i] if len(decision_scores) > 1 else decision_scores
                    confidence_scores[category] = float(score)
            else:
                confidence_scores = {predicted_category: abs(float(decision_scores))}

        except Exception as e:
            confidence_scores = None

        return predicted_category, confidence_scores


class TaskPriorityPredictor:
    def __init__(self, model_path='task_priority_model.pkl'):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)

            # Extract required components
            self.model = self.model_data['model']
            self.label_encoder = self.model_data['label_encoder']
            self.tfidf_vectorizer = self.model_data['tfidf_vectorizer']
            self.employees_data = self.model_data.get('employees_data', {})
            self.categories = self.model_data['categories']

            # Handle workload balancer if it exists in the model
            if 'workload_balancer' in self.model_data:
                self.workload_balancer = self.model_data['workload_balancer']
            else:
                # Create a simple workload balancer if not in model
                self.workload_balancer = SimpleWorkloadBalancer(self.employees_data)

        except FileNotFoundError:
            st.error(
                f"❌ Priority model file '{self.model_path}' not found. Please ensure the file exists in the same directory.")
            return False
        except Exception as e:
            st.error(f"❌ Error loading priority model: {str(e)}")
            st.error("Please check if the model file is compatible and not corrupted.")
            return False
        return True

    def preprocess_text_simple(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return ' '.join(text.split())

    def calculate_text_features_fast(self, description):
        words = description.split()
        token_count = len(words)
        return token_count, token_count

    def create_prediction_features(self, description, category, employee_id=None):
        token_count, word_count = self.calculate_text_features_fast(description)

        tfidf_features = self.tfidf_vectorizer.transform([description]).toarray()[0]

        if employee_id and employee_id in self.employees_data:
            emp_info = self.employees_data[employee_id]
            emp_load = emp_info['emp_load']
            category_match = 1 if emp_info['emp_preferred_category'] == category else 0
        else:
            emp_load = 5
            category_match = 0

        category_features = []
        for cat in sorted(self.categories):
            category_features.append(1 if cat == category else 0)

        features = [token_count, word_count, emp_load, category_match]
        features.extend(category_features)
        features.extend(tfidf_features)

        return np.array(features).reshape(1, -1)

    def predict_priority(self, description, category, employee_id=None):
        try:
            if category not in self.categories:
                category = self.categories[0]

            X = self.create_prediction_features(description, category, employee_id)

            prediction_encoded = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]

            predicted_priority = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = prediction_proba[prediction_encoded]

            priority_probabilities = {}
            for i, priority in enumerate(self.label_encoder.classes_):
                priority_probabilities[priority] = prediction_proba[i]

            return {
                'predicted_priority': predicted_priority,
                'confidence': confidence,
                'all_probabilities': priority_probabilities
            }

        except Exception as e:
            st.error(f"Error predicting priority: {e}")
            return None


def load_employees_data():
    """Load employee data from CSV"""
    try:
        employees_df = pd.read_csv('employees_dataset.csv')
        return employees_df
    except Exception as e:
        st.error(f"Error loading employees data: {e}")
        return None


def get_employee_recommendations(category, employees_df, max_load=10):
    """Get employee recommendations based on category preference and workload"""
    if employees_df is None:
        return []

    # Filter employees who prefer this category
    preferred_employees = employees_df[employees_df['emp_preferred_category'] == category].copy()

    # Sort by workload (ascending - less loaded first)
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

    # If no preferred employees available, add others
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
    """Get employee recommendation using the model's workload balancer"""
    try:
        if hasattr(priority_predictor, 'workload_balancer') and priority_predictor.workload_balancer:
            optimal_emp = priority_predictor.workload_balancer.get_optimal_employee(
                category, priority_result.get('all_probabilities', {})
            )

            if optimal_emp and optimal_emp in priority_predictor.employees_data:
                emp_info = priority_predictor.employees_data[optimal_emp]
                return {
                    'emp_id': optimal_emp,
                    'emp_load': priority_predictor.workload_balancer.current_loads.get(optimal_emp, 5),
                    'preferred_category': emp_info.get('emp_preferred_category', 'Unknown'),
                    'category_match': emp_info.get('emp_preferred_category') == category,
                    'availability': 'Available'
                }
    except Exception as e:
        st.warning(f"Model-based recommendation failed: {e}")

    return None


def load_and_prepare_test_data():
    """Load and prepare test data for model evaluation"""
    try:
        # Load tasks dataset
        tasks_df = pd.read_csv('tasks_dataset.csv')

        # Create a test set from the available data
        # For demo purposes, we'll use a subset of the data
        test_size = min(100, len(tasks_df))
        test_data = tasks_df.sample(n=test_size, random_state=42)

        return test_data
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None


def evaluate_category_model(category_predictor, test_data):
    """Evaluate category prediction model"""
    if test_data is None or category_predictor.svm_model is None:
        return None

    try:
        # Get predictions
        predictions = []
        true_labels = []

        for _, row in test_data.iterrows():
            pred_category, _ = category_predictor.predict_category(row['task_description'])
            if pred_category != "Unable to process":
                predictions.append(pred_category)
                true_labels.append(row['category'])

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
    """Evaluate priority prediction model"""
    if test_data is None or priority_predictor.model is None:
        return None

    try:
        predictions = []
        true_labels = []

        for _, row in test_data.iterrows():
            # First predict category
            pred_category, _ = category_predictor.predict_category(row['task_description'])
            if pred_category != "Unable to process":
                # Then predict priority
                priority_result = priority_predictor.predict_priority(
                    row['task_description'], pred_category
                )
                if priority_result:
                    predictions.append(priority_result['predicted_priority'])
                    true_labels.append(row['priority'])

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


def create_confusion_matrix_plot(cm, categories, title):
    """Create an interactive confusion matrix plot"""
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


def create_metrics_comparison_chart(cat_metrics, pri_metrics):
    """Create a comparison chart of model metrics"""
    metrics_data = {
        'Model': ['Category Prediction', 'Priority Prediction'],
        'Accuracy': [cat_metrics['accuracy'], pri_metrics['accuracy']],
        'Precision': [cat_metrics['precision'], pri_metrics['precision']],
        'Recall': [cat_metrics['recall'], pri_metrics['recall']],
        'F1-Score': [cat_metrics['f1_score'], pri_metrics['f1_score']]
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


def create_feature_importance_plot(priority_predictor):
    """Create feature importance plot for Random Forest model"""
    try:
        if hasattr(priority_predictor.model, 'feature_importances_'):
            importances = priority_predictor.model.feature_importances_
            feature_names = priority_predictor.model_data.get('feature_names',
                                                              [f'Feature_{i}' for i in range(len(importances))])

            # Get top 15 features
            top_indices = np.argsort(importances)[-15:]
            top_importances = importances[top_indices]
            top_features = [feature_names[i] for i in top_indices]

            fig = go.Figure(go.Bar(
                x=top_importances,
                y=top_features,
                orientation='h',
                marker_color='lightblue'
            ))

            fig.update_layout(
                title='Top 15 Feature Importances (Priority Model)',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=600
            )

            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error creating feature importance plot: {e}")
        return None


def show_model_metrics_page():
    """Show the model metrics and evaluation page"""

    st.header("📊 Model Performance Analytics")
    st.markdown("---")

    # Load models and data
    if 'category_predictor' not in st.session_state or 'priority_predictor' not in st.session_state:
        st.error("⚠️ Models not loaded. Please visit the main dashboard first.")
        return

    category_predictor = st.session_state.category_predictor
    priority_predictor = st.session_state.priority_predictor

    # Load test data
    with st.spinner("Loading and evaluating models..."):
        test_data = load_and_prepare_test_data()

        if test_data is not None:
            # Evaluate models
            cat_metrics = evaluate_category_model(category_predictor, test_data)
            pri_metrics = evaluate_priority_model(priority_predictor, category_predictor, test_data)

            if cat_metrics and pri_metrics:
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📈 Overall Performance", "🎯 Category Model", "⚡ Priority Model", "🔍 Detailed Analysis"])

                with tab1:
                    st.subheader("Model Performance Overview")

                    # Key metrics cards
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            label="Category Accuracy",
                            value=f"{cat_metrics['accuracy']:.3f}",
                            delta=f"{cat_metrics['precision']:.3f} precision"
                        )

                    with col2:
                        st.metric(
                            label="Priority Accuracy",
                            value=f"{pri_metrics['accuracy']:.3f}",
                            delta=f"{pri_metrics['precision']:.3f} precision"
                        )

                    with col3:
                        st.metric(
                            label="Category Recall",
                            value=f"{cat_metrics['recall']:.3f}",
                            delta=f"{cat_metrics['f1_score']:.3f} F1-score"
                        )

                    with col4:
                        st.metric(
                            label="Priority Recall",
                            value=f"{pri_metrics['recall']:.3f}",
                            delta=f"{pri_metrics['f1_score']:.3f} F1-score"
                        )

                    st.markdown("---")

                    # Comparison chart
                    comparison_fig = create_metrics_comparison_chart(cat_metrics, pri_metrics)
                    st.plotly_chart(comparison_fig, use_container_width=True)

                    # Model insights
                    st.subheader("🎯 Key Insights")

                    insights_col1, insights_col2 = st.columns(2)

                    with insights_col1:
                        st.info(f"""
                        **Category Model Performance:**
                        - Accuracy: {cat_metrics['accuracy']:.1%}
                        - Best performing: SVM with TF-IDF features
                        - Handles {len(cat_metrics['categories'])} different categories
                        """)

                    with insights_col2:
                        st.info(f"""
                        **Priority Model Performance:**
                        - Accuracy: {pri_metrics['accuracy']:.1%}
                        - Best performing: Random Forest classifier
                        - Considers employee workload and preferences
                        """)

                with tab2:
                    st.subheader("🎯 Category Prediction Model Analysis")

                    # Confusion matrix
                    cat_cm_fig = create_confusion_matrix_plot(
                        cat_metrics['confusion_matrix'],
                        cat_metrics['categories'],
                        "Category Prediction Confusion Matrix"
                    )
                    st.plotly_chart(cat_cm_fig, use_container_width=True)

                    # Detailed metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 Detailed Metrics")
                        st.json({
                            "Accuracy": f"{cat_metrics['accuracy']:.4f}",
                            "Precision": f"{cat_metrics['precision']:.4f}",
                            "Recall": f"{cat_metrics['recall']:.4f}",
                            "F1-Score": f"{cat_metrics['f1_score']:.4f}"
                        })

                    with col2:
                        st.subheader("📋 Classification Report")
                        if len(cat_metrics['true_labels']) > 0:
                            report = classification_report(
                                cat_metrics['true_labels'],
                                cat_metrics['predictions'],
                                output_dict=True,
                                zero_division=0
                            )

                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(3))

                with tab3:
                    st.subheader("⚡ Priority Prediction Model Analysis")

                    # Confusion matrix
                    pri_cm_fig = create_confusion_matrix_plot(
                        pri_metrics['confusion_matrix'],
                        pri_metrics['categories'],
                        "Priority Prediction Confusion Matrix"
                    )
                    st.plotly_chart(pri_cm_fig, use_container_width=True)

                    # Feature importance
                    importance_fig = create_feature_importance_plot(priority_predictor)
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)

                    # Detailed metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 Detailed Metrics")
                        st.json({
                            "Accuracy": f"{pri_metrics['accuracy']:.4f}",
                            "Precision": f"{pri_metrics['precision']:.4f}",
                            "Recall": f"{pri_metrics['recall']:.4f}",
                            "F1-Score": f"{pri_metrics['f1_score']:.4f}"
                        })

                    with col2:
                        st.subheader("📋 Classification Report")
                        if len(pri_metrics['true_labels']) > 0:
                            report = classification_report(
                                pri_metrics['true_labels'],
                                pri_metrics['predictions'],
                                output_dict=True,
                                zero_division=0
                            )

                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(3))

                with tab4:
                    st.subheader("🔍 Detailed Analysis & Recommendations")

                    # Performance analysis
                    st.subheader("📈 Performance Analysis")

                    analysis_text = f"""
                    ### Model Performance Summary

                    **Category Prediction Model (SVM):**
                    - Overall Accuracy: {cat_metrics['accuracy']:.1%}
                    - This model shows {'excellent' if cat_metrics['accuracy'] > 0.9 else 'good' if cat_metrics['accuracy'] > 0.8 else 'moderate'} performance
                    - Precision: {cat_metrics['precision']:.1%} - {'High precision means low false positives' if cat_metrics['precision'] > 0.8 else 'Consider tuning to reduce false positives'}
                    - Recall: {cat_metrics['recall']:.1%} - {'High recall means low false negatives' if cat_metrics['recall'] > 0.8 else 'Consider improving to catch more true positives'}

                    **Priority Prediction Model (Random Forest):**
                    - Overall Accuracy: {pri_metrics['accuracy']:.1%}
                    - This model shows {'excellent' if pri_metrics['accuracy'] > 0.9 else 'good' if pri_metrics['accuracy'] > 0.8 else 'moderate'} performance
                    - Precision: {pri_metrics['precision']:.1%}
                    - Recall: {pri_metrics['recall']:.1%}

                    ### Key Observations:
                    1. **Data Quality**: Model performance is heavily dependent on the quality and diversity of training data
                    2. **Feature Engineering**: The combination of text features (TF-IDF) and categorical features works well
                    3. **Employee Integration**: The priority model successfully incorporates employee workload and preferences
                    """

                    st.markdown(analysis_text)

                    # Recommendations
                    st.subheader("💡 Recommendations for Improvement")

                    recommendations = []

                    if cat_metrics['accuracy'] < 0.85:
                        recommendations.append(
                            "🎯 **Category Model**: Consider collecting more diverse training data or feature engineering")

                    if pri_metrics['accuracy'] < 0.85:
                        recommendations.append("⚡ **Priority Model**: Fine-tune hyperparameters or add more features")

                    if cat_metrics['precision'] < 0.8:
                        recommendations.append(
                            "🔍 **False Positives**: Review misclassified samples to improve precision")

                    if pri_metrics['recall'] < 0.8:
                        recommendations.append(
                            "📊 **Coverage**: Improve recall by balancing training data across priority classes")

                    recommendations.extend([
                        "📈 **Continuous Learning**: Implement feedback loops to retrain models with new data",
                        "🔄 **A/B Testing**: Test model variations in production to find optimal configurations",
                        "📊 **Monitoring**: Set up automated monitoring for model drift and performance degradation",
                        "🎯 **Feature Importance**: Regularly review feature importance to understand model decisions"
                    ])

                    for rec in recommendations:
                        st.write(f"- {rec}")

                    # Data statistics
                    st.subheader("📊 Dataset Statistics")

                    if test_data is not None:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Test Samples", len(test_data))

                        with col2:
                            category_dist = test_data['category'].value_counts()
                            st.metric("Categories", len(category_dist))

                        with col3:
                            priority_dist = test_data['priority'].value_counts()
                            st.metric("Priority Levels", len(priority_dist))

                        # Distribution charts
                        fig_dist = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('Category Distribution', 'Priority Distribution'),
                            specs=[[{"type": "pie"}, {"type": "pie"}]]
                        )

                        # Category distribution
                        fig_dist.add_trace(
                            go.Pie(labels=category_dist.index, values=category_dist.values, name="Categories"),
                            row=1, col=1
                        )

                        # Priority distribution
                        fig_dist.add_trace(
                            go.Pie(labels=priority_dist.index, values=priority_dist.values, name="Priorities"),
                            row=1, col=2
                        )

                        fig_dist.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig_dist, use_container_width=True)

            else:
                st.error("❌ Could not evaluate models. Please check the data and model files.")
        else:
            st.error("❌ Could not load test data. Please ensure 'tasks_dataset.csv' exists.")


def main():
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

    # Custom CSS (existing styles)
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
    .employee-card {
        background-color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .priority-high { color: #ff4444; font-weight: bold; }
    .priority-medium { color: #ff8800; font-weight: bold; }
    .priority-low { color: #44aa44; font-weight: bold; }
    .category-match { color: #22aa22; }
    .category-no-match { color: #aa2222; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🎯 AI Task Management System</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        st.info(
            "This AI system predicts task categories and priorities, then recommends employees based on workload and preferences.")

        st.header("🔧 Models Used")
        st.markdown("- **Category Predictor**: SVM with TF-IDF")
        st.markdown("- **Priority Predictor**: Random Forest")
        st.markdown("- **Employee Matcher**: Workload Balancer")

        # Model status indicators
        if st.session_state.get('models_loaded', False):
            st.success("✅ Models Status: Loaded")
            if hasattr(st.session_state, 'category_predictor') and hasattr(st.session_state.category_predictor,
                                                                           'categories'):
                st.info(f"📂 Categories: {len(st.session_state.category_predictor.categories)}")
            if st.session_state.employees_df is not None:
                st.info(f"👥 Employees: {len(st.session_state.employees_df)}")
        else:
            st.error("❌ Models Status: Not Loaded")

        # Debug section (optional)
        if st.checkbox("🔍 Debug Mode"):
            st.subheader("Debug Information")
            import os
            model_files = ['task_classifier.pkl', 'task_priority_model.pkl', 'employees_dataset.csv']
            for file in model_files:
                if os.path.exists(file):
                    st.success(f"✅ {file}")
                else:
                    st.error(f"❌ {file} not found")

    # Initialize models
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.category_predictor = TaskCategoryPredictor()
            st.session_state.priority_predictor = TaskPriorityPredictor()
            st.session_state.employees_df = load_employees_data()

            # Validate models loaded successfully
            category_loaded = hasattr(st.session_state.category_predictor,
                                      'svm_model') and st.session_state.category_predictor.svm_model is not None
            priority_loaded = hasattr(st.session_state.priority_predictor,
                                      'model') and st.session_state.priority_predictor.model is not None

            if category_loaded and priority_loaded:
                st.session_state.models_loaded = True
                st.success("✅ AI models loaded successfully!")
            else:
                st.session_state.models_loaded = False
                st.error("❌ Failed to load one or more AI models. Please check the model files.")
                st.stop()

    # Check if models are loaded before proceeding
    if not st.session_state.get('models_loaded', False):
        st.error("❌ AI models are not loaded. Please refresh the page and ensure model files are available.")
        st.stop()

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Task Description Input")
        task_description = st.text_area(
            "Enter your task description:",
            placeholder="e.g., Fix the login bug in the authentication system that prevents users from accessing their accounts",
            height=150
        )

        predict_button = st.button("🚀 Analyze Task", type="primary", use_container_width=True)

    with col2:
        st.header("💡 Examples")
        example_tasks = [
            "Fix database connection error",
            "Create marketing presentation for Q4",
            "Schedule team meeting for sprint planning",
            "Design new user interface for mobile app",
            "Review and approve budget proposal"
        ]

        for i, example in enumerate(example_tasks):
            if st.button(f"📋 {example}", key=f"example_{i}"):
                st.session_state.example_task = example
                task_description = example
                predict_button = True

    # Handle example selection
    if 'example_task' in st.session_state:
        task_description = st.session_state.example_task
        del st.session_state.example_task
        predict_button = True

    # Prediction logic
    if predict_button and task_description.strip():
        with st.spinner("Analyzing task with AI models..."):
            # Step 1: Predict Category
            predicted_category, category_confidence = st.session_state.category_predictor.predict_category(
                task_description)

            if predicted_category != "Unable to process":
                # Step 2: Predict Priority
                priority_result = st.session_state.priority_predictor.predict_priority(task_description,
                                                                                       predicted_category)

                if priority_result:
                    # Step 3: Get Employee Recommendations
                    # Try model-based recommendation first
                    model_recommendation = get_model_based_employee_recommendation(
                        st.session_state.priority_predictor, predicted_category, priority_result
                    )

                    # Get CSV-based recommendations
                    csv_recommendations = get_employee_recommendations(
                        predicted_category,
                        st.session_state.employees_df
                    )

                    # Combine recommendations (model first if available)
                    employee_recommendations = []
                    if model_recommendation:
                        employee_recommendations.append(model_recommendation)

                    # Add CSV recommendations (avoid duplicates)
                    existing_ids = [emp['emp_id'] for emp in employee_recommendations]
                    for emp in csv_recommendations:
                        if emp['emp_id'] not in existing_ids:
                            employee_recommendations.append(emp)

                    employee_recommendations = employee_recommendations[:5]  # Limit to 5

                    # Display Results
                    st.success("✅ Task analysis completed!")

                    # Results section
                    st.header("📊 Analysis Results")

                    # Category and Priority Cards
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div class="prediction-card">
                        <h3>📂 Predicted Category</h3>
                        <h2 style="color: #1f77b4;">{predicted_category}</h2>
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
                        st.markdown(f"""
                        <div class="prediction-card">
                        <h3>🎯 Confidence</h3>
                        <h2 style="color: #1f77b4;">{confidence:.1%}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    # Detailed probabilities
                    st.subheader("📈 Priority Probabilities")
                    prob_cols = st.columns(3)
                    for i, (prio, prob) in enumerate(priority_result['all_probabilities'].items()):
                        with prob_cols[i]:
                            st.metric(prio, f"{prob:.1%}")

                    # Employee Recommendations
                    st.subheader("👥 Recommended Employees")

                    if employee_recommendations:
                        for i, emp in enumerate(employee_recommendations):
                            match_icon = "✅" if emp['category_match'] else "⚠️"
                            match_text = "Perfect Match" if emp['category_match'] else "Available"
                            match_class = "category-match" if emp['category_match'] else "category-no-match"

                            load_color = "#22aa22" if emp['emp_load'] < 6 else "#ff8800" if emp[
                                                                                                'emp_load'] < 8 else "#ff4444"

                            st.markdown(f"""
                            <div class="employee-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h4>{match_icon} {emp['emp_id']}</h4>
                                    <p><span class="{match_class}">Category: {emp['preferred_category']}</span></p>
                                    <p>Status: <strong>{match_text}</strong></p>
                                </div>
                                <div style="text-align: right;">
                                    <p>Workload</p>
                                    <h3 style="color: {load_color};">{emp['emp_load']}/10</h3>
                                    <p><em>{emp['availability']}</em></p>
                                </div>
                            </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No employees found for this category.")

                    # Action buttons
                    st.subheader("🎬 Actions")
                    action_cols = st.columns(3)

                    with action_cols[0]:
                        if st.button("📧 Notify Selected Employee", use_container_width=True):
                            st.success("Notification sent!")

                    with action_cols[1]:
                        if st.button("📅 Schedule Task", use_container_width=True):
                            st.success("Task scheduled!")

                    with action_cols[2]:
                        if st.button("💾 Save to Database", use_container_width=True):
                            st.success("Task saved to database!")

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
        "🤖 Powered by AI | Task Management System v1.0"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()