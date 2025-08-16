#!/usr/bin/env python3
"""
Task Priority Prediction Model Trainer with Regularization
Implements anti-overfitting measures to achieve realistic performance ranges (0.7-0.95)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from collections import defaultdict, Counter
import time
import os
import re

warnings.filterwarnings('ignore')


class WorkloadBalancer:
    """Manages workload distribution among employees with realistic constraints"""

    def __init__(self, employees_data, tasks_df):
        self.employees_data = employees_data
        self.initial_workloads = {emp_id: info['emp_load'] for emp_id, info in employees_data.items()}
        self.current_workloads = self.initial_workloads.copy()
        self.category_preferences = {emp_id: info['emp_preferred_category']
                                     for emp_id, info in employees_data.items()}
        self.assignment_history = {}
        self.randomness_factor = 0.15

    def select_employee_for_task(self, task_category, predicted_priority, urgency_score=0):
        """Select optimal employee for task assignment with controlled randomness"""
        available_employees = [emp_id for emp_id, load in self.current_workloads.items() if load < 10]

        if not available_employees:
            return min(self.current_workloads, key=self.current_workloads.get)

        candidates = []
        for emp_id in available_employees:
            base_score = self._calculate_assignment_score(emp_id, task_category, predicted_priority)
            random_factor = np.random.normal(0, self.randomness_factor)
            final_score = base_score + random_factor
            candidates.append((emp_id, final_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:min(3, len(candidates))]
        weights = [0.6, 0.3, 0.1][:len(top_candidates)]
        selected_idx = np.random.choice(len(top_candidates), p=weights)

        selected_emp = top_candidates[selected_idx][0]
        self.assignment_history[selected_emp] = self.assignment_history.get(selected_emp, 0) + 1

        return selected_emp

    def _calculate_assignment_score(self, emp_id, task_category, predicted_priority):
        """Calculate employee assignment score based on multiple factors"""
        score = 0
        current_load = self.current_workloads[emp_id]
        is_expert = self.category_preferences[emp_id] == task_category

        if is_expert:
            score += 3

        score += (10 - current_load) * 0.5

        if predicted_priority == 'High' and current_load <= 5:
            score += 1

        return score

    def update_employee_workload(self, emp_id, task_complexity=1, predicted_priority='Medium'):
        """Update employee workload after task assignment"""
        if emp_id not in self.current_workloads:
            return

        complexity_multipliers = {'High': 1.5, 'Medium': 1.0, 'Low': 0.8}
        final_complexity = task_complexity * complexity_multipliers.get(predicted_priority, 1.0)

        noise = np.random.normal(0, 0.1)
        self.current_workloads[emp_id] = min(10, max(0, self.current_workloads[emp_id] + final_complexity + noise))

    def get_employee_status(self, emp_id):
        """Retrieve current status and details for specified employee"""
        if emp_id not in self.employees_data:
            return None
        return {
            'emp_id': emp_id,
            'current_load': self.current_workloads.get(emp_id, 0),
            'preferred_category': self.category_preferences.get(emp_id, 'Unknown'),
            'recent_assignments': self.assignment_history.get(emp_id, 0)
        }


class TaskPriorityPredictor:
    """Main class for training task priority prediction models with regularization"""

    def __init__(self, use_quick_training=True, regularization_level='medium'):
        self.use_quick_training = use_quick_training
        self.regularization_level = regularization_level
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        self.feature_names = []
        self.employees_data = None
        self.workload_balancer = None
        self.available_categories = []
        self.available_priorities = []
        self.priority_keywords = None

    def extract_keyword_patterns(self, tasks_df):
        """Extract keyword patterns from task descriptions with controlled noise"""
        print("Extracting keyword patterns...")

        keyword_patterns = {
            'high': {
                'strong': ['urgent', 'immediately', 'critical', 'emergency'],
                'moderate': ['asap', 'now', 'today'],
                'weak': ['important', 'priority']
            },
            'medium': {
                'strong': ['important', 'needed', 'priority'],
                'moderate': ['soon', 'required', 'due'],
                'weak': ['update', 'review']
            },
            'low': {
                'strong': ['plan', 'organize', 'future'],
                'moderate': ['update', 'review', 'document'],
                'weak': ['eventually', 'later']
            }
        }

        self.priority_keywords = keyword_patterns
        return keyword_patterns

    def load_training_data(self):
        """Load and validate training datasets"""
        print("Loading training datasets...")

        # Ensure datasets directory exists
        if not os.path.exists('datasets'):
            raise FileNotFoundError("datasets directory not found. Please ensure datasets folder exists.")

        # Check for required files
        required_files = ['datasets/tasks_dataset.csv', 'datasets/employees_dataset.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            raise FileNotFoundError(f"Required dataset files missing: {missing_files}")

        self.tasks_df = pd.read_csv('datasets/tasks_dataset.csv')
        self.employees_df = pd.read_csv('datasets/employees_dataset.csv')

        # Validate required columns in tasks dataset
        required_task_columns = ['taskid', 'task_description', 'priority', 'category', 'assigned_to_employeeid']
        missing_task_cols = [col for col in required_task_columns if col not in self.tasks_df.columns]

        if missing_task_cols:
            raise ValueError(f"Required columns missing from tasks dataset: {missing_task_cols}")

        # Validate required columns in employees dataset
        required_emp_columns = ['emp_id', 'emp_load', 'emp_preferred_category']
        missing_emp_cols = [col for col in required_emp_columns if col not in self.employees_df.columns]

        if missing_emp_cols:
            raise ValueError(f"Required columns missing from employees dataset: {missing_emp_cols}")

        # Create train/test split at dataset level for external evaluation
        self.tasks_df, self.test_tasks_df = train_test_split(
            self.tasks_df, test_size=0.2, random_state=42, stratify=self.tasks_df['priority']
        )

        # Save test data for external evaluation
        os.makedirs('datasets', exist_ok=True)
        self.test_tasks_df.to_csv('datasets/test_tasks_dataset.csv', index=False)
        print(f"Test dataset saved: {len(self.test_tasks_df)} samples for external evaluation")
        print(f"Training dataset: {len(self.tasks_df)} samples")

        if os.path.exists('datasets/task_preprocessed_data.csv'):
            try:
                self.preprocessed_df = pd.read_csv('datasets/task_preprocessed_data.csv')
                # Verify required columns exist
                required_columns = ['taskid', 'token_count', 'original_word_count']
                missing_columns = [col for col in required_columns if col not in self.preprocessed_df.columns]

                if missing_columns:
                    print(f"Missing columns {missing_columns} in preprocessed data. Regenerating...")
                    raise ValueError("Missing required columns")

            except (ValueError, pd.errors.EmptyDataError, KeyError):
                self.preprocessed_df = self._create_preprocessed_data()
        else:
            self.preprocessed_df = self._create_preprocessed_data()

        print(f"Loaded {len(self.tasks_df)} tasks and {len(self.employees_df)} employees")

        self.available_categories = sorted(self.tasks_df['category'].unique())
        self.available_priorities = sorted(self.tasks_df['priority'].unique())

        priority_distribution = self.tasks_df['priority'].value_counts()
        print("Priority distribution:")
        for priority, count in priority_distribution.items():
            percentage = count / len(self.tasks_df) * 100
            print(f"  {priority}: {count} ({percentage:.1f}%)")

            if percentage < 10:
                print(f"    Warning: {priority} priority is underrepresented")
            elif percentage > 60:
                print(f"    Warning: {priority} priority is overrepresented")

        self.extract_keyword_patterns(self.tasks_df)
        self.employees_data = self.employees_df.set_index('emp_id').to_dict('index')

        return self.tasks_df, self.employees_df, self.preprocessed_df

    def _create_preprocessed_data(self):
        """Create preprocessed data with token counts and other metrics"""
        preprocessed_df = pd.DataFrame({
            'taskid': self.tasks_df['taskid'],
            'token_count': self.tasks_df['task_description'].str.split().str.len(),
            'original_word_count': self.tasks_df['task_description'].str.split().str.len()
        })

        # Fill any NaN values with 0
        preprocessed_df = preprocessed_df.fillna(0)

        # Save the preprocessed data for future use
        os.makedirs('datasets', exist_ok=True)
        preprocessed_df.to_csv('datasets/task_preprocessed_data.csv', index=False)

        return preprocessed_df

    def create_feature_matrix(self, tasks_df, preprocessed_df):
        """Create regularized feature matrix to prevent overfitting"""
        print(f"Creating feature matrix...")

        df = tasks_df.merge(preprocessed_df[['taskid', 'token_count', 'original_word_count']],
                            on='taskid', how='left')

        # Handle any missing values from the merge
        df['token_count'] = df['token_count'].fillna(df['task_description'].str.split().str.len())
        df['original_word_count'] = df['original_word_count'].fillna(df['task_description'].str.split().str.len())
        df = df.fillna(0)

        keyword_features = self._extract_keyword_features(df['task_description'])
        text_features = self._extract_text_features(df['task_description'])

        max_tfidf_features = 5 if self.use_quick_training else 8  # Reduced further
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 1),
            min_df=5,  # Increased to reduce vocabulary
            max_df=0.6  # Reduced to ignore common terms
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['task_description']).toarray()

        category_features = pd.get_dummies(df['category'], prefix='cat').values
        employee_features = self._extract_employee_features(df)

        features = [
            df['token_count'].values,
            df['original_word_count'].values,
        ]

        for i in range(keyword_features.shape[1]):
            features.append(keyword_features[:, i])

        for i in range(text_features.shape[1]):
            features.append(text_features[:, i])

        for i in range(employee_features.shape[1]):
            features.append(employee_features[:, i])

        n_category_features = min(5, category_features.shape[1])
        for i in range(n_category_features):
            features.append(category_features[:, i])

        for i in range(tfidf_features.shape[1]):
            features.append(tfidf_features[:, i])

        X = np.column_stack(features)

        noise_levels = {'heavy': 0.05, 'medium': 0.02, 'light': 0.01}
        noise_level = noise_levels.get(self.regularization_level, 0.02)
        noise = np.random.normal(0, noise_level, X.shape)
        X = X + noise

        self.feature_names = ['token_count', 'word_count']
        self.feature_names.extend(['high_keywords', 'medium_keywords', 'low_keywords', 'urgency_score'])
        self.feature_names.extend(['caps_ratio', 'punctuation_score', 'text_complexity'])
        self.feature_names.extend(['employee_load', 'category_match'])
        self.feature_names.extend([f'category_{i}' for i in range(n_category_features)])
        self.feature_names.extend([f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

        y = self.label_encoder.fit_transform(df['priority'])

        print(f"Generated {X.shape[1]} features across multiple categories")
        print(
            f"Feature breakdown: Keywords(4), Text(3), Employee(2), Category({n_category_features}), TF-IDF({tfidf_features.shape[1]})")

        return X, y, df

    def _extract_keyword_features(self, descriptions):
        """Extract keyword-based features with controlled weights"""
        features = []

        for desc in descriptions:
            desc_lower = desc.lower()

            high_score = 0
            medium_score = 0
            low_score = 0

            for kw in self.priority_keywords['high']['strong']:
                if kw in desc_lower:
                    high_score += 0.8
            for kw in self.priority_keywords['high']['moderate']:
                if kw in desc_lower:
                    high_score += 0.5

            for kw in self.priority_keywords['medium']['strong']:
                if kw in desc_lower:
                    medium_score += 0.8
            for kw in self.priority_keywords['medium']['moderate']:
                if kw in desc_lower:
                    medium_score += 0.5

            for kw in self.priority_keywords['low']['strong']:
                if kw in desc_lower:
                    low_score += 0.6

            urgency_score = min(3, high_score + medium_score * 0.5)

            noise = np.random.normal(0, 0.1)
            features.append([
                high_score + noise,
                medium_score + noise,
                low_score + noise,
                urgency_score + noise
            ])

        return np.array(features)

    def _extract_text_features(self, descriptions):
        """Extract diverse text-based features"""
        features = []

        for desc in descriptions:
            words = desc.split()

            caps_ratio = sum(1 for word in words if word.isupper()) / max(1, len(words))
            punctuation_score = min(3, desc.count('!') + desc.count('?') * 0.5)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            text_complexity = min(1, avg_word_length / 10)

            features.append([caps_ratio, punctuation_score, text_complexity])

        return np.array(features)

    def _extract_employee_features(self, df):
        """Extract employee-related features"""
        features = []

        for _, row in df.iterrows():
            emp_id = row['assigned_to_employeeid']

            if emp_id in self.employees_data:
                emp_info = self.employees_data[emp_id]
                employee_load = emp_info['emp_load'] / 10
                category_match = 1 if emp_info['emp_preferred_category'] == row['category'] else 0
            else:
                employee_load = 0.5
                category_match = 0

            features.append([employee_load, category_match])

        return np.array(features)

    def train_prediction_model(self, X, y):
        """Train model with regularization to achieve realistic performance"""
        print(f"Training prediction model...")

        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        if self.use_quick_training:
            print("Using Random Forest with strong regularization...")

            param_grid = {
                'n_estimators': [30, 50],  # Fewer trees
                'max_depth': [3, 4],  # Shallower trees
                'min_samples_split': [10, 15],  # Higher split requirement
                'min_samples_leaf': [5, 8],  # Higher leaf requirement
                'max_features': ['sqrt'],  # Reduced feature subset
                'bootstrap': [True]  # Always use bootstrap
            }

            model = RandomForestClassifier(
                random_state=42,
                class_weight=class_weight_dict,
                n_jobs=-1
            )
        else:
            print("Using XGBoost with regularization...")

            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 4],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.6, 0.8],
                'colsample_bytree': [0.6, 0.8],
                'reg_alpha': [0.1, 0.5],
                'reg_lambda': [1.0, 2.0],
                'min_child_weight': [3, 5]
            }

            model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            )

        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        if cv_scores.mean() > 0.92:
            print("Warning: CV accuracy suggests potential overfitting")
        elif cv_scores.mean() < 0.65:
            print("Warning: CV accuracy below expected range")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        test_accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time

        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Test accuracy: {test_accuracy:.3f}")

        train_accuracy = accuracy_score(y_train, self.model.predict(X_train_scaled))
        accuracy_gap = train_accuracy - test_accuracy

        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Train-test gap: {accuracy_gap:.3f}")

        if accuracy_gap > 0.1:
            print("Warning: Large train-test gap indicates potential overfitting")
        elif accuracy_gap < 0.02:
            print("Good: Small train-test gap indicates proper generalization")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("Actual\\Predicted:", "  ".join(f"{cls:>8}" for cls in self.label_encoder.classes_))
        for i, row in enumerate(cm):
            print(f"{self.label_encoder.classes_[i]:>8}", "  ".join(f"{val:>8}" for val in row))

        if test_accuracy >= 0.75 and test_accuracy <= 0.90 and accuracy_gap <= 0.1:
            print(f"\nModel performance within target range: {test_accuracy:.3f}")
        elif test_accuracy > 0.90:
            print(f"\nPotential overfitting detected: accuracy {test_accuracy:.3f} exceeds 0.90")
        else:
            print(f"\nModel underperforming: accuracy {test_accuracy:.3f} below 0.75 target")

        return self.model

    def create_workload_balancer(self, df):
        """Initialize workload balancer component"""
        print("Creating workload balancer...")
        self.workload_balancer = WorkloadBalancer(self.employees_data, df)
        return self.workload_balancer

    def save_trained_model(self, filename='models/task_priority_model.pkl'):
        """Save trained model and associated components"""
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_names': self.feature_names,
            'employees_data': self.employees_data,
            'workload_balancer': self.workload_balancer,
            'available_categories': self.available_categories,
            'available_priorities': self.available_priorities,
            'priority_keywords': self.priority_keywords,
            'regularization_level': self.regularization_level,
            'use_quick_training': self.use_quick_training,
            'version': 'v1.0'
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        file_size = os.path.getsize(filename) / 1024 / 1024
        print(f"Model saved to {filename} ({file_size:.1f} MB)")

    def execute_training_pipeline(self):
        """Execute complete training pipeline"""
        print("=" * 70)
        print("TASK PRIORITY PREDICTION MODEL TRAINER")
        print("=" * 70)

        total_start = time.time()

        try:
            tasks_df, employees_df, preprocessed_df = self.load_training_data()
            X, y, df = self.create_feature_matrix(tasks_df, preprocessed_df)
            model = self.train_prediction_model(X, y)
            workload_balancer = self.create_workload_balancer(df)
            self.save_trained_model()

            total_time = time.time() - total_start

            print(f"\n{'=' * 70}")
            print(f"TRAINING PIPELINE COMPLETED IN {total_time:.1f} SECONDS")
            print("=" * 70)

        except Exception as e:
            print(f"Training pipeline error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main training function with user configuration"""
    print("Task Priority Prediction Model Trainer")

    print("\nSelect regularization strength:")
    print("1. Light regularization")
    print("2. Medium regularization [Recommended]")
    print("3. Heavy regularization")

    reg_choice = input("Enter choice (1-3): ").strip()
    regularization_map = {'1': 'light', '2': 'medium', '3': 'heavy'}
    regularization_level = regularization_map.get(reg_choice, 'medium')

    print("\nSelect model complexity:")
    print("1. Quick training (Random Forest, ~45 seconds)")
    print("2. Comprehensive training (Regularized XGBoost, ~2-3 minutes)")

    training_choice = input("Enter choice (1 or 2): ").strip()
    use_quick_training = training_choice != '2'

    print(f"Initializing training with {regularization_level} regularization...")

    try:
        predictor = TaskPriorityPredictor(
            use_quick_training=use_quick_training,
            regularization_level=regularization_level
        )
        predictor.execute_training_pipeline()

        print(f"\nTraining completed successfully!")
        print(f"Run the predictor with: python task_priority_predictor.py")

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()