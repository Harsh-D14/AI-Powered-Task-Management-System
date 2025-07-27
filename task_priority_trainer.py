#!/usr/bin/env python3
"""
Efficient Task Priority Prediction Model
Optimized version with reduced complexity for faster training
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from collections import defaultdict
import time
import os

warnings.filterwarnings('ignore')


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


class EfficientTaskPriorityPredictor:
    def __init__(self, quick_mode=True):
        self.quick_mode = quick_mode
        self.model = None
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20 if quick_mode else 50,  # Reduced features
            stop_words='english',
            ngram_range=(1, 1)  # Only unigrams for speed
        )
        self.feature_names = []
        self.employees_data = None
        self.workload_balancer = None

    def load_data(self):
        """Load and preprocess datasets efficiently"""
        print("Loading datasets...")

        # Load only necessary columns to save memory
        self.tasks_df = pd.read_csv('tasks_dataset.csv')
        self.employees_df = pd.read_csv('employees_dataset.csv')

        # Check if preprocessed data exists, if not create basic features
        if os.path.exists('task_preprocessed_data.csv'):
            preprocessed_df = pd.read_csv('task_preprocessed_data.csv')[
                ['taskid', 'token_count', 'original_word_count']
            ]
        else:
            # Create basic text features if preprocessed data doesn't exist
            print("Creating basic text features...")
            preprocessed_df = pd.DataFrame({
                'taskid': self.tasks_df['taskid'],
                'token_count': self.tasks_df['task_description'].str.split().str.len(),
                'original_word_count': self.tasks_df['task_description'].str.split().str.len()
            })

        self.preprocessed_df = preprocessed_df

        print(f"Loaded {len(self.tasks_df)} tasks and {len(self.employees_df)} employees")

        # Store employees data for workload balancing
        self.employees_data = self.employees_df.set_index('emp_id').to_dict('index')

        return self.tasks_df, self.employees_df, self.preprocessed_df

    def create_features_efficient(self, tasks_df, preprocessed_df):
        """Create streamlined feature set for better performance"""
        print("Creating efficient features...")

        # Merge datasets
        df = tasks_df.merge(preprocessed_df, on='taskid', how='left')

        # Fill missing values
        df['token_count'] = df['token_count'].fillna(df['task_description'].str.split().str.len())
        df['original_word_count'] = df['original_word_count'].fillna(df['token_count'])

        # 1. Text features using TF-IDF (reduced)
        print("Processing text features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['task_description']).toarray()

        # 2. Simple categorical features
        categories = sorted(df['category'].unique())
        category_encoded = pd.get_dummies(df['category'], prefix='cat').values

        # 3. Employee features (simplified)
        employee_loads = []
        category_matches = []

        for _, row in df.iterrows():
            emp_id = row['assigned_to_employeeid']
            if emp_id in self.employees_data:
                emp_info = self.employees_data[emp_id]
                employee_loads.append(emp_info['emp_load'])
                category_matches.append(1 if emp_info['emp_preferred_category'] == row['category'] else 0)
            else:
                employee_loads.append(5)  # Default
                category_matches.append(0)

        # Combine features efficiently
        features = [
            df['token_count'].values,
            df['original_word_count'].values,
            np.array(employee_loads),
            np.array(category_matches)
        ]

        # Add category features
        for i in range(category_encoded.shape[1]):
            features.append(category_encoded[:, i])

        # Add TF-IDF features
        for i in range(tfidf_features.shape[1]):
            features.append(tfidf_features[:, i])

        X = np.column_stack(features)

        # Create feature names
        self.feature_names = ['token_count', 'word_count', 'emp_load', 'cat_match']
        self.feature_names.extend([f'cat_{i}' for i in range(category_encoded.shape[1])])
        self.feature_names.extend([f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

        # Encode target
        y = self.label_encoder.fit_transform(df['priority'])

        print(f"Created {X.shape[1]} features for {X.shape[0]} samples")
        return X, y, df

    def train_model_efficient(self, X, y):
        """Train model with efficient hyperparameter search"""
        print(f"Training model ({'Quick' if self.quick_mode else 'Thorough'} mode)...")

        start_time = time.time()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if self.quick_mode:
            # Quick training with Random Forest (faster than XGBoost)
            print("Using Random Forest for quick training...")

            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }

            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            cv_folds = 3

        else:
            # More thorough training with XGBoost
            print("Using XGBoost for thorough training...")

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            }

            model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False,
                n_jobs=-1
            )
            cv_folds = 3  # Reduced from 5

        # Efficient Grid Search
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0  # Reduced verbosity
        )

        print("Running hyperparameter search...")
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        # Evaluate
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.1f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test accuracy: {test_accuracy:.3f}")

        # Quick classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        return self.model

    def create_simple_workload_balancer(self, df):
        """Create simplified workload balancer"""
        print("Creating workload balancer...")

        self.workload_balancer = SimpleWorkloadBalancer(self.employees_data)
        return self.workload_balancer

    def save_model(self, filename='task_priority_model.pkl'):
        """Save model efficiently"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_names': self.feature_names,
            'employees_data': self.employees_data,
            'workload_balancer': self.workload_balancer,
            'categories': sorted(self.tasks_df['category'].unique()),
            'quick_mode': self.quick_mode
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        file_size = os.path.getsize(filename) / 1024 / 1024
        print(f"Model saved as {filename} ({file_size:.1f} MB)")

    def train_complete_pipeline(self):
        """Run efficient training pipeline"""
        print("=" * 50)
        print("EFFICIENT TASK PRIORITY MODEL TRAINING")
        print("=" * 50)

        total_start = time.time()

        # Load data
        tasks_df, employees_df, preprocessed_df = self.load_data()

        # Create features
        X, y, df = self.create_features_efficient(tasks_df, preprocessed_df)

        # Train model
        model = self.train_model_efficient(X, y)

        # Create workload balancer
        workload_balancer = self.create_simple_workload_balancer(df)

        # Save model
        self.save_model()

        total_time = time.time() - total_start

        print(f"\n{'=' * 50}")
        print(f"TRAINING COMPLETED IN {total_time:.1f} SECONDS!")
        print("You can now use the predictor program.")
        print("=" * 50)


def main():
    """Main function with mode selection"""
    print("Choose training mode:")
    print("1. Quick mode (Fast, Random Forest, ~30 seconds)")
    print("2. Thorough mode (Slower, XGBoost, ~2-3 minutes)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '2':
        quick_mode = False
        print("Using thorough mode...")
    else:
        quick_mode = True
        print("Using quick mode...")

    # Check for required files
    required_files = ['tasks_dataset.csv', 'employees_dataset.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Error: Missing files: {missing_files}")
        return

    # Train model
    predictor = EfficientTaskPriorityPredictor(quick_mode=quick_mode)
    predictor.train_complete_pipeline()


if __name__ == "__main__":
    main()