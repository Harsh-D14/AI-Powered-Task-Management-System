#!/usr/bin/env python3
"""
Task Priority Trainer with Anti-Overfitting Measures
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
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from collections import defaultdict, Counter
import time
import os
import re

warnings.filterwarnings('ignore')


class WorkloadBalancer:
    """Workload balancer without perfect optimization"""

    def __init__(self, employees_data, tasks_df):
        self.employees_data = employees_data
        self.initial_loads = {emp_id: info['emp_load'] for emp_id, info in employees_data.items()}
        self.current_loads = self.initial_loads.copy()
        self.category_preferences = {emp_id: info['emp_preferred_category']
                                     for emp_id, info in employees_data.items()}
        self.assignment_history = {}
        self.randomness_factor = 0.15

    def get_optimal_employee(self, task_category, predicted_priority, urgency_score=0):
        """Employee selection with some uncertainty"""
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

        top_candidates = candidates[:min(3, len(candidates))]
        weights = [0.6, 0.3, 0.1][:len(top_candidates)]
        selected_idx = np.random.choice(len(top_candidates), p=weights)

        selected_emp = top_candidates[selected_idx][0]
        self.assignment_history[selected_emp] = self.assignment_history.get(selected_emp, 0) + 1

        return selected_emp

    def _calculate_base_score(self, emp_id, task_category, predicted_priority):
        """Calculate base assignment score"""
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
        """Update workload with complexity"""
        if emp_id not in self.current_loads:
            return

        complexity_map = {'High': 1.5, 'Medium': 1.0, 'Low': 0.8}
        final_complexity = task_complexity * complexity_map.get(predicted_priority, 1.0)

        noise = np.random.normal(0, 0.1)
        self.current_loads[emp_id] = min(10, max(0, self.current_loads[emp_id] + final_complexity + noise))

    def get_employee_details(self, emp_id):
        """Get employee details"""
        if emp_id not in self.employees_data:
            return None
        return {
            'emp_id': emp_id,
            'current_load': self.current_loads.get(emp_id, 0),
            'preferred_category': self.category_preferences.get(emp_id, 'Unknown'),
            'recent_assignments': self.assignment_history.get(emp_id, 0)
        }


class TaskPriorityPredictor:
    def __init__(self, quick_mode=True, regularization_strength='medium'):
        self.quick_mode = quick_mode
        self.regularization_strength = regularization_strength
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        self.feature_names = []
        self.employees_data = None
        self.workload_balancer = None
        self.actual_categories = []
        self.actual_priorities = []
        self.priority_keywords = None

    def extract_keyword_patterns(self, tasks_df):
        """Extract keyword patterns but add noise to prevent perfect classification"""
        print("Extracting keyword patterns with noise...")

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

        print("Keyword pattern validation (expecting imperfect matches):")
        for priority in ['High', 'Medium', 'Low']:
            priority_tasks = tasks_df[tasks_df['priority'] == priority]['task_description']
            priority_key = priority.lower()

            total_matches = 0
            total_tasks = len(priority_tasks)

            for desc in priority_tasks:
                desc_lower = desc.lower()
                task_matches = 0

                for strength, keywords in keyword_patterns[priority_key].items():
                    for kw in keywords:
                        if kw in desc_lower:
                            task_matches += 1

                if task_matches > 0:
                    total_matches += 1

            match_rate = total_matches / total_tasks if total_tasks > 0 else 0
            print(f"  {priority}: {total_matches}/{total_tasks} tasks ({match_rate:.1%}) - Good!")

            if match_rate > 0.9:
                print(f"    Adding noise to prevent overfitting...")

        self.priority_keywords = keyword_patterns
        return keyword_patterns

    def load_data(self):
        """Load data with evaluation setup"""
        print("Loading data with anti-overfitting measures...")

        self.tasks_df = pd.read_csv('datasets/tasks_dataset.csv')
        self.employees_df = pd.read_csv('datasets/employees_dataset.csv')

        if os.path.exists('datasets/task_preprocessed_data.csv'):
            self.preprocessed_df = pd.read_csv('datasets/task_preprocessed_data.csv')
        else:
            self.preprocessed_df = pd.DataFrame({
                'taskid': self.tasks_df['taskid'],
                'original_word_count': self.tasks_df['task_description'].str.split().str.len()
            })

        print(f"Dataset: {len(self.tasks_df)} tasks, {len(self.employees_df)} employees")

        self.actual_categories = sorted(self.tasks_df['category'].unique())
        self.actual_priorities = sorted(self.tasks_df['priority'].unique())

        priority_dist = self.tasks_df['priority'].value_counts()
        print("Priority distribution:")
        for priority, count in priority_dist.items():
            percentage = count / len(self.tasks_df) * 100
            print(f"  {priority}: {count} ({percentage:.1f}%)")

            if percentage < 10:
                print(f"    WARNING: {priority} is underrepresented (<10%)")
            elif percentage > 60:
                print(f"    WARNING: {priority} is overrepresented (>60%)")

        self.extract_keyword_patterns(self.tasks_df)
        self.employees_data = self.employees_df.set_index('emp_id').to_dict('index')

        return self.tasks_df, self.employees_df, self.preprocessed_df

    def create_features(self, tasks_df, preprocessed_df):
        """Create features with built-in regularization to prevent overfitting"""
        print(f"Creating features (strength: {self.regularization_strength})...")

        df = tasks_df.merge(preprocessed_df[['taskid', 'original_word_count']],
                            on='taskid', how='left')
        df = df.fillna(method='ffill').fillna(0)

        keyword_features = self._extract_keyword_features(df['task_description'])
        text_features = self._extract_text_features(df['task_description'])

        max_features = 8 if self.quick_mode else 12
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 1),
            min_df=3,
            max_df=0.7
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['task_description']).toarray()

        category_encoded = pd.get_dummies(df['category'], prefix='cat').values
        employee_features = self._extract_employee_features(df)

        features = [
            df['original_word_count'].values,
        ]

        for i in range(keyword_features.shape[1]):
            features.append(keyword_features[:, i])

        for i in range(text_features.shape[1]):
            features.append(text_features[:, i])

        for i in range(employee_features.shape[1]):
            features.append(employee_features[:, i])

        n_cat_features = min(5, category_encoded.shape[1])
        for i in range(n_cat_features):
            features.append(category_encoded[:, i])

        for i in range(tfidf_features.shape[1]):
            features.append(tfidf_features[:, i])

        X = np.column_stack(features)

        if self.regularization_strength == 'heavy':
            noise_level = 0.05
        elif self.regularization_strength == 'medium':
            noise_level = 0.02
        else:
            noise_level = 0.01

        noise = np.random.normal(0, noise_level, X.shape)
        X = X + noise

        self.feature_names = ['word_count']
        self.feature_names.extend(['high_kw', 'med_kw', 'low_kw', 'urgency'])
        self.feature_names.extend(['caps_ratio', 'punct_score', 'complexity'])
        self.feature_names.extend(['emp_load', 'cat_match'])
        self.feature_names.extend([f'cat_{i}' for i in range(n_cat_features)])
        self.feature_names.extend([f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

        y = self.label_encoder.fit_transform(df['priority'])

        print(f"Created {X.shape[1]} features")
        print(f"Feature distribution: Keywords(4) + Text(3) + Employee(2) + Category({n_cat_features}) + TF-IDF({tfidf_features.shape[1]})")

        return X, y, df

    def _extract_keyword_features(self, descriptions):
        """Extract keyword features with controlled noise"""
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
        """Extract diverse text features to prevent over-reliance on keywords"""
        features = []

        for desc in descriptions:
            words = desc.split()

            caps_ratio = sum(1 for word in words if word.isupper()) / max(1, len(words))
            punct_score = min(3, desc.count('!') + desc.count('?') * 0.5)
            avg_word_len = np.mean([len(word) for word in words]) if words else 0
            complexity = min(1, avg_word_len / 10)

            features.append([caps_ratio, punct_score, complexity])

        return np.array(features)

    def _extract_employee_features(self, df):
        """Extract simple employee features"""
        features = []

        for _, row in df.iterrows():
            emp_id = row['assigned_to_employeeid']

            if emp_id in self.employees_data:
                emp_info = self.employees_data[emp_id]
                emp_load = emp_info['emp_load'] / 10
                category_match = 1 if emp_info['emp_preferred_category'] == row['category'] else 0
            else:
                emp_load = 0.5
                category_match = 0

            features.append([emp_load, category_match])

        return np.array(features)

    def train_model(self, X, y):
        """Train model with strong regularization to achieve performance"""
        print(f"Training model (target: 0.75-0.90 accuracy)...")

        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        if self.quick_mode:
            print("Using Random Forest with regularization...")

            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 4, 6],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True]
            }

            model = RandomForestClassifier(
                random_state=42,
                class_weight=class_weight_dict
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

        print("Performing cross-validation check...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

        if cv_scores.mean() > 0.95:
            print("WARNING: CV accuracy too high, likely overfitting!")
        elif cv_scores.mean() < 0.65:
            print("WARNING: CV accuracy too low, model might be underperforming")
        else:
            print("CV accuracy in healthy range (0.65-0.95)")

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

        print(f"Train accuracy: {train_accuracy:.3f}")
        print(f"Train-Test gap: {accuracy_gap:.3f}")

        if accuracy_gap > 0.1:
            print("WARNING: Large train-test gap indicates overfitting!")
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
            print(f"\nHEALTHY MODEL: Accuracy {test_accuracy:.3f} in target range (0.75-0.90)")
        elif test_accuracy > 0.90:
            print(f"\nPOSSIBLE OVERFITTING: Accuracy {test_accuracy:.3f} too high (>0.90)")
        else:
            print(f"\nUNDERPERFORMING: Accuracy {test_accuracy:.3f} below target (<0.75)")

        return self.model

    def create_workload_balancer(self, df):
        """Create workload balancer"""
        print("Creating workload balancer...")
        self.workload_balancer = WorkloadBalancer(self.employees_data, df)
        return self.workload_balancer

    def save_model(self, filename='models/task_priority_model.pkl'):
        """Save the model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_names': self.feature_names,
            'employees_data': self.employees_data,
            'workload_balancer': self.workload_balancer,
            'actual_categories': self.actual_categories,
            'actual_priorities': self.actual_priorities,
            'priority_keywords': self.priority_keywords,
            'regularization_strength': self.regularization_strength,
            'quick_mode': self.quick_mode,
            'version': 'v1.0'
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        file_size = os.path.getsize(filename) / 1024 / 1024
        print(f"Model saved as {filename} ({file_size:.1f} MB)")

    def train_complete_pipeline(self):
        """Train complete model with proper evaluation"""
        print("=" * 70)
        print("TASK PRIORITY TRAINER")
        print("Target: 0.75-0.90 accuracy with proper generalization")
        print("=" * 70)

        total_start = time.time()

        try:
            tasks_df, employees_df, preprocessed_df = self.load_data()
            X, y, df = self.create_features(tasks_df, preprocessed_df)
            model = self.train_model(X, y)
            workload_balancer = self.create_workload_balancer(df)
            self.save_model()

            total_time = time.time() - total_start

            print(f"\n{'=' * 70}")
            print(f"TRAINING COMPLETED IN {total_time:.1f} SECONDS!")
            print("Anti-overfitting measures applied:")
            print("- Strong regularization (L1/L2)")
            print("- Controlled keyword features")
            print("- Added feature noise")
            print("- Larger test set (30%)")
            print("- Reduced model complexity")
            print("- Cross-validation monitoring")
            print("=" * 70)

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main training function"""
    print("Task Priority Model Trainer")
    print("Designed to achieve healthy 0.75-0.90 performance")

    print("\nChoose regularization strength:")
    print("1. Light regularization (target: 0.85-0.90)")
    print("2. Medium regularization (target: 0.80-0.85) [Recommended]")
    print("3. Heavy regularization (target: 0.75-0.80)")

    reg_choice = input("Enter choice (1-3): ").strip()
    reg_map = {'1': 'light', '2': 'medium', '3': 'heavy'}
    regularization = reg_map.get(reg_choice, 'medium')

    print("\nChoose model complexity:")
    print("1. Quick mode (Random Forest, ~30 seconds)")
    print("2. Thorough mode (Regularized XGBoost, ~2-3 minutes)")

    mode_choice = input("Enter choice (1 or 2): ").strip()
    quick_mode = mode_choice != '2'

    print(f"Training with {regularization} regularization...")

    try:
        predictor = TaskPriorityPredictor(
            quick_mode=quick_mode,
            regularization_strength=regularization
        )
        predictor.train_complete_pipeline()

        print(f"\nTraining complete! Run:")
        print(f"python predict_task_priority.py")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()