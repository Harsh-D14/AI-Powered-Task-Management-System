#!/usr/bin/env python3
"""
Advanced Task Priority Trainer with Keyword-Based Intelligence
Optimized for keyword-enhanced datasets with sophisticated workload balancing
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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


class AdvancedWorkloadBalancer:
    """Advanced workload balancer with intelligent employee assignment"""

    def __init__(self, employees_data, tasks_df):
        self.employees_data = employees_data
        self.initial_loads = {emp_id: info['emp_load'] for emp_id, info in employees_data.items()}
        self.current_loads = self.initial_loads.copy()
        self.category_preferences = {emp_id: info['emp_preferred_category']
                                     for emp_id, info in employees_data.items()}
        self.assignment_history = {}
        self.category_experts = self._build_expert_mapping()
        self.performance_scores = self._calculate_performance_scores(tasks_df)

    def _build_expert_mapping(self):
        """Build mapping of categories to expert employees"""
        expert_mapping = defaultdict(list)
        for emp_id, pref in self.category_preferences.items():
            expert_mapping[pref].append(emp_id)
        return dict(expert_mapping)

    def _calculate_performance_scores(self, tasks_df):
        """Calculate employee performance scores based on historical assignments"""
        performance_scores = {}

        for emp_id in self.employees_data.keys():
            # Base performance score (lower initial load = better performer assumption)
            base_score = (10 - self.initial_loads[emp_id]) / 10

            # Count successful category matches in historical data
            emp_tasks = tasks_df[tasks_df['assigned_to_employeeid'] == emp_id]
            if len(emp_tasks) > 0:
                category_matches = sum(1 for _, task in emp_tasks.iterrows()
                                       if self.category_preferences[emp_id] == task['category'])
                match_ratio = category_matches / len(emp_tasks)
                performance_scores[emp_id] = base_score * 0.7 + match_ratio * 0.3
            else:
                performance_scores[emp_id] = base_score

        return performance_scores

    def get_optimal_employee(self, task_category, predicted_priority, urgency_score=0):
        """Advanced employee selection algorithm"""
        # Get all available employees (not completely overloaded)
        available_employees = [emp_id for emp_id, load in self.current_loads.items() if load < 10]

        if not available_employees:
            # Emergency fallback: reset least loaded employee
            least_loaded = min(self.current_loads, key=self.current_loads.get)
            self.current_loads[least_loaded] = 8  # Reset to manageable level
            return least_loaded

        # Score each available employee
        candidates = []

        for emp_id in available_employees:
            score = self._calculate_assignment_score(emp_id, task_category, predicted_priority, urgency_score)
            candidates.append({
                'emp_id': emp_id,
                'score': score,
                'current_load': self.current_loads[emp_id],
                'is_expert': self.category_preferences[emp_id] == task_category,
                'performance': self.performance_scores[emp_id]
            })

        # Sort by score (higher is better) and return best candidate
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidates[0]

        # Update assignment history
        self.assignment_history[best_candidate['emp_id']] = \
            self.assignment_history.get(best_candidate['emp_id'], 0) + 1

        return best_candidate['emp_id']

    def _calculate_assignment_score(self, emp_id, task_category, predicted_priority, urgency_score):
        """Calculate comprehensive assignment score for an employee"""
        score = 0
        current_load = self.current_loads[emp_id]
        is_expert = self.category_preferences[emp_id] == task_category

        # 1. Expertise bonus (most important factor)
        if is_expert:
            score += 50  # Strong expertise bonus
        else:
            score -= 10  # Penalty for non-expert

        # 2. Workload factor (inverse relationship)
        workload_score = (10 - current_load) * 3  # 0-30 points
        score += workload_score

        # 3. Performance history
        performance_bonus = self.performance_scores.get(emp_id, 0.5) * 20  # 0-20 points
        score += performance_bonus

        # 4. Priority-based adjustments
        if predicted_priority == 'High':
            if current_load <= 5:  # Prefer less loaded for high priority
                score += 15
            elif current_load >= 8:  # Avoid overloaded for high priority
                score -= 20

            # Extra bonus for expert + low load combination for high priority
            if is_expert and current_load <= 4:
                score += 25

        elif predicted_priority == 'Medium':
            if 3 <= current_load <= 7:  # Optimal range for medium priority
                score += 10

        # 5. Urgency adjustments
        if urgency_score > 2:  # High urgency task
            if current_load <= 4:
                score += 10
            elif current_load >= 8:
                score -= 15

        # 6. Load balancing - prevent consecutive assignments
        recent_assignments = self.assignment_history.get(emp_id, 0)
        if recent_assignments >= 3:
            score -= recent_assignments * 5  # Increasing penalty

        # 7. Avoid overloading
        if current_load >= 9:
            score -= 30  # Strong penalty for near-overload
        elif current_load >= 7:
            score -= 10  # Moderate penalty for high load

        return score

    def update_workload(self, emp_id, task_complexity=1, predicted_priority='Medium'):
        """Update employee workload with priority-based complexity"""
        if emp_id not in self.current_loads:
            return

        # Adjust complexity based on priority
        complexity_multiplier = {
            'High': 2.0,
            'Medium': 1.0,
            'Low': 0.7
        }

        final_complexity = task_complexity * complexity_multiplier.get(predicted_priority, 1.0)
        self.current_loads[emp_id] = min(10, self.current_loads[emp_id] + final_complexity)

        # Periodic load balancing - reset assignment history
        total_assignments = sum(self.assignment_history.values())
        if total_assignments > 0 and total_assignments % 20 == 0:
            self.assignment_history = {}

    def get_employee_details(self, emp_id):
        """Get comprehensive employee details"""
        if emp_id not in self.employees_data:
            return None

        return {
            'emp_id': emp_id,
            'current_load': self.current_loads.get(emp_id, 0),
            'initial_load': self.initial_loads.get(emp_id, 0),
            'preferred_category': self.category_preferences.get(emp_id, 'Unknown'),
            'performance_score': self.performance_scores.get(emp_id, 0.5),
            'recent_assignments': self.assignment_history.get(emp_id, 0),
            'load_increase': self.current_loads.get(emp_id, 0) - self.initial_loads.get(emp_id, 0)
        }

    def get_workload_summary(self):
        """Get overall workload distribution summary"""
        loads = list(self.current_loads.values())
        return {
            'total_employees': len(loads),
            'avg_load': np.mean(loads),
            'max_load': max(loads),
            'min_load': min(loads),
            'overloaded_count': sum(1 for load in loads if load >= 9),
            'underutilized_count': sum(1 for load in loads if load <= 3)
        }


class KeywordBasedTaskPriorityPredictor:
    def __init__(self, quick_mode=True):
        self.quick_mode = quick_mode
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
        self.urgency_patterns = None

    def extract_keyword_patterns(self, tasks_df):
        """Extract comprehensive keyword patterns from the enhanced dataset"""
        print("Analyzing keyword patterns in your enhanced dataset...")

        # Define comprehensive keyword patterns based on your data
        keyword_patterns = {
            'high': {
                'primary': ['urgent', 'immediately', 'emergency', 'critical', 'asap'],
                'secondary': ['immediate', 'now', 'today', 'right away', 'priority high'],
                'urgency_indicators': ['urgent', 'immediately', 'emergency', 'asap', 'critical']
            },
            'medium': {
                'primary': ['important', 'needed', 'priority', 'soon', 'required'],
                'secondary': ['due near', 'needed:', 'priority)', 'important:', 'due soon'],
                'urgency_indicators': ['needed', 'important', 'required', 'priority']
            },
            'low': {
                'primary': ['update', 'review', 'plan', 'organize', 'document'],
                'secondary': ['next quarter', 'when possible', 'eventually', 'future'],
                'urgency_indicators': []
            }
        }

        # Validate patterns against actual data
        print("Validating keyword patterns against your data:")
        for priority in ['High', 'Medium', 'Low']:
            priority_tasks = tasks_df[tasks_df['priority'] == priority]['task_description']
            priority_key = priority.lower()

            primary_matches = 0
            secondary_matches = 0

            for desc in priority_tasks:
                desc_lower = desc.lower()
                primary_matches += sum(1 for kw in keyword_patterns[priority_key]['primary']
                                       if kw in desc_lower)
                secondary_matches += sum(1 for kw in keyword_patterns[priority_key]['secondary']
                                         if kw in desc_lower)

            total_tasks = len(priority_tasks)
            print(
                f"  {priority}: {primary_matches} primary + {secondary_matches} secondary matches in {total_tasks} tasks")

        self.priority_keywords = keyword_patterns
        return keyword_patterns

    def load_data(self):
        """Load and analyze the enhanced datasets"""
        print("Loading enhanced datasets...")

        # Load main datasets
        self.tasks_df = pd.read_csv('datasets/tasks_dataset.csv')
        self.employees_df = pd.read_csv('datasets/employees_dataset.csv')

        # Load preprocessed data if available
        if os.path.exists('datasets/task_preprocessed_data.csv'):
            print("Found preprocessed data file, using enhanced features...")
            self.preprocessed_df = pd.read_csv('datasets/task_preprocessed_data.csv')
        else:
            print("Creating basic preprocessed features...")
            self.preprocessed_df = pd.DataFrame({
                'taskid': self.tasks_df['taskid'],
                'token_count': self.tasks_df['task_description'].str.split().str.len(),
                'original_word_count': self.tasks_df['task_description'].str.split().str.len()
            })

        print(f"Loaded {len(self.tasks_df)} tasks and {len(self.employees_df)} employees")

        # Extract data characteristics
        self.actual_categories = sorted(self.tasks_df['category'].unique())
        self.actual_priorities = sorted(self.tasks_df['priority'].unique())

        print(f"Categories: {self.actual_categories}")
        print(f"Priorities: {self.actual_priorities}")

        # Analyze priority distribution
        priority_dist = self.tasks_df['priority'].value_counts()
        print("Priority distribution:")
        for priority, count in priority_dist.items():
            print(f"  {priority}: {count} ({count / len(self.tasks_df) * 100:.1f}%)")

        # Extract keyword patterns
        self.extract_keyword_patterns(self.tasks_df)

        # Store employees data
        self.employees_data = self.employees_df.set_index('emp_id').to_dict('index')

        return self.tasks_df, self.employees_df, self.preprocessed_df

    def create_enhanced_features(self, tasks_df, preprocessed_df):
        """Create enhanced features leveraging keywords and preprocessing"""
        print("Creating enhanced features with keyword intelligence...")

        # Merge datasets
        df = tasks_df.merge(preprocessed_df[['taskid', 'token_count', 'original_word_count']],
                            on='taskid', how='left')

        # Handle missing values
        df['token_count'] = df['token_count'].fillna(df['task_description'].str.split().str.len())
        df['original_word_count'] = df['original_word_count'].fillna(df['token_count'])

        # 1. Keyword-based features (most important)
        keyword_features = self._extract_keyword_features(df['task_description'])

        # 2. Advanced text features
        text_features = self._extract_advanced_text_features(df['task_description'])

        # 3. TF-IDF features (reduced importance due to keyword focus)
        max_features = 15 if self.quick_mode else 25
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['task_description']).toarray()

        # 4. Category features
        category_encoded = pd.get_dummies(df['category'], prefix='cat').values

        # 5. Employee features
        employee_features = self._extract_employee_features(df)

        # Combine all features
        features = [
            df['token_count'].values,
            df['original_word_count'].values,
        ]

        # Add keyword features (most important)
        for i in range(keyword_features.shape[1]):
            features.append(keyword_features[:, i])

        # Add text features
        for i in range(text_features.shape[1]):
            features.append(text_features[:, i])

        # Add employee features
        for i in range(employee_features.shape[1]):
            features.append(employee_features[:, i])

        # Add category features
        for i in range(category_encoded.shape[1]):
            features.append(category_encoded[:, i])

        # Add TF-IDF features (less weight)
        for i in range(tfidf_features.shape[1]):
            features.append(tfidf_features[:, i])

        X = np.column_stack(features)

        # Create feature names
        self.feature_names = ['token_count', 'word_count']
        self.feature_names.extend(['high_keywords', 'medium_keywords', 'low_keywords',
                                   'urgency_score', 'keyword_confidence'])
        self.feature_names.extend(['caps_ratio', 'punct_score', 'avg_word_len', 'complexity_score'])
        self.feature_names.extend(['emp_load', 'cat_match', 'emp_performance'])
        self.feature_names.extend([f'cat_{cat}' for cat in self.actual_categories])
        self.feature_names.extend([f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

        # Encode target
        y = self.label_encoder.fit_transform(df['priority'])

        print(f"Created {X.shape[1]} enhanced features for {X.shape[0]} samples")
        print(f"Feature distribution: {len(self.feature_names)} total features")
        print(f"  - Keyword features: 5")
        print(f"  - Text features: 4")
        print(f"  - Employee features: 3")
        print(f"  - Category features: {len(self.actual_categories)}")
        print(f"  - TF-IDF features: {tfidf_features.shape[1]}")

        return X, y, df

    def _extract_keyword_features(self, descriptions):
        """Extract keyword-based features"""
        features = []

        for desc in descriptions:
            desc_lower = desc.lower()

            # Count keywords for each priority level
            high_score = sum(1 for kw in self.priority_keywords['high']['primary'] +
                             self.priority_keywords['high']['secondary'] if kw in desc_lower)

            medium_score = sum(1 for kw in self.priority_keywords['medium']['primary'] +
                               self.priority_keywords['medium']['secondary'] if kw in desc_lower)

            low_score = sum(1 for kw in self.priority_keywords['low']['primary'] +
                            self.priority_keywords['low']['secondary'] if kw in desc_lower)

            # Overall urgency score
            urgency_score = sum(1 for kw in self.priority_keywords['high']['urgency_indicators'] +
                                self.priority_keywords['medium']['urgency_indicators'] if kw in desc_lower)

            # Keyword confidence (how clear the priority signal is)
            total_keywords = high_score + medium_score + low_score
            if total_keywords > 0:
                confidence = max(high_score, medium_score, low_score) / total_keywords
            else:
                confidence = 0

            features.append([high_score, medium_score, low_score, urgency_score, confidence])

        return np.array(features)

    def _extract_advanced_text_features(self, descriptions):
        """Extract advanced text analysis features"""
        features = []

        for desc in descriptions:
            words = desc.split()

            # Capitalization ratio (urgency indicator)
            caps_ratio = sum(1 for word in words if word.isupper()) / len(words) if words else 0

            # Punctuation urgency score
            punct_score = desc.count('!') * 2 + desc.count('?') + desc.count('-') * 0.5

            # Average word length
            avg_word_len = np.mean([len(word) for word in words]) if words else 0

            # Text complexity score
            complex_words = sum(1 for word in words if len(word) > 7)
            complexity_score = complex_words / len(words) if words else 0

            features.append([caps_ratio, punct_score, avg_word_len, complexity_score])

        return np.array(features)

    def _extract_employee_features(self, df):
        """Extract employee-related features"""
        features = []

        for _, row in df.iterrows():
            emp_id = row['assigned_to_employeeid']

            if emp_id in self.employees_data:
                emp_info = self.employees_data[emp_id]
                emp_load = emp_info['emp_load']
                category_match = 1 if emp_info['emp_preferred_category'] == row['category'] else 0

                # Employee performance estimate (lower initial load = better performer)
                emp_performance = (10 - emp_load) / 10
            else:
                emp_load = 5
                category_match = 0
                emp_performance = 0.5

            features.append([emp_load, category_match, emp_performance])

        return np.array(features)

    def train_enhanced_model(self, X, y):
        """Train model optimized for keyword-enhanced data"""
        print(f"Training enhanced model ({'Quick' if self.quick_mode else 'Thorough'} mode)...")

        start_time = time.time()

        # Data split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Handle class imbalance (though it should be better now)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        print(f"Class weights: {class_weight_dict}")

        # Model selection with keyword-optimized parameters
        if self.quick_mode:
            print("Using Random Forest optimized for keyword features...")
            param_grid = {
                'n_estimators': [150, 250],  # More trees for keyword stability
                'max_depth': [10, 15],  # Deeper trees for keyword combinations
                'min_samples_split': [2, 3],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']  # Feature selection
            }

            model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight_dict
            )
        else:
            print("Using XGBoost optimized for keyword features...")
            param_grid = {
                'n_estimators': [200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }

            model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False,
                n_jobs=-1
            )

        # Grid search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # More folds for stability

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        print("Training with keyword-optimized parameters...")
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_

        # Comprehensive evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        test_accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.1f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test accuracy: {test_accuracy:.3f}")

        # Detailed evaluation
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("Actual\\Predicted:", "  ".join(f"{cls:>8}" for cls in self.label_encoder.classes_))
        for i, row in enumerate(cm):
            print(f"{self.label_encoder.classes_[i]:>8}", "  ".join(f"{val:>8}" for val in row))

        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 15 Most Important Features:")
            print(feature_importance.head(15).to_string(index=False))

            # Analyze keyword feature importance
            keyword_features = feature_importance[
                feature_importance['feature'].str.contains('keyword|urgency|confidence')
            ]
            if not keyword_features.empty:
                print(f"\nKeyword Features Performance:")
                print(keyword_features.to_string(index=False))

        return self.model

    def create_advanced_workload_balancer(self, df):
        """Create the advanced workload balancer"""
        print("Creating advanced workload balancer...")
        self.workload_balancer = AdvancedWorkloadBalancer(self.employees_data, df)

        # Show initial workload summary
        summary = self.workload_balancer.get_workload_summary()
        print(f"Workload balancer initialized:")
        print(f"  Average load: {summary['avg_load']:.1f}")
        print(f"  Load range: {summary['min_load']}-{summary['max_load']}")
        print(f"  Experts available for all {len(self.actual_categories)} categories")

        return self.workload_balancer

    def save_model(self, filename='models/task_priority_model.pkl'):
        """Save the enhanced model"""
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
            'quick_mode': self.quick_mode,
            'version': 'keyword_enhanced_v1.0'
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        file_size = os.path.getsize(filename) / 1024 / 1024
        print(f"Enhanced model saved as {filename} ({file_size:.1f} MB)")

    def train_complete_pipeline(self):
        """Run the complete advanced training pipeline"""
        print("=" * 70)
        print("ADVANCED KEYWORD-BASED TASK PRIORITY TRAINER")
        print("Optimized for keyword-enhanced datasets")
        print("=" * 70)

        total_start = time.time()

        try:
            # Load enhanced data
            tasks_df, employees_df, preprocessed_df = self.load_data()

            # Create enhanced features
            X, y, df = self.create_enhanced_features(tasks_df, preprocessed_df)

            # Train enhanced model
            model = self.train_enhanced_model(X, y)

            # Create advanced workload balancer
            workload_balancer = self.create_advanced_workload_balancer(df)

            # Save enhanced model
            self.save_model()

            total_time = time.time() - total_start

            print(f"\n{'=' * 70}")
            print(f"ADVANCED TRAINING COMPLETED IN {total_time:.1f} SECONDS!")
            print("Enhanced features implemented:")
            print("✓ Keyword-based priority detection")
            print("✓ Advanced workload balancing with expertise matching")
            print("✓ Performance-based employee scoring")
            print("✓ Urgency pattern recognition")
            print("✓ Intelligent category-expert matching")
            print("✓ Load balancing with priority consideration")
            print("=" * 70)

        except Exception as e:
            print(f"Error during advanced training: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main training function"""
    print("Advanced Task Priority Model Trainer")
    print("Optimized for keyword-enhanced datasets")
    print("\nChoose training mode:")
    print("1. Quick mode (Random Forest, ~60 seconds)")
    print("2. Thorough mode (XGBoost, ~4-5 minutes)")

    choice = input("Enter choice (1 or 2): ").strip()

    quick_mode = choice != '2'
    mode_name = "quick" if quick_mode else "thorough"
    print(f"Using {mode_name} mode with keyword optimization...")

    try:
        # Train advanced model
        predictor = KeywordBasedTaskPriorityPredictor(quick_mode=quick_mode)
        predictor.train_complete_pipeline()

        print(f"\nTraining complete! Run the predictor:")
        print(f"python task_priority_predictor.py")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your CSV files are in the current directory.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()