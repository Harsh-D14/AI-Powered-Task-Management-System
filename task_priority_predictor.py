#!/usr/bin/env python3
"""
Efficient Task Priority Predictor
Optimized version with faster processing and simpler interface
"""

import pickle
import numpy as np
import pandas as pd
import re
import warnings
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
    def __init__(self, model_path='task_priority_model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.load_model()

    def load_model(self):
        """Load the saved model efficiently"""
        try:
            print("Loading model...")
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)

            # Extract components
            self.model = self.model_data['model']
            self.label_encoder = self.model_data['label_encoder']
            self.tfidf_vectorizer = self.model_data['tfidf_vectorizer']
            self.feature_names = self.model_data['feature_names']
            self.employees_data = self.model_data['employees_data']
            self.workload_balancer = self.model_data['workload_balancer']
            self.categories = self.model_data['categories']
            self.quick_mode = self.model_data.get('quick_mode', True)

            model_type = "Random Forest" if self.quick_mode else "XGBoost"
            print(f"✅ {model_type} model loaded successfully!")
            print(f"📂 Available categories: {', '.join(self.categories)}")

        except FileNotFoundError:
            print(f"❌ Error: Model file '{self.model_path}' not found.")
            print("Please run the trainer program first.")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def preprocess_text_simple(self, text):
        """Simple and fast text preprocessing"""
        # Basic cleaning
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return ' '.join(text.split())

    def calculate_text_features_fast(self, description):
        """Fast text feature calculation"""
        words = description.split()
        token_count = len(words)
        return token_count, token_count  # Use same for both counts

    def create_prediction_features(self, description, category, employee_id=None):
        """Create feature vector efficiently"""
        # Text features
        token_count, word_count = self.calculate_text_features_fast(description)

        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([description]).toarray()[0]

        # Employee features
        if employee_id and employee_id in self.employees_data:
            emp_info = self.employees_data[employee_id]
            emp_load = emp_info['emp_load']
            category_match = 1 if emp_info['emp_preferred_category'] == category else 0
        else:
            emp_load = 5  # Default
            category_match = 0

        # Category encoding
        category_features = []
        for cat in sorted(self.categories):
            category_features.append(1 if cat == category else 0)

        # Combine features in the right order
        features = [token_count, word_count, emp_load, category_match]
        features.extend(category_features)
        features.extend(tfidf_features)

        return np.array(features).reshape(1, -1)

    def predict_priority_fast(self, description, category, employee_id=None):
        """Fast priority prediction"""
        try:
            # Validate category
            if category not in self.categories:
                print(f"⚠️  '{category}' not recognized. Using {self.categories[0]}")
                category = self.categories[0]

            # Create features
            X = self.create_prediction_features(description, category, employee_id)

            # Make prediction
            prediction_encoded = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]

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
                'all_probabilities': priority_probabilities
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def recommend_employee_fast(self, category, priority_probabilities):
        """Fast employee recommendation"""
        try:
            optimal_emp = self.workload_balancer.get_optimal_employee(category, priority_probabilities)

            if optimal_emp and optimal_emp in self.employees_data:
                emp_info = self.employees_data[optimal_emp]
                current_load = self.workload_balancer.current_loads[optimal_emp]
                preferred_category = emp_info['emp_preferred_category']

                return {
                    'employee_id': optimal_emp,
                    'current_load': current_load,
                    'preferred_category': preferred_category,
                    'category_match': preferred_category == category
                }
            return None
        except Exception as e:
            print(f"❌ Employee recommendation error: {e}")
            return None

    def single_prediction(self, description, category, employee_id=None):
        """Make a single prediction with results display"""
        print(f"\n📝 Task: {description[:60]}{'...' if len(description) > 60 else ''}")
        print(f"📂 Category: {category}")

        # Predict priority
        result = self.predict_priority_fast(description, category, employee_id)

        if not result:
            return None

        # Display prediction
        priority = result['predicted_priority']
        confidence = result['confidence']

        # Priority emoji
        priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        emoji = priority_emoji.get(priority, '⚪')

        print(f"\n🎯 PRIORITY: {emoji} {priority} ({confidence:.0%} confidence)")

        # Show probability bars
        print("📊 Probabilities:")
        for p in ['High', 'Medium', 'Low']:
            if p in result['all_probabilities']:
                prob = result['all_probabilities'][p]
                bar_length = int(prob * 15)
                bar = "█" * bar_length + "░" * (15 - bar_length)
                print(f"   {p:6}: {prob:.0%} |{bar}|")

        # Recommend employee if not specified
        if not employee_id:
            emp_rec = self.recommend_employee_fast(category, result['all_probabilities'])
            if emp_rec:
                match_status = "✅" if emp_rec['category_match'] else "❌"
                print(f"\n👤 RECOMMENDED: {emp_rec['employee_id']}")
                print(f"   Load: {emp_rec['current_load']}/10 | Match: {match_status}")

                # Update workload
                task_weight = {'High': 2, 'Medium': 1, 'Low': 1}
                self.workload_balancer.update_workload(
                    emp_rec['employee_id'],
                    task_weight.get(priority, 1)
                )

        return result

    def quick_interactive(self):
        """Streamlined interactive mode"""
        print("\n" + "=" * 50)
        print("🚀 QUICK TASK PRIORITY PREDICTOR")
        print("=" * 50)
        print("💡 Tip: Type 'q' to quit, 'help' for commands")

        while True:
            try:
                print("\n" + "-" * 30)

                # Get input
                user_input = input("📝 Task description: ").strip()

                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == 'help':
                    print(f"📂 Categories: {', '.join(self.categories)}")
                    print("💡 Just enter task description, then category")
                    continue

                if not user_input:
                    continue

                # Get category
                category = input(f"📂 Category: ").strip()
                if not category:
                    category = self.categories[0]

                # Make prediction
                self.single_prediction(user_input, category)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def demo_mode(self):
        """Quick demo with sample tasks"""
        print("\n🎮 DEMO MODE - Sample Predictions")
        print("=" * 40)

        demo_tasks = [
            ("Fix login bug causing crashes", "Bug Fix"),
            ("Write user documentation", "Documentation"),
            ("Design new dashboard layout", "UI/UX"),
            ("Deploy to production server", "Deployment"),
            ("Review code changes", "Code Review")
        ]

        for i, (desc, cat) in enumerate(demo_tasks, 1):
            print(f"\n[{i}/5]")
            self.single_prediction(desc, cat)

            if i < len(demo_tasks):
                input("Press Enter for next task...")

        print(f"\n🏁 Demo completed!")


def main():
    """Streamlined main function"""
    try:
        predictor = EfficientTaskPriorityPredictor()

        print("\n🎯 Choose mode:")
        print("1. Interactive mode")
        print("2. Single prediction")
        print("3. Demo mode")

        choice = input("\nMode (1-3): ").strip()

        if choice == '1':
            predictor.quick_interactive()

        elif choice == '2':
            desc = input("📝 Task description: ")
            cat = input(f"📂 Category ({'/'.join(predictor.categories[:3])}...): ")
            if not cat:
                cat = predictor.categories[0]
            predictor.single_prediction(desc, cat)

        elif choice == '3':
            predictor.demo_mode()

        else:
            print("Starting interactive mode...")
            predictor.quick_interactive()

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()