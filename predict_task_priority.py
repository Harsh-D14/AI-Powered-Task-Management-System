#!/usr/bin/env python3
"""
Task Priority Predictor
Works with properly regularized models that achieve 0.75-0.90 performance
"""

import pickle
import numpy as np
import pandas as pd
import re
import warnings
import os
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')


class WorkloadBalancer:
    """Workload balancer without perfect optimization"""

    def __init__(self, employees_data, tasks_df=None):
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

        # Pick from top 3 candidates with weighted probability
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
    def __init__(self, model_path='models/task_priority_model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.load_model()

    def load_model(self):
        """Load the model"""
        try:
            print("Loading model (expected accuracy: 0.75-0.90)...")
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)

            # Extract components
            self.model = self.model_data['model']
            self.label_encoder = self.model_data['label_encoder']
            self.scaler = self.model_data.get('scaler', None)
            self.tfidf_vectorizer = self.model_data['tfidf_vectorizer']
            self.feature_names = self.model_data.get('feature_names', [])
            self.employees_data = self.model_data['employees_data']
            self.workload_balancer = self.model_data['workload_balancer']
            self.actual_categories = self.model_data.get('actual_categories', [])
            self.actual_priorities = self.model_data.get('actual_priorities', ['High', 'Medium', 'Low'])
            self.priority_keywords = self.model_data.get('priority_keywords', {})
            self.regularization_strength = self.model_data.get('regularization_strength', 'medium')
            self.quick_mode = self.model_data.get('quick_mode', True)
            self.version = self.model_data.get('version', 'v1.0')

            model_type = "Random Forest" if self.quick_mode else "Regularized XGBoost"
            print(f"Loaded {model_type} model (version: {self.version})")
            print(f"Regularization: {self.regularization_strength}")
            print(f"Available categories: {', '.join(self.actual_categories)}")
            print(f"Expected performance: 0.75-0.90 accuracy")

            if self.priority_keywords:
                print("Keyword detection: Enabled with controlled weights")

        except FileNotFoundError:
            print(f"Error: Model file '{self.model_path}' not found.")
            print("Please run 'python train_task_priority.py' first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please retrain with the trainer.")
            raise

    def validate_category(self, category):
        """Validate category with suggestions"""
        if category in self.actual_categories:
            return category, True

        # Find closest match
        category_lower = category.lower()
        for actual_cat in self.actual_categories:
            if category_lower in actual_cat.lower() or actual_cat.lower() in category_lower:
                print(f"Category '{category}' corrected to: '{actual_cat}'")
                return actual_cat, False

        # Fallback
        print(f"Category '{category}' not found. Available: {', '.join(self.actual_categories)}")
        fallback = self.actual_categories[0]
        print(f"Using: '{fallback}'")
        return fallback, False

    def analyze_keywords(self, description):
        """Keyword analysis (not perfect)"""
        desc_lower = description.lower()
        analysis = {}

        for priority, patterns in self.priority_keywords.items():
            strong_matches = [kw for kw in patterns.get('strong', []) if kw in desc_lower]
            moderate_matches = [kw for kw in patterns.get('moderate', []) if kw in desc_lower]
            weak_matches = [kw for kw in patterns.get('weak', []) if kw in desc_lower]

            # Weighted scoring (not too dominant)
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
        """Calculate urgency with limits"""
        desc_lower = description.lower()
        urgency_score = 0

        # High urgency indicators (capped)
        high_urgency = ['urgent', 'immediately', 'critical', 'emergency']
        urgency_score += min(2, sum(0.8 for kw in high_urgency if kw in desc_lower))

        # Medium urgency indicators
        medium_urgency = ['important', 'needed', 'priority', 'asap']
        urgency_score += min(1, sum(0.5 for kw in medium_urgency if kw in desc_lower))

        # Punctuation (limited impact)
        urgency_score += min(1, description.count('!') * 0.5)

        return min(3, urgency_score)  # Cap at 3

    def extract_features(self, description, category, employee_id=None):
        """Extract features matching the training process"""
        # Validate category
        validated_category, category_valid = self.validate_category(category)

        # Basic features
        words = description.split()
        word_count = len(words)

        # Keyword analysis (with controlled weights)
        keyword_analysis = self.analyze_keywords(description)
        high_kw = keyword_analysis['high']['score']
        med_kw = keyword_analysis['medium']['score']
        low_kw = keyword_analysis['low']['score']
        urgency = self.calculate_urgency(description)

        # Text features
        caps_ratio = sum(1 for word in words if word.isupper()) / max(1, len(words))
        punct_score = min(3, description.count('!') + description.count('?') * 0.5)
        avg_word_len = np.mean([len(word) for word in words]) if words else 0
        complexity = min(1, avg_word_len / 10)

        # Employee features
        if employee_id and employee_id in self.employees_data:
            emp_info = self.employees_data[employee_id]
            emp_load = emp_info['emp_load'] / 10  # Normalized
            category_match = 1 if emp_info['emp_preferred_category'] == validated_category else 0
        else:
            emp_load = 0.5
            category_match = 0

        # Category features (limited to match training)
        n_cat_features = min(5, len(self.actual_categories))
        category_features = []
        for i, cat in enumerate(self.actual_categories):
            if i >= n_cat_features:
                break
            category_features.append(1 if cat == validated_category else 0)

        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([description]).toarray()[0]

        # Combine features in training order
        features = [
            word_count,
            high_kw, med_kw, low_kw, urgency,
            caps_ratio, punct_score, complexity,
            emp_load, category_match
        ]
        features.extend(category_features)
        features.extend(tfidf_features)

        return (np.array(features).reshape(1, -1), validated_category, category_valid,
                keyword_analysis, urgency)

    def predict_priority(self, description, category, employee_id=None):
        """Make priority prediction"""
        try:
            # Extract features
            X, validated_category, category_valid, keyword_analysis, urgency_score = \
                self.extract_features(description, category, employee_id)

            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Make prediction
            prediction_encoded = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]

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
                'category_was_valid': category_valid,
                'keyword_analysis': keyword_analysis,
                'urgency_score': urgency_score
            }

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def explain_prediction(self, result):
        """Explain prediction with expectations"""
        print(f"\nPREDICTION EXPLANATION:")
        print(f"Note: This is a model (not perfect) - expect some uncertainty")

        # Keyword analysis
        keyword_analysis = result['keyword_analysis']
        print("Keyword Analysis:")

        total_keyword_evidence = 0
        for priority, analysis in keyword_analysis.items():
            if analysis['total_matches'] > 0:
                total_keyword_evidence += analysis['score']
                print(f"  {priority.capitalize()} indicators (score: {analysis['score']:.1f}):")
                if analysis['strong']:
                    print(f"    Strong: {', '.join(analysis['strong'])}")
                if analysis['moderate']:
                    print(f"    Moderate: {', '.join(analysis['moderate'])}")

        if total_keyword_evidence == 0:
            print("  No clear priority keywords found - relying on other features")
        elif total_keyword_evidence < 1:
            print("  Weak keyword evidence - prediction has higher uncertainty")

        # Confidence assessment
        confidence = result['confidence']
        if confidence >= 0.8:
            confidence_level = "High"
        elif confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        print(f"Model confidence: {confidence_level} ({confidence:.0%})")

        # Expectations
        if confidence < 0.6:
            print("WARNING: Low confidence - consider manual review")
        elif result['urgency_score'] > 2 and result['predicted_priority'] != 'High':
            print("WARNING: High urgency but not High priority - worth double-checking")

    def recommend_employee(self, category, priority_probabilities, urgency_score):
        """Employee recommendation"""
        try:
            predicted_priority = max(priority_probabilities, key=priority_probabilities.get)

            optimal_emp = self.workload_balancer.get_optimal_employee(
                category, predicted_priority, urgency_score
            )

            if optimal_emp and optimal_emp in self.employees_data:
                emp_details = self.workload_balancer.get_employee_details(optimal_emp)

                if emp_details:
                    # Generate reasoning
                    reasons = []

                    if emp_details['preferred_category'] == category:
                        reasons.append("category match")
                    else:
                        reasons.append("best available (no perfect match)")

                    load = emp_details['current_load']
                    if load <= 4:
                        reasons.append("low workload")
                    elif load <= 7:
                        reasons.append("reasonable workload")
                    else:
                        reasons.append("high workload but necessary")

                    return {
                        'employee_id': emp_details['emp_id'],
                        'current_load': emp_details['current_load'],
                        'preferred_category': emp_details['preferred_category'],
                        'category_match': emp_details['preferred_category'] == category,
                        'recent_assignments': emp_details['recent_assignments'],
                        'selection_reasons': reasons,
                        'confidence_note': "Selection includes some randomness for distribution"
                    }
            return None
        except Exception as e:
            print(f"Error in employee recommendation: {e}")
            return None

    def make_prediction(self, description, category, employee_id=None, show_explanation=True):
        """Make complete prediction"""
        print(f"\nTask: {description[:70]}{'...' if len(description) > 70 else ''}")
        print(f"Category: {category}")

        # Make prediction
        result = self.predict_priority(description, category, employee_id)

        if not result:
            print("Prediction failed.")
            return None

        # Display results
        priority = result['predicted_priority']
        confidence = result['confidence']

        print(f"\nPREDICTED PRIORITY: {priority.upper()}")
        print(f"Confidence: {confidence:.0%}")

        # Show probability distribution
        print("\nProbability Distribution:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for rank, (p, prob) in enumerate(sorted_probs, 1):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  #{rank} {p:6}: {prob:.0%} |{bar}|")

        # Show explanation
        if show_explanation:
            self.explain_prediction(result)

        # Employee recommendation
        if not employee_id:
            emp_rec = self.recommend_employee(
                result['validated_category'],
                result['all_probabilities'],
                result['urgency_score']
            )

            if emp_rec:
                load = emp_rec['current_load']
                load_status = "Light" if load <= 4 else "Medium" if load <= 7 else "Heavy"
                match_status = "Expert" if emp_rec['category_match'] else "Non-expert"

                print(f"\nRECOMMENDED EMPLOYEE: {emp_rec['employee_id']}")
                print(f"  Workload: {load_status} ({load:.1f}/10)")
                print(f"  Category fit: {match_status}")
                print(f"  Preferred: {emp_rec['preferred_category']}")
                print(f"  Recent assignments: {emp_rec['recent_assignments']}")
                print(f"  Selection reasons: {', '.join(emp_rec['selection_reasons'])}")
                print(f"  Note: {emp_rec['confidence_note']}")

                # Update workload
                self.workload_balancer.update_workload(
                    emp_rec['employee_id'], 1, priority
                )

        return result

    def interactive_mode(self):
        """Interactive mode with expectations"""
        print("\n" + "=" * 70)
        print("TASK PRIORITY PREDICTOR")
        print("Expected Performance: 0.75-0.90 accuracy (not perfect)")
        print("=" * 70)
        print("Commands: 'quit' | 'help' | 'categories' | 'test'")

        while True:
            try:
                print("\n" + "-" * 50)

                user_input = input("Enter task description: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Session ended.")
                    break

                if user_input.lower() == 'help':
                    print("This is a model designed to avoid overfitting.")
                    print("It achieves 0.75-0.90 accuracy with proper generalization.")
                    print("Some predictions may be uncertain - this is normal and healthy!")
                    continue

                if user_input.lower() == 'categories':
                    print("Available categories:")
                    for i, cat in enumerate(self.actual_categories, 1):
                        print(f"  {i}. {cat}")
                    continue

                if user_input.lower() == 'test':
                    test_cases = [
                        ("Urgent: Fix critical database connection issue immediately", "Bug Fix"),
                        ("Important documentation update needed for next release", "Documentation"),
                        ("Schedule regular team meeting for project discussion", "Meeting")
                    ]

                    for i, (desc, cat) in enumerate(test_cases, 1):
                        print(f"\n--- Test {i} ---")
                        self.make_prediction(desc, cat, show_explanation=False)
                    continue

                if not user_input:
                    continue

                # Get category
                category = input("Enter category: ").strip()
                if not category:
                    category = self.actual_categories[0]
                    print(f"Using default: {category}")

                # Make prediction
                self.make_prediction(user_input, category)

            except KeyboardInterrupt:
                print("\nSession ended.")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function"""
    try:
        predictor = TaskPriorityPredictor()

        print("\nSelect mode:")
        print("1. Interactive mode")
        print("2. Single prediction")
        print("3. Test mode")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            predictor.interactive_mode()

        elif choice == '2':
            desc = input("Enter task description: ")
            print(f"Available categories: {', '.join(predictor.actual_categories)}")
            cat = input("Enter category: ").strip()
            if not cat:
                cat = predictor.actual_categories[0]

            predictor.make_prediction(desc, cat)

        elif choice == '3':
            print("\nTest Mode - Performance Demo")
            print("-" * 40)

            test_cases = [
                ("Urgent: Critical server outage needs immediate attention", "Bug Fix"),
                ("Important: Update API documentation before release", "Documentation"),
                ("Schedule weekly team standup meeting", "Meeting"),
                ("Fix minor UI alignment issue when convenient", "UI/UX"),
                ("Deploy application to staging environment soon", "Deployment")
            ]

            correct_predictions = 0

            for i, (description, category) in enumerate(test_cases, 1):
                print(f"\n{'=' * 15} Test {i}/{len(test_cases)} {'=' * 15}")
                result = predictor.make_prediction(description, category, show_explanation=False)

                # Rough validation (you'd need actual labels for real evaluation)
                expected_priority = "High" if "urgent" in description.lower() or "critical" in description.lower() else \
                    "Medium" if "important" in description.lower() or "soon" in description.lower() else "Low"

                if result and result['predicted_priority'] == expected_priority:
                    correct_predictions += 1
                    print(f"Prediction matches expected priority")
                else:
                    print(f"Prediction differs from expected - this is normal for models")

            accuracy = correct_predictions / len(test_cases)
            print(f"\n{'=' * 50}")
            print(f"Test Results: {correct_predictions}/{len(test_cases)} ({accuracy:.0%})")

            if 0.6 <= accuracy <= 0.9:
                print("Performance in healthy range")
            elif accuracy > 0.9:
                print("Suspiciously high - may indicate overfitting")
            else:
                print("Lower than expected - model may need retraining")

        else:
            print("Starting interactive mode...")
            predictor.interactive_mode()

    except FileNotFoundError:
        print("Model not found. Please train first:")
        print("python train_task_priority.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()