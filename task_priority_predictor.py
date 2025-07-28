#!/usr/bin/env python3
"""
Advanced Task Priority Predictor
Works with keyword-enhanced models and sophisticated workload balancing
"""

import pickle
import numpy as np
import pandas as pd
import re
import warnings
import os
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')


# Import the workload balancer class for pickle compatibility
class AdvancedWorkloadBalancer:
    """Advanced workload balancer with intelligent employee assignment"""

    def __init__(self, employees_data, tasks_df=None):
        self.employees_data = employees_data
        self.initial_loads = {emp_id: info['emp_load'] for emp_id, info in employees_data.items()}
        self.current_loads = self.initial_loads.copy()
        self.category_preferences = {emp_id: info['emp_preferred_category']
                                     for emp_id, info in employees_data.items()}
        self.assignment_history = {}
        self.category_experts = self._build_expert_mapping()
        # Initialize performance scores with basic logic if tasks_df not available
        self.performance_scores = self._calculate_performance_scores(
            tasks_df) if tasks_df is not None else self._default_performance_scores()

    def _build_expert_mapping(self):
        """Build mapping of categories to expert employees"""
        expert_mapping = defaultdict(list)
        for emp_id, pref in self.category_preferences.items():
            expert_mapping[pref].append(emp_id)
        return dict(expert_mapping)

    def _default_performance_scores(self):
        """Default performance scores when no historical data available"""
        performance_scores = {}
        for emp_id in self.employees_data.keys():
            base_score = (10 - self.initial_loads[emp_id]) / 10
            performance_scores[emp_id] = base_score
        return performance_scores

    def _calculate_performance_scores(self, tasks_df):
        """Calculate employee performance scores based on historical assignments"""
        performance_scores = {}

        for emp_id in self.employees_data.keys():
            base_score = (10 - self.initial_loads[emp_id]) / 10

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
        available_employees = [emp_id for emp_id, load in self.current_loads.items() if load < 10]

        if not available_employees:
            least_loaded = min(self.current_loads, key=self.current_loads.get)
            self.current_loads[least_loaded] = 8
            return least_loaded

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

        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidates[0]

        self.assignment_history[best_candidate['emp_id']] = \
            self.assignment_history.get(best_candidate['emp_id'], 0) + 1

        return best_candidate['emp_id']

    def _calculate_assignment_score(self, emp_id, task_category, predicted_priority, urgency_score):
        """Calculate comprehensive assignment score for an employee"""
        score = 0
        current_load = self.current_loads[emp_id]
        is_expert = self.category_preferences[emp_id] == task_category

        # Expertise bonus
        if is_expert:
            score += 50
        else:
            score -= 10

        # Workload factor
        workload_score = (10 - current_load) * 3
        score += workload_score

        # Performance history
        performance_bonus = self.performance_scores.get(emp_id, 0.5) * 20
        score += performance_bonus

        # Priority-based adjustments
        if predicted_priority == 'High':
            if current_load <= 5:
                score += 15
            elif current_load >= 8:
                score -= 20

            if is_expert and current_load <= 4:
                score += 25

        elif predicted_priority == 'Medium':
            if 3 <= current_load <= 7:
                score += 10

        # Urgency adjustments
        if urgency_score > 2:
            if current_load <= 4:
                score += 10
            elif current_load >= 8:
                score -= 15

        # Load balancing
        recent_assignments = self.assignment_history.get(emp_id, 0)
        if recent_assignments >= 3:
            score -= recent_assignments * 5

        # Avoid overloading
        if current_load >= 9:
            score -= 30
        elif current_load >= 7:
            score -= 10

        return score

    def update_workload(self, emp_id, task_complexity=1, predicted_priority='Medium'):
        """Update employee workload with priority-based complexity"""
        if emp_id not in self.current_loads:
            return

        complexity_multiplier = {'High': 2.0, 'Medium': 1.0, 'Low': 0.7}
        final_complexity = task_complexity * complexity_multiplier.get(predicted_priority, 1.0)
        self.current_loads[emp_id] = min(10, self.current_loads[emp_id] + final_complexity)

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


class AdvancedTaskPriorityPredictor:
    def __init__(self, model_path='models/task_priority_model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.load_model()

    def load_model(self):
        """Load the enhanced model and all components"""
        try:
            print("Loading advanced keyword-based model...")
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)

            # Extract components with version handling
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
            self.quick_mode = self.model_data.get('quick_mode', True)
            self.version = self.model_data.get('version', 'unknown')

            model_type = "Random Forest" if self.quick_mode else "XGBoost"
            print(f"Successfully loaded {model_type} model (version: {self.version})")
            print(f"Available categories: {', '.join(self.actual_categories)}")
            print(f"Priority levels: {', '.join(self.actual_priorities)}")

            if self.priority_keywords:
                print("Keyword-based priority detection enabled")
                # Show some keyword examples
                for priority, patterns in self.priority_keywords.items():
                    primary_keywords = patterns.get('primary', [])[:3]
                    print(f"  {priority.capitalize()}: {', '.join(primary_keywords)}...")

        except FileNotFoundError:
            print(f"Error: Model file '{self.model_path}' not found.")
            print("Please run 'python task_priority_trainer.py' first to train the model.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please retrain the model with the updated trainer.")
            raise

    def validate_category(self, category):
        """Smart category validation with suggestions"""
        if category in self.actual_categories:
            return category, True

        # Try exact partial matching
        category_lower = category.lower()
        for actual_cat in self.actual_categories:
            if category_lower in actual_cat.lower() or actual_cat.lower() in category_lower:
                print(f"Category '{category}' corrected to: '{actual_cat}'")
                return actual_cat, False

        # Try word-level matching
        category_words = set(category_lower.split())
        best_match = None
        max_overlap = 0

        for actual_cat in self.actual_categories:
            actual_words = set(actual_cat.lower().split())
            overlap = len(category_words.intersection(actual_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = actual_cat

        if best_match and max_overlap > 0:
            print(f"Category '{category}' auto-corrected to: '{best_match}'")
            return best_match, False

        # Show available categories and use fallback
        print(f"Category '{category}' not recognized.")
        print(f"Available: {', '.join(self.actual_categories)}")
        fallback = self.actual_categories[0]
        print(f"Using fallback: '{fallback}'")
        return fallback, False

    def analyze_keywords(self, description):
        """Advanced keyword analysis"""
        desc_lower = description.lower()
        keyword_analysis = {}

        for priority, patterns in self.priority_keywords.items():
            primary_matches = [kw for kw in patterns.get('primary', []) if kw in desc_lower]
            secondary_matches = [kw for kw in patterns.get('secondary', []) if kw in desc_lower]
            urgency_matches = [kw for kw in patterns.get('urgency_indicators', []) if kw in desc_lower]

            total_score = len(primary_matches) * 3 + len(secondary_matches) * 2 + len(urgency_matches) * 4

            keyword_analysis[priority] = {
                'score': total_score,
                'primary': primary_matches,
                'secondary': secondary_matches,
                'urgency': urgency_matches,
                'total_matches': len(primary_matches) + len(secondary_matches) + len(urgency_matches)
            }

        return keyword_analysis

    def calculate_urgency_score(self, description):
        """Calculate overall urgency score"""
        desc_lower = description.lower()
        urgency_score = 0

        # High urgency indicators
        high_urgency = ['urgent', 'immediately', 'emergency', 'critical', 'asap']
        urgency_score += sum(4 for kw in high_urgency if kw in desc_lower)

        # Medium urgency indicators
        medium_urgency = ['important', 'needed', 'priority', 'soon']
        urgency_score += sum(2 for kw in medium_urgency if kw in desc_lower)

        # Punctuation urgency
        urgency_score += description.count('!') * 2
        urgency_score += description.count('?') * 1

        # Capitalization urgency
        words = description.split()
        caps_words = sum(1 for word in words if word.isupper())
        urgency_score += caps_words * 1.5

        return urgency_score

    def extract_enhanced_features(self, description, category, employee_id=None):
        """Extract all enhanced features for prediction"""
        # Validate category
        validated_category, category_valid = self.validate_category(category)

        # Basic text features
        words = description.split()
        token_count = len(words)
        word_count = len(words)

        # Keyword analysis
        keyword_analysis = self.analyze_keywords(description)
        high_keywords = keyword_analysis['high']['score']
        medium_keywords = keyword_analysis['medium']['score']
        low_keywords = keyword_analysis['low']['score']
        urgency_score = self.calculate_urgency_score(description)

        # Keyword confidence
        total_keyword_score = high_keywords + medium_keywords + low_keywords
        if total_keyword_score > 0:
            keyword_confidence = max(high_keywords, medium_keywords, low_keywords) / total_keyword_score
        else:
            keyword_confidence = 0

        # Advanced text features
        caps_ratio = sum(1 for word in words if word.isupper()) / len(words) if words else 0
        punct_score = description.count('!') * 2 + description.count('?') + description.count('-') * 0.5
        avg_word_len = np.mean([len(word) for word in words]) if words else 0
        complex_words = sum(1 for word in words if len(word) > 7)
        complexity_score = complex_words / len(words) if words else 0

        # Employee features
        if employee_id and employee_id in self.employees_data:
            emp_info = self.employees_data[employee_id]
            emp_load = emp_info['emp_load']
            category_match = 1 if emp_info['emp_preferred_category'] == validated_category else 0
            emp_performance = (10 - emp_load) / 10
        else:
            emp_load = 5
            category_match = 0
            emp_performance = 0.5

        # Category encoding
        category_features = []
        for cat in self.actual_categories:
            category_features.append(1 if cat == validated_category else 0)

        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([description]).toarray()[0]

        # Combine features in training order
        features = [
            token_count, word_count,
            high_keywords, medium_keywords, low_keywords, urgency_score, keyword_confidence,
            caps_ratio, punct_score, avg_word_len, complexity_score,
            emp_load, category_match, emp_performance
        ]
        features.extend(category_features)
        features.extend(tfidf_features)

        return (np.array(features).reshape(1, -1), validated_category, category_valid,
                keyword_analysis, urgency_score)

    def predict_priority_advanced(self, description, category, employee_id=None):
        """Advanced priority prediction with keyword intelligence"""
        try:
            # Extract enhanced features
            X, validated_category, category_valid, keyword_analysis, urgency_score = \
                self.extract_enhanced_features(description, category, employee_id)

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
        """Provide detailed explanation of the prediction"""
        print(f"\nPREDICTION EXPLANATION:")

        # Keyword analysis explanation
        keyword_analysis = result['keyword_analysis']
        print("Keyword Analysis:")

        for priority, analysis in keyword_analysis.items():
            if analysis['total_matches'] > 0:
                print(f"  {priority.capitalize()} priority indicators:")
                if analysis['primary']:
                    print(f"    Primary keywords: {', '.join(analysis['primary'])}")
                if analysis['urgency']:
                    print(f"    Urgency keywords: {', '.join(analysis['urgency'])}")
                print(f"    Total score: {analysis['score']}")

        # Overall assessment
        urgency_score = result['urgency_score']
        if urgency_score > 6:
            urgency_level = "Very High"
        elif urgency_score > 3:
            urgency_level = "High"
        elif urgency_score > 1:
            urgency_level = "Medium"
        else:
            urgency_level = "Low"

        print(f"Overall urgency level: {urgency_level} (score: {urgency_score:.1f})")

        # Category validation
        if not result['category_was_valid']:
            print(f"Note: Category was auto-corrected to '{result['validated_category']}'")

    def recommend_employee_advanced(self, category, priority_probabilities, urgency_score):
        """Advanced employee recommendation with detailed reasoning"""
        try:
            predicted_priority = max(priority_probabilities, key=priority_probabilities.get)

            optimal_emp = self.workload_balancer.get_optimal_employee(
                category, predicted_priority, urgency_score
            )

            if optimal_emp and optimal_emp in self.employees_data:
                emp_details = self.workload_balancer.get_employee_details(optimal_emp)

                if emp_details:
                    # Generate selection reasoning
                    reasons = []

                    if emp_details['preferred_category'] == category:
                        reasons.append("category expert")

                    load = emp_details['current_load']
                    if load <= 3:
                        reasons.append("low workload")
                    elif load <= 6:
                        reasons.append("moderate workload")
                    elif load >= 8:
                        reasons.append("high workload but best available")

                    if predicted_priority == 'High' and load <= 5:
                        reasons.append("suitable for high priority")

                    if emp_details['performance_score'] > 0.7:
                        reasons.append("high performer")

                    if urgency_score > 3 and load <= 4:
                        reasons.append("available for urgent tasks")

                    if not reasons:
                        reasons.append("best overall match")

                    return {
                        'employee_id': emp_details['emp_id'],
                        'current_load': emp_details['current_load'],
                        'initial_load': emp_details['initial_load'],
                        'preferred_category': emp_details['preferred_category'],
                        'category_match': emp_details['preferred_category'] == category,
                        'performance_score': emp_details['performance_score'],
                        'recent_assignments': emp_details['recent_assignments'],
                        'load_increase': emp_details['load_increase'],
                        'selection_reasons': reasons
                    }
            return None
        except Exception as e:
            print(f"Error in employee recommendation: {e}")
            return None

    def make_complete_prediction(self, description, category, employee_id=None, show_explanation=True):
        """Make a complete prediction with analysis and recommendation"""
        print(f"\nTask: {description[:80]}{'...' if len(description) > 80 else ''}")
        print(f"Category: {category}")

        # Make prediction
        result = self.predict_priority_advanced(description, category, employee_id)

        if not result:
            print("Prediction failed. Please check your input.")
            return None

        # Display results
        priority = result['predicted_priority']
        confidence = result['confidence']

        print(f"\nPREDICTED PRIORITY: {priority.upper()}")
        print(f"Confidence: {confidence:.0%}")

        # Show probability breakdown
        print("\nProbability Distribution:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for rank, (p, prob) in enumerate(sorted_probs, 1):
            bar_length = int(prob * 25)
            bar = "█" * bar_length + "░" * (25 - bar_length)
            print(f"  #{rank} {p:6}: {prob:.0%} |{bar}|")

        # Show explanation
        if show_explanation:
            self.explain_prediction(result)

        # Employee recommendation
        if not employee_id:
            emp_rec = self.recommend_employee_advanced(
                result['validated_category'],
                result['all_probabilities'],
                result['urgency_score']
            )

            if emp_rec:
                load = emp_rec['current_load']
                load_status = "Light" if load <= 3 else "Medium" if load <= 6 else "Heavy"
                match_status = "Expert match" if emp_rec['category_match'] else "No category match"

                print(f"\nRECOMMENDED EMPLOYEE: {emp_rec['employee_id']}")
                print(f"  Current workload: {load_status} ({load:.1f}/10)")
                print(f"  Initial workload: {emp_rec['initial_load']}/10")
                print(f"  Load increase: +{emp_rec['load_increase']:.1f}")
                print(f"  Category expertise: {match_status}")
                print(f"  Preferred category: {emp_rec['preferred_category']}")
                print(f"  Performance score: {emp_rec['performance_score']:.2f}")
                print(f"  Recent assignments: {emp_rec['recent_assignments']}")
                print(f"  Selection reasons: {', '.join(emp_rec['selection_reasons'])}")

                # Update workload
                self.workload_balancer.update_workload(
                    emp_rec['employee_id'], 1, priority
                )

                # Show updated workload
                new_load = self.workload_balancer.current_loads[emp_rec['employee_id']]
                print(f"  Updated workload: {new_load:.1f}/10")

        return result

    def interactive_mode(self):
        """Enhanced interactive mode"""
        print("\n" + "=" * 80)
        print("ADVANCED TASK PRIORITY PREDICTOR")
        print("Keyword-Enhanced with Intelligent Workload Balancing")
        print("=" * 80)
        print("Commands: 'quit' to exit | 'help' for help | 'categories' to show available")
        print("          'workload' for team status | 'examples' for sample predictions")

        while True:
            try:
                print("\n" + "-" * 60)

                user_input = input("Enter task description: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Session ended. Team workload summary:")
                    summary = self.workload_balancer.get_workload_summary()
                    print(f"  Average load: {summary['avg_load']:.1f}")
                    print(f"  Overloaded employees: {summary['overloaded_count']}")
                    print(f"  Underutilized employees: {summary['underutilized_count']}")
                    break

                if user_input.lower() == 'help':
                    print("This advanced predictor uses keyword analysis for accurate priority detection.")
                    print("It includes intelligent workload balancing with expertise matching.")
                    print("Enter detailed task descriptions with priority keywords for best results.")
                    continue

                if user_input.lower() == 'categories':
                    print("Available categories:")
                    for i, cat in enumerate(self.actual_categories, 1):
                        experts = len([emp for emp, pref in self.workload_balancer.category_preferences.items()
                                       if pref == cat])
                        print(f"  {i:2}. {cat} ({experts} experts available)")
                    continue

                if user_input.lower() == 'workload':
                    print("Current team workload status:")
                    summary = self.workload_balancer.get_workload_summary()
                    print(f"  Total employees: {summary['total_employees']}")
                    print(f"  Average workload: {summary['avg_load']:.1f}/10")
                    print(f"  Workload range: {summary['min_load']:.1f} - {summary['max_load']:.1f}")
                    print(f"  Overloaded (≥9): {summary['overloaded_count']} employees")
                    print(f"  Underutilized (≤3): {summary['underutilized_count']} employees")
                    continue

                if user_input.lower() == 'examples':
                    examples = [
                        ("Fix urgent database connection issue immediately", "Bug Fix"),
                        ("Important: Update user documentation needed for release", "Documentation"),
                        ("Schedule team meeting for project planning next week", "Meeting")
                    ]

                    for i, (desc, cat) in enumerate(examples, 1):
                        print(f"\n--- Example {i} ---")
                        self.make_complete_prediction(desc, cat, show_explanation=False)
                    continue

                if not user_input:
                    continue

                # Get category with smart suggestions
                category = input("Enter category (or press Enter for suggestions): ").strip()

                if not category:
                    # Smart category suggestion based on keywords
                    desc_lower = user_input.lower()
                    suggestions = []

                    if any(word in desc_lower for word in ['fix', 'bug', 'error', 'issue']):
                        suggestions.append('Bug Fix')
                    if any(word in desc_lower for word in ['document', 'manual', 'guide']):
                        suggestions.append('Documentation')
                    if any(word in desc_lower for word in ['design', 'ui', 'interface']):
                        suggestions.append('UI/UX')
                    if any(word in desc_lower for word in ['deploy', 'release', 'server']):
                        suggestions.append('Deployment')
                    if any(word in desc_lower for word in ['meeting', 'discuss', 'plan']):
                        suggestions.append('Meeting')

                    if suggestions:
                        available_suggestions = [s for s in suggestions if s in self.actual_categories]
                        if available_suggestions:
                            category = available_suggestions[0]
                            print(f"Auto-suggested category: {category}")
                        else:
                            category = self.actual_categories[0]
                            print(f"Using default category: {category}")
                    else:
                        print(f"Available: {', '.join(self.actual_categories[:5])}...")
                        category = input("Choose category: ").strip() or self.actual_categories[0]

                # Make prediction
                self.make_complete_prediction(user_input, category)

            except KeyboardInterrupt:
                print("\nSession ended.")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again with different input.")


def main():
    """Main function for the advanced predictor"""
    try:
        predictor = AdvancedTaskPriorityPredictor()

        print("\nSelect mode:")
        print("1. Interactive mode (recommended)")
        print("2. Single prediction")
        print("3. Demo mode with examples")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            predictor.interactive_mode()

        elif choice == '2':
            desc = input("Enter task description: ")
            print(f"Available categories: {', '.join(predictor.actual_categories)}")
            cat = input("Enter category: ").strip()
            if not cat:
                cat = predictor.actual_categories[0]
                print(f"Using default: {cat}")

            predictor.make_complete_prediction(desc, cat)

        elif choice == '3':
            print("\nDemo mode - Advanced Keyword-Based Predictions")
            print("-" * 50)

            demo_tasks = [
                ("Urgent: Fix critical login authentication bug immediately", "Bug Fix"),
                ("Important documentation update needed for API release", "Documentation"),
                ("Design new user interface for dashboard analytics", "UI/UX"),
                ("Schedule weekly team meeting for project planning", "Meeting"),
                ("Deploy application to production environment", "Deployment")
            ]

            for i, (description, category) in enumerate(demo_tasks, 1):
                print(f"\n{'=' * 20} Demo {i}/{len(demo_tasks)} {'=' * 20}")
                predictor.make_complete_prediction(description, category, show_explanation=False)

                if i < len(demo_tasks):
                    input("\nPress Enter to continue...")

            print(f"\nDemo completed! The model shows excellent keyword recognition.")

        else:
            print("Invalid choice. Starting interactive mode...")
            predictor.interactive_mode()

    except FileNotFoundError:
        print("Model file not found. Please train the model first:")
        print("python task_priority_trainer.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()