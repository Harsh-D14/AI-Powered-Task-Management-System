import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define categories and other constants
CATEGORIES = [
    "Deployment", "Code Review", "Email", "Meeting", "UI/UX",
    "Feature Update", "Call", "Bug Fix", "Report", "Documentation"
]

PRIORITIES = ["Low", "Medium", "High"]

# Priority-specific keywords for ML training
PRIORITY_KEYWORDS = {
    "High": ["urgent", "immediate", "critical", "emergency", "asap", "immediately"],
    "Medium": ["important", "due near", "expected", "priority", "soon", "needed"],
    "Low": []  # No special keywords for low priority
}

# Task description templates for each category
TASK_TEMPLATES = {
    "Deployment": [
        "Deploy application to production environment",
        "Configure server infrastructure for new release",
        "Setup continuous integration pipeline",
        "Migrate database to new version",
        "Configure load balancers for high availability",
        "Update production servers with latest patches",
        "Rollback deployment due to issues",
        "Scale infrastructure for increased load"
    ],
    "Code Review": [
        "Review pull request for authentication module",
        "Conduct code review for API endpoints",
        "Review security implementation in payment gateway",
        "Validate coding standards in user interface",
        "Review database optimization queries",
        "Audit code for security vulnerabilities",
        "Review performance optimization changes",
        "Validate new feature implementation"
    ],
    "Email": [
        "Send project status update to stakeholders",
        "Prepare weekly newsletter for team",
        "Draft announcement for new feature release",
        "Compose client communication regarding updates",
        "Send meeting invitation for planning session",
        "Notify team about system maintenance",
        "Send progress report to management",
        "Communicate deadline changes to team"
    ],
    "Meeting": [
        "Schedule sprint planning meeting",
        "Organize stakeholder review session",
        "Conduct team retrospective meeting",
        "Arrange client demonstration session",
        "Plan architecture discussion meeting",
        "Coordinate incident response meeting",
        "Schedule performance review session",
        "Organize team building session"
    ],
    "UI/UX": [
        "Design user interface for mobile application",
        "Create wireframes for dashboard layout",
        "Develop user experience flow for checkout",
        "Design icons and visual elements",
        "Conduct usability testing session",
        "Redesign problematic user interface",
        "Create responsive design for mobile",
        "Fix accessibility issues in interface"
    ],
    "Feature Update": [
        "Implement new search functionality",
        "Add notification system to application",
        "Develop user profile management feature",
        "Create advanced reporting dashboard",
        "Build real-time chat functionality",
        "Update payment processing system",
        "Add multi-language support",
        "Implement data export feature"
    ],
    "Call": [
        "Schedule client consultation call",
        "Conduct technical support call",
        "Arrange vendor negotiation call",
        "Plan team sync call for project updates",
        "Organize incident response call",
        "Schedule customer feedback call",
        "Arrange contract renewal discussion",
        "Plan emergency escalation call"
    ],
    "Bug Fix": [
        "Fix login authentication error",
        "Resolve payment processing bug",
        "Debug memory leak in application",
        "Fix responsive design issues",
        "Resolve data synchronization problem",
        "Address security vulnerability",
        "Fix performance bottleneck",
        "Resolve system crash issues"
    ],
    "Report": [
        "Generate monthly performance analytics",
        "Create project progress summary report",
        "Prepare security audit documentation",
        "Compile user feedback analysis report",
        "Generate system health monitoring report",
        "Create incident post-mortem report",
        "Prepare quarterly business review",
        "Document compliance audit results"
    ],
    "Documentation": [
        "Update API documentation for new endpoints",
        "Create user manual for new features",
        "Write technical specification document",
        "Update installation and setup guide",
        "Prepare troubleshooting documentation",
        "Document incident resolution procedures",
        "Create onboarding documentation",
        "Update system architecture documentation"
    ]
}


def generate_employees(num_employees=100):
    """Generate employee dataset with realistic distribution"""
    employees = []

    # Ensure we have employees for each category (at least 3 per category)
    category_assignments = []
    for category in CATEGORIES:
        category_assignments.extend([category] * 3)  # At least 3 employees per category

    # Fill remaining slots with random categories
    remaining_slots = num_employees - len(category_assignments)
    if remaining_slots > 0:
        category_assignments.extend(np.random.choice(CATEGORIES, remaining_slots))

    # Shuffle to randomize order
    random.shuffle(category_assignments)

    for i in range(num_employees):
        emp_id = f"EMP{i + 1:03d}"
        # Employee load between 1-10 (current workload)
        emp_load = np.random.randint(1, 11)
        emp_preferred_category = category_assignments[i]

        employees.append({
            'emp_id': emp_id,
            'emp_load': emp_load,
            'emp_preferred_category': emp_preferred_category
        })

    return pd.DataFrame(employees)


def add_priority_keywords(description, priority):
    """Add priority-specific keywords to task descriptions for ML training"""
    if priority == "High":
        keyword = np.random.choice(PRIORITY_KEYWORDS["High"])
        # Different ways to incorporate urgent keywords
        patterns = [
            f"{keyword.capitalize()} - {description}",
            f"{description} - {keyword} action required",
            f"{description} ({keyword})",
            f"Need to {description.lower()} {keyword}",
            f"{keyword.capitalize()}: {description}"
        ]
        return np.random.choice(patterns)

    elif priority == "Medium":
        keyword = np.random.choice(PRIORITY_KEYWORDS["Medium"])
        # Different ways to incorporate medium priority keywords
        patterns = [
            f"{description} - {keyword}",
            f"{keyword.capitalize()} to {description.lower()}",
            f"{description} ({keyword})",
            f"{description} - {keyword} for next phase",
            f"{keyword.capitalize()}: {description}"
        ]
        return np.random.choice(patterns)

    else:  # Low priority
        # Add neutral descriptive words that don't indicate urgency
        neutral_additions = [
            "",  # No addition
            "when convenient",
            "for future release",
            "as time permits",
            "in spare time",
            "for next quarter"
        ]
        addition = np.random.choice(neutral_additions)
        if addition:
            return f"{description} - {addition}"
        return description


def generate_tasks(employees_df, num_tasks=400):
    """Generate task dataset with proper assignment logic and priority-specific keywords"""
    tasks = []

    for i in range(num_tasks):
        task_id = f"TASK{i + 1:04d}"

        # Select random category
        category = np.random.choice(CATEGORIES)

        # Generate base task description
        base_description = np.random.choice(TASK_TEMPLATES[category])

        # Select priority first (before adding keywords)
        priority = np.random.choice(PRIORITIES, p=[0.3, 0.5, 0.2])  # Medium most common

        # Add priority-specific keywords to description
        task_description = add_priority_keywords(base_description, priority)

        # Add some additional context variations (without conflicting keywords)
        neutral_variations = [
            task_description,
            f"{task_description} for current sprint",
            f"{task_description} - milestone deliverable",
            f"{task_description} (updated scope)",
            f"{task_description} - phase {np.random.randint(1, 4)}"
        ]

        # Only apply neutral variations to avoid keyword conflicts
        if priority == "Low" and np.random.random() < 0.3:  # 30% chance for low priority
            task_description = np.random.choice(neutral_variations[:1] + neutral_variations[1:])

        # Find employees who prefer this category
        suitable_employees = employees_df[employees_df['emp_preferred_category'] == category]

        if len(suitable_employees) > 0:
            # Among suitable employees, prefer those with lower workload
            weights = 1 / (suitable_employees['emp_load'] + 1)  # Inverse weight
            weights = weights / weights.sum()  # Normalize

            assigned_employee = np.random.choice(
                suitable_employees['emp_id'].values,
                p=weights
            )
        else:
            # Fallback: assign to any employee (shouldn't happen with our generation)
            assigned_employee = np.random.choice(employees_df['emp_id'].values)

        tasks.append({
            'taskid': task_id,
            'task_description': task_description,
            'priority': priority,
            'category': category,
            'assigned_to_employeeid': assigned_employee
        })

    return pd.DataFrame(tasks)


def fix_existing_dataset(csv_file_path):
    """Fix the existing dataset by separating and correcting assignments with priority keywords"""
    try:
        # Read the existing dataset
        df = pd.read_csv(csv_file_path)

        print("Original dataset shape:", df.shape)
        print("Original mismatch analysis:")

        # Extract unique employees from the existing data
        employee_data = df[['User_ID', 'User_Load', 'User_Pref_Category']].drop_duplicates()
        employee_data.columns = ['emp_id', 'emp_load', 'emp_preferred_category']

        # Create tasks dataset with corrected assignments and priority keywords
        tasks_fixed = []

        for _, row in df.iterrows():
            # Find employees who prefer this task's category
            suitable_employees = employee_data[
                employee_data['emp_preferred_category'] == row['Category']
                ]

            if len(suitable_employees) > 0:
                # Assign to employee with lowest load among suitable ones
                best_employee = suitable_employees.loc[
                    suitable_employees['emp_load'].idxmin(), 'emp_id'
                ]
            else:
                # Fallback to original assignment
                best_employee = row['Assigned_To']

            # Add priority keywords to task description
            enhanced_description = add_priority_keywords(row['Task_Description'], row['Urgency'])

            tasks_fixed.append({
                'taskid': row['Task_ID'],
                'task_description': enhanced_description,
                'priority': row['Urgency'],
                'category': row['Category'],
                'assigned_to_employeeid': best_employee
            })

        tasks_df = pd.DataFrame(tasks_fixed)

        # Verify the fix
        merged_check = tasks_df.merge(
            employee_data,
            left_on='assigned_to_employeeid',
            right_on='emp_id'
        )
        matches = (merged_check['category'] == merged_check['emp_preferred_category']).sum()
        total = len(merged_check)

        print(f"Fixed dataset: {matches}/{total} matches ({matches / total * 100:.1f}%)")

        return tasks_df, employee_data

    except Exception as e:
        print(f"Error reading existing file: {e}")
        return None, None


def analyze_priority_keywords(tasks_df):
    """Analyze the distribution of priority keywords in task descriptions"""
    high_keywords = PRIORITY_KEYWORDS["High"]
    medium_keywords = PRIORITY_KEYWORDS["Medium"]

    keyword_stats = {
        "High": {"total": 0, "with_keywords": 0, "keyword_counts": {}},
        "Medium": {"total": 0, "with_keywords": 0, "keyword_counts": {}},
        "Low": {"total": 0, "with_keywords": 0, "keyword_counts": {}}
    }

    for _, task in tasks_df.iterrows():
        priority = task['priority']
        description = task['task_description'].lower()

        keyword_stats[priority]["total"] += 1

        # Check for high priority keywords
        found_high_keywords = [kw for kw in high_keywords if kw in description]
        # Check for medium priority keywords
        found_medium_keywords = [kw for kw in medium_keywords if kw in description]

        if priority == "High":
            if found_high_keywords:
                keyword_stats[priority]["with_keywords"] += 1
                for kw in found_high_keywords:
                    keyword_stats[priority]["keyword_counts"][kw] = keyword_stats[priority]["keyword_counts"].get(kw,
                                                                                                                  0) + 1

        elif priority == "Medium":
            if found_medium_keywords:
                keyword_stats[priority]["with_keywords"] += 1
                for kw in found_medium_keywords:
                    keyword_stats[priority]["keyword_counts"][kw] = keyword_stats[priority]["keyword_counts"].get(kw,
                                                                                                                  0) + 1

        elif priority == "Low":
            # Low priority should NOT have high/medium keywords
            if found_high_keywords or found_medium_keywords:
                keyword_stats[priority]["with_keywords"] += 1
                # This is actually bad for low priority tasks

    # Print analysis results
    for priority in ["High", "Medium", "Low"]:
        stats = keyword_stats[priority]
        total = stats["total"]
        with_kw = stats["with_keywords"]

        if priority == "Low":
            print(
                f"{priority} Priority: {total} tasks, {with_kw} incorrectly contain urgency keywords ({(with_kw / total * 100):.1f}%)")
        else:
            print(
                f"{priority} Priority: {total} tasks, {with_kw} contain appropriate keywords ({(with_kw / total * 100):.1f}%)")
            if stats["keyword_counts"]:
                print(f"  Keyword distribution: {dict(stats['keyword_counts'])}")

    # Show some examples
    print("\n=== Sample Task Descriptions by Priority ===")
    for priority in ["High", "Medium", "Low"]:
        priority_tasks = tasks_df[tasks_df['priority'] == priority].head(3)
        print(f"\n{priority} Priority Examples:")
        for _, task in priority_tasks.iterrows():
            print(f"  • {task['task_description']}")


# Main execution
if __name__ == "__main__":
    print("=== AI Task Management Dataset Generator ===\n")

    # Option 1: Fix existing dataset
    print("1. Attempting to fix existing dataset...")
    tasks_fixed, employees_fixed = fix_existing_dataset('ai_task_management_dataset_400.csv')

    if tasks_fixed is not None and employees_fixed is not None:
        print("✓ Successfully fixed existing dataset")
        tasks_fixed.to_csv('fixed_tasks.csv', index=False)
        employees_fixed.to_csv('fixed_employees.csv', index=False)
        print("✓ Saved: fixed_tasks.csv and fixed_employees.csv")

    print("\n2. Generating new clean datasets...")

    # Option 2: Generate completely new datasets
    employees_df = generate_employees(100)
    tasks_df = generate_tasks(employees_df, 450)  # Generate 450 tasks

    # Verify the assignment quality
    merged_df = tasks_df.merge(employees_df, left_on='assigned_to_employeeid', right_on='emp_id')
    correct_assignments = (merged_df['category'] == merged_df['emp_preferred_category']).sum()
    total_assignments = len(merged_df)

    print(f"✓ Generated {len(employees_df)} employees")
    print(f"✓ Generated {len(tasks_df)} tasks")
    print(
        f"✓ Assignment accuracy: {correct_assignments}/{total_assignments} ({correct_assignments / total_assignments * 100:.1f}%)")

    # Save the new datasets
    employees_df.to_csv('employees_dataset.csv', index=False)
    tasks_df.to_csv('tasks_dataset.csv', index=False)

    print("✓ Saved: employees_dataset.csv and tasks_dataset.csv")

    # Display sample data
    print("\n=== Sample Employee Data ===")
    print(employees_df.head(10))

    print("\n=== Sample Task Data ===")
    print(tasks_df.head(10))

    # Show category distribution
    print("\n=== Category Distribution ===")
    print("Employee preferences:")
    print(employees_df['emp_preferred_category'].value_counts())
    print("\nTask categories:")
    print(tasks_df['category'].value_counts())

    print("\n=== Priority Distribution ===")
    print(tasks_df['priority'].value_counts())

    print("\n=== Employee Load Distribution ===")
    print(f"Average employee load: {employees_df['emp_load'].mean():.2f}")
    print(f"Load range: {employees_df['emp_load'].min()} - {employees_df['emp_load'].max()}")

    # Analyze priority keyword distribution for ML training verification
    print("\n=== Priority Keyword Analysis (for ML Training) ===")
    analyze_priority_keywords(tasks_df)


def analyze_priority_keywords(tasks_df):
    """Analyze the distribution of priority keywords in task descriptions"""
    high_keywords = PRIORITY_KEYWORDS["High"]
    medium_keywords = PRIORITY_KEYWORDS["Medium"]

    keyword_stats = {
        "High": {"total": 0, "with_keywords": 0, "keyword_counts": {}},
        "Medium": {"total": 0, "with_keywords": 0, "keyword_counts": {}},
        "Low": {"total": 0, "with_keywords": 0, "keyword_counts": {}}
    }

    for _, task in tasks_df.iterrows():
        priority = task['priority']
        description = task['task_description'].lower()

        keyword_stats[priority]["total"] += 1

        # Check for high priority keywords
        found_high_keywords = [kw for kw in high_keywords if kw in description]
        # Check for medium priority keywords
        found_medium_keywords = [kw for kw in medium_keywords if kw in description]

        if priority == "High":
            if found_high_keywords:
                keyword_stats[priority]["with_keywords"] += 1
                for kw in found_high_keywords:
                    keyword_stats[priority]["keyword_counts"][kw] = keyword_stats[priority]["keyword_counts"].get(kw,
                                                                                                                  0) + 1

        elif priority == "Medium":
            if found_medium_keywords:
                keyword_stats[priority]["with_keywords"] += 1
                for kw in found_medium_keywords:
                    keyword_stats[priority]["keyword_counts"][kw] = keyword_stats[priority]["keyword_counts"].get(kw,
                                                                                                                  0) + 1

        elif priority == "Low":
            # Low priority should NOT have high/medium keywords
            if found_high_keywords or found_medium_keywords:
                keyword_stats[priority]["with_keywords"] += 1
                # This is actually bad for low priority tasks

    # Print analysis results
    for priority in ["High", "Medium", "Low"]:
        stats = keyword_stats[priority]
        total = stats["total"]
        with_kw = stats["with_keywords"]

        if priority == "Low":
            print(
                f"{priority} Priority: {total} tasks, {with_kw} incorrectly contain urgency keywords ({(with_kw / total * 100):.1f}%)")
        else:
            print(
                f"{priority} Priority: {total} tasks, {with_kw} contain appropriate keywords ({(with_kw / total * 100):.1f}%)")
            if stats["keyword_counts"]:
                print(f"  Keyword distribution: {dict(stats['keyword_counts'])}")

    # Show some examples
    print("\n=== Sample Task Descriptions by Priority ===")
    for priority in ["High", "Medium", "Low"]:
        priority_tasks = tasks_df[tasks_df['priority'] == priority].head(3)
        print(f"\n{priority} Priority Examples:")
        for _, task in priority_tasks.iterrows():
            print(f"  • {task['task_description']}")