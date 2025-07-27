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

# Task description templates for each category
TASK_TEMPLATES = {
    "Deployment": [
        "Deploy application to production environment",
        "Configure server infrastructure for new release",
        "Setup continuous integration pipeline",
        "Migrate database to new version",
        "Configure load balancers for high availability"
    ],
    "Code Review": [
        "Review pull request for authentication module",
        "Conduct code review for API endpoints",
        "Review security implementation in payment gateway",
        "Validate coding standards in user interface",
        "Review database optimization queries"
    ],
    "Email": [
        "Send project status update to stakeholders",
        "Prepare weekly newsletter for team",
        "Draft announcement for new feature release",
        "Compose client communication regarding updates",
        "Send meeting invitation for planning session"
    ],
    "Meeting": [
        "Schedule sprint planning meeting",
        "Organize stakeholder review session",
        "Conduct team retrospective meeting",
        "Arrange client demonstration session",
        "Plan architecture discussion meeting"
    ],
    "UI/UX": [
        "Design user interface for mobile application",
        "Create wireframes for dashboard layout",
        "Develop user experience flow for checkout",
        "Design icons and visual elements",
        "Conduct usability testing session"
    ],
    "Feature Update": [
        "Implement new search functionality",
        "Add notification system to application",
        "Develop user profile management feature",
        "Create advanced reporting dashboard",
        "Build real-time chat functionality"
    ],
    "Call": [
        "Schedule client consultation call",
        "Conduct technical support call",
        "Arrange vendor negotiation call",
        "Plan team sync call for project updates",
        "Organize emergency incident response call"
    ],
    "Bug Fix": [
        "Fix login authentication error",
        "Resolve payment processing bug",
        "Debug memory leak in application",
        "Fix responsive design issues",
        "Resolve data synchronization problem"
    ],
    "Report": [
        "Generate monthly performance analytics",
        "Create project progress summary report",
        "Prepare security audit documentation",
        "Compile user feedback analysis report",
        "Generate system health monitoring report"
    ],
    "Documentation": [
        "Update API documentation for new endpoints",
        "Create user manual for new features",
        "Write technical specification document",
        "Update installation and setup guide",
        "Prepare troubleshooting documentation"
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


def generate_tasks(employees_df, num_tasks=400):
    """Generate task dataset with proper assignment logic"""
    tasks = []

    for i in range(num_tasks):
        task_id = f"TASK{i + 1:04d}"

        # Select random category
        category = np.random.choice(CATEGORIES)

        # Generate task description
        task_description = np.random.choice(TASK_TEMPLATES[category])

        # Add some variation to task descriptions
        variations = [
            f"{task_description}",
            f"{task_description} for Q{np.random.randint(1, 5)} milestone",
            f"{task_description} - priority project",
            f"{task_description} (updated requirements)",
            f"{task_description} - phase {np.random.randint(1, 4)}"
        ]
        task_description = np.random.choice(variations)

        # Select priority
        priority = np.random.choice(PRIORITIES, p=[0.3, 0.5, 0.2])  # Medium most common

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
    """Fix the existing dataset by separating and correcting assignments"""
    try:
        # Read the existing dataset
        df = pd.read_csv(csv_file_path)

        print("Original dataset shape:", df.shape)
        print("Original mismatch analysis:")

        # Extract unique employees from the existing data
        employee_data = df[['User_ID', 'User_Load', 'User_Pref_Category']].drop_duplicates()
        employee_data.columns = ['emp_id', 'emp_load', 'emp_preferred_category']

        # Create tasks dataset with corrected assignments
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

            tasks_fixed.append({
                'taskid': row['Task_ID'],
                'task_description': row['Task_Description'],
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