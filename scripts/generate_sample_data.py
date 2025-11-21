"""
Generate sample loan application data for Spark training demonstration.
Creates a synthetic dataset with realistic credit risk features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_loan_data(n_samples=10000):
    """
    Generate synthetic loan application data.
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with loan application features
    """
    print(f"Generating {n_samples} sample loan applications...")
    
    # Generate applicant demographics
    data = {
        'application_id': [f'APP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'applicant_name': [f'Applicant_{i}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(21, 70, n_samples),
        'annual_income': np.random.randint(20000, 200000, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n_samples),
    }
    
    # Generate credit information
    data['credit_score'] = np.random.randint(300, 850, n_samples)
    data['debt_to_income_ratio'] = np.round(np.random.uniform(0.05, 0.6, n_samples), 2)
    data['num_credit_lines'] = np.random.randint(1, 20, n_samples)
    data['num_derogatory_marks'] = np.random.poisson(0.5, n_samples)
    data['months_since_last_delinquency'] = np.random.choice(
        list(range(0, 120)) + [np.nan] * (n_samples // 2), 
        n_samples
    )
    
    # Generate loan information
    data['loan_amount'] = np.random.randint(5000, 100000, n_samples)
    data['loan_term'] = np.random.choice([36, 60, 84, 120], n_samples)
    data['loan_purpose'] = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 
         'major_purchase', 'car', 'business', 'education'], 
        n_samples
    )
    data['interest_rate'] = np.round(np.random.uniform(3.5, 25.0, n_samples), 2)
    
    # Generate bank statement features (simplified)
    data['avg_monthly_balance'] = np.random.randint(500, 50000, n_samples)
    data['num_overdrafts'] = np.random.poisson(0.3, n_samples)
    data['num_late_fees'] = np.random.poisson(0.2, n_samples)
    data['monthly_income_deposits'] = np.random.randint(1000, 20000, n_samples)
    
    # Generate target variable (loan_status: 0 = repaid, 1 = defaulted)
    # Create realistic correlations
    default_prob = (
        0.1 +  # Base default rate
        (data['credit_score'] < 600) * 0.3 +  # Low credit score
        (data['debt_to_income_ratio'] > 0.45) * 0.2 +  # High DTI
        (data['num_derogatory_marks'] > 2) * 0.15 +  # Multiple derogatory marks
        (data['num_overdrafts'] > 3) * 0.1 +  # Frequent overdrafts
        (data['annual_income'] < 30000) * 0.1  # Low income
    )
    
    data['loan_status'] = np.random.binomial(1, np.clip(default_prob, 0, 0.8))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_cols = ['months_since_last_delinquency', 'employment_length']
    for col in missing_cols:
        if col in df.columns:
            mask = np.random.random(len(df)) < 0.1
            df.loc[mask, col] = np.nan
    
    print(f"✅ Generated {len(df)} records")
    print(f"   - Features: {len(df.columns)}")
    print(f"   - Default rate: {df['loan_status'].mean():.2%}")
    
    return df


def main():
    """Generate and save sample data."""
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data' / 'sample'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    df = generate_sample_loan_data(n_samples=10000)
    
    # Save to CSV
    output_path = data_dir / 'loan_data_sample.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Sample data saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    # Print summary statistics
    print("\n📊 Dataset Summary:")
    print(df.describe())
    
    print("\n🎯 Target Distribution:")
    print(df['loan_status'].value_counts())


if __name__ == "__main__":
    main()
