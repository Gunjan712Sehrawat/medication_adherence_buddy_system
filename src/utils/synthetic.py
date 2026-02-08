"""
Synthetic Data Generator for Medication Adherence
Generates realistic training data when real data is limited
"""

import numpy as np
import pandas as pd
from faker import Faker
from typing import Dict, Optional
import random


class SyntheticDataGenerator:
    """
    Generate synthetic patient data for medication adherence modeling.
    
    Uses realistic distributions and correlations to create training data
    that mimics real-world patterns.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """Initialize generator with random seed"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
    
    def generate(self, n_samples: int = 1000, 
                class_balance: float = 0.3) -> pd.DataFrame:
        """
        Generate synthetic patient dataset
        
        Args:
            n_samples: Number of patients to generate
            class_balance: Proportion of non-adherent patients (0-1)
            
        Returns:
            DataFrame with patient features and labels
        """
        print(f"Generating {n_samples} synthetic patient records...")
        
        # Determine adherence status
        n_nonadherent = int(n_samples * class_balance)
        n_adherent = n_samples - n_nonadherent
        
        labels = np.concatenate([
            np.ones(n_nonadherent),
            np.zeros(n_adherent)
        ])
        np.random.shuffle(labels)
        
        data = []
        
        for i, label in enumerate(labels):
            patient = self._generate_patient(label, patient_idx=i)
            data.append(patient)
        
        df = pd.DataFrame(data)
        
        print(f"✓ Generated {n_samples} samples")
        print(f"  - Adherent: {n_adherent} ({(1-class_balance)*100:.1f}%)")
        print(f"  - Non-adherent: {n_nonadherent} ({class_balance*100:.1f}%)")
        
        return df
    
    def _generate_patient(self, is_nonadherent: int, patient_idx: int) -> Dict:
        """Generate a single patient record with realistic correlations"""
        
        # Base demographics
        age = self._sample_age(is_nonadherent)
        num_medications = self._sample_num_medications(age, is_nonadherent)
        days_since_rx = np.random.randint(1, 90)
        doses_per_day = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        
        # IVR call patterns (correlated with adherence)
        ivr_calls_total = np.random.randint(3, 12)
        
        if is_nonadherent:
            # Non-adherent: higher miss rate
            miss_rate = np.random.beta(6, 3)  # Skewed toward high
            ivr_calls_missed = int(ivr_calls_total * miss_rate)
        else:
            # Adherent: lower miss rate
            miss_rate = np.random.beta(2, 5)  # Skewed toward low
            ivr_calls_missed = int(ivr_calls_total * miss_rate)
        
        ivr_calls_answered = ivr_calls_total - ivr_calls_missed
        
        # Generate response pattern (temporal)
        ivr_response_pattern = self._generate_response_pattern(
            ivr_calls_total, is_nonadherent
        )
        
        # SMS patterns (correlated with adherence and IVR)
        sms_delivered = np.random.randint(5, 20)
        
        if is_nonadherent:
            # Non-adherent: lower read rate
            read_rate = np.random.beta(2, 5) if miss_rate > 0.5 else np.random.beta(3, 4)
            sms_read = int(sms_delivered * read_rate)
            sms_clicked = int(sms_read * np.random.beta(2, 6))
            
            # Slower response times
            sms_avg_response_time_hrs = np.random.gamma(3, 8)  # Mean ~24 hrs
        else:
            # Adherent: higher read rate
            read_rate = np.random.beta(5, 2)
            sms_read = int(sms_delivered * read_rate)
            sms_clicked = int(sms_read * np.random.beta(4, 3))
            
            # Faster response times
            sms_avg_response_time_hrs = np.random.gamma(2, 3)  # Mean ~6 hrs
        
        # Prescription refill behavior
        expected_refill_day = 30
        
        if is_nonadherent:
            days_since_refill_due = max(0, int(np.random.normal(5, 3)))
        else:
            days_since_refill_due = max(-5, int(np.random.normal(-2, 2)))
        
        # Weekend vs weekday behavior
        if is_nonadherent:
            weekend_response_rate = np.random.beta(2, 5)
        else:
            weekend_response_rate = np.random.beta(4, 3)
        
        # Chronic conditions (more = higher complexity = higher risk)
        if age > 65:
            num_conditions = np.random.poisson(2) + 1
        else:
            num_conditions = np.random.poisson(1)
        
        # Add some noise to make it realistic
        noise_factor = np.random.normal(0, 0.05)
        
        patient = {
            'patient_id': f'PT{patient_idx:05d}',
            'age': age,
            'num_medications': num_medications,
            'doses_per_day': doses_per_day,
            'days_since_prescription': days_since_rx,
            'chronic_conditions': num_conditions,
            
            # IVR features
            'ivr_calls_total': ivr_calls_total,
            'ivr_calls_answered': ivr_calls_answered,
            'ivr_calls_missed': ivr_calls_missed,
            'ivr_response_pattern': ivr_response_pattern,
            
            # SMS features
            'sms_delivered': sms_delivered,
            'sms_read': sms_read,
            'sms_clicked': sms_clicked,
            'sms_avg_response_time_hrs': round(sms_avg_response_time_hrs, 2),
            
            # Behavioral features
            'days_since_refill_due': days_since_refill_due,
            'weekend_response_rate': round(weekend_response_rate, 3),
            
            # Label
            'label': int(is_nonadherent)
        }
        
        return patient
    
    def _sample_age(self, is_nonadherent: int) -> int:
        """Sample age with realistic distribution"""
        if is_nonadherent:
            # Non-adherent skews younger and older
            if np.random.rand() < 0.3:
                age = int(np.random.normal(28, 5))  # Young adults
            else:
                age = int(np.random.normal(68, 8))  # Older adults
        else:
            # Adherent: middle age
            age = int(np.random.normal(55, 12))
        
        return np.clip(age, 18, 95)
    
    def _sample_num_medications(self, age: int, is_nonadherent: int) -> int:
        """Sample number of medications based on age"""
        if age > 65:
            mean_meds = 5 if is_nonadherent else 4
        elif age > 45:
            mean_meds = 3 if is_nonadherent else 2
        else:
            mean_meds = 2 if is_nonadherent else 1
        
        return max(1, int(np.random.poisson(mean_meds)))
    
    def _generate_response_pattern(self, n_calls: int, 
                                   is_nonadherent: int) -> list:
        """Generate temporal response pattern (1=responded, 0=missed)"""
        pattern = []
        
        if is_nonadherent:
            # Declining pattern
            prob_start = 0.5
            prob_end = 0.2
        else:
            # Stable or improving pattern
            prob_start = 0.7
            prob_end = 0.8
        
        for i in range(n_calls):
            progress = i / max(n_calls - 1, 1)
            prob = prob_start + (prob_end - prob_start) * progress
            response = 1 if np.random.rand() < prob else 0
            pattern.append(response)
        
        return pattern
    
    def add_missing_data(self, df: pd.DataFrame, 
                        missing_rate: float = 0.2) -> pd.DataFrame:
        """
        Add realistic missing data patterns
        
        Args:
            df: Complete dataframe
            missing_rate: Proportion of values to make missing (0-1)
            
        Returns:
            DataFrame with missing values
        """
        df_missing = df.copy()
        
        # Features that can have missing data
        missable_features = [
            'sms_read', 'sms_clicked', 'sms_avg_response_time_hrs',
            'ivr_calls_answered', 'weekend_response_rate'
        ]
        
        for feature in missable_features:
            if feature in df_missing.columns:
                n_missing = int(len(df_missing) * missing_rate)
                missing_idx = np.random.choice(
                    df_missing.index, size=n_missing, replace=False
                )
                df_missing.loc[missing_idx, feature] = np.nan
        
        print(f"Added missing data (~{missing_rate*100:.0f}% per feature)")
        
        return df_missing
    
    def generate_time_series(self, patient_id: str, 
                            days: int = 30,
                            is_nonadherent: bool = False) -> pd.DataFrame:
        """
        Generate time-series data for a single patient
        
        Args:
            patient_id: Patient identifier
            days: Number of days to generate
            is_nonadherent: Whether patient is non-adherent
            
        Returns:
            DataFrame with daily measurements
        """
        data = []
        
        for day in range(days):
            # Simulate daily medication taking
            if is_nonadherent:
                took_medication = np.random.rand() < 0.6  # 60% adherence
            else:
                took_medication = np.random.rand() < 0.9  # 90% adherence
            
            # Simulate daily contact attempts
            ivr_attempted = np.random.rand() < 0.3
            ivr_answered = ivr_attempted and (
                np.random.rand() < (0.4 if is_nonadherent else 0.7)
            )
            
            sms_sent = np.random.rand() < 0.5
            sms_read = sms_sent and (
                np.random.rand() < (0.5 if is_nonadherent else 0.8)
            )
            
            data.append({
                'patient_id': patient_id,
                'day': day,
                'took_medication': int(took_medication),
                'ivr_attempted': int(ivr_attempted),
                'ivr_answered': int(ivr_answered),
                'sms_sent': int(sms_sent),
                'sms_read': int(sms_read),
                'is_nonadherent': int(is_nonadherent)
            })
        
        return pd.DataFrame(data)


def augment_dataset(df: pd.DataFrame, 
                   augmentation_factor: int = 2) -> pd.DataFrame:
    """
    Augment dataset by adding noise to existing samples
    Useful for expanding small datasets
    
    Args:
        df: Original dataset
        augmentation_factor: How many times to augment (total = original * factor)
        
    Returns:
        Augmented dataset
    """
    augmented = [df]
    
    for _ in range(augmentation_factor - 1):
        df_aug = df.copy()
        
        # Add small noise to numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [c for c in numerical_cols if c != 'label']
        
        for col in numerical_cols:
            noise = np.random.normal(0, df[col].std() * 0.1, len(df))
            df_aug[col] = df[col] + noise
            df_aug[col] = df_aug[col].clip(lower=0)  # No negative values
        
        augmented.append(df_aug)
    
    result = pd.concat(augmented, ignore_index=True)
    print(f"Dataset augmented: {len(df)} → {len(result)} samples")
    
    return result


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate standard dataset
    data = generator.generate(n_samples=500, class_balance=0.3)
    
    print("\nDataset Info:")
    print(data.info())
    print("\nFirst few rows:")
    print(data.head())
    
    print("\nClass distribution:")
    print(data['label'].value_counts())
    
    # Add missing data
    data_missing = generator.add_missing_data(data, missing_rate=0.15)
    print(f"\nMissing values:")
    print(data_missing.isnull().sum()[data_missing.isnull().sum() > 0])
    
    # Generate time-series for one patient
    ts_data = generator.generate_time_series('PT00001', days=30, is_nonadherent=True)
    print(f"\nTime-series sample:")
    print(ts_data.head(10))
    
    # Save
    data.to_csv('data/synthetic/training_data.csv', index=False)
    print("\n✓ Data saved to data/synthetic/training_data.csv")
