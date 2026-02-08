"""
API Testing Script
Test the Flask REST API endpoints
"""

import requests
import json
from pprint import pprint

# API base URL
BASE_URL = "http://localhost:5000"


def test_health_check():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_single_prediction():
    """Test single patient prediction"""
    print("\n=== Testing Single Prediction ===")
    
    patient_data = {
        "patient_id": "TEST_001",
        "ivr_calls_total": 5,
        "ivr_calls_answered": 2,
        "ivr_calls_missed": 3,
        "sms_delivered": 10,
        "sms_read": 3,
        "sms_avg_response_time_hrs": 18.5,
        "age": 68,
        "num_medications": 5,
        "days_since_prescription": 7,
        "days_since_refill_due": 4
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_batch_prediction():
    """Test batch prediction"""
    print("\n=== Testing Batch Prediction ===")
    
    patients = {
        "patients": [
            {
                "patient_id": "TEST_002",
                "ivr_calls_total": 8,
                "ivr_calls_answered": 6,
                "ivr_calls_missed": 2,
                "sms_delivered": 12,
                "sms_read": 10,
                "sms_avg_response_time_hrs": 5.5,
                "age": 45,
                "num_medications": 2,
                "days_since_prescription": 30
            },
            {
                "patient_id": "TEST_003",
                "ivr_calls_total": 6,
                "ivr_calls_answered": 1,
                "ivr_calls_missed": 5,
                "sms_delivered": 8,
                "sms_read": 2,
                "sms_avg_response_time_hrs": 36.0,
                "age": 72,
                "num_medications": 8,
                "days_since_prescription": 14
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict_batch",
        json=patients,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_explanation():
    """Test detailed explanation"""
    print("\n=== Testing Explanation ===")
    
    patient_data = {
        "patient_id": "TEST_004",
        "ivr_calls_total": 7,
        "ivr_calls_answered": 2,
        "ivr_calls_missed": 5,
        "sms_delivered": 10,
        "sms_read": 3,
        "sms_avg_response_time_hrs": 28.0,
        "age": 65,
        "num_medications": 6,
        "days_since_prescription": 10,
        "days_since_refill_due": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/api/explain",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_rules_only():
    """Test rule-based prediction only"""
    print("\n=== Testing Rules-Only Prediction ===")
    
    patient_data = {
        "patient_id": "TEST_005",
        "ivr_calls_total": 5,
        "ivr_calls_answered": 1,
        "ivr_calls_missed": 4,
        "sms_delivered": 8,
        "sms_read": 2,
        "age": 70,
        "num_medications": 7
    }
    
    response = requests.post(
        f"{BASE_URL}/api/rules_only",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_demo_data():
    """Test demo data generation"""
    print("\n=== Testing Demo Data Generation ===")
    
    response = requests.get(f"{BASE_URL}/api/demo_data?n=5")
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Generated {result['count']} samples")
    print("\nFirst sample:")
    pprint(result['data'][0])


def run_all_tests():
    """Run all API tests"""
    print("=" * 60)
    print("🧪 Medication Adherence API Test Suite")
    print("=" * 60)
    
    try:
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_explanation()
        test_rules_only()
        test_demo_data()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the Flask server is running:")
        print("  python src/api/flask_app.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    run_all_tests()
