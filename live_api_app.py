"""
Script to test the live API deployment using the requests module.

This script sends a POST request to the live API endpoint and displays
both the status code and the model inference result.

Usage:
    python test_live_api.py [API_URL]

    If no URL is provided, it will prompt for the API URL.
    Example: python test_live_api.py https://your-api.onrender.com/inference
"""

import sys
import requests


def test_live_api(api_url: str) -> dict:
    """
    Send POST request to live API and return status code and prediction.

    Args:
        api_url: Full URL to the API inference endpoint
                 (e.g., https://your-api.onrender.com/inference)

    Returns:
        dict: Contains 'status_code' and 'prediction' keys
    """
    # Sample data for testing - using data that typically leads to high income
    test_data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 83311,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    try:
        # Send POST request to the live API
        response = requests.post(api_url, json=test_data, timeout=30)

        # Get status code
        status_code = response.status_code

        # Get prediction result
        if status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "No prediction in response")
        else:
            prediction = f"Error: {response.text}"

        return {
            "status_code": status_code,
            "prediction": prediction
        }

    except requests.exceptions.RequestException as e:
        return {
            "status_code": None,
            "prediction": f"Request failed: {str(e)}"
        }


def main():
    """Main function to run the live API test."""
    # Get API URL from command line argument or prompt user
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = input("Enter the full API URL (e.g., https://your-api.onrender.com/inference): ").strip()

    # Ensure URL is not empty
    if not api_url:
        print("Error: API URL cannot be empty.")
        sys.exit(1)

    print(f"\nTesting live API at: {api_url}")
    print("Sending POST request...\n")

    # Test the API
    result = test_live_api(api_url)

    # Display results
    print("=" * 50)
    print("API Test Results:")
    print("=" * 50)
    print(f"Status Code: {result['status_code']}")
    print(f"Prediction:  {result['prediction']}")
    print("=" * 50)

    # Return appropriate exit code
    if result['status_code'] == 200:
        print("\n✓ API test successful!")
        sys.exit(0)
    else:
        print("\n✗ API test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

