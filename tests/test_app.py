import pytest
import json
import sys
import os

# Add the parent directory to sys.path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert json.loads(response.data) == {"status": "healthy"}

def test_simulate_missing_fields(client):
    """Test the simulate endpoint with missing fields."""
    # We are missing tax_rate
    payload = {
        "interest_rate": 5.0,
        "gov_spending": 500.0
    }
    response = client.post('/simulate', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Missing required field" in data["error"]

def test_simulate_invalid_values(client):
    """Test the simulate endpoint with invalid non-numeric values."""
    payload = {
        "interest_rate": "five",
        "gov_spending": 500.0,
        "tax_rate": 20.0
    }
    response = client.post('/simulate', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Invalid input values" in data["error"]
    
# Note: We won't test a successful /simulate response in unit tests directly 
# unless we mock the models, because the models might not be trained in the CI 
# environment unless we run train.py first.
