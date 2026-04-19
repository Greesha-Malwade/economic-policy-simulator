from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load models at startup
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
try:
    model_inflation = joblib.load(os.path.join(MODEL_DIR, 'model_inflation.pkl'))
    model_unemployment = joblib.load(os.path.join(MODEL_DIR, 'model_unemployment.pkl'))
except FileNotFoundError:
    model_inflation = None
    model_unemployment = None
    print("Warning: Models not found. Please run the training script first.")

def get_policy_recommendation(inflation, unemployment):
    """Simple rule-based recommendation engine."""
    if inflation > 5.0 and unemployment < 4.0:
        return "Contractionary monetary policy recommended to cool down inflation."
    elif inflation < 2.0 and unemployment > 6.0:
        return "Expansionary fiscal/monetary policy recommended to stimulate growth and reduce unemployment."
    else:
        return "Maintain current policies, the economy appears relatively stable."

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/simulate', methods=['POST'])
def simulate():
    if model_inflation is None or model_unemployment is None:
        return jsonify({"error": "Models are not loaded. Train models first."}), 500
        
    try:
        data = request.get_json()
        
        # Extract features
        interest_rate = float(data['interest_rate'])
        gov_spending = float(data['gov_spending'])
        tax_rate = float(data['tax_rate'])
        
        # Prepare input for models (2D array expected)
        features = [[interest_rate, gov_spending, tax_rate]]
        
        # Predict
        predicted_inflation = model_inflation.predict(features)[0]
        predicted_unemployment = model_unemployment.predict(features)[0]
        
        # Get recommendation
        recommendation = get_policy_recommendation(predicted_inflation, predicted_unemployment)
        
        response = {
            "input": {
                "interest_rate": interest_rate,
                "gov_spending": gov_spending,
                "tax_rate": tax_rate
            },
            "predictions": {
                "inflation_percent": round(predicted_inflation, 2),
                "unemployment_percent": round(predicted_unemployment, 2)
            },
            "policy_recommendation": recommendation
        }
        
        return jsonify(response), 200
        
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except ValueError:
        return jsonify({"error": "Invalid input values. Please provide numerical values."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
