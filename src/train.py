import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib


# ── Data Generation ──────────────────────────────────────────────────────────

def generate_synthetic_data(num_samples=1000):
    """Generates synthetic economic data."""
    np.random.seed(42)

    # Independent variables
    interest_rate = np.random.uniform(1.0, 10.0, num_samples)
    gov_spending  = np.random.uniform(100.0, 1000.0, num_samples)
    tax_rate      = np.random.uniform(10.0, 40.0, num_samples)

    # Dependent variables (with noise)
    # High interest rate  -> Low inflation,  High unemployment
    # High gov spending   -> High inflation,  Low unemployment
    # High tax rate       -> Low inflation,   High unemployment

    noise_inflation   = np.random.normal(0, 0.5, num_samples)
    inflation         = 2.0 - 0.5 * interest_rate + 0.01 * gov_spending \
                        - 0.1 * tax_rate + noise_inflation

    noise_unemployment = np.random.normal(0, 0.5, num_samples)
    unemployment       = 4.0 + 0.3 * interest_rate - 0.005 * gov_spending \
                         + 0.1 * tax_rate + noise_unemployment

    df = pd.DataFrame({
        'interest_rate': interest_rate,
        'gov_spending':  gov_spending,
        'tax_rate':      tax_rate,
        'inflation':     inflation,
        'unemployment':  unemployment
    })
    return df


# ── Experiment Logging ───────────────────────────────────────────────────────

def log_experiment(timestamp, inflation_r2, unemployment_r2,
                   inflation_mae, unemployment_mae):
    """Appends experiment metrics to experiment_log.txt."""
    log_path = "experiment_log.txt"
    write_header = not os.path.exists(log_path)

    with open(log_path, "a") as f:
        if write_header:
            f.write("Timestamp | Inflation R2 | Unemployment R2 | "
                    "Inflation MAE | Unemployment MAE\n")
            f.write("-" * 85 + "\n")
        f.write(
            f"{timestamp} | {inflation_r2:.4f} | {unemployment_r2:.4f} | "
            f"{inflation_mae:.4f} | {unemployment_mae:.4f}\n"
        )


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_actual_inf, y_pred_inf,
                              y_actual_unemp, y_pred_unemp):
    """Plots Actual vs Predicted for Inflation and Unemployment side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Actual vs Predicted — Linear Regression Models",
                 fontsize=14, fontweight="bold", y=1.01)

    datasets = [
        (axes[0], y_actual_inf,   y_pred_inf,   "Inflation",   "#2196F3"),
        (axes[1], y_actual_unemp, y_pred_unemp, "Unemployment", "#FF5722"),
    ]

    for ax, y_actual, y_pred, title, colour in datasets:
        ax.scatter(y_actual, y_pred, alpha=0.35, s=15,
                   color=colour, edgecolors="none", label="Predictions")

        # Perfect-prediction reference line
        all_vals = np.concatenate([y_actual, y_pred])
        lo, hi   = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect fit")

        r2  = round(r2_score(y_actual, y_pred), 2)
        mae = round(mean_absolute_error(y_actual, y_pred), 2)

        ax.set_title(f"{title} Model\nR2 = {r2}  |  MAE = {mae}",
                     fontsize=11, pad=10)
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# ── Evaluation Helper ─────────────────────────────────────────────────────────

def evaluate_model(model_name, y_test, y_pred):
    """Calculates MSE, RMSE, and R2 Score; prints and logs results."""
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    # ── Console output ───────────────────────────────────────
    print(f"\n---- {model_name} Model Evaluation ----")
    print(f"MSE:      {mse:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # ── Append metrics block to experiment_log.txt ───────────
    with open("experiment_log.txt", "a") as f:
        f.write(f"\n[{model_name}]\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write("--------\n")

    return r2, mean_absolute_error(y_test, y_pred)


# ── Training Pipeline ────────────────────────────────────────────────────────

def train_and_save_models():
    """Trains both regression models, evaluates, visualises, and saves them."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*55}")
    print(f"  Experiment Run Started: {timestamp}")
    print(f"{'='*55}")

    # ── Generate & save data ─────────────────────────────────
    print("\nGenerating synthetic data...")
    df = generate_synthetic_data()
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/economic_data.csv', index=False)

    X = df[['interest_rate', 'gov_spending', 'tax_rate']]

    # ── Train / Test Split ───────────────────────────────────
    print("\nSplitting data into train (80%) and test (20%) sets...")
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    y_inflation    = df['inflation']
    y_unemployment = df['unemployment']

    y_inf_train, y_inf_test     = train_test_split(y_inflation,    test_size=0.2, random_state=42)
    y_unemp_train, y_unemp_test = train_test_split(y_unemployment, test_size=0.2, random_state=42)

    # ── Model 1: Inflation ───────────────────────────────────
    model_inflation = LinearRegression()
    model_inflation.fit(X_train, y_inf_train)

    # Predictions on test set for evaluation
    pred_inf_test  = model_inflation.predict(X_test)
    # Predictions on full set for visualization
    pred_inflation = model_inflation.predict(X)

    # ── Model 2: Unemployment ────────────────────────────────
    model_unemployment = LinearRegression()
    model_unemployment.fit(X_train, y_unemp_train)

    # Predictions on test set for evaluation
    pred_unemp_test   = model_unemployment.predict(X_test)
    # Predictions on full set for visualization
    pred_unemployment = model_unemployment.predict(X)

    # ── Print evaluation metrics (MSE, RMSE, R2) ─────────────
    print(f"\n{'='*55}")
    print("  Model Evaluation (on held-out test set):")
    print(f"{'='*55}")

    inflation_r2, inflation_mae = evaluate_model(
        "Inflation", y_inf_test, pred_inf_test
    )
    unemployment_r2, unemployment_mae = evaluate_model(
        "Unemployment", y_unemp_test, pred_unemp_test
    )

    print(f"\n{'='*55}")

    # ── Save models ──────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_inflation,    'models/model_inflation.pkl')
    joblib.dump(model_unemployment, 'models/model_unemployment.pkl')
    print("\n[OK] Models saved to 'models/' directory.")

    # ── Log experiment ───────────────────────────────────────
    log_experiment(timestamp, inflation_r2, unemployment_r2,
                   inflation_mae, unemployment_mae)
    print("[LOG] Experiment results appended to 'experiment_log.txt'.")
    print(f"{'='*55}\n")

    # ── Visualise Actual vs Predicted ────────────────────────
    print("[PLOT] Displaying Actual vs Predicted plots...")
    plot_actual_vs_predicted(
        y_inflation,    pred_inflation,
        y_unemployment, pred_unemployment
    )


if __name__ == "__main__":
    train_and_save_models()
