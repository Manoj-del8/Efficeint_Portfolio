import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Streamlit title and description
st.title("Efficient Portfolio Optimization Using Markowitz Model")
st.write("Upload historical stock data (percentage returns) to get the optimal portfolio.")

# Sample data preview
sample_data = {
    'Date': ['01-01-2023', '01-02-2023', '01-03-2023'],
    'AAPL': [2.5, 3.0, 1.5],
    'GOOG': [1.8, 2.2, 2.0],
    'AMZN': [1.0, 2.5, 3.2]
}
sample_df = pd.DataFrame(sample_data)
st.write("Sample Data Format:")
st.write(sample_df)

# File uploader to upload stock data
uploaded_file = st.file_uploader("Upload your historical stock data (CSV)", type="csv")
if uploaded_file is not None:
    try:
        # Read data
        data = pd.read_csv(uploaded_file)

        # Validate and process the data
        if 'Date' not in data.columns:
            st.error("The uploaded file must contain a 'Date' column.")
            st.stop()

        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
        returns_data = data.drop(columns=['Date']).apply(pd.to_numeric, errors='coerce').dropna()

        if returns_data.empty:
            st.error("No valid numerical data found. Please check your file.")
            st.stop()

        returns_data = returns_data / 100  # Convert percentages to decimals
        st.write("Uploaded Data Preview:", returns_data.head())

        # Time Frame Selection
        time_frame = st.selectbox("Select Time Frame for Analysis", ['1 Month', '3 Months', '6 Months', '1 Year'])
        rows = {'1 Month': 30, '3 Months': 90, '6 Months': 180, '1 Year': 365}[time_frame]
        returns_data = returns_data.tail(rows)

        # User input for minimum weights
        st.subheader("Set Constraints")
        min_weight = st.number_input("Enter Minimum Weight for Each Stock (e.g., 0.05 for 5%)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        risk_free_rate = st.number_input("Enter Risk-Free Rate (e.g., 0.02 for 2%)", min_value=0.0, max_value=1.0, value=0.02) / 252  # Convert annual to daily

        # Portfolio calculations
        mean_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        daily_std = returns_data.std()

        # Portfolio Performance Calculation
        def portfolio_performance(weights, mean_returns, cov_matrix):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_volatility, portfolio_return

        # Generate Efficient Frontier
        num_assets = len(mean_returns)
        results = []
        weights_record = []

        for _ in range(1000):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights = np.maximum(weights, min_weight)  # Apply minimum weight constraint
            weights /= np.sum(weights)  # Re-normalize to sum to 1
            vol, ret = portfolio_performance(weights, mean_returns, cov_matrix)
            results.append([ret * 21, vol * np.sqrt(21), weights])  # Monthly return and volatility
            weights_record.append(weights)

        # Organize Efficient Frontier Results
        results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Weights'])
        results_df['Sharpe'] = (results_df['Return'] - risk_free_rate * 21) / results_df['Volatility']  # Sharpe for monthly returns
        results_df.sort_values(by='Sharpe', ascending=False, inplace=True)

        # Extract Optimal Portfolio
        optimal_portfolio = results_df.iloc[0]
        optimal_weights = optimal_portfolio['Weights']

        # Tangency Portfolio (Sharpe Ratio)
        tangency_sharpe = (optimal_portfolio['Return'] - risk_free_rate * 21) / optimal_portfolio['Volatility']

        # Capital Market Line (CML) Data
        cml_volatility = np.linspace(0, optimal_portfolio['Volatility'], 100)
        cml_return = risk_free_rate * 21 + tangency_sharpe * cml_volatility

        # Plot Efficient Frontier with CML
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', alpha=0.7)
        plt.colorbar(label='Sharpe Ratio')
        plt.plot(cml_volatility, cml_return, color='orange', linestyle='--', label='Capital Market Line (CML)')
        plt.scatter(optimal_portfolio['Volatility'], optimal_portfolio['Return'], color='red', marker='*', s=200, label='Optimal Portfolio')
        plt.title('Efficient Frontier with Monthly Return and CML')
        plt.xlabel('Volatility (Risk) - Monthly')
        plt.ylabel('Return - Monthly')
        plt.legend()
        st.pyplot(plt)

        # Display Portfolio Details
        st.subheader("Optimal Portfolio Details")
        weights_table = pd.DataFrame({
            'Stock': returns_data.columns,
            'Weight': optimal_weights
        })
        st.table(weights_table)

        # Enhanced Portfolio Statistics Display
        daily_return = optimal_portfolio['Return'] / 21
        yearly_return = daily_return * 252  # Approx 252 trading days in a year

        st.write("Optimal Portfolio Statistics:")
        st.metric(label="Expected Monthly Return", value=f"{optimal_portfolio['Return']:.4f}")
        st.metric(label="Expected Yearly Return", value=f"{yearly_return:.4f}")
        st.metric(label="Portfolio Volatility (Monthly)", value=f"{optimal_portfolio['Volatility']:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
