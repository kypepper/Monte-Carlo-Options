import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function for American option pricing using a binomial tree (simplified for illustration)
def binomial_tree_american_call(S, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    prices = np.zeros((N+1, N+1))
    option = np.zeros((N+1, N+1))

    for i in range(N+1):
        prices[i, N] = S * (u ** (N - i)) * (d ** i)
        option[i, N] = max(0, prices[i, N] - K)

    for j in range(N-1, -1, -1):
        for i in range(j+1):
            prices[i, j] = S * (u ** (j - i)) * (d ** i)
            hold = np.exp(-r * dt) * (p * option[i, j+1] + (1-p) * option[i+1, j+1])
            exercise = max(0, prices[i, j] - K)
            option[i, j] = max(hold, exercise)

    return option[0, 0]

# Monte Carlo simulation for visualization
def monte_carlo_paths(S, T, r, sigma, n_paths=50, steps=100):
    dt = T / steps
    paths = np.zeros((steps + 1, n_paths))
    paths[0] = S
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

# Placeholder function for random values
def rand_val():
    return round(np.random.uniform(1.0, 10.0), 4)

st.title("American Option Pricing Interface")

st.markdown("""
This tool compares **American option prices** using the **Binomial Tree**, **Trinomial Tree**, **Finite Difference**, and **Heston** models.  
Also visualized are asset paths from a Monte Carlo simulation.
""")

S = st.slider("Initial Stock Price (S₀)", 20.0, 1000.0, 100.0)
K = st.slider("Strike Price (K)", 20.0, 1000.0, 100.0)
T = st.slider("Time to Maturity (T in years)", 0.1, 5.0, 1.0, step=0.1)
r = st.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.03, step=0.005)
sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
N = st.slider("Number of Steps (N)", 10, 1000, 100, step=10)

# Calculate Binomial Tree prices
binomial_call = round(binomial_tree_american_call(S, K, T, r, sigma, N), 4)
binomial_put = rand_val()  # Placeholder

# Placeholder prices for other methods
trinomial_call = rand_val()
trinomial_put = rand_val()
finite_diff_call = rand_val()
finite_diff_put = rand_val()
heston_call = rand_val()
heston_put = rand_val()

# Data for the table
data = pd.DataFrame({
    "Type": ["American Call", "American Put"],
    "Binomial Tree": [binomial_call, binomial_put],
    "Trinomial Tree": [trinomial_call, trinomial_put],
    "Finite Difference": [finite_diff_call, finite_diff_put],
    "Heston Model": [heston_call, heston_put]
})

st.subheader("Option Pricing Table")
st.dataframe(data)

# Plot Monte Carlo asset paths with more trials
st.subheader("Monte Carlo Asset Paths")
paths = monte_carlo_paths(S, T, r, sigma, n_paths=50, steps=100)
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(paths.shape[1]):
    ax.plot(np.linspace(0, T, paths.shape[0]), paths[:, i], lw=0.8, alpha=0.6)
ax.set_title("Simulated Asset Price Paths (50 trials)")
ax.set_xlabel("Time (Years)")
ax.set_ylabel("Asset Price")
st.pyplot(fig)

# Separate bar charts for each method
methods = ["Binomial Tree", "Trinomial Tree", "Finite Difference", "Heston Model"]
calls = [binomial_call, trinomial_call, finite_diff_call, heston_call]
puts = [binomial_put, trinomial_put, finite_diff_put, heston_put]

st.subheader("Option Prices by Method")
fig2, ax2 = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(methods))

bars1 = ax2.bar(index, calls, bar_width, label='Call Price')
bars2 = ax2.bar(index + bar_width, puts, bar_width, label='Put Price')

ax2.set_xlabel('Method')
ax2.set_ylabel('Option Price')
ax2.set_title('American Option Prices by Pricing Method')
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(methods)
ax2.legend()
st.pyplot(fig2)

# Additional: Binomial price vs number of steps (to show convergence)
st.subheader("Binomial Tree Call Price vs Number of Steps")

steps_range = list(range(10, N+1, max(1, N//20)))  # about 20 points max
prices_vs_steps = []
for steps in steps_range:
    price = binomial_tree_american_call(S, K, T, r, sigma, steps)
    prices_vs_steps.append(price)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(steps_range, prices_vs_steps, marker='o', linestyle='-')
ax3.set_xlabel("Number of Steps (N)")
ax3.set_ylabel("Option Price")
ax3.set_title("Binomial Tree American Call Option Price Convergence")
st.pyplot(fig3)

st.markdown("**Note:** Trinomial, Finite Difference, and Heston models are shown with placeholder random values.")
