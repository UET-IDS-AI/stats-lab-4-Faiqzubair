"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    CDF: F_X(x) = (1 - e^{-x}) * u(x)   →   X ~ Exp(λ=1)

    STEP 1 — Analytical computation
        P(X > 5)       = 1 - F(5)  = e^{-5}
        P(X < 5)       = F(5)      = 1 - e^{-5}
        P(3 < X < 7)   = F(7) - F(3) = e^{-3} - e^{-7}

    STEP 2 — Simulate 100 000 samples from Exp(1)

    STEP 3 — Estimate P(X > 5) via Monte Carlo

    RETURN
        analytic_gt5      : float
        analytic_lt5      : float
        analytic_interval : float
        simulated_gt5     : float
    """
    # ---- STEP 1 : Analytical ----------------------------------------
    analytic_gt5      = math.exp(-5)                   # P(X > 5)
    analytic_lt5      = 1 - math.exp(-5)               # P(X < 5)
    analytic_interval = math.exp(-3) - math.exp(-7)    # P(3 < X < 7)

    # ---- STEP 2 : Simulation ----------------------------------------
    rng     = np.random.default_rng(seed=42)
    samples = rng.exponential(scale=1.0, size=100_000)

    # ---- STEP 3 : Monte Carlo estimate ------------------------------
    simulated_gt5 = np.mean(samples > 5)

    # ---- Summary print ----------------------------------------------
    print("=== Q1 — CDF Probabilities ===")
    print(f"  Analytic  P(X > 5)       = {analytic_gt5:.6f}")
    print(f"  Analytic  P(X < 5)       = {analytic_lt5:.6f}")
    print(f"  Analytic  P(3 < X < 7)   = {analytic_interval:.6f}")
    print(f"  Simulated P(X > 5)       = {simulated_gt5:.6f}")

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF: f(x) = 2x * e^{-x^2}  for x >= 0

    STEP 1 — Verify non-negativity
        For x >= 0: 2x >= 0 and e^{-x^2} > 0  →  f(x) >= 0  ✓

    STEP 2 — Compute integral from 0 to ∞
        Let u = x^2, du = 2x dx
        ∫₀^∞ 2x e^{-x²} dx = ∫₀^∞ e^{-u} du = 1  ✓

    STEP 3 — Determine validity (integral == 1 and non-negative)

    STEP 4 — Plot f(x) on [0, 3]

    RETURN
        integral_value : float
        is_valid_pdf   : bool
    """
    def f(x):
        return 2 * x * np.exp(-x ** 2)

    # ---- STEP 1 : Non-negativity (checked analytically, confirmed numerically) --
    x_check = np.linspace(0, 10, 1000)
    non_negative = bool(np.all(f(x_check) >= 0))

    # ---- STEP 2 : Numerical integration ---------------------------------
    integral_value, _ = quad(f, 0, np.inf)

    # ---- STEP 3 : Validity check ----------------------------------------
    is_valid_pdf = non_negative and math.isclose(integral_value, 1.0, abs_tol=1e-6)

    # ---- STEP 4 : Plot --------------------------------------------------
    x_plot = np.linspace(0, 3, 300)
    y_plot = f(x_plot)

    plt.figure(figsize=(7, 4))
    plt.plot(x_plot, y_plot, color="royalblue", linewidth=2.5, label=r"$f(x)=2x\,e^{-x^2}$")
    plt.fill_between(x_plot, y_plot, alpha=0.15, color="royalblue")
    plt.title("Q2 — Candidate PDF: $f(x) = 2x\\,e^{-x^2}$", fontsize=13)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("q2_pdf_plot.png", dpi=150)
    plt.show()

    # ---- Summary print ----------------------------------------------
    print("=== Q2 — PDF Validation ===")
    print(f"  Non-negative : {non_negative}")
    print(f"  Integral     : {integral_value:.6f}")
    print(f"  Valid PDF    : {is_valid_pdf}")

    return integral_value, is_valid_pdf


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(λ=1)   →   PDF: f(x) = e^{-x},  CDF: F(x) = 1 - e^{-x}

    STEP 1 — Analytical
        P(X > 5)       = e^{-5}
        P(1 < X < 3)   = e^{-1} - e^{-3}

    STEP 2 — Simulate 100 000 samples

    STEP 3 — Monte Carlo estimates

    RETURN
        analytic_gt5        : float
        analytic_interval   : float
        simulated_gt5       : float
        simulated_interval  : float
    """
    # ---- STEP 1 : Analytical ----------------------------------------
    analytic_gt5      = math.exp(-5)                   # P(X > 5)
    analytic_interval = math.exp(-1) - math.exp(-3)    # P(1 < X < 3)

    # ---- STEP 2 : Simulation ----------------------------------------
    rng     = np.random.default_rng(seed=42)
    samples = rng.exponential(scale=1.0, size=100_000)

    # ---- STEP 3 : Monte Carlo estimates -----------------------------
    simulated_gt5      = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    # ---- Summary print ----------------------------------------------
    print("=== Q3 — Exponential Distribution ===")
    print(f"  Analytic  P(X > 5)     = {analytic_gt5:.6f}")
    print(f"  Simulated P(X > 5)     = {simulated_gt5:.6f}")
    print(f"  Analytic  P(1<X<3)     = {analytic_interval:.6f}")
    print(f"  Simulated P(1<X<3)     = {simulated_interval:.6f}")

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(μ=10, σ²=4)   →   σ = 2

    STEP 1 — Standardise:  Z = (X - 10) / 2

    STEP 2 — Analytical (using standard normal CDF Φ)
        P(X ≤ 12)      = Φ((12-10)/2) = Φ(1)
        P(8 < X < 12)  = Φ((12-10)/2) - Φ((8-10)/2) = Φ(1) - Φ(-1)

    STEP 3 — Simulate 100 000 samples from N(10, 4)

    STEP 4 — Monte Carlo estimates

    RETURN
        analytic_le12       : float
        analytic_interval   : float
        simulated_le12      : float
        simulated_interval  : float
    """
    mu, sigma = 10.0, 2.0

    # ---- STEP 2 : Analytical ----------------------------------------
    analytic_le12      = norm.cdf((12 - mu) / sigma)           # Φ(1)
    analytic_interval  = norm.cdf((12 - mu) / sigma) - norm.cdf((8 - mu) / sigma)  # Φ(1)-Φ(-1)

    # ---- STEP 3 : Simulation ----------------------------------------
    rng     = np.random.default_rng(seed=42)
    samples = rng.normal(loc=mu, scale=sigma, size=100_000)

    # ---- STEP 4 : Monte Carlo estimates -----------------------------
    simulated_le12      = np.mean(samples <= 12)
    simulated_interval  = np.mean((samples > 8) & (samples < 12))

    # ---- Summary print ----------------------------------------------
    print("=== Q4 — Gaussian Distribution  X ~ N(10, 2²) ===")
    print(f"  Analytic  P(X ≤ 12)    = {analytic_le12:.6f}")
    print(f"  Simulated P(X ≤ 12)    = {simulated_le12:.6f}")
    print(f"  Analytic  P(8<X<12)    = {analytic_interval:.6f}")
    print(f"  Simulated P(8<X<12)    = {simulated_interval:.6f}")

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval


# =========================================================
# Run all questions
# =========================================================

if __name__ == "__main__":
    print()
    cdf_probabilities()
    print()
    pdf_validation_plot()
    print()
    exponential_probabilities()
    print()
    gaussian_probabilities()
    print()
