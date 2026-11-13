"""
Example 2: Parameter Estimation with Concentration Data (A → B → C)

Demonstrates:
- Adding experimental concentration data
- Running NLS parameter estimation
- Comparing estimated vs. true parameters
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from kipetax import ReactionModel


def generate_synthetic_data(k1_true=2.0, k2_true=0.5, t_end=5.0, n_points=20, noise=0.005):
    """Generate noisy synthetic concentration data."""
    from scipy.integrate import odeint

    def odes(y, t, k1, k2):
        A, B, C = y
        return [-k1 * A, k1 * A - k2 * B, k2 * B]

    t = np.linspace(0, t_end, n_points)
    sol = odeint(odes, [1.0, 0.0, 0.0], t, args=(k1_true, k2_true))

    rng = np.random.default_rng(42)
    sol += rng.normal(0, noise, size=sol.shape)
    sol = np.clip(sol, 0, None)

    return pd.DataFrame({'A': sol[:, 0], 'B': sol[:, 1], 'C': sol[:, 2]}, index=t)


def main():
    # True parameters (what we want to recover)
    k1_true = 2.0
    k2_true = 0.5

    print(f"True parameters: k1={k1_true}, k2={k2_true}")

    # Generate synthetic data
    df = generate_synthetic_data(k1_true=k1_true, k2_true=k2_true)
    print(f"Generated {len(df)} measurement points")

    # Create the reaction model
    r1 = ReactionModel('Ex_2_ABC_estimation')

    # Define parameters to be estimated
    k1 = r1.parameter('k1', value=1.0, bounds=(0.1, 10))
    k2 = r1.parameter('k2', value=0.2, bounds=(0.01, 5))

    # Define components
    A = r1.component('A', value=1.0)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)

    # Define ODEs
    r1.add_ode('A', -k1 * A)
    r1.add_ode('B',  k1 * A - k2 * B)
    r1.add_ode('C',  k2 * B)

    # Add measurement data
    r1.add_data(name='concentration_data', data=df)

    # Set discretization
    r1.set_discretization(nfe=40, ncp=3)

    # Known variances (from measurement noise)
    variances = {'A': 1e-4, 'B': 1e-4, 'C': 1e-4}

    # Run parameter estimation
    print("\nRunning parameter estimation...")
    r1.run_opt(variances=variances)

    ro = r1.results_dict['p_estimator']
    print(f"\nEstimation status: {ro.status}")
    print(f"Objective value: {ro.objective:.6e}")
    print(f"\nParameter comparison:")
    print(f"  {'Parameter':12s} {'True':>10s} {'Estimated':>12s} {'Error%':>10s}")
    print(f"  {'-'*48}")
    for pname, p_true in [('k1', k1_true), ('k2', k2_true)]:
        p_est = ro.P.get(pname, float('nan'))
        err_pct = abs(p_est - p_true) / p_true * 100 if p_true != 0 else float('nan')
        print(f"  {pname:12s} {p_true:>10.4f} {p_est:>12.4f} {err_pct:>9.2f}%")

    return ro


if __name__ == '__main__':
    ro = main()

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in ['A', 'B', 'C']:
            ax.plot(ro.Z.index, ro.Z[col], label=f'{col} (estimated)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.set_title('A → B → C Parameter Estimation')
        ax.legend()
        plt.tight_layout()
        plt.savefig('Ex_2_estimation.png', dpi=100)
        print("\nPlot saved to Ex_2_estimation.png")
        plt.show()
    except ImportError:
        pass
