"""
Example 1: ODE Simulation (A → B → C)

Demonstrates:
- Defining components, parameters, and ODEs
- Running a pure simulation with kipetax
- Plotting results
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from kipetax import ReactionModel


def main():
    # Create the reaction model
    r1 = ReactionModel('Ex_1_ABC_simulation')

    # Define parameters (fixed for simulation)
    k1 = r1.parameter('k1', value=2.0, bounds=(0, 10), fixed=True)
    k2 = r1.parameter('k2', value=0.5, bounds=(0, 5), fixed=True)

    # Define components with initial conditions
    A = r1.component('A', value=1.0)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)

    # Define ODEs: A → B → C
    r1.add_ode('A', -k1 * A)
    r1.add_ode('B',  k1 * A - k2 * B)
    r1.add_ode('C',  k2 * B)

    # Set simulation time
    r1.set_time(10.0)

    # Set discretization (more elements = more accurate)
    r1.set_discretization(nfe=50, ncp=3)

    # Run simulation
    print("Running simulation...")
    r1.simulate()

    # Access results
    ro = r1.results_dict['simulator']
    print(f"\nSimulation status: {ro.status}")
    print(f"Final concentrations at t=10:")
    print(f"  A = {ro.Z['A'].iloc[-1]:.6f}")
    print(f"  B = {ro.Z['B'].iloc[-1]:.6f}")
    print(f"  C = {ro.Z['C'].iloc[-1]:.6f}")
    print(f"  Sum = {(ro.Z['A'] + ro.Z['B'] + ro.Z['C']).iloc[-1]:.6f} (should be ~1.0)")

    # Check mass balance
    total = ro.Z['A'] + ro.Z['B'] + ro.Z['C']
    max_err = abs(total - 1.0).max()
    print(f"\nMax mass balance error: {max_err:.2e}")

    return ro


if __name__ == '__main__':
    ro = main()

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in ['A', 'B', 'C']:
            ax.plot(ro.Z.index, ro.Z[col], label=col)
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.set_title('A → B → C Simulation')
        ax.legend()
        plt.tight_layout()
        plt.savefig('Ex_1_simulation.png', dpi=100)
        print("\nPlot saved to Ex_1_simulation.png")
        plt.show()
    except ImportError:
        print("\n(matplotlib not installed, skipping plot)")
