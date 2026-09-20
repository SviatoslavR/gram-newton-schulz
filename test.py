import subprocess
import sys
import os

def run_experiment(name, cmd_args):
    print(f"\n==================================================")
    print(f" [Experiment] {name}")
    print(f" Command: python {' '.join(cmd_args)}")
    print(f"==================================================")
    
    result = subprocess.run([sys.executable] + cmd_args)
    if result.returncode != 0:
        print(f"[Error] Experiment '{name}' failed with exit code {result.returncode}")
    else:
        print(f"Experiment '{name}' completed successfully.")

if __name__ == "__main__":
    print("Starting Gram Newton-Schulz Stability & INT8 Test Suite...")
    
    base_script = "tests/stress_test.py"

    common_flags = ["--pure-pytorch"]
    run_experiment(
        name="1. INT8 Simulation WITHOUT Restart (Expected Blowup)",
        cmd_args=[base_script] + common_flags + ["--simulate-int8", "--restarts", "99", "-o", "plots_int8_no_restart"]
    )
    
    run_experiment(
        name="2. INT8 Simulation WITH Restart=2",
        cmd_args=[base_script] + common_flags + ["--simulate-int8", "--restarts", "2", "-o", "plots_int8_restart_2"]
    )

    run_experiment(
        name="3. Native Precision Baseline (BF16) WITH Restart=2",
        cmd_args=[base_script] + common_flags + ["--restarts", "2", "-o", "plots_baseline_bf16"]
    )
    
    print("\n All test suites have finished executing!")
    print("Check the generated folders ('plots_int8_no_restart/', 'plots_int8_restart_2/', 'plots_baseline_bf16/') for comparison plots.")