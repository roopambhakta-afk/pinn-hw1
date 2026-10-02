import subprocess
import sys

scripts = [
    "b1_blowup.py",
    "b1_convergence.py",
    "b1_dependent_D.py",
]

for script in scripts:
    print(f"\n===== {script} =====")
    subprocess.run([sys.executable, script], check=True)

print("\nAll figures are in figures/, all numbers are in results/")
