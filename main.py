#main.py
from experiments.dynamics_experiment import run as run_dynamics
from experiments.training_experiment import run as run_training
from experiments.memory_experiment import run as run_memory
from experiments.encoding_experiment import run as run_encoding
from experiments.robustness_experiment import run as run_robustness


def main():

    print("=" * 60)
    print("LIQUID NEURAL DYNAMICS EXPERIMENTS")
    print("=" * 60)

    print("\n[1/5] Liquid dynamics")
    run_dynamics()

    print("\n[2/5] Temporal prediction")
    run_training()

    print("\n[3/5] Temporal memory")
    run_memory()

    print("\n[4/5] Oscillator / temporal encoding")
    run_encoding()

    print("\n[5/5] Hardware-inspired robustness")
    run_robustness()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()