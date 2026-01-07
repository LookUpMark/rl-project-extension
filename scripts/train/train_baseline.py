"""Baseline training - no domain randomization."""
from utils import set_seed, create_envs, train_and_evaluate, print_results, SEED

def main():
    set_seed()
    env_src, env_tgt = create_envs(udr_source=False)
    
    print(f"=== BASELINE (Seed: {SEED}) ===")
    results = train_and_evaluate(env_src, env_tgt, "./logs/baseline/", model_name="ppo_hopper_baseline")
    print_results(results, "Baseline")
    
    env_src.close()
    env_tgt.close()

if __name__ == "__main__":
    main()
