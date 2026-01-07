"""UDR training - Uniform Domain Randomization with fixed ±30% ranges."""
from utils import set_seed, create_envs, train_and_evaluate, print_results, SEED

def main():
    set_seed()
    env_src, env_tgt = create_envs(udr_source=True)  # UDR enabled
    
    print(f"=== UDR ±30% (Seed: {SEED}) ===")
    results = train_and_evaluate(env_src, env_tgt, "./logs/udr/", model_name="ppo_hopper_udr")
    print_results(results, "UDR")
    
    env_src.close()
    env_tgt.close()

if __name__ == "__main__":
    main()
