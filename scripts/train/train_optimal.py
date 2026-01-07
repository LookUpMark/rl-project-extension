"""Train with optimal parameters from auto-selection."""
import json
from utils import set_seed, create_envs, train_and_evaluate, print_results, SEED
from callbacks.adr_callback import ADRCallback


def main(config_path="configs/optimal_adr.json", timesteps=5_000_000):
    set_seed()
    
    # Load optimal config
    with open(config_path) as f:
        config = json.load(f)
    enabled = config['adr_enabled_params']
    
    env_src, env_tgt = create_envs(udr_source=False)
    env_src.unwrapped.adr_enabled_params = enabled
    
    print(f"=== OPTIMAL ADR (Seed: {SEED}) ===")
    print(f"Enabled: {enabled}")
    
    callback = ADRCallback(check_freq=2048)
    results = train_and_evaluate(
        env_src, env_tgt, "./logs/optimal/",
        timesteps=timesteps, callback=callback, model_name="ppo_hopper_optimal"
    )
    
    # Save results
    final_adr = env_src.unwrapped.get_adr_info()
    output = {
        'enabled_params': enabled,
        'source': results['source'], 'target': results['target'],
        'transfer_gap': results['gap'], 'final_adr': final_adr
    }
    with open("./logs/optimal/results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Final ADR: {final_adr}")
    print_results(results, "Optimal")
    
    env_src.close()
    env_tgt.close()


if __name__ == '__main__':
    main()
