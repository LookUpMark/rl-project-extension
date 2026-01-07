"""ADR training - Automatic Domain Randomization with adaptive ranges."""
import argparse
from utils import set_seed, create_envs, train_and_evaluate, print_results, SEED
from callbacks.adr_callback import ADRCallback

RUNS = {'2_5M': 2_500_000, '5M': 5_000_000, '10M': 10_000_000}

def main(run='5M'):
    set_seed()
    timesteps = RUNS[run]
    env_src, env_tgt = create_envs(udr_source=False)
    
    print(f"=== ADR {run} (Seed: {SEED}) ===")
    callback = ADRCallback(check_freq=2048)
    results = train_and_evaluate(
        env_src, env_tgt, f"./logs/adr/run_{run}/",
        timesteps=timesteps, callback=callback, model_name=f"ppo_hopper_adr_{run}"
    )
    
    # Print ADR state
    adr_state = env_src.unwrapped.get_adr_info()
    print(f"Final ADR ranges: {adr_state}")
    print_results(results, f"ADR {run}")
    
    env_src.close()
    env_tgt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', choices=RUNS.keys(), default='5M')
    main(parser.parse_args().run)
