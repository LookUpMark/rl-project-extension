"""Ablation study - tests individual parameter contributions."""
import argparse
import json
import os
from datetime import datetime
from utils import set_seed, create_envs, train_and_evaluate, print_results, SEED, TIMESTEPS_DEFAULT
from callbacks.adr_callback import ADRCallback

# 10 configurations: baseline, UDR, and 8 ADR variants
CONFIGS = {
    'baseline':      {'mass': False, 'damping': False, 'friction': False, 'mode': 'baseline'},
    'udr':           {'mass': True,  'damping': True,  'friction': True,  'mode': 'udr'},
    'adr_none':      {'mass': False, 'damping': False, 'friction': False, 'mode': 'adr'},
    'adr_mass':      {'mass': True,  'damping': False, 'friction': False, 'mode': 'adr'},
    'adr_damp':      {'mass': False, 'damping': True,  'friction': False, 'mode': 'adr'},
    'adr_fric':      {'mass': False, 'damping': False, 'friction': True,  'mode': 'adr'},
    'adr_mass_damp': {'mass': True,  'damping': True,  'friction': False, 'mode': 'adr'},
    'adr_mass_fric': {'mass': True,  'damping': False, 'friction': True,  'mode': 'adr'},
    'adr_damp_fric': {'mass': False, 'damping': True,  'friction': True,  'mode': 'adr'},
    'adr_all':       {'mass': True,  'damping': True,  'friction': True,  'mode': 'adr'},
}


def main(config_name):
    set_seed()
    config = CONFIGS[config_name]
    mode = config['mode']
    log_dir = f"./logs/ablation/{config_name}/"
    
    # Create env based on mode
    env_src, env_tgt = create_envs(udr_source=(mode == 'udr'))
    
    # Set ADR enabled params if ADR mode
    if mode == 'adr':
        env_src.unwrapped.adr_enabled_params = {
            k: config[k] for k in ['mass', 'damping', 'friction']
        }
    
    print(f"=== ABLATION: {config_name} ({mode.upper()}) ===")
    print(f"Params: M={config['mass']}, D={config['damping']}, F={config['friction']}")
    
    # Train with callback only for ADR mode
    callback = ADRCallback(check_freq=2048) if mode == 'adr' else None
    results = train_and_evaluate(
        env_src, env_tgt, log_dir,
        callback=callback, model_name=f"ppo_ablation_{config_name}"
    )
    
    # Save JSON results
    final_adr = env_src.unwrapped.get_adr_info() if mode == 'adr' else {}
    output = {
        'config_name': config_name, 'mode': mode, 'config': config,
        'seed': SEED, 'timesteps': TIMESTEPS_DEFAULT,
        'source_mean': results['source'][0], 'source_std': results['source'][1],
        'target_mean': results['target'][0], 'target_std': results['target'][1],
        'transfer_gap': results['gap'], 'final_adr_state': final_adr,
        'timestamp': datetime.now().isoformat()
    }
    with open(f"{log_dir}results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print_results(results, config_name)
    if mode == 'adr':
        print(f"Final ADR: {final_adr}")
    
    env_src.close()
    env_tgt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=CONFIGS.keys())
    main(parser.parse_args().config)
