"""Automated parameter selection based on statistical analysis."""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))
from statistical_analysis import load_all_results, compute_marginal_contributions


class ParameterSelector:
    """Selects optimal parameters based on ablation study results."""
    
    def __init__(self, results_dir="logs/ablation/"):
        self.results_dir = results_dir
        self.contributions = None
        self.selection = None
    
    def analyze(self):
        """Run statistical analysis."""
        df = load_all_results()
        if len(df) == 0:
            raise ValueError("No ablation results found")
        self.contributions = compute_marginal_contributions(df)
        return self
    
    def select(self, strategy='significant', threshold=0.05):
        """Select parameters based on strategy.
        
        Strategies:
        - 'significant': Statistically significant positive contributors
        - 'positive': All positive contributors  
        - 'top_N': Top N contributors by magnitude
        """
        if self.contributions is None:
            self.analyze()
        
        if strategy == 'significant':
            self.selection = {
                p: True for p, d in self.contributions.items()
                if d['p_value'] < threshold and d['marginal_contribution'] > 0
            }
        elif strategy == 'positive':
            self.selection = {
                p: True for p, d in self.contributions.items()
                if d['marginal_contribution'] > 0
            }
        elif strategy.startswith('top_'):
            n = int(strategy.split('_')[1])
            sorted_params = sorted(
                self.contributions.items(),
                key=lambda x: x[1]['marginal_contribution'],
                reverse=True
            )
            self.selection = {p: True for p, _ in sorted_params[:n]}
        
        # Fill non-selected
        for p in ['mass', 'damping', 'friction']:
            if p not in self.selection:
                self.selection[p] = False
        
        return self.selection
    
    def get_recommendation(self):
        """Generate recommendation."""
        if self.selection is None:
            self.select()
        
        enabled = [p for p, v in self.selection.items() if v]
        return {
            'enabled_params': enabled,
            'config': self.selection,
            'rationale': {p: self.contributions[p] for p in enabled if p in self.contributions}
        }
    
    def export_config(self, output_path="configs/optimal_adr.json"):
        """Export as config file."""
        if self.contributions is None:
            self.analyze()
        if self.selection is None:
            self.select()
            
        recommendation = self.get_recommendation()
        
        config = {
            'adr_enabled_params': recommendation['config'],
            'selection_method': 'statistical_significance',
            'contributions': {
                p: {
                    'contribution': d['marginal_contribution'],
                    'p_value': d['p_value']
                }
                for p, d in self.contributions.items()
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Config saved to {output_path}")
        return config


if __name__ == '__main__':
    try:
        selector = ParameterSelector()
        recommendation = selector.analyze().get_recommendation()
        
        print("=== AUTOMATED PARAMETER SELECTION ===")
        print(f"Recommended: {recommendation['enabled_params']}")
        print(f"Config: {recommendation['config']}")
        
        selector.export_config()
    except ValueError as e:
        print(f"Error: {e}")
        print("Run ablation experiments first.")
