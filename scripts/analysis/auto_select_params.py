"""Automated parameter selection based on ablation analysis."""
import os
import json
from statistical_analysis import load_results, compute_contributions


class ParameterSelector:
    """Select optimal ADR parameters from ablation study."""
    
    def __init__(self):
        self.contributions = None
        self.selection = None
    
    def analyze(self):
        """Load data and compute contributions."""
        df = load_results()
        if len(df) == 0:
            raise ValueError("No ablation results found")
        self.contributions = compute_contributions(df)
        return self
    
    def select(self, strategy='significant', threshold=0.05):
        """Select parameters.
        
        Strategies: 'significant', 'positive', 'top_N'
        """
        if self.contributions is None:
            self.analyze()
        
        if strategy == 'significant':
            self.selection = {p: d['p_value'] < threshold and d['marginal_contribution'] > 0
                             for p, d in self.contributions.items()}
        elif strategy == 'positive':
            self.selection = {p: d['marginal_contribution'] > 0 for p, d in self.contributions.items()}
        elif strategy.startswith('top_'):
            n = int(strategy.split('_')[1])
            ranked = sorted(self.contributions.items(), key=lambda x: x[1]['marginal_contribution'], reverse=True)
            self.selection = {p: i < n for i, (p, _) in enumerate(ranked)}
        
        return self.selection
    
    def export(self, path="configs/optimal_adr.json"):
        """Save config file."""
        if self.contributions is None: self.analyze()
        if self.selection is None: self.select()
        
        config = {
            'adr_enabled_params': self.selection,
            'selection_method': 'statistical_significance',
            'contributions': {p: {'contribution': d['marginal_contribution'], 'p_value': d['p_value']}
                             for p, d in self.contributions.items()}
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved: {path}")
        return config


if __name__ == '__main__':
    try:
        selector = ParameterSelector().analyze()
        selection = selector.select()
        enabled = [p for p, v in selection.items() if v]
        
        print("=== AUTO PARAMETER SELECTION ===")
        print(f"Recommended: {enabled}")
        print(f"Config: {selection}")
        selector.export()
    except ValueError as e:
        print(f"Error: {e}")
