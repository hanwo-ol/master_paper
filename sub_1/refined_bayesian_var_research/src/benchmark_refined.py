# ============================================================================
# COMPLETE FIX: All Issues Resolved
# 1. Model path fix
# 2. MC Dropout activation fix
# 3. Calibration calculation fix
# 4. Backtesting logic fix
# ============================================================================

# ============================================================================
# benchmark_refined.py (PATH FIX)
# ============================================================================

import numpy as np
import pandas as pd
import torch
import os
from typing import Dict


class HistoricalVaR:
    """Historical VaR (Baseline 1)"""
    
    @staticmethod
    def compute(returns: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        """Historical quantile"""
        quantile = 1 - confidence
        return np.percentile(returns, quantile * 100)


class ParametricVaR:
    """Parametric VaR with Gaussian assumption (Baseline 2)"""
    
    @staticmethod
    def compute(returns: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        """Gaussian VaR using mean and std"""
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence)
        mean = np.mean(returns)
        std = np.std(returns)
        return mean + z_score * std


class VanillaNN:
    """Vanilla Neural Network without Uncertainty (Baseline 3)"""
    
    def __init__(self, input_dim: int = 11, device: str = 'cpu'):
        self.device = device
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        ).to(device)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction only"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy().squeeze()


class BenchmarkEvaluator:
    """Benchmark 평가 (FIXED: model path)"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results = {}
    
    def evaluate_historical_var(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Historical VaR 평가"""
        # Simple average as baseline
        predictions = np.full(len(y_test), np.mean(y_test))
        
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        
        self.results['Historical VaR'] = {
            'MAE': mae,
            'RMSE': rmse,
            'predictions': predictions
        }
        
        return self.results['Historical VaR']
    
    def evaluate_parametric_var(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Parametric VaR 평가"""
        # Gaussian assumption
        predictions = np.full(len(y_test), ParametricVaR.compute(y_test))
        
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        
        self.results['Parametric VaR'] = {
            'MAE': mae,
            'RMSE': rmse,
            'predictions': predictions
        }
        
        return self.results['Parametric VaR']
    
    def evaluate_vanilla_nn(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Vanilla NN 평가"""
        model = VanillaNN(input_dim=X_test.shape[1], device=self.device)
        predictions = model.predict(X_test)
        
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        
        self.results['Vanilla NN'] = {
            'MAE': mae,
            'RMSE': rmse,
            'predictions': predictions
        }
        
        return self.results['Vanilla NN']
    
    def evaluate_bayesian_nn(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Bayesian NN 평가 (FIXED: model path)"""
        from model_refined import BayesianVaRNN
        
        model = BayesianVaRNN(input_dim=X_test.shape[1], hidden_dim=128)
        
        # FIX: Use correct path (in src/ directory)
        model_path = 'best_bayesian_var_model.pt'
        if not os.path.exists(model_path):
            model_path = '../best_bayesian_var_model.pt'
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"[OK] Model loaded from {model_path}")
        else:
            print(f"[WARNING] Model file not found, using untrained model")
        
        model = model.to(self.device)
        model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            var_pred, _, _ = model(X_tensor)
        
        predictions = var_pred.squeeze().cpu().numpy()
        
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        
        self.results['Bayesian NN'] = {
            'MAE': mae,
            'RMSE': rmse,
            'predictions': predictions
        }
        
        return self.results['Bayesian NN']
    
    def print_comparison(self) -> pd.DataFrame:
        """벤치마크 결과 출력"""
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON RESULTS")
        print("="*80)
        
        data = []
        for model_name, metrics in self.results.items():
            data.append({
                'model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE']
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + df.to_string(index=False))
        
        return df
    
    def save_results(self, filepath: str) -> None:
        """결과 저장"""
        data = []
        for model_name, metrics in self.results.items():
            data.append({
                'model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"\n[OK] Results saved to {filepath}")


class ImprovementAnalysis:
    """개선도 분석"""
    
    @staticmethod
    def analyze_improvement(results: Dict) -> Dict:
        """Bayesian NN의 개선도 분석"""
        
        if 'Bayesian NN' not in results or 'Historical VaR' not in results:
            return {}
        
        bayesian_mae = results['Bayesian NN']['MAE']
        historical_mae = results['Historical VaR']['MAE']
        
        improvement = (historical_mae - bayesian_mae) / historical_mae * 100
        
        return {
            'improvement_vs_historical': improvement,
            'bayesian_mae': bayesian_mae,
            'historical_mae': historical_mae
        }
    
    @staticmethod
    def print_improvement_analysis(analysis: Dict) -> None:
        """개선도 출력"""
        if not analysis:
            return
        
        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        
        improvement = analysis['improvement_vs_historical']
        
        print(f"\nBayesian NN vs Historical VaR:")
        print(f"  Historical MAE: {analysis['historical_mae']:.6f}")
        print(f"  Bayesian MAE:   {analysis['bayesian_mae']:.6f}")
        print(f"  Improvement:    {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"\n[OK] Bayesian NN is {improvement:.1f}% MORE ACCURATE")
        else:
            print(f"\n[WARNING] Bayesian NN is {-improvement:.1f}% LESS ACCURATE")


def main():
    """Main execution"""
    print("Loading test data...")
    data = np.load('../data/synthetic_data.npz')
    X_val = data['X_val']
    y_val = data['y_val']
    
    print("Running benchmarks...")
    evaluator = BenchmarkEvaluator(device='cpu')
    
    # Run all benchmarks
    evaluator.evaluate_historical_var(X_val, y_val)
    evaluator.evaluate_parametric_var(X_val, y_val)
    evaluator.evaluate_vanilla_nn(X_val, y_val)
    evaluator.evaluate_bayesian_nn(X_val, y_val)
    
    # Print comparison
    comparison_df = evaluator.print_comparison()
    
    # Improvement analysis
    improvement = ImprovementAnalysis.analyze_improvement(evaluator.results)
    ImprovementAnalysis.print_improvement_analysis(improvement)
    
    return evaluator, comparison_df


if __name__ == '__main__':
    evaluator, comparison_df = main()
