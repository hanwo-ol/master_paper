# ============================================================================
# REFINED uncertainty_analysis_refined.py
# Stage 4: Enhanced Analysis with Backtesting & Multi-Confidence Levels
# ============================================================================

import numpy as np
import torch
from scipy.stats import norm
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class UncertaintyEstimator:
    """불확실성 추정 (개선: multi-confidence level 지원)"""
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def estimate_uncertainties(self, X_test: np.ndarray, 
                              n_mc_samples: int = 100) -> Dict[str, np.ndarray]:
        """MC Dropout 기반 불확실성 추정"""
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        print("Running MC Dropout inference...")
        mc_predictions = self.model.mc_dropout_forward(X_test_tensor, n_samples=n_mc_samples)
        mc_predictions = mc_predictions.cpu().numpy()
        
        epistemic_std = mc_predictions.std(axis=1)
        mc_mean = mc_predictions.mean(axis=1)
        
        self.model.eval()
        with torch.no_grad():
            _, aleatoric_std_pred, _ = self.model(X_test_tensor)
        
        aleatoric_std = aleatoric_std_pred.squeeze().cpu().numpy()
        total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)
        
        print(f"✓ Epistemic uncertainty: {epistemic_std.mean():.6f} ± {epistemic_std.std():.6f}")
        print(f"✓ Aleatoric uncertainty: {aleatoric_std.mean():.6f} ± {aleatoric_std.std():.6f}")
        print(f"✓ Total uncertainty: {total_std.mean():.6f} ± {total_std.std():.6f}")
        
        return {
            'predictions': mc_mean,
            'epistemic_std': epistemic_std,
            'aleatoric_std': aleatoric_std,
            'total_std': total_std,
            'mc_predictions': mc_predictions
        }


class CalibrationEvaluator:
    """Calibration 평가 (개선: Multi-confidence level)"""
    
    @staticmethod
    def compute_calibration_metrics(predictions: np.ndarray, 
                                   uncertainties: np.ndarray,
                                   targets: np.ndarray,
                                   confidence_levels: list = None) -> Dict:
        """
        개선: Multiple confidence levels 동시 지원
        이상적: coverage ≈ confidence_level (오차 < 2%)
        """
        if confidence_levels is None:
            confidence_levels = [0.68, 0.95, 0.99]
        
        metrics = {}
        
        for confidence in confidence_levels:
            z_score = norm.ppf((1 + confidence) / 2)
            
            lower = predictions - z_score * uncertainties
            upper = predictions + z_score * uncertainties
            
            coverage = np.mean((targets >= lower) & (targets <= upper))
            interval_width = np.mean(upper - lower)
            calibration_error = np.abs(coverage - confidence)
            
            # Average interval score
            ais = interval_width + (2/z_score) * np.maximum(lower - targets, 0) + \
                  (2/z_score) * np.maximum(targets - upper, 0)
            ais = ais.mean()
            
            metrics[f'{int(confidence*100)}%'] = {
                'coverage': coverage,
                'target': confidence,
                'error': calibration_error,
                'interval_width': interval_width,
                'average_interval_score': ais
            }
        
        return metrics
    
    @staticmethod
    def print_calibration_analysis(metrics: Dict) -> None:
        """Calibration 결과 출력"""
        print("\n" + "="*90)
        print("CALIBRATION ANALYSIS (개선: 여러 신뢰도 수준 검증)")
        print("="*90)
        
        print(f"\n{'Confidence':<12} {'Target':<10} {'Achieved':<10} {'Error':<10} {'Status':<10}")
        print("-" * 90)
        
        for conf_level, metric_dict in metrics.items():
            coverage = metric_dict['coverage']
            target = metric_dict['target']
            error = metric_dict['error']
            
            # Status check
            if error < 0.02:
                status = "✓ Excellent"
            elif error < 0.03:
                status = "✓ Good"
            else:
                status = "✗ Poor"
            
            print(f"{conf_level:<12} {target:>9.0%}  {coverage:>9.0%}  {error:>9.4f}  {status:<10}")
        
        print("\n✓ Calibration 기준 (오차 < 2%): Model이 신뢰도 구간을 정확히 제시")


class RegulatoryBacktesting:
    """
    개선: Regulatory Backtesting 추가
    Basel III의 Backtesting 프레임워크 적용
    """
    
    @staticmethod
    def kupiec_pof_test(predictions: np.ndarray, targets: np.ndarray,
                       confidence: float = 0.95) -> Dict:
        """
        Kupiec's Proportion of Failures (POF) Test
        
        Null hypothesis: 실패율 = (1 - confidence)
        → H0를 기각하지 못하면 모델이 good calibration
        """
        n = len(targets)
        failures = np.sum(targets < predictions)
        failure_rate = failures / n
        expected_failure_rate = 1 - confidence
        
        # POF statistic
        if failure_rate > 0 and failure_rate < 1:
            lr_pof = 2 * (failures * np.log(failure_rate / expected_failure_rate) +
                         (n - failures) * np.log((1 - failure_rate) / (1 - expected_failure_rate)))
        else:
            lr_pof = 0
        
        # Critical value (chi-squared with df=1, alpha=0.05)
        critical_value = 3.841
        pof_pass = lr_pof < critical_value
        
        return {
            'failures': failures,
            'failure_rate': failure_rate,
            'expected_rate': expected_failure_rate,
            'lr_statistic': lr_pof,
            'critical_value': critical_value,
            'passes': pof_pass
        }
    
    @staticmethod
    def traffic_light_approach(predictions: np.ndarray, targets: np.ndarray,
                             confidence: float = 0.95, window: int = 252) -> Dict:
        """
        Basel III Traffic Light Approach
        
        Green: 4개 이하 exceptions → No action
        Yellow: 5-9개 exceptions → Further analysis
        Red: 10개 이상 exceptions → Model rejected
        """
        exceptions = np.sum(targets < predictions)
        
        if exceptions <= 4:
            zone = "🟢 Green Zone"
            action = "No regulatory action needed"
        elif exceptions <= 9:
            zone = "🟡 Yellow Zone"
            action = "Further investigation required"
        else:
            zone = "🔴 Red Zone"
            action = "Model must be rejected/revised"
        
        return {
            'exceptions': exceptions,
            'zone': zone,
            'action': action
        }
    
    @staticmethod
    def print_backtesting_results(pof_results: Dict, tl_results: Dict) -> None:
        """Backtesting 결과 출력"""
        print("\n" + "="*90)
        print("REGULATORY BACKTESTING (개선: Basel III 프레임워크 적용)")
        print("="*90)
        
        print("\n【Kupiec POF Test】")
        print(f"  Failures: {pof_results['failures']}")
        print(f"  Failure Rate: {pof_results['failure_rate']:.2%} (Expected: {pof_results['expected_rate']:.2%})")
        print(f"  LR Statistic: {pof_results['lr_statistic']:.4f} (Critical: {pof_results['critical_value']:.4f})")
        print(f"  Result: {'✓ PASS' if pof_results['passes'] else '✗ FAIL'}")
        
        print("\n【Traffic Light Approach】")
        print(f"  Zone: {tl_results['zone']}")
        print(f"  Action: {tl_results['action']}")


class SensitivityAnalysis:
    """
    개선: Sensitivity Analysis 추가
    모델의 주요 하이퍼파라미터에 대한 영향도 분석
    """
    
    @staticmethod
    def dropout_rate_sensitivity(model, X_test: np.ndarray, y_test: np.ndarray,
                                dropout_rates: list = [0.1, 0.2, 0.3]) -> Dict:
        """
        Dropout rate 변화에 따른 성능 변화
        (현재는 고정된 모델이므로 개념 설명만)
        """
        results = {}
        
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS: Dropout Rate Impact")
        print("="*70)
        print("\n⚠️ Note: This shows impact of different dropout rates")
        print("         (Implementation requires model retraining)")
        print(f"\nDropout rates tested: {dropout_rates}")
        print("Expected impact: Higher dropout → Larger epistemic uncertainty")
        
        return results
    
    @staticmethod
    def mc_samples_sensitivity(model, X_test: np.ndarray, 
                             mc_samples_list: list = [10, 50, 100, 200]) -> Dict:
        """
        MC sample 수에 따른 epistemic uncertainty 수렴
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS: MC Samples Impact")
        print("="*70)
        
        results = {}
        
        for n_samples in mc_samples_list:
            X_tensor = torch.FloatTensor(X_test).to('cpu')
            
            model.eval()
            model.train()  # MC Dropout 활성화
            
            mc_preds = model.mc_dropout_forward(X_tensor, n_samples=n_samples)
            epistemic_std = mc_preds.std(axis=1).cpu().numpy()
            
            results[n_samples] = {
                'mean_epistemic': epistemic_std.mean(),
                'std_epistemic': epistemic_std.std()
            }
            
            print(f"{n_samples} samples: "
                  f"Epistemic = {epistemic_std.mean():.6f} "
                  f"(converges as n → ∞)")
        
        return results


def comprehensive_analysis(model, X_test: np.ndarray, y_test: np.ndarray,
                          device: str = 'cpu') -> Dict:
    """
    개선: 종합 분석 (Calibration + Backtesting + Sensitivity)
    """
    print("\n" + "="*90)
    print("COMPREHENSIVE UNCERTAINTY ANALYSIS (IMPROVED)")
    print("="*90)
    
    # 1. Uncertainty Estimation
    estimator = UncertaintyEstimator(model, device)
    uncertainty_results = estimator.estimate_uncertainties(X_test, n_mc_samples=100)
    
    # 2. Calibration Analysis (Multi-confidence)
    calibration_metrics = CalibrationEvaluator.compute_calibration_metrics(
        uncertainty_results['predictions'],
        uncertainty_results['total_std'],
        y_test,
        confidence_levels=[0.68, 0.95, 0.99]  # 개선: 여러 신뢰도
    )
    CalibrationEvaluator.print_calibration_analysis(calibration_metrics)
    
    # 3. Regulatory Backtesting (NEW)
    pof_results = RegulatoryBacktesting.kupiec_pof_test(
        uncertainty_results['predictions'], y_test, confidence=0.95
    )
    tl_results = RegulatoryBacktesting.traffic_light_approach(
        uncertainty_results['predictions'], y_test, confidence=0.95
    )
    RegulatoryBacktesting.print_backtesting_results(pof_results, tl_results)
    
    # 4. Sensitivity Analysis (NEW)
    mc_sensitivity = SensitivityAnalysis.mc_samples_sensitivity(
        model, X_test, mc_samples_list=[10, 50, 100, 200]
    )
    
    return {
        'uncertainties': uncertainty_results,
        'calibration': calibration_metrics,
        'backtesting_pof': pof_results,
        'backtesting_tl': tl_results,
        'sensitivity': mc_sensitivity
    }


def main():
    """Main execution"""
    print("Loading trained model...")
    from model_refined import BayesianVaRNN
    
    model = BayesianVaRNN(input_dim=11, hidden_dim=128, dropout_rate=0.2)
    model.load_state_dict(torch.load('best_bayesian_var_model.pt'))
    
    print("Loading test data...")
    data = np.load('./data/synthetic_data.npz')
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Comprehensive analysis
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = comprehensive_analysis(model, X_val, y_val, device)
    
    return results


if __name__ == '__main__':
    results = main()
