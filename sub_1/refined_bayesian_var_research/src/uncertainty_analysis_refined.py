# ============================================================================
# uncertainty_analysis_refined.py (FINAL FIX)
# Stage 4: Enhanced uncertainty analysis with backtesting
# ALL ISSUES RESOLVED:
# 1. MC Dropout 강제 활성화 + 검증
# 2. Gradient detach 추가
# 3. Calibration 계산 완전 수정
# 4. Backtesting 로직 완전 수정
# ============================================================================

import torch
import numpy as np
from typing import Dict
from scipy.stats import chi2


def comprehensive_analysis(model, X_test: np.ndarray, y_test: np.ndarray, 
                          device: str = 'cpu') -> Dict:
    """
    완전히 수정된 불확실성 분석
    """
    print("\n" + "="*90)
    print("COMPREHENSIVE UNCERTAINTY ANALYSIS (FINAL FIX)")
    print("="*90)
    
    model = model.to(device)
    X_tensor = torch.FloatTensor(X_test[:10000]).to(device)  # 샘플링 (빠른 실행)
    y_sample = y_test[:10000]
    
    # ========================================================================
    # FIX 1: MC Dropout inference with FORCED train mode
    # ========================================================================
    print("Running MC Dropout inference...")
    
    # 강제로 모든 dropout 활성화
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Force dropout to be active
    
    n_samples = 100
    mc_predictions = []
    aleatoric_stds = []
    
    for i in range(n_samples):
        with torch.no_grad():
            var_pred, aleatoric_std, _ = model(X_tensor)
            # FIX 2: detach() 추가
            mc_predictions.append(var_pred.detach().cpu().numpy())
            aleatoric_stds.append(aleatoric_std.detach().cpu().numpy())
    
    mc_predictions = np.array(mc_predictions).squeeze()
    aleatoric_stds = np.array(aleatoric_stds).squeeze()
    
    # Shape 확인
    if mc_predictions.ndim == 1:
        mc_predictions = mc_predictions.reshape(n_samples, -1)
    if aleatoric_stds.ndim == 1:
        aleatoric_stds = aleatoric_stds.reshape(n_samples, -1)
    
    # Epistemic uncertainty (variation across MC samples)
    epistemic_std = np.std(mc_predictions, axis=0)
    
    # Aleatoric uncertainty (average of predicted uncertainties)
    aleatoric_std_mean = np.mean(aleatoric_stds, axis=0)
    
    # Total uncertainty
    total_std = np.sqrt(epistemic_std**2 + aleatoric_std_mean**2)
    
    # Mean prediction
    mean_predictions = np.mean(mc_predictions, axis=0)
    
    print(f"✓ Epistemic uncertainty: {epistemic_std.mean():.6f} ± {epistemic_std.std():.6f}")
    print(f"✓ Aleatoric uncertainty: {aleatoric_std_mean.mean():.6f} ± {aleatoric_std_mean.std():.6f}")
    print(f"✓ Total uncertainty: {total_std.mean():.6f} ± {total_std.std():.6f}")
    
    if epistemic_std.mean() < 1e-6:
        print("\n⚠️ WARNING: Epistemic uncertainty is near zero!")
        print("   This suggests MC Dropout may not be working properly.")
        print("   Possible causes:")
        print("   - Dropout layers not active during inference")
        print("   - Model overfitted to point estimates")
        print("   - Need higher dropout rate or more stochastic layers")
    
    # ========================================================================
    # FIX 3: Calibration analysis (완전 수정)
    # ========================================================================
    print("\n" + "="*90)
    print("CALIBRATION ANALYSIS (FIXED)")
    print("="*90)
    
    confidence_levels = [0.68, 0.95, 0.99]
    z_scores = [1.0, 1.96, 2.576]
    
    print(f"\n{'Confidence':<15} {'Target':<10} {'Achieved':<10} {'Error':<10} {'Status':<10}")
    print("-" * 90)
    
    calibration_results = {}
    for conf, z in zip(confidence_levels, z_scores):
        # Calculate confidence intervals using total uncertainty
        lower = mean_predictions - z * total_std
        upper = mean_predictions + z * total_std
        
        # Check coverage
        in_interval = ((y_sample >= lower) & (y_sample <= upper))
        achieved_coverage = in_interval.mean()
        
        error = abs(achieved_coverage - conf)
        
        # Status based on error
        if error < 0.02:
            status = "✓ Excellent"
        elif error < 0.05:
            status = "○ Good"
        elif error < 0.10:
            status = "△ Fair"
        else:
            status = "✗ Poor"
        
        print(f"{conf:.0%}             {conf:.0%}        {achieved_coverage:.0%}       {error:.4f}     {status}")
        
        calibration_results[f'{conf:.0%}'] = {
            'target': conf,
            'achieved': achieved_coverage,
            'error': error
        }
    
    avg_error = np.mean([r['error'] for r in calibration_results.values()])
    
    if avg_error < 0.02:
        print(f"\n✓ EXCELLENT: Average calibration error = {avg_error:.4f} < 2%")
    elif avg_error < 0.05:
        print(f"\n○ GOOD: Average calibration error = {avg_error:.4f} < 5%")
    else:
        print(f"\n✗ POOR: Average calibration error = {avg_error:.4f} > 5%")
        print("   Model's uncertainty estimates are not well-calibrated.")
    
    # ========================================================================
    # FIX 4: Regulatory backtesting (완전 수정)
    # ========================================================================
    print("\n" + "="*90)
    print("REGULATORY BACKTESTING (BASEL III - FIXED)")
    print("="*90)
    
    # VaR 95% backtesting
    confidence = 0.95
    alpha = 1 - confidence  # 5%
    
    # VaR violations: actual loss exceeds predicted VaR
    # VaR는 음수 (손실), y_sample < mean_predictions means actual loss > predicted VaR
    violations = (y_sample < mean_predictions).sum()
    n_total = len(y_sample)
    violation_rate = violations / n_total
    
    print(f"\n【Basic Statistics】")
    print(f"  Total observations: {n_total}")
    print(f"  VaR violations: {violations}")
    print(f"  Violation rate: {violation_rate:.2%} (Expected: {alpha:.0%})")
    
    # Kupiec POF test
    if violations > 0 and violations < n_total:
        p = violations / n_total
        likelihood_ratio = -2 * (
            violations * np.log(alpha) + (n_total - violations) * np.log(1 - alpha) -
            violations * np.log(p) - (n_total - violations) * np.log(1 - p)
        )
    else:
        likelihood_ratio = np.inf if violations == n_total else 0.0
    
    critical_value = chi2.ppf(0.95, df=1)  # 3.841
    pof_pass = likelihood_ratio < critical_value
    
    print(f"\n【Kupiec POF Test (Proportional-of-Failure)】")
    print(f"  LR Statistic: {likelihood_ratio:.4f}")
    print(f"  Critical value (95%): {critical_value:.4f}")
    print(f"  Result: {'✓ PASS' if pof_pass else '✗ FAIL'}")
    
    if not pof_pass:
        print(f"  → VaR model is NOT accurate at 95% confidence level")
    
    # Basel III Traffic Light Approach
    # Green: ≤4 violations, Yellow: 5-9, Red: ≥10 (for 250 observations)
    # Scale to current sample size
    scaled_green = int(4 * n_total / 250)
    scaled_yellow = int(9 * n_total / 250)
    
    if violations <= scaled_green:
        zone = "🟢 Green Zone"
        action = "No action required - Model performs well"
    elif violations <= scaled_yellow:
        zone = "🟡 Yellow Zone"
        action = "Model requires monitoring and possible revision"
    else:
        zone = "🔴 Red Zone"
        action = "Model must be rejected/revised immediately"
    
    print(f"\n【Basel III Traffic Light Approach】")
    print(f"  Violation threshold (Green): ≤{scaled_green}")
    print(f"  Violation threshold (Yellow): ≤{scaled_yellow}")
    print(f"  Actual violations: {violations}")
    print(f"  Zone: {zone}")
    print(f"  Action: {action}")
    
    # ========================================================================
    # Sensitivity analysis (with detach)
    # ========================================================================
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: MC Samples Impact")
    print("="*70)
    
    for n_mc in [10, 50, 100, 200]:
        mc_temp = []
        for _ in range(n_mc):
            with torch.no_grad():
                var_pred, _, _ = model(X_tensor)
                mc_temp.append(var_pred.detach().cpu().numpy())
        
        mc_temp = np.array(mc_temp).squeeze()
        if mc_temp.ndim == 1:
            mc_temp = mc_temp.reshape(n_mc, -1)
        
        epistemic_temp = np.std(mc_temp, axis=0).mean()
        print(f"{n_mc:3d} samples: Epistemic = {epistemic_temp:.6f}")
    
    return {
        'uncertainties': {
            'epistemic_std': epistemic_std,
            'aleatoric_std': aleatoric_std_mean,
            'total_std': total_std,
            'predictions': mean_predictions,
            'mc_predictions': mc_predictions
        },
        'calibration': calibration_results,
        'backtesting': {
            'violations': violations,
            'violation_rate': violation_rate,
            'lr_statistic': likelihood_ratio,
            'pof_pass': pof_pass,
            'zone': zone,
            'avg_calibration_error': avg_error
        }
    }


def main():
    """Main execution"""
    import os
    from model_refined import BayesianVaRNN
    
    print("Loading data...")
    data = np.load('../data/synthetic_data.npz')
    X_val = data['X_val']
    y_val = data['y_val']
    
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BayesianVaRNN(input_dim=11, hidden_dim=128, dropout_rate=0.2)
    
    model_path = 'best_bayesian_var_model.pt'
    if not os.path.exists(model_path):
        model_path = '../best_bayesian_var_model.pt'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"[OK] Model loaded from {model_path}")
    else:
        print(f"[WARNING] Using untrained model")
    
    print("\nRunning comprehensive analysis...")
    results = comprehensive_analysis(model, X_val, y_val, device)
    
    return results


if __name__ == '__main__':
    results = main()
