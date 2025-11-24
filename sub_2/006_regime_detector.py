# 006_regime_detector.py
import numpy as np
import pandas as pd
import pickle
import os
import torch

class RegimeDetector:
    """
    논문의 Challenge 3 (Model Uncertainty) 대응.
    Test 시점에 미래의 Regime Label을 보지 않고,
    과거 데이터로 학습된 K-Means를 통해 현재 Regime의 확률 분포 P(s_t)를 추론함.
    """
    def __init__(self, k=4, tau=0.60, base_dir='./processed_data'):
        self.k = k
        self.model_path = os.path.join(base_dir, f'K{k}_tau{tau:.2f}', f'kmeans_model_K{k}.pkl')
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.scaler = data['scaler']
            
        # K-Means의 Centroid 저장
        self.centroids = self.kmeans.cluster_centers_ # (K, n_features)
        print(f"✓ Regime Detector Loaded (K={k})")

    def detect_regime(self, market_features_df, temperature=1.0):
        """
        입력: Market Summary DataFrame (features columns must match training)
        출력: Soft Probabilities (N_samples, K)
        
        Temperature가 낮을수록 Hard Assignment에 가까워짐.
        """
        # Feature 순서 보장 및 Scaling
        # 학습 때 사용한 feature list (preprocessing.py 참조)
        feature_cols = [
            'market_return', 'market_volatility', 'vix',
            'treasury_10y', 'yield_spread',
            'ma_return_5d', 'ma_vol_5d',
            'ma_return_20d', 'ma_vol_20d'
        ]
        
        X = market_features_df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # 거리 계산 (Euclidean Distance)
        # dists: (N_samples, K)
        dists = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids, axis=2)
        
        # Softmax with negative distance
        # 거리가 가까울수록 확률이 높음
        logits = -dists / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        return torch.FloatTensor(probs)

if __name__ == "__main__":
    # 테스트
    try:
        detector = RegimeDetector(k=4, tau=0.60)
        
        # 임시 데이터 로드해서 테스트
        df = pd.read_csv(f'./processed_data/K4_tau0.60/market_regimes_K4.csv', index_col=0)
        sample = df.iloc[-5:] # 마지막 5일치
        
        probs = detector.detect_regime(sample)
        print("\n[Sample Detection]")
        print("Regime Probabilities:\n", probs)
        print("Sum check:", probs.sum(dim=1))
    except Exception as e:
        print(f"Error: {e}")
        print("먼저 005_preprocessing.py를 실행하여 K4_tau0.60 데이터를 생성해야 합니다.")