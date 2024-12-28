# Hierarchical Q-Learning for Network Threat Detection

A deep reinforcement learning approach to network intrusion detection using hierarchical Q-learning. This project implements a two-stage classification system: a binary classifier for initial threat detection and a multi-class classifier for specific attack categorization.

## Project Overview

### Performance Metrics

#### Binary Classification Results
- Overall Accuracy: 99.977%
- Benign Traffic (1,935,399 samples):
  - Precision: 99.978%
  - Recall: 99.996%
  - F1-Score: 99.987%
- Malicious Traffic (242,405 samples):
  - Precision: 99.970%
  - Recall: 99.822%
  - F1-Score: 99.896%

#### Multi-Class Classification Results
- Overall Accuracy: 99.684%
- Attack-specific Performance:
  - Botnet (28,907 samples):
    - Precision: 99.787%
    - Recall: 98.862%
    - F1-Score: 99.322%
  - Brute-force (18,820 samples):
    - Precision: 98.703%
    - Recall: 99.915%
    - F1-Score: 99.306%
  - DDoS (155,191 samples):
    - Precision: 99.717%
    - Recall: 100.000%
    - F1-Score: 99.858%
  - DoS (39,314 samples):
    - Precision: 99.959%
    - Recall: 99.372%
    - F1-Score: 99.665%
  - Web Attacks (173 samples):
    - Requires additional training data

### Architecture

1. **Binary Classification Stage**
   - Distinguishes between benign and malicious traffic
   - Fast initial filtering of network traffic
   - Optimized for high recall to minimize false negatives
   - Uses XGBoost-based classification with SHAP explanations
   - Achieves 99.97% accuracy on test data

2. **Multi-Class Classification Stage**
   - Detailed classification of malicious traffic into specific attack types
   - Categories: DoS/DDoS, Web Attacks, Brute Force, Botnet attacks
   - Only processes traffic flagged as malicious by binary classifier
   - Achieves 99.68% overall accuracy on attack classification

## Technical Architecture

### Agent Design

1. **Binary Classification Agent**
   - XGBoost-based classifier optimized for fast initial screening
   - Features:
     - TreeExplainer integration for SHAP value generation
     - Real-time feature importance tracking
     - Configurable class weight balancing
     - Automated feature selection and preprocessing
   - Decision Process:
     - Input: Network flow features (packet length, protocol, flow metrics)
     - Output: Binary classification (benign/malicious)
     - Confidence scoring with SHAP-based explanations
     - Threshold-based decision boundaries with configurable sensitivity
   - Binary Agent Local Structure:
      - XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping=10,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric='logloss', feature_types=None, gamma=None,
              grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=100, n_jobs=None,
              num_parallel_tree=None, ...)
2. **Multi-Class RL Agent**
   - Hierarchical Q-Learning architecture
   - State Space:
     - Network flow features
     - Binary agent confidence scores
     - Historical context window
   - Action Space:
     - Attack type classification (DoS/DDoS, Botnet, etc.)
     - Confidence threshold adjustment
     - Feature importance weighting
   - Reward Structure:
     - Immediate reward for correct classification
     - Penalty for false positives/negatives
     - Time-based efficiency bonus
     - Confidence calibration rewards

   QNetwork(
   (shared_layers): Sequential(
      (0): Linear(in_features=40, out_features=128, bias=True)
      (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=128, out_features=64, bias=True)
      (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
   )
   (value_fc): Linear(in_features=64, out_features=1, bias=True)
   (advantage_fc): Linear(in_features=64, out_features=5, bias=True)
   )

### Feature Engineering

1. **Network Flow Features**
   - Temporal Features:
     - Flow duration
     - Inter-arrival time statistics
     - Time-based traffic patterns
   - Volume Features:
     - Bytes/packets per second
     - Packet size distribution
     - Flow volume metrics
   - Protocol Features:
     - Protocol type encoding
     - Port number analysis
     - Flag combinations
   - Behavioral Features:
     - Connection patterns
     - Service request frequencies
     - Error rate analysis

### Learning Process

1. **Training Pipeline**
   - Two-stage training approach:
     1. Binary Agent Training:
        - XGBoost optimization with cross-validation
        - SHAP-guided feature selection
        - Class weight optimization
        - Model performance monitoring
     2. Multi-Class Agent Training:
        - Experience replay buffer management
        - Epsilon-greedy exploration strategy
        - Dynamic Q-value updates
        - Reward shaping based on confidence

2. **Optimization Techniques**
   - Batch processing optimization
   - GPU acceleration support
   - Memory-efficient data handling
   - Dynamic batch sizing
   - Early stopping with validation monitoring

### Real-time Operations

1. **Inference Pipeline**

### System Benefits

1. **High Accuracy**
   - Binary classification accuracy of 99.977% across 2.17M samples
   - Multi-class classification accuracy of 99.684% for attack types
   - Extremely low false positive rate (0.022% for benign traffic)

2. **Scalability**
   - Efficient two-stage processing pipeline
   - Handles large-scale traffic analysis (2.17M+ samples tested)
   - Optimized for real-time classification

3. **Interpretability**
   - SHAP-based explanation system
   - Feature importance tracking
   - Decision confidence metrics
   - Misclassification analysis

4. **Adaptability**
   - Dynamic threshold adjustment
   - Continuous learning capabilities
   - Performance monitoring and adaptation
