# Hierarchical Q-Learning for Network Threat Detection

A deep reinforcement learning approach to network intrusion detection using hierarchical Q-learning. This project implements a two-stage classification system: a binary classifier for initial threat detection and a multi-class classifier for specific attack categorization.

## Project Overview

### Architecture

1. **Binary Classification Stage**
   - Distinguishes between benign and malicious traffic
   - Fast initial filtering of network traffic
   - Optimized for high recall to minimize false negatives

2. **Multi-Class Classification Stage**
   - Detailed classification of malicious traffic into specific attack types
   - Categories include: DoS/DDoS, Web Attacks, Brute Force, Botnet, etc.
   - Only processes traffic flagged as malicious by the binary classifier

### Key Features

- Deep Q-Network (DQN) implementation with experience replay
- Hierarchical classification approach
- Batch processing support
- Comprehensive preprocessing pipeline
- Evaluation metrics and visualizations
- Support for both full training and quick overfitting tests

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (optional but recommended)
