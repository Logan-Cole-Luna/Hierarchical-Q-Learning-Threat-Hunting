# Hierarchical-Q-Learning-Threat-Hunting

This project implements a hierarchical Q-learning framework for network threat detection. It leverages two agents—a high-level agent for categorizing broad attack types and a low-level agent for identifying specific threat patterns. This approach improves the detection and prioritization of network threats by allowing the model to focus on both general attack classes and fine-grained anomalies.

## Project Overview

### Agents

1. **High-Level Agent:**
   - **Purpose:** Classifies broad attack categories like DoS/DDoS, Web Attacks, and Brute Force.
   - **Features:** Uses high-level network features, such as packet size and flow duration.
   - **Reward Mechanism:** Rewards are given for accurate identification of common attack types and for filtering out benign traffic.

2. **Low-Level Agent:**
   - **Purpose:** Conducts detailed scans to identify specific anomalies within the broad categories, such as DoS subtypes or Brute Force variants.
   - **Features:** Uses more granular features like packet length variance and flag counts.
   - **Reward Mechanism:** Rewards are given based on the accuracy and efficiency of detecting specific threats within the high-level categories.

### Why Q-Learning?

The use of Q-learning provides several advantages for this threat-hunting framework:
- **Adaptability:** Q-learning adapts well to complex and dynamic network environments.
- **Hierarchical Structure:** Supports a structured approach to decision-making, enhancing both efficiency and scalability.
- **Exploration-Exploitation Trade-off:** Balances exploration of new attack types with exploitation of known patterns.
- **Incremental Learning:** Allows real-time, continuous learning to adapt to evolving threats.

### Dataset

The model is trained and evaluated using the **CIC-IDS2017 Intrusion Detection Evaluation Dataset**, a comprehensive dataset containing various attack types and normal traffic for network intrusion detection research.

## Project Structure

- `agents/`: Contains the agent classes (`HighLevelAgent`, `LowLevelAgent`) which define the behavior and learning of each agent.
- `scripts/`: Contains additional scripts, including the `Evaluator.py` for evaluating the agents after training.
- `Trainer.py`: Manages the main training loop, coordinating interactions between agents and the environment.
- `overfit.py`: A script designed to test the Q-networks on a small subset of data, allowing for quick verification and overfitting.
- `replay_buffer.py`: Implements a replay buffer for experience replay, aiding in stable training.
- `RewardCalculator.py`: Contains the reward calculation logic used to incentivize agents based on their performance.

## Running the Project

### Prerequisites

Make sure the following dependencies are installed:
- Python 3.x
- NumPy
- PyTorch
- Scikit-learn
- Matplotlib
- Seaborn

You can install the necessary packages with:

```bash
pip install -r requirements.txt
```

## Preprocess Data
Download dataset from link below and extract files into data folder
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?resource=download

```bash
python scripts/preprocess.py
```

## Running the Main Script
To train both the high-level and low-level agents, execute the following command:

## Running the Overfitting Script
The overfit.py script is designed to overfit the agents on a small subset of the dataset, allowing for quick verification that the Q-networks can learn basic patterns. This can be useful for debugging or validating the initial setup before full training.

To run the overfitting script:

```bash
python overfit.py
```

This script:

1. Loads a subset of the data to train the agents.
2. Monitors training loss, applies early stopping, and logs progress at intervals.
3. Evaluates the agents on the subset data, displaying classification reports, confusion matrices, and optional ROC curves for detailed analysis.
4. Key Considerations
5. Data Preparation: Ensure that the dataset is preprocessed and saved in the appropriate directory structure. Run any necessary preprocessing scripts to prepare the data files.
6. Hyperparameter Tuning: Adjust hyperparameters (e.g., learning rate, epsilon decay) in Train.py and overfit.py for optimal performance on the full dataset or subset.


```bash
python Train.py
```

This script:

1. Loads preprocessed training and test data.
2. Initializes the environment and agents with the specified hyperparameters.
3. Starts training both agents, logging progress periodically.
4. Saves the trained models (high_level_agent.h5 and low_level_agent.h5) upon completion.
5. Evaluates the agents using the test data and prints a detailed classification report.
