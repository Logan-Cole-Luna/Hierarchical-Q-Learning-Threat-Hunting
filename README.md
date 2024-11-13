# Hierarchical-Q-Learning-Threat-Hunting

Two Agents:

- High level incentivized to classify attack types accurately, while the low-level agent is rewarded for identifying detailed patterns, enhancing overall model efficiency in prioritizing and detecting network threats
- High Level Agent:
    - Prioritizes broader attack categories, such as DoS/DDoS, Web Attacks, or Brute Force,
    - Uses features like packet size and flow duration, and is rewarded for quickly identifying common attack types
    - Will filter out benign traffic
- Low Level Agent:
    - Performs focused scans, investigating specific anomalies with detailed metrics
    - Packet length variance, flag counts, and is rewarded based on detection accuracy and efficiency.
    - DoS Subtypes, Brute Force variants, etc

Why Q Learning:

- Adaptability to Complex, changing Landscapes
- Hierarchical Structure and Sequential Decision-Making
- Exploration-Exploitation Trade-off
- Scalability and Efficiency
- Supports Incremental Learning in Real Time

Dataset:

- **Intrusion detection evaluation dataset (CIC-IDS2017)**