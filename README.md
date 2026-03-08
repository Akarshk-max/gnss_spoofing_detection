# GNSS Anti-Spoofing AI Hackathon — NyneOS Technologies 2026

**Team**: [Your Team Name]  
**Round 1 Submission** (Deadline: 8 March 2026, 11:59 PM IST)  
**Model**: XGBoost (Physics-Informed Feature Engineering)  
**Evaluation Metric**: Weighted F1 Score  
**Final Validation Score**: [Insert your Weighted F1 here after training]

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU%20Accelerated-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 1. Problem Understanding

Global Navigation Satellite Systems (GNSS) are the backbone of modern infrastructure, yet they are inherently vulnerable to **spoofing attacks** — where an adversary transmits counterfeit signals to mislead the receiver’s position, velocity, and timing (PVT) solution.  

This hackathon (organised by NyneOS Technologies in collaboration with Kaizen 2026, EES & ARIES IIT Delhi) requires an **AI-driven detection system** that analyses receiver observables to distinguish authentic signals from spoofed ones. The dataset is **tabular** (receiver tracking outputs), highly imbalanced (6:1 ratio), and evaluated on **Weighted F1 Score** to ensure fair performance on the minority spoofed class.

Our solution focuses on **physics-first detection**: we do not treat the problem as a black-box classification task. Instead, we explicitly engineer features that capture the fundamental inconsistencies introduced by spoofers (correlation peak distortion, Doppler–pseudorange mismatch, sudden phase jumps, etc.).

---

## 2. Physics-First Feature Engineering (Core Innovation)

Raw features alone are insufficient. Following GNSS signal-processing literature and the official problem statement, we derived **12 high-signal features** that directly target spoofing mechanisms (lift-off, drag-off, takeover attacks).

### 2.1 Correlator Distortion Metrics (SQM)
Authentic signals maintain symmetry in the correlation function (`EC ≈ LC`). Spoofers distort this peak.

```python
correlator_diff = EC - LC
correlator_ratio = EC / (LC + 1e-6)
prompt_balance = PC / (EC + LC + 1e-6)
PC_magnitude = √(PIP² + PQP²)
