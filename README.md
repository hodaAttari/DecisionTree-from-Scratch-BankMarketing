# DecisionTree-from-Scratch-BankMarketing

An educational project implementing the **ID3 Decision Tree algorithm from scratch**, applied to the **Bank Marketing Dataset** (Portugal). The notebook reconstructs tree learning using **entropy** and **information gain**, providing full transparency over each computational step.

---

## Overview
This project rebuilds the ID3 (Iterative Dichotomiser 3) algorithm entirely in Python without relying on `DecisionTreeClassifier`.  
At each split, the feature yielding maximum **information gain**—the largest entropy reduction—is selected to partition the dataset.  
Recursion continues until nodes become pure or stopping thresholds are met.

$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$

---

## Structure
├── hw1.ipynb → implementation and analysis

├── bank.csv → dataset (~11k samples, 17 features)

├── report.html → exploratory profiling (YData Profiling)

└── README.md → documentation

---

## Implementation
### recursively split using information gain (ID3)

### traverse the learned tree to classify samples
Includes:

- Custom entropy and information gain computation
- Recursive tree building
- scikit-learn API compatibility
- Visual tree tracing and node statistics
- Dataset: Real marketing data of banking clients contacted for term deposits.

Features include demographic info (age, education, marital), financial metrics (balance, loan, housing), and communication details (contact, duration, campaign, etc.).

Target variable deposit indicates whether the client subscribed (“yes”/“no”).

Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn ydata-profiling
```
### Learning Outcomes
Understanding and reproducing:

- Entropy and information gain as measures of uncertainty reduction
- Recursive model construction logic
- Transparent decision boundaries and interpretability
- Exploratory data profiling and correlation insight
