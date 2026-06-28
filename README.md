# Aviation Parts Repairability Classifier

A machine learning proof-of-concept that predicts whether an aircraft part should be classified as **repairable** or **unrepairable** based on its attributes — automating a manual classification task common in aviation MRO (Maintenance, Repair, and Overhaul) systems.

> **Note:** This project uses 100% synthetic, randomly generated data. No real company data, part numbers, or proprietary information is used or represented anywhere in this repository.

---

## Why This Project

In aviation MRO operations, whether a part is repairable or treated as a one-time-use consumable is typically decided by finance and planning teams, based on factors like cost, manufacturer approval status, and material type. Once that decision is made, the resulting status is manually applied to each part in the system — often across long lists, one part at a time.

In my role, I am the one applying those updates. Watching this process repeatedly raised a natural question: if the underlying decision follows a consistent pattern, could that pattern be learned? This project is a proof-of-concept exploring exactly that — not to replace the decision-making process, but to test whether the criteria behind it are predictable from the part's attributes alone.

The labeling logic used here is grounded in real MRO industry practices: high-value, manufacturer-approved, non-tool components are more likely to be classified as repairable, while low-cost, unapproved, or tool items are more likely treated as consumables — consistent with standard rotable/repairable/consumable classification practices used across the industry.


---

## Dataset

500 synthetic aircraft parts were generated with the following attributes:

| Feature | Description |
|---|---|
| `material_type` | Standard Part, Tool, Component, Raw Material, or Non-aircraft Material |
| `has_pma` | Whether the part has Parts Manufacturer Approval |
| `is_tool` | Whether the part is classified as a tool rather than an aircraft part |
| `is_fixed_asset` | Whether the part is tracked as a fixed asset |
| `ata_chapter` | Aircraft system classification code |
| `sales_price_sar` | Unit price in Saudi Riyals |
| `weight_kg` | Part weight |

**Realistic noise was deliberately introduced** to avoid an artificially perfect dataset:
- 8% of labels were randomly flipped, simulating human classification inconsistency
- 5% of manufacturer-approval values were marked missing and then imputed, simulating incomplete real-world records

---

## Models & Results

Two models were trained and compared on identical train/test splits:

| Model | Accuracy |
|---|---|
| **Random Forest** | **88%** |
| Logistic Regression | 79% |

Random Forest outperformed Logistic Regression by 9 percentage points. This gap is likely explained by conditional relationships in the labeling logic — for example, a part being classified as a "tool" overrides its cost or approval status entirely. Tree-based models naturally capture this kind of conditional, rule-like logic, while linear models assume more uniform relationships across features.

**Detailed performance (Random Forest):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Unrepairable | 0.90 | 0.93 | 0.91 |
| Repairable | 0.84 | 0.79 | 0.81 |

The model shows stronger recall for unrepairable parts, likely reflecting class imbalance in the dataset (more unrepairable than repairable examples) — a common and realistic challenge in MRO data.

---

## Visualizations

- `feature_importance.png` — which part attributes most influenced the model's decisions
- `confusion_matrix.png` — a breakdown of correct vs. incorrect predictions on the test set

---

## Project Structure

```
aviation-parts-repairability-classifier/
├── aviation_classifier.ipynb   # Full notebook: data generation, training, evaluation
├── feature_importance.png      # Feature importance chart
├── confusion_matrix.png        # Model evaluation visualization
└── README.md
```

---

## How It Works (Pipeline Summary)

1. Generate synthetic part data with realistic attributes
2. Apply a domain-informed scoring rule to assign repairable/unrepairable labels
3. Introduce label noise and missing data to simulate real-world imperfection
4. Split data into training (80%) and testing (20%) sets
5. Train and compare Random Forest and Logistic Regression models
6. Evaluate using accuracy, precision, recall, and F1-score
7. Visualize feature importance and prediction accuracy
8. Provide an interactive function to predict on new, hypothetical parts

---

## Future Extensions

- Extend from binary classification to the full material class taxonomy (Rotable / Repairable / Consumable)
- Train on real (anonymized) MRO datasets where available
- Deploy as a simple web interface for live predictions
- Add cross-validation for more rigorous performance estimates

---

## Author

Built by Raghad Baeshen — IT Specialist in aviation MRO, exploring how AI can support enterprise and operational decision-making.

[LinkedIn](https://www.linkedin.com/in/raghadbaeshen)
