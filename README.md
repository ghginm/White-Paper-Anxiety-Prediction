# White Paper: Anxiety Prediction ([Read the Paper (PDF)](Developing_A_Machine_Learning_Model_to_Predict_Anxiety_Levels.pdf))

This project explores the use of **machine learning** to predict **anxiety levels** using **mobile sensing data**. Our aim is to improve early detection of anxiety disorders, leveraging real-world behavioral patterns instead of traditional clinical assessments.

## The Challenge

**Problem:**
Traditional medical tests struggle with **low disease prevalence**, resulting in **high false positive rates** that burden healthcare systems.

**Goal:**
Develop a **reliable ML model** that can **identify anxiety disorders** with high precision, especially under **class imbalance** conditions.

## Data Overview

* **Source:** `PROSIT` app
* **Participants:** 161 individuals
* **Duration:** 2 weeks
* **Observations:** 322 data points
* **Features:**

  * Mobile phone usage
  * Sleep patterns
  * GPS mobility
  * Self-reported well-being
* **Class Imbalance:** \~1:4 (Anxiety vs Healthy)

## Understanding Model Probabilities

In healthcare, **well-calibrated probabilities** are crucial. They aid in informed decision-making, especially when dealing with rare conditions.

### Why Probabilities Matter:

* Allow for **threshold-independent** classification
* Reflect the **true prior** in imbalanced datasets
* Help understand the **likelihood** of actual diagnosis

### Example Calculation

Sure! Here’s the updated example snippet with the full formula included:

> If:
>
> * Prevalence (prior) = 0.05
> * True Positive Rate (TPR) = 0.9
> * False Positive Rate (FPR) = 0.1
>
> Then:
>
> $$
> P(\text{anxiety} \mid +) = \frac{P(+ \mid \text{anxiety}) \ P(\text{anxiety})}{P(+ \mid \text{anxiety}) \ P(\text{anxiety}) + P(+ \mid \text{healthy}) \ P(\text{healthy})}
> $$
>
> Numerically:
>
> $$
> P(\text{anxiety} \mid +) = \frac{0.9 \times 0.05}{0.9 \times 0.05 + 0.1 \times 0.95} \approx 0.346
> $$

This means that **even a strong test yields modest confidence** under low prevalence.

* As the **prior increases**, posterior probabilities improve
* Even with a **99% TPR**, a low prior suppresses confidence

## Traditional Tests vs Machine Learning

To some degree, ML models are more robust to this issue because, if the training set is representative of operational conditions, the classifier’s output will provide a good estimate of the probability of class membership under those conditions, i.e., the posterior probability $P(\text{anxiety} \mid +).$ This also depends on the cost function, though to a lesser degree. However, this robustness is not guaranteed, particularly when the training procedure is modified to account for data imbalance. Such modifications can improve accuracy for the minority class but may result in output probabilities that are not well calibrated. Furthermore, having a representative training set is essential yet challenging, especially in the context of small datasets, which are often supplemented with synthetic data.

In our view, it is crucial to keep these considerations in mind, even if a given model results in a low out-of-sample error. For example, while being an accurate classifier, Random Forest (RF) is known to produce poorly calibrated probabilities, which can mislead downstream decision-making processes, especially when these probabilities are interpreted as confidence scores. This limitation becomes particularly problematic in healthcare applications, where decisions often depend on reliable probabilistic estimates rather than simple binary predictions.

| Aspect                   | Traditional Tests | ML Models                     |
| ------------------------ | ----------------- | ----------------------------- |
| Prior Awareness          | ❌                 | ✅                             |
| Threshold Flexibility    | ❌                 | ✅                             |
| Probability Calibration  | ❌                 | ⚠️ Depends on model           |
| Class Imbalance Handling | Manual            | Built-in options (with risks) |

**Important:** Oversampling or balancing may improve performance **but distort probability estimates**.

### Our Results:

* Probabilities were **well-calibrated**
* Posterior matched **true class distribution**

Proper probabilities suggest that the model has effectively incorporated the prior, which is crucial in case of substantial class imbalance. It also indicates that the classifier is less reliant on a specific classification threshold.

<div align="center">
  <img alt="Probability Calibration" src="figures/probabilities.png" width="650">
</div>

## Exploring Model Weights

We analyzed **first-layer weights** to estimate **feature importance** across 15 runs, loosely interpreting them as some form of feature importance. For more information refer to the paper.

### Caveats:

* Early weights are affected by deeper layers
* Neural networks are **nonlinear**, making direct interpretation difficult

### Key Findings:

* **Alcohol consumption** was the most influential feature
* Weight patterns aligned with **AdamW decay dynamics**. We were interested in either those that increased or decreased but substantially (see the figure below)

<div align="center">
  <img alt="Feature Importance" src="figures/var_imp.png" width="650">
</div>

<div align="center">
  <img alt="Weight Visualization" src="figures/weights.png" width="650">
</div>

## Summary

This project demonstrates the **potential of ML in mental health diagnostics**, especially in navigating challenges like:

* Class imbalance
* Low disease prevalence
* Probability calibration
