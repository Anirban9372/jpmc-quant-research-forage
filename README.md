# JPMorgan Chase — Quantitative Research Job Simulation

[![Forage](https://img.shields.io/badge/Forage-JPMorgan%20Chase-blue?style=flat-square)](https://www.theforage.com)
[![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python)](https://python.org)
[![Completed](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)](https://github.com)
[![Certificate](https://img.shields.io/badge/Certificate-July%202026-orange?style=flat-square)](https://github.com)

A complete implementation of JPMorgan Chase's Quantitative Research virtual experience program on Forage. The simulation covers two parallel workstreams: **commodity derivatives pricing** and **credit risk modeling** — both core functions of a real quantitative research team.

---

## Overview

| Task | Domain | Problem | Method |
|------|--------|---------|--------|
| 1 | Commodity Desk | Extrapolate monthly gas prices to any date | Sinusoidal regression + linear trend |
| 2 | Commodity Desk | Price a natural gas storage contract | Cash flow model with physical constraints |
| 3 | Retail Banking Risk | Predict borrower probability of default | Logistic regression + Expected Loss (Basel III) |
| 4 | Retail Banking Risk | Optimally bucket FICO scores | Dynamic programming + log-likelihood maximization |

---

## The Business Story

### Commodity Desk (Tasks 1 & 2)

A VP named **Alex** wants to trade natural gas storage contracts — agreements where a client buys cheap summer gas, stores it, and sells it at higher winter prices. Before any contract can be quoted, the desk needs two things:

1. **A price model** — market data feeds only provide one price per month, but contracts specify exact injection and withdrawal dates. Task 1 builds a continuous price curve that estimates gas prices on any date.

2. **A contract pricer** — once the price curve exists, Task 2 uses it to compute the net value of any storage contract, accounting for purchase costs, storage fees, injection and withdrawal costs, and physical capacity constraints.

### Retail Banking / Risk Team (Tasks 3 & 4)

An associate named **Charlie** is analyzing the bank's loan book. Default rates are higher than expected, and regulation requires the bank to hold capital reserves proportional to expected losses. Two models are needed:

3. **A PD model** — given a borrower's characteristics (income, FICO score, outstanding debt, etc.), predict their probability of default and compute expected loss under the Basel III formula: `EL = PD × LGD × EAD`.

4. **A FICO bucketer** — Charlie's mortgage model requires categorical inputs. Task 4 finds the optimal way to split the continuous 300–850 FICO range into *n* discrete risk buckets, using dynamic programming to maximize log-likelihood — meaning each bucket has the most distinct and internally consistent default rate possible.

---

## Task 1 — Natural Gas Price Analysis

**File:** `notebook1.ipynb` | **Data:** `Nat_Gas.csv`

### The Model

Natural gas prices have two components: a long-term upward drift and a predictable annual seasonal cycle (winter demand spike, summer dip). Both are captured in a single equation:

$$P(t) = A + Bt + C_1 \sin\left(\frac{2\pi t}{12}\right) + C_2 \cos\left(\frac{2\pi t}{12}\right)$$

The key mathematical trick: instead of fitting $C \sin(\omega t + \phi)$ directly (which is nonlinear in $\phi$), we use the angle addition identity to rewrite it as $C_1 \sin(\omega t) + C_2 \cos(\omega t)$, making all parameters linear and fittable via `scipy.optimize.curve_fit`.

### Fitted Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| A | 10.1349 | Base price at t=0 (Oct 2020) |
| B | +0.0456 | Price rises ~$0.046/month (~$0.55/year) |
| C (amplitude) | 0.691 | ±$0.69 seasonal swing around trend |

### Key Output

```python
estimate_price('2024-06-15')   # → 11.51 (summer, below trend)
estimate_price('2024-01-15')   # → 12.54 (winter, above trend)
estimate_price('2025-06-30')   # → 12.12 (1-year extrapolation)
```

---

## Task 2 — Storage Contract Pricing

**File:** `notebook2.ipynb` | **Data:** `Nat_Gas__part2.csv`

### Cash Flow Model

$$V = \underbrace{\sum P_{\text{sell}}(t_i) \cdot V_i}_{\text{revenue}} - \underbrace{\sum P_{\text{buy}}(t_j) \cdot V_j}_{\text{purchase}} - \underbrace{r_s \cdot d \cdot V}_{\text{storage}} - \underbrace{r_i \cdot V_{\text{in}}}_{\text{injection}} - \underbrace{r_w \cdot V_{\text{out}}}_{\text{withdrawal}}$$

Physical constraints enforced:
- Cannot inject more than remaining storage capacity: `volume = min(rate, max_storage − current)`
- Cannot withdraw more than currently stored: `volume = min(rate, current)`

### Sample Output

```python
# Buy summer 2023, sell winter 2023/24, realistic costs
price_contract(
    injection_dates   = ['2023-06-30', '2023-07-31'],
    withdrawal_dates  = ['2023-12-31', '2024-01-31'],
    injection_rate    = 1_000_000,
    withdrawal_rate   = 1_000_000,
    max_storage       = 3_000_000,
    storage_cost_per_day      = 0.001,
    injection_cost_per_mmbtu  = 0.005,
    withdrawal_cost_per_mmbtu = 0.005
)
# → Positive value: winter prices exceed summer + all costs
```

---

## Task 3 — Credit Risk / Probability of Default

**File:** `notebook_3.ipynb` | **Data:** `Task_3_and_4_Loan_Data.csv`

### Dataset

10,000 borrowers | 6 features | 18.5% default rate

| Feature | Correlation with Default |
|---------|--------------------------|
| credit_lines_outstanding | +0.863 (strongest) |
| total_debt_outstanding | +0.759 |
| fico_score | −0.325 |
| years_employed | −0.285 |
| loan_amt_outstanding | +0.099 |
| income | +0.016 |

### Model

Logistic regression with standard train/test split (80/20, stratified) and StandardScaler.

$$P(\text{default}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}$$

**ROC-AUC: 1.0000** — Note: this reflects a synthetic dataset with a clean decision boundary, not real-world performance (real credit models achieve 0.70–0.85).

### Expected Loss Function

```python
# Basel III: EL = PD × LGD × EAD
expected_loss(credit_lines_outstanding=5,
              loan_amt_outstanding=5000,
              total_debt_outstanding=20000,
              income=25000,
              years_employed=1,
              fico_score=480)
# Probability of Default : 1.0000
# Expected Loss          : $4,500.00

expected_loss(credit_lines_outstanding=0,
              loan_amt_outstanding=5000,
              total_debt_outstanding=3000,
              income=90000,
              years_employed=8,
              fico_score=780)
# Probability of Default : 0.0000
# Expected Loss          : $0.00
```

---

## Task 4 — FICO Score Bucketing (Dynamic Programming)

**File:** `notebook4.ipynb` | **Data:** `Task_3_and_4_Loan_Data.csv`

### The Optimization Problem

Find *n* bucket boundaries that maximize total Bernoulli log-likelihood:

$$\mathcal{L} = \sum_{i=1}^{n} \left[ k_i \log p_i + (n_i - k_i) \log(1 - p_i) \right]$$

where $n_i$ = borrowers in bucket $i$, $k_i$ = defaults, $p_i = k_i / n_i$.

### Why Dynamic Programming

Brute force for 5 buckets over 374 unique FICO values: $\binom{374}{4} \approx 760$ million combinations.  
DP recurrence: $dp[i][j] = \max_{m < i}(dp[m][j-1] + \text{bucket\_ll}(m+1, i))$  
Complexity: $O(N^2 \cdot k)$ where $N=374$, $k=5$ → ~700,000 operations.

### Optimal Boundaries (5 buckets)

| Rating | FICO Range | Borrowers | Default Rate |
|--------|-----------|-----------|--------------|
| 5 (Worst) | 408 – 521 | 301 | **66.1%** |
| 4 | 521 – 581 | 1,407 | 38.1% |
| 3 | 581 – 641 | 3,438 | 20.4% |
| 2 | 641 – 697 | 3,197 | 10.5% |
| 1 (Best) | 697 – 850 | 1,657 | **4.6%** |

```python
fico_to_rating(420)  # → 5 (worst)
fico_to_rating(540)  # → 4
fico_to_rating(610)  # → 3
fico_to_rating(670)  # → 2
fico_to_rating(750)  # → 1 (best)
```

---

## Repository Structure

```
jpmc-quant-research-forage/
├── notebook1.ipynb          # Task 1: Price analysis & extrapolation
├── notebook2.ipynb          # Task 2: Storage contract pricing
├── notebook_3.ipynb         # Task 3: Credit risk / PD modeling
├── notebook4.ipynb          # Task 4: FICO bucketing via DP
├── Nat_Gas.csv              # Monthly gas prices Oct 2020 – Sep 2024
├── Nat_Gas__part2.csv       # Gas price data for Task 2
├── Task_3_and_4_Loan_Data.csv  # 10,000 borrower records
└── README.md
```

---

## Skills Demonstrated

- **Time series modeling** — sinusoidal regression, trend decomposition, forward extrapolation
- **Derivatives pricing** — cash flow modeling, physical constraint handling, cost structure design
- **Credit risk** — logistic regression PD model, Basel III EL formula (PD × LGD × EAD)
- **Dynamic programming** — optimal interval partitioning, prefix sums, backtracking
- **Statistical inference** — maximum likelihood estimation, Bernoulli log-likelihood, ROC-AUC
- **ML best practices** — stratified train/test split, StandardScaler (fit on train only), data leakage prevention
- **Python** — pandas, numpy, scipy.optimize, sklearn, matplotlib

---

## Certificate

**JPMorgan Chase & Co. Quantitative Research Virtual Experience Program** — Forage, July 2026

---

*Completed by Anirban Pal | B.Tech Information Technology | IIEST Shibpur | Class of 2029*
