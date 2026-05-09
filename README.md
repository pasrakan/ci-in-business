# Causal Inference for Business Decision Making — Reference Code

Reference code accompanying the course *Causal Inference for Business Decision Making*.

Each session has a paired `sessionN.py` and `sessionN.R` file with end-to-end runnable examples.

## Layout

```
.
├── 26S/                    # Spring 2026 session scripts
│   ├── session3.{py,R}     # Randomized experiments / A-B testing
│   ├── session4.{py,R}     # Matching methods & propensity scores
│   ├── session5.{py,R}     # IPW & doubly robust methods
│   ├── session6.{py,R}     # Regression-based adjustment
│   ├── session7.{py,R}     # Instrumental variables
│   ├── session8.{py,R}     # Difference-in-differences
│   ├── session9.{py,R}     # Synthetic control
│   ├── session10.{py,R}    # Regression discontinuity
│   ├── session11.{py,R}    # Panel methods / fixed effects
│   ├── session12.{py,R}    # Heterogeneous treatment effects
│   └── session13.{py,R}    # Sensitivity analysis
└── data/
    └── lalonde_psid.csv    # Dehejia–Wahba LaLonde-PSID1 sample
```

The `data/` folder sits one level above the scripts. Sessions 4 and 5 load
`../data/lalonde_psid.csv`.
