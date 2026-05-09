# Causal Inference for Business Decision Making — Reference Code

Reference code accompanying the course *Causal Inference for Business Decision Making* at TUM.

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
`../data/lalonde_psid.csv` (with a fallback to `transcripts/code/data/...`
when run from the course root).

## Running the scripts

### Python

```bash
cd 26S
python session4.py
```

Required packages vary by session — see the header of each file. Common ones:
`pandas`, `numpy`, `statsmodels`, `matplotlib`, `causalinference`,
`linearmodels`, `econml`.

### R

```bash
cd 26S
Rscript session4.R
```

Required packages are listed at the top of each file (e.g. `MatchIt`, `cobalt`,
`WeightIt`, `did`, `Synth`, `rdrobust`, `plm`, `grf`, `sensemakr`).

## Data

`data/lalonde_psid.csv` is the Dehejia & Wahba (1999) LaLonde-PSID1 sample:
185 NSW treated units + 2,490 PSID-1 observational controls. Used in sessions 4
and 5 to illustrate matching and weighting.
