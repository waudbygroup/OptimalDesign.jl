# OptimalDesign.jl

**Bayesian optimal experimental design for nonlinear models.**

OptimalDesign.jl helps you decide *where* and *how many times* to measure in order to learn model parameters as efficiently as possible. It works with any nonlinear model you can write as a Julia function, handles prior uncertainty via particle-based Bayesian averaging, and supports both pre-planned batch designs and fully adaptive sequential experiments.

## Features

- **Batch design** — compute an optimal allocation of measurements before acquiring any data
- **Adaptive design** — sequentially choose the next measurement based on what you've learned so far
- **Bayesian averaging** — designs are robust to parameter uncertainty, not optimised at a single guess
- **Multiple criteria** — D-optimal (overall precision), Ds-optimal (subset of parameters), A-optimal, E-optimal
- **Particle posterior** — lightweight sequential Monte Carlo inference with likelihood tempering
- **Optimality verification** — Gateaux derivative and the General Equivalence Theorem
- **Built-in plotting** — corner plots, credible bands, design allocation, posterior evolution animations

## Getting started

Head to the [Quickstart](@ref) for a minimal working example, or read the [Guide](@ref "Workflows") for a fuller picture of what the package can do.

## Related packages

- [Kirstine.jl](https://github.com/lsandig/Kirstine.jl) — locally optimal designs for nonlinear regression models. Computes classical (non-Bayesian) D-optimal designs using the Fedorov--Wynn algorithm. A good complement if you want to compare Bayesian and frequentist designs.
- [ExperimentalDesign.jl](https://github.com/phrb/ExperimentalDesign.jl) — factorial and screening designs (different scope: linear models, discrete factors).
- [PopED](https://andrewhooker.github.io/PopED/) — optimal design for nonlinear mixed-effects models in pharmacometrics (R/Fortran).
