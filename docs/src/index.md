# OptimalDesign.jl

**Adaptive and static Bayesian optimal experimental design for nonlinear models.**

OptimalDesign.jl helps you decide *where* and *how many times* to measure in order to learn model parameters as efficiently as possible. It works with any nonlinear model you can write as a Julia function, handles prior uncertainty via particle-based Bayesian averaging, and supports both pre-planned batch designs and fully adaptive sequential experiments.

## Features

- **Batch design** — compute an optimal allocation of measurements before acquiring any data
- **Adaptive design** — sequentially choose the next measurement based on what you've learned so far
- **Bayesian averaging** — designs are optimised across priors (specified through Distributions.jl)
- **Multiple criteria** — D-optimal, Ds-optimal (parameter subsets), A-optimal, E-optimal, and **EIG** (Expected Information Gain) for broad or multi-modal priors
- **Particle posterior** — inference with likelihood tempering and pluggable resampling strategies (Liu-West, systematic, and Gaussian mixture model)
- **Optimality verification** — via the Gateaux derivative and the General Equivalence Theorem (FIM-based criteria)
- **Built-in plotting** — credible bands, posterior distribution, corner plots, design allocation, and animated evolution

## Getting started

Head to the [Quickstart](@ref) for a minimal working example, or read the [Guide](@ref "Workflows") for a fuller picture of what the package can do.

## Related packages

- [Kirstine.jl](https://github.com/lsandig/Kirstine.jl) — static Bayesian optimal designs.
- [ExperimentalDesign.jl](https://github.com/phrb/ExperimentalDesign.jl) — design of experiments.
