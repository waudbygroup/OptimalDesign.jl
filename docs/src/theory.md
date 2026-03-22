# Theory

This page gives a concise overview of the ideas behind optimal experimental design. For a thorough treatment, see the references at the end.

## What is experimental design?

An experiment measures a system at chosen conditions — a time point, a dose, a frequency, a temperature. The **design** is the set of conditions and how many times each is measured. A good design extracts as much information as possible from a limited measurement budget.

Optimal experimental design formalises "as much information as possible" using the **Fisher Information Matrix** and a scalar criterion.

## The Fisher Information Matrix

For a model ``y = f(\\theta, \\xi) + \\varepsilon`` with Gaussian noise ``\\varepsilon \\sim \\mathcal{N}(0, \\sigma^2(\\theta, \\xi))``, the Fisher Information Matrix (FIM) at a single design point ``\\xi`` and parameter value ``\\theta`` is:

```math
M(\\theta, \\xi) = \\frac{1}{\\sigma^2(\\theta, \\xi)} J(\\theta, \\xi)^\\top J(\\theta, \\xi)
```

where ``J = \\partial f / \\partial \\theta`` is the Jacobian. For a design with ``n`` measurements at points ``\\xi_1, \\ldots, \\xi_n``, the total FIM is the sum:

```math
M(\\theta, d) = \\sum_{i=1}^{n} M(\\theta, \\xi_i)
```

The FIM captures how sensitively the model responds to each parameter at each design point. A larger FIM (in some matrix sense) means more precise parameter estimates.

## Design criteria

A **criterion** maps the ``p \\times p`` FIM to a scalar measuring design quality:

| Criterion | Formula | Interpretation |
|-----------|---------|---------------|
| **D-optimal** | ``\\log \\det M`` | Minimise the volume of the confidence ellipsoid |
| **A-optimal** | ``-\\operatorname{tr}(M^{-1})`` | Minimise the average parameter variance |
| **E-optimal** | ``\\lambda_{\\min}(M)`` | Minimise the worst-case parameter variance |

### Ds-optimality

Often you care about a subset of parameters — say, a rate constant — while others (amplitudes, baselines) are nuisance parameters. **Ds-optimality** targets the subset by applying a transformation ``g(\\theta)`` that extracts the parameters of interest, then optimising the D-criterion on the transformed FIM via the delta method.

In OptimalDesign.jl, this is specified with `select`:

```julia
transformation = select(:R₂)            # interested in R₂ only
transformation = select(:R₂₁, :R₂₂)     # interested in two rates
```

## Bayesian (pseudo-Bayesian) design

Classical optimal design assumes the parameters are known (or guessed). But the FIM — and therefore the optimal design — depends on the true parameter values, which are unknown before the experiment.

**Bayesian optimal design** (also called pseudo-Bayesian or robust design) averages the criterion over prior parameter uncertainty:

```math
\\Phi(d) = \\mathbb{E}_{\\theta \\sim \\pi} \\left[ \\phi\\bigl(M(\\theta, d)\\bigr) \\right]
```

where ``\\pi`` is the prior distribution and ``\\phi`` is the criterion (e.g., ``\\log\\det``). This produces designs that are good on average across plausible parameter values, rather than optimal only at a single guess.

OptimalDesign.jl approximates this expectation by averaging over particles drawn from the prior (or from the current posterior, in the adaptive case).

## The exchange algorithm

Finding the optimal design is an optimisation problem over the space of probability measures on the candidate set. The **exchange algorithm** (Fedorov 1972, Wynn 1972) solves this iteratively:

1. Start with uniform weights over all candidates
2. At each step, transfer weight from the worst candidate to the best, using the Gateaux derivative as a guide
3. Repeat until convergence

For batch designs with ``n`` measurements, the continuous weights are discretised into integer counts using an apportionment method.

## The General Equivalence Theorem

The **General Equivalence Theorem** (Kiefer & Wolfowitz 1960) provides a certificate of optimality. For a D-optimal design with transformed dimension ``q``, the Gateaux (directional) derivative ``\\phi'(d, \\xi)`` satisfies:

- ``\\phi'(d, \\xi) \\leq q`` for all candidates ``\\xi``
- ``\\phi'(d, \\xi) = q`` at the support points of the design

If the maximum Gateaux derivative exceeds ``q``, the design is not optimal and the maximiser indicates which candidate should receive more weight.

OptimalDesign.jl computes and plots the Gateaux derivative via `verify_optimality` and `plot_gateaux`.

## Efficiency

The **efficiency** of a design ``d_1`` relative to ``d_2`` is the ratio of their criterion values, raised to the power ``1/q``:

```math
\\text{eff}(d_1, d_2) = \\left( \\frac{\\det M(d_1)}{\\det M(d_2)} \\right)^{1/q}
```

An efficiency of 0.5 means ``d_1`` would need roughly twice as many measurements to match the precision of ``d_2``.

## Adaptive design

In an **adaptive** (sequential) experiment, the design is updated after each observation:

1. **Design**: choose the next measurement(s) using the current posterior
2. **Acquire**: measure at the chosen design point
3. **Update**: incorporate the observation into the posterior via Bayes' rule

This creates a feedback loop where the experiment learns from its own data. Adaptive designs can outperform batch designs when the optimal measurement locations depend strongly on the (initially unknown) parameter values.

OptimalDesign.jl implements adaptive design using a particle filter posterior. After each observation, particle weights are updated by the likelihood and, when necessary, particles are resampled and jittered to maintain diversity (see [Posterior Inference](@ref)).

## References

- Atkinson, A.C., Donev, A.N. & Tobias, R.D. (2007). *Optimum Experimental Designs, with SAS*. Oxford University Press.
- Chaloner, K. & Verdinelli, I. (1995). Bayesian experimental design: A review. *Statistical Science*, 10(3), 273–304.
- Ryan, E.G., Drovandi, C.C., McGree, J.M. & Pettitt, A.N. (2016). A review of modern computational algorithms for Bayesian optimal design. *International Statistical Review*, 84(1), 128–154.
- Fedorov, V.V. (1972). *Theory of Optimal Experiments*. Academic Press.
- Kiefer, J. & Wolfowitz, J. (1960). The equivalence of two extremum problems. *Canadian Journal of Mathematics*, 12, 363–366.
