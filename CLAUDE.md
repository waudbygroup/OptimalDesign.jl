# OptimalDesign.jl — Development Notes

## Logging convention

Use Julia's standard logging macros (`@info`, `@debug`, `@warn`) throughout.
No `verbose` keyword arguments — callers control visibility via the logging system.

- **`@info`** — high-level process milestones: experiment start/complete, exchange
  algorithm phase transitions, convergence status, final results.
- **`@debug`** — per-iteration detail: FW gap, support size, step transfers,
  per-observation updates in adaptive loops.
- **`@warn`** — singular FIM, resampling triggers, non-convergence.

To see debug output in example scripts or the REPL:

```julia
ENV["JULIA_DEBUG"] = OptimalDesign
```

## Code organisation

- **Core type definitions go in `types.jl`** — abstract types, domain structs (problem,
  design, results, particles), and type aliases. Abstract types must always be in `types.jl`.
- **Strategy/plugin structs** (e.g. resampling strategies, design criteria) live in their
  own implementation files alongside their methods. This is the established pattern for
  `LiuWestResampling`, `SystematicResampling`, `GMMResampling`, etc.
