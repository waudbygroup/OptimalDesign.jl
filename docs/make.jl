using Pkg
Pkg.activate(@__DIR__)  # Activate the docs environment

using Documenter
using OptimalDesign

makedocs(;
    sitename="OptimalDesign.jl",
    format=Documenter.HTML(),
    modules=[OptimalDesign],
    warnonly=[:missing_docs, :cross_references],
    pages=[
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
        "Guide" => [
            "Workflows" => "guide/workflow.md",
            "Defining Problems" => "guide/problems.md",
            "Posterior Inference" => "guide/posteriors.md",
            "Plotting" => "guide/plotting.md",
        ],
        "Theory" => "theory.md",
        "Examples" => [
            "Batch Design" => "examples/batch_design.md",
            "Vector Observations" => "examples/vector_observation.md",
            "Adaptive Design" => "examples/adaptive_design.md",
            "Switching Costs" => "examples/switching_costs.md",
        ],
        "API" => "api.md",
    ],
)
