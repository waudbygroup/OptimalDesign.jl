using OptimalDesign
using Test
using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using Random

Random.seed!(42)

# Resolve name conflicts with Distributions.jl
const od_sample = OptimalDesign.sample
const od_loglikelihood = OptimalDesign.loglikelihood

@testset "OptimalDesign.jl" begin

    @testset "draw" begin
        params = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5))
        θ = draw(params)
        @test θ isa ComponentArray
        @test haskey(θ, :A)
        @test haskey(θ, :R₂)

        θs = draw(params, 100)
        @test length(θs) == 100
        @test all(θ -> θ isa ComponentArray, θs)
    end

    @testset "DesignProblem construction" begin
        # Minimal construction
        prob = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
        )
        @test prob.predict isa Function
        @test prob.jacobian === nothing
        @test prob.transformation isa Identity
        @test prob.cost((t=0.1,), (t=0.2,)) == 1.0
        @test prob.constraint((t=0.1,), nothing) == true

        # Full construction
        prob2 = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            jacobian = (θ, ξ) -> [exp(-θ.R₂ * ξ.t)  -θ.A * ξ.t * exp(-θ.R₂ * ξ.t)],
            sigma = (θ, ξ) -> 0.05,
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            transformation = DeltaMethod(θ -> ComponentArray(R₂=θ.R₂)),
            cost = (prev, ξ) -> ξ.t + 0.1,
        )
        @test prob2.jacobian !== nothing
    end

    @testset "Criteria" begin
        M = [2.0 0.5; 0.5 3.0]

        d = DCriterion()(M)
        @test d ≈ log(det(M))

        a = ACriterion()(M)
        @test a ≈ -tr(inv(M))

        e = ECriterion()(Symmetric(M))
        @test e ≈ eigmin(Symmetric(M))
    end

    @testset "weighted_fim" begin
        J = [1.0 2.0; 3.0 4.0]

        # Scalar sigma
        F1 = weighted_fim(J, 0.5)
        @test F1 ≈ J' * J / 0.25

        # Vector sigma
        F2 = weighted_fim(J, [0.5, 1.0])
        W = Diagonal([1/0.25, 1/1.0])
        @test F2 ≈ J' * W * J

        # Matrix sigma
        Σ = [1.0 0.2; 0.2 0.5]
        F3 = weighted_fim(J, Σ)
        @test F3 ≈ J' * inv(Σ) * J

        # Vector J (scalar observation)
        Jv = [1.0, 2.0]
        F4 = weighted_fim(Jv, 1.0)
        @test F4 ≈ reshape(Jv, 1, 2)' * reshape(Jv, 1, 2)
    end

    @testset "Example 1: Exponential decay — FIM" begin
        prob = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )

        θ = ComponentArray(A=1.0, R₂=10.0)
        ξ = (t=0.1,)

        M = information(prob, θ, ξ)
        @test size(M) == (2, 2)
        @test issymmetric(M) || M ≈ M'
        # Rank 1 from single scalar observation — one eigenvalue is zero (up to float)
        @test all(eigvals(Symmetric(M)) .>= -1e-10)

        # Verify FIM by hand:
        # y = A * exp(-R₂*t)
        # ∂y/∂A = exp(-R₂*t) = exp(-1) ≈ 0.3679
        # ∂y/∂R₂ = -A*t*exp(-R₂*t) = -0.1*exp(-1) ≈ -0.03679
        e = exp(-10.0 * 0.1)
        J_expected = [e  -1.0*0.1*e]
        F_expected = J_expected' * J_expected / 0.05^2
        @test M ≈ F_expected atol=1e-10
    end

    @testset "Example 2: Inversion recovery — analytic vs ForwardDiff Jacobian" begin
        predict = (θ, ξ) -> θ.A - θ.B * exp(-θ.R₁ * ξ.τ)
        jac = (θ, ξ) -> begin
            e = exp(-θ.R₁ * ξ.τ)
            [1.0  -e  θ.B * ξ.τ * e]
        end

        prob_ad = DesignProblem(
            predict,
            parameters = (A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogNormal(0, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )

        prob_analytic = DesignProblem(
            predict,
            jacobian = jac,
            parameters = (A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogNormal(0, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )

        θ = ComponentArray(A=1.0, B=2.0, R₁=1.0)
        ξ = (τ=0.5,)

        M_ad = information(prob_ad, θ, ξ)
        M_analytic = information(prob_analytic, θ, ξ)

        @test M_ad ≈ M_analytic atol=1e-10
        @test size(M_ad) == (3, 3)
    end

    @testset "DeltaMethod transformation" begin
        # Use a full-rank FIM by summing over multiple design points
        prob = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            transformation = select(:R₂),
            sigma = (θ, ξ) -> 0.05,
        )

        θ = ComponentArray(A=1.0, R₂=10.0)

        # Sum FIM over two well-separated time points for full rank
        M = information(prob, θ, (t=0.05,)) + information(prob, θ, (t=0.2,))
        @test isposdef(Symmetric(M))

        Mt = transform(prob, M, θ)

        # Transformed matrix should be 1×1 for single parameter of interest
        @test size(Mt) == (1, 1)
        @test Mt[1, 1] > 0
    end

    @testset "ParticlePosterior" begin
        prob = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )

        post = ParticlePosterior(prob, 500)
        @test length(post.particles) == 500
        @test length(post.log_weights) == 500
        @test all(isfinite, post.log_weights)

        # ESS should be n for uniform weights
        ess = effective_sample_size(post)
        @test ess ≈ 500.0 atol=1.0

        # Posterior mean should be close to prior mean
        μ = posterior_mean(post)
        @test μ isa ComponentArray
        @test abs(μ.A - 1.0) < 0.1  # prior mean of A is 1.0

        # Sample (qualified to avoid conflict with Distributions.sample)
        s = od_sample(post, 10)
        @test length(s) == 10
    end

    @testset "loglikelihood" begin
        prob = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )

        θ = ComponentArray(A=1.0, R₂=10.0)
        ξ = (t=0.1,)
        ŷ = prob.predict(θ, ξ)

        # Perfect observation: highest likelihood
        ll_perfect = od_loglikelihood(prob, θ, ξ, ŷ)
        ll_noisy = od_loglikelihood(prob, θ, ξ, ŷ + 0.1)
        @test ll_perfect > ll_noisy

        # Structured observation
        ll_struct = od_loglikelihood(prob, θ, ξ, (value=ŷ, σ=0.05))
        @test ll_struct ≈ ll_perfect
    end

    @testset "update! posterior" begin
        prob = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )

        post = ParticlePosterior(prob, 1000)
        θ_true = ComponentArray(A=1.0, R₂=10.0)
        ξ = (t=0.1,)
        y = prob.predict(θ_true, ξ) + 0.05 * randn()

        update!(post, prob, ξ, y)

        μ = posterior_mean(post)
        @test μ isa ComponentArray
    end

    @testset "expected_utility" begin
        # Use vector observation so single-point FIM is full rank (2 obs, 2 params)
        prob = DesignProblem(
            (θ, ξ) -> [θ.A * exp(-θ.R₂ * ξ.t), θ.A * exp(-θ.R₂ * ξ.t * 2)],
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> [0.05, 0.05],
        )

        particles = draw(prob.parameters, 100)
        ξ = (t=0.1,)

        u = expected_utility(prob, DCriterion(), particles, ξ; batch_size=50)
        @test isfinite(u)

        # Score multiple candidates
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        scores = score_candidates(prob, DCriterion(), particles, candidates; batch_size=50)
        @test length(scores) == 20
        @test all(isfinite, scores)

        # Scalar observation (rank-deficient FIM) — should not error
        prob_scalar = DesignProblem(
            (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
            parameters = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )
        u_scalar = expected_utility(prob_scalar, DCriterion(), particles, ξ; batch_size=50)
        @test !isnan(u_scalar)
    end

    @testset "Example 3: Vector observation" begin
        prob = DesignProblem(
            (θ, ξ) -> [θ.A₁ * exp(-θ.R₂₁ * ξ.t),
                        θ.A₂ * exp(-θ.R₂₂ * ξ.t)],
            parameters = (A₁=Normal(1, 0.1), R₂₁=LogNormal(2, 0.5),
                          A₂=Normal(1, 0.1), R₂₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> [0.05, 0.05],
        )

        θ = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=25.0)
        ξ = (t=0.1,)

        M = information(prob, θ, ξ)
        @test size(M) == (4, 4)
        @test all(eigvals(Symmetric(M)) .>= -1e-10)
    end

    @testset "Example 4: Selective observation — block sparsity" begin
        prob = DesignProblem(
            (θ, ξ) -> if ξ.i == 1
                θ.A₁ * exp(-θ.R₂₁ * ξ.t)
            else
                θ.A₂ * exp(-θ.R₂₂ * ξ.t)
            end,
            parameters = (A₁=Normal(1, 0.1), R₂₁=LogNormal(2, 0.5),
                          A₂=Normal(1, 0.1), R₂₂=LogNormal(2, 0.5)),
            sigma = (θ, ξ) -> 0.05,
        )

        θ = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=25.0)

        # Measuring decay 1: only A₁, R₂₁ contribute
        M1 = information(prob, θ, (i=1, t=0.1))
        @test size(M1) == (4, 4)
        # Columns/rows for A₂, R₂₂ (indices 3,4) should be zero
        @test norm(M1[3:4, :]) < 1e-10
        @test norm(M1[:, 3:4]) < 1e-10

        # Measuring decay 2: only A₂, R₂₂ contribute
        M2 = information(prob, θ, (i=2, t=0.1))
        @test norm(M2[1:2, :]) < 1e-10
        @test norm(M2[:, 1:2]) < 1e-10

        # Sum gives full-rank (if time chosen well)
        M_sum = M1 + M2
        @test rank(M_sum) == 2  # rank 2: each measurement contributes rank 1
    end

end
