"""
    GMMResampling(; ess_threshold=0.5, k_max=10, log_dir=nothing)

Resampling by fitting a Gaussian Mixture Model (full covariance) to the current
particle distribution and drawing fresh samples from it. Avoids particle
impoverishment while capturing multi-modal posterior structure.

BIC-guided model selection determines the number of components (up to `k_max`).
Fitting is performed in unconstrained parameter space to handle bounded priors correctly.

- `ess_threshold`: fraction of N below which resampling is triggered.
- `k_max`: maximum number of GMM components to evaluate.
- `log_dir`: if set, saves a 3-panel diagnostic PNG for every resample event.
"""
struct GMMResampling <: ResamplingStrategy
    ess_threshold::Float64
    k_max::Int
    log_dir::Union{String, Nothing}
end
GMMResampling(; ess_threshold::Float64=0.5, k_max::Int=10,
               log_dir::Union{String,Nothing}=nothing) =
    GMMResampling(ess_threshold, k_max, log_dir)

export GMMResampling

"""Fit GMMs for K=1,2,...,k_max and return the model with the lowest BIC."""
function _gmm_select(Z::Matrix{Float64}, k_max::Int)
    n = size(Z, 2)
    # Per-dimension jitter scaled to each axis — prevents degenerate within-cluster
    # covariances when the posterior has narrowed to a thin strip in some directions.
    σ_dims = vec(std(Z; dims=2))
    noise  = max.(σ_dims, 1e-4) .* 1e-2
    Z = Z .+ noise .* randn(size(Z))
    X = Matrix(Z')  # GaussianMixtures expects n×d
    best_gmm = nothing
    best_bic = Inf
    for K in 1:min(k_max, n)
        # Full-covariance fit can fail via Cholesky when a cluster is nearly degenerate.
        # parallel=false keeps all work on the calling thread so exceptions propagate
        # into this try-catch. Stop the search at the first failure.
        gmm = try
            GaussianMixtures.GMM(K, X; method=:kmeans, kind=:full, parallel=false)
        catch
            @debug "GMM K=$K: fit failed — stopping search"
            break
        end
        # llpg returns log p(x_i | component k), shape n×K (without mixing weights)
        ll = GaussianMixtures.llpg(gmm, X)
        log_π = log.(GaussianMixtures.weights(gmm))
        total_ll = sum(logsumexp(log_π .+ ll[i, :]) for i in 1:n)
        bic = -2total_ll + GaussianMixtures.nparams(gmm) * log(n)
        @debug "GMM BIC search" K bic total_ll
        if bic < best_bic
            best_bic = bic
            best_gmm = gmm
        else
            break
        end
    end
    # If every full-covariance attempt failed (very degenerate posterior), fall back to
    # a single diagonal Gaussian which is always numerically stable.
    if best_gmm === nothing
        @debug "GMM: all full-covariance fits failed — falling back to K=1 diagonal"
        best_gmm = GaussianMixtures.GMM(1, X; method=:kmeans, kind=:diag)
    end
    K_best = length(GaussianMixtures.weights(best_gmm))
    @debug "GMM selected" K_best best_bic
    best_gmm
end

function _gmm_save_fig(Z::Matrix{Float64}, log_weights::Vector{Float64},
                        gmm, Z_new::Matrix{Float64}, pnames, log_dir::String)
    d, n = size(Z)
    if d < 2
        @debug "GMM log: skipping figure (d < 2)"
        return
    end
    mkpath(log_dir)
    fig_idx = length(readdir(log_dir)) + 1

    lw_n = log_weights .- maximum(log_weights)
    K    = length(GaussianMixtures.weights(gmm))
    π_k  = GaussianMixtures.weights(gmm)
    μ    = GaussianMixtures.means(gmm)   # K×d
    Σ_raw = GaussianMixtures.covars(gmm) # K×d (diag) or Vector{Matrix} (full)

    z1_in,  z2_in  = Z[1, :],     Z[2, :]
    z1_out, z2_out = Z_new[1, :], Z_new[2, :]

    pad = 0.3
    xlo = min(minimum(z1_in), minimum(z1_out)) - pad
    xhi = max(maximum(z1_in), maximum(z1_out)) + pad
    ylo = min(minimum(z2_in), minimum(z2_out)) - pad
    yhi = max(maximum(z2_in), maximum(z2_out)) + pad
    xg  = range(xlo, xhi; length=100)
    yg  = range(ylo, yhi; length=100)

    # 2D marginal of GMM over first two parameters
    dists_2d = if GaussianMixtures.kind(gmm) == :diag
        [MvNormal(μ[k, 1:2], Diagonal(Σ_raw[k, 1:2])) for k in 1:K]
    else
        [MvNormal(μ[k, 1:2], Symmetric(Σ_raw[k][1:2, 1:2])) for k in 1:K]
    end
    # dens is (length(yg) × length(xg)); contour! wants (length(xg) × length(yg))
    dens = [sum(π_k[k] * pdf(dists_2d[k], [x, y]) for k in 1:K)
            for y in yg, x in xg]'

    name1  = string(pnames[1])
    name2  = string(pnames[2])
    xlab   = d > 2 ? "$name1 (unc.) [first of $d]" : "$name1 (unc.)"
    ylab   = "$name2 (unc.)"
    cr     = (max(-100.0, minimum(lw_n)), 0.0)

    fig = CairoMakie.Figure(size=(1400, 460))
    CairoMakie.Label(fig[0, 1:3], "Resample $fig_idx  —  GMM K=$K  —  n=$n";
                     fontsize=13, font=:bold)

    # (a) incoming particles coloured by log-weight
    ax1 = CairoMakie.Axis(fig[1, 1]; title="(a) incoming", xlabel=xlab, ylabel=ylab)
    sc  = CairoMakie.scatter!(ax1, z1_in, z2_in;
                              color=lw_n, colormap=:plasma, colorrange=cr, markersize=4)
    CairoMakie.Colorbar(fig[1, 1][1, 2], sc; label="log w")

    # (b) incoming + GMM contours + component means
    ax2 = CairoMakie.Axis(fig[1, 2]; title="(b) GMM fit (K=$K)", xlabel=xlab)
    CairoMakie.scatter!(ax2, z1_in, z2_in;
                        color=lw_n, colormap=:plasma, colorrange=cr, markersize=4, alpha=0.5)
    CairoMakie.contour!(ax2, xg, yg, dens; levels=8, color=:black, linewidth=1.5)
    CairoMakie.scatter!(ax2, μ[:, 1], μ[:, 2];
                        color=:yellow, marker=:cross, markersize=14, strokewidth=2)

    # (c) outgoing particles
    ax3 = CairoMakie.Axis(fig[1, 3]; title="(c) outgoing", xlabel=xlab)
    CairoMakie.scatter!(ax3, z1_out, z2_out; color=:dodgerblue, markersize=4)

    for ax in (ax1, ax2, ax3)
        CairoMakie.xlims!(ax, xlo, xhi)
        CairoMakie.ylims!(ax, ylo, yhi)
    end

    fname = joinpath(log_dir, "resample_$(lpad(fig_idx, 4, '0')).png")
    CairoMakie.save(fname, fig)
    @info "GMM resample figure saved" fname K
end

function resample!(posterior::Particles{T, GMMResampling};
        prob::Union{AbstractDesignProblem, Nothing}=nothing) where {T}
    n = length(posterior.particles)
    d = length(first(posterior.particles))
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    pnames = keys(first(posterior.particles))

    transforms = prob !== nothing ? _param_transforms(prob.parameters) : nothing

    # Map particles to unconstrained space (d×n)
    Z = Matrix{Float64}(undef, d, n)
    for i in 1:n
        θ = posterior.particles[i]
        for (ki, k) in enumerate(pnames)
            val = getproperty(θ, k)
            Z[ki, i] = transforms !== nothing ? transforms[ki].forward(val) : val
        end
    end

    # Equalise weights via systematic resample before GMM fitting
    Z_rs = Z[:, systematic_resample(w, n)]

    # Fit GMM with BIC-guided K selection
    gmm = _gmm_select(Z_rs, posterior.resampling.k_max)

    # Draw n fresh particles from GMM (rand returns n×d; transpose to d×n)
    Z_new = Matrix(rand(gmm, n)')

    # Optional diagnostic figure
    if posterior.resampling.log_dir !== nothing
        _gmm_save_fig(Z, copy(posterior.log_weights), gmm, Z_new, pnames,
                      posterior.resampling.log_dir)
    end

    # Back-transform and rebuild ComponentArrays
    for i in 1:n
        vals = ntuple(d) do ki
            z = Z_new[ki, i]
            transforms !== nothing ? transforms[ki].inverse(z) : z
        end
        posterior.particles[i] = ComponentArray(NamedTuple{pnames}(vals))
    end

    fill!(posterior.log_weights, -log(n))
    K_used = length(GaussianMixtures.weights(gmm))
    @debug "Resampled with GMM" K_used
    posterior
end

"""
    update!(posterior, prob, data, ::GMMResampling)

Direct importance weighting update. Applies likelihoods observation-by-observation,
normalises, and resamples (via GMM) when ESS drops below `ess_threshold × n`.
"""
function update!(posterior::Particles{T, GMMResampling}, prob::AbstractDesignProblem,
        data::AbstractVector{<:NamedTuple}, strategy::GMMResampling) where {T}
    n = length(posterior.particles)
    for d in data
        for i in 1:n
            posterior.log_weights[i] += loglikelihood(prob, posterior.particles[i], d.x, d.y)
        end
        posterior.log_weights .-= logsumexp(posterior.log_weights)
        if effective_sample_size(posterior) < strategy.ess_threshold * n
            resample!(posterior; prob=prob)
        end
    end
    posterior
end
