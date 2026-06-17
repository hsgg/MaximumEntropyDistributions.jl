"""
Maximum Entropy reconstruction of a PDF on a bounded interval [xmin, xmax].

The MaxEnt PDF takes the form:
    P(x) ∝ exp(-∑ λₙ xⁿ)

The Lagrange multipliers λ are found by Newton's method, matching target moments.
"""
module MaximumEntropyDistributions

export MaxEntPDF


using LinearAlgebra
using FastGaussQuadrature


# ── Result struct ─────────────────────────────────────────────────────────────

"""
    MaxEntPDF{T<:AbstractFloat}

Stores the result of a MaxEnt fit on [xmin, xmax].

Fields:
  - `xmin`, `xmax` : interval edges
  - `λ`            : Lagrange multipliers (length N)
  - `Z`            : partition function (normalisation constant)
  - `μ_target`     : target moments supplied by the caller
  - `μ_fit`        : moments achieved by the fit
"""
struct MaxEntPDF{T<:AbstractFloat}
    xmin     :: T
    xmax     :: T
    λ        :: Vector{T}
    Z        :: T
    μ_target :: Vector{T}
    μ_fit    :: Vector{T}
end

"""
    (m::MaxEntPDF)(x)

Evaluate the MaxEnt PDF at `x`.
"""
function (m::MaxEntPDF)(x)
    (x < m.xmin || x > m.xmax) && return zero(m.Z)
    return exp(-sum(λ * x^n for (n, λ) in enumerate(m.λ))) / m.Z
end


# ── Public constructor ────────────────────────────────────────────────────────

"""
    MaxEntPDF(xmin, xmax, μ_target; n_quad=64, tol=1e-12, maxiter=200)

Fit a MaxEnt distribution on [xmin, xmax] matching the supplied moments
`μ_target`, where `μ_target[n]` = ⟨xⁿ⟩ for n = 1, …, N.

The returned `MaxEntPDF` object is callable: `m(x)` evaluates the PDF at `x`.
"""
function MaxEntPDF(xmin::T, xmax::T, μ_target::AbstractVector{T}; kwargs...) where {T<:AbstractFloat}
    λ, Z, μ_fit = fit_maxent_lambdas(xmin, xmax, μ_target; kwargs...)
    return MaxEntPDF{T}(xmin, xmax, λ, Z, collect(μ_target), μ_fit)
end

# Convenience: promote mixed Real inputs to a common float type
function MaxEntPDF(xmin::Real, xmax::Real, μ_target::AbstractVector; kwargs...)
    T = float(promote_type(typeof(xmin), typeof(xmax), eltype(μ_target)))
    return MaxEntPDF(T(xmin), T(xmax), convert(Vector{T}, μ_target); kwargs...)
end


# ── Lambda solver ─────────────────────────────────────────────────────────────

"""
    fit_maxent_lambdas(xmin, xmax, μ_target; n_quad=64, tol=1e-12, maxiter=200)
        -> λ, Z, μ_fit

Solve for the Lagrange multipliers λ such that the MaxEnt distribution
    P(x) ∝ exp(-∑ λₙ xⁿ)
on [xmin, xmax] matches the supplied moments `μ_target[n]` = ⟨xⁿ⟩, n = 1…N.

Returns the multipliers `λ`, the partition function `Z`, and the achieved
moments `μ_fit`.
"""
function fit_maxent_lambdas(xmin, xmax, μ_target;
                            n_quad   = 64,
                            tol      = 1e-12,
                            maxiter  = 200)

    N = length(μ_target)

    # Affine map from the native [-1,1] quadrature nodes to [xmin, xmax]
    u, w    = gausslegendre(n_quad)
    x_nodes = @. (xmax - xmin) / 2 * u + (xmax + xmin) / 2
    weights = @. (xmax - xmin) / 2 * w

    # Powers of x_nodes: cols = moment orders 1…N  (n_quad × N)
    X_pow = [x^n for x in x_nodes, n in 1:N]

    function compute_moments_and_cov(λ)
        f  = exp.(-(X_pow * λ))
        Z  = dot(weights, f)
        pw = (f ./ Z) .* weights          # probability-weighted quadrature

        μ  = X_pow' * pw                  # model moments (N,)

        E2 = [dot(pw, x_nodes .^ (n + m)) for n in 1:N, m in 1:N]
        H  = E2 .- μ * μ'                 # Hessian = Cov(xⁿ, xᵐ)

        return μ, H, Z
    end

    # ── Newton iterations ─────────────────────────────────────────────────────

    λ = zeros(N)

    for _ in 1:maxiter
        μ, H, _ = compute_moments_and_cov(λ)
        g = μ .- μ_target

        norm(g) < tol && break

        Δλ = H \ g
        g_norm = norm(g)
        step = 1.0
        for _ in 1:20
            λ_new = λ .+ step .* Δλ
            f_new = exp.(-(X_pow * λ_new))
            if all(isfinite, f_new) && dot(weights, f_new) > 0
                Z_new = dot(weights, f_new)
                μ_new = X_pow' * ((f_new ./ Z_new) .* weights)
                if norm(μ_new .- μ_target) < g_norm
                    λ = λ_new
                    break
                end
            end
            step *= 0.5
        end
    end

    μ_fit, _, Z = compute_moments_and_cov(λ)
    return λ, Z, μ_fit
end


end
