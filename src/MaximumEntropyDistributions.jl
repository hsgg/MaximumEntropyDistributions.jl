"""
Maximum Entropy reconstruction of a PDF on a bounded interval [xmin, xmax].

The MaxEnt PDF takes the form:
    P(x) ∝ exp(-∑ λₙ xⁿ)

The Lagrange multipliers λ are found by Newton's method, matching target moments.
"""
module MaximumEntropyDistributions

export MaxEntPDF, MaxEntPDF_cumulants, moments_to_cumulants, cumulants_to_moments


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


# ── Moment ↔ Cumulant conversion ─────────────────────────────────────────────

"""
    moments_to_cumulants(μ) -> κ

Convert raw moments `μ = [μ₁, μ₂, …, μₙ]` where `μₖ = ⟨xᵏ⟩` to cumulants
`κ = [κ₁, κ₂, …, κₙ]` using the standard recursive formula.
"""
function moments_to_cumulants(μ::AbstractVector)
    N = length(μ)
    κ = similar(μ)
    for n in 1:N
        s = μ[n]
        for j in 1:n-1
            s -= binomial(n - 1, j - 1) * κ[j] * μ[n - j]
        end
        κ[n] = s
    end
    return κ
end

"""
    cumulants_to_moments(κ) -> μ

Convert cumulants `κ = [κ₁, κ₂, …, κₙ]` to raw moments
`μ = [μ₁, μ₂, …, μₙ]` where `μₖ = ⟨xᵏ⟩`, using the standard recursive formula.
"""
function cumulants_to_moments(κ::AbstractVector)
    N = length(κ)
    μ = similar(κ)
    for n in 1:N
        s = κ[n]
        for j in 1:n-1
            s += binomial(n - 1, j - 1) * κ[j] * μ[n - j]
        end
        μ[n] = s
    end
    return μ
end

"""
    moment_cumulant_jacobian(μ, κ) -> M

Compute the N×N Jacobian matrix `M[n,k] = ∂κₙ/∂μₖ` of the moment→cumulant
map.  `M` is lower-triangular with unit diagonal and is always invertible.
"""
function moment_cumulant_jacobian(μ::AbstractVector, κ::AbstractVector)
    N = length(μ)
    T = promote_type(eltype(μ), eltype(κ))
    M = zeros(T, N, N)
    for n in 1:N
        M[n, n] = one(T)
        for k in 1:n-1
            s = zero(T)
            for j in 1:n-1
                s -= binomial(n - 1, j - 1) * M[j, k] * μ[n - j]
            end
            s -= binomial(n - 1, n - k - 1) * κ[n - k]
            M[n, k] = s
        end
    end
    return M
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

        local Δλ
        try
            Δλ = H \ g
        catch e
            e isa SingularException || rethrow(e)
            Δλ = pinv(H) * g
        end
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


# ── Cumulant-based lambda solver ──────────────────────────────────────────────

"""
    fit_maxent_from_cumulants(xmin, xmax, κ_target; n_quad=64, tol=1e-12, maxiter=200)
        -> λ, Z, μ_fit

Solve for the Lagrange multipliers λ such that the MaxEnt distribution
    P(x) ∝ exp(-∑ λₙ xⁿ)
on [xmin, xmax] matches the supplied cumulants `κ_target[n]` = κₙ, n = 1…N.

Internally converts cumulants to moments via `cumulants_to_moments` and
delegates to `fit_maxent_lambdas`.  The cumulant→moment→cumulant path is
numerically stable because the moment-space Newton iteration uses the
covariance matrix H as its natural preconditioner.

Returns the multipliers `λ`, the partition function `Z`, and the achieved
moments `μ_fit` (convert via `moments_to_cumulants(μ_fit)` to check cumulant
matching).
"""
function fit_maxent_from_cumulants(xmin, xmax, κ_target; kwargs...)
    μ_target = cumulants_to_moments(κ_target)
    return fit_maxent_lambdas(xmin, xmax, μ_target; kwargs...)
end


# ── Cumulant-based public constructor ─────────────────────────────────────────

"""
    MaxEntPDF_cumulants(xmin, xmax, κ_target; n_quad=64, tol=1e-12, maxiter=200)

Fit a MaxEnt distribution on [xmin, xmax] matching the supplied cumulants
`κ_target`, where `κ_target[n]` = κₙ for n = 1, …, N.

Internally converts cumulants to moments and delegates to the moment-based
Newton solver for numerical stability.

The returned `MaxEntPDF` object is callable: `m(x)` evaluates the PDF at `x`.
Cumulant accuracy can be checked via `moments_to_cumulants(m.μ_fit) ≈ κ_target`.
"""
function MaxEntPDF_cumulants(xmin::T, xmax::T, κ_target::AbstractVector{T}; kwargs...) where {T<:AbstractFloat}
    λ, Z, μ_fit = fit_maxent_from_cumulants(xmin, xmax, κ_target; kwargs...)
    μ_target = cumulants_to_moments(κ_target)
    return MaxEntPDF{T}(xmin, xmax, λ, Z, μ_target, μ_fit)
end

# Convenience: promote mixed Real inputs to a common float type
function MaxEntPDF_cumulants(xmin::Real, xmax::Real, κ_target::AbstractVector; kwargs...)
    T = float(promote_type(typeof(xmin), typeof(xmax), eltype(κ_target)))
    return MaxEntPDF_cumulants(T(xmin), T(xmax), convert(Vector{T}, κ_target); kwargs...)
end


end
