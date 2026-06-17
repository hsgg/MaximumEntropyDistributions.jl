# MaximumEntropyDistributions

[![Build Status](https://github.com/hsgg/MaximumEntropyDistributions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/hsgg/MaximumEntropyDistributions.jl/actions/workflows/CI.yml?query=branch%3Amain)

Reconstruct a continuous probability distribution on a bounded interval
`[xmin, xmax]` from a small set of known moments, using the maximum entropy
principle. Given moments ⟨x⟩, ⟨x²⟩, …, ⟨xᴺ⟩ — estimated from data or
computed analytically — the package returns a callable PDF that matches those
moments exactly while remaining maximally noncommittal about everything else.


## Background

### The maximum entropy principle

When we know only a few summary statistics of a distribution, many PDFs are
consistent with those constraints. The **maximum entropy principle** (Jaynes
1957) provides a principled way to choose among them: pick the distribution
that has the *largest Shannon entropy*,

```math
H[P] = -\int P(x) \ln P(x) dx
```

subject to the moment constraints. Entropy measures how "spread out" or
"uninformative" a distribution is; maximizing it produces the distribution that
encodes the given constraints and *nothing more*. Any other choice would
implicitly inject information that was not in the data.

### The constrained optimization problem

Formally, we solve:

```
maximize  H[P] = -∫ P(x) ln P(x) dx
subject to  ∫ P(x) xⁿ dx = μₙ ,  n = 1, …, N
            ∫ P(x) dx = 1
            P(x) = 0  outside [xmin, xmax]
```

Introducing Lagrange multipliers λ₁, …, λₙ for the moment constraints (and
λ₀ for normalization), the calculus of variations gives the unique solution:

```
P(x) = exp(-λ₁ x - λ₂ x² - … - λₙ xᴺ) / Z
```

where the partition function `Z = ∫ exp(-∑ λₙ xⁿ) dx` ensures normalization.
This is an **exponential family** distribution whose sufficient statistics are
the monomials `x, x², …, xᴺ`.

### Connection to familiar distributions

The MaxEnt form recovers well-known distributions as special cases:

| Constraints supplied | MaxEnt distribution |
|---|---|
| None (only normalization, bounded support) | Uniform |
| Mean ⟨x⟩ on [0, ∞) | Exponential |
| Mean ⟨x⟩ and variance ⟨x²⟩ on (−∞, ∞) | Gaussian |
| N moments on [xmin, xmax] | MaxEnt polynomial-exponential |

Each of these is the *least-structured* distribution compatible with its
constraints. Providing more moments lets the MaxEnt PDF capture skewness,
kurtosis, and multimodality beyond what a Gaussian can represent.

### Why it works numerically

Finding the Lagrange multipliers λ is a **convex** dual optimization problem,
so it has a unique solution and Newton's method converges reliably. The
Hessian of the dual objective equals the covariance matrix Cov(xⁿ, xᵐ) under
the current distribution — positive definite, well-conditioned, and cheap to
compute via quadrature.


## Algorithm

1. **Gauss-Legendre quadrature** maps the interval `[xmin, xmax]` to the
   native `[-1, 1]` nodes and evaluates all integrals numerically (default:
   64 points; increase for sharp or multimodal distributions).

2. **Newton's method** iterates on the Lagrange multipliers λ. At each step
   it computes the moment residual `g = ⟨xⁿ⟩ - μₙ` and the Hessian
   `H_nm = Cov(xⁿ, xᵐ)`, then updates `λ ← λ + H⁻¹ g`.

3. **Line search** halves the step size if the proposed update would make the
   distribution non-finite or increase the residual, ensuring convergence even
   when the initial `λ = 0` (uniform distribution) is far from the solution.

Convergence is declared when `‖g‖ < tol` (default `1e-12`).


## Installation

```julia
using Pkg
Pkg.add("MaximumEntropyDistributions")
```


## Usage

```julia
using MaximumEntropyDistributions
using Statistics: mean

# Suppose we have a sample and want to fit a MaxEnt PDF on [0, 1]
data = ...                       # your data, assumed to lie in [0, 1]
N    = 4                         # number of moments to match

μ_target = [mean(data .^ k) for k in 1:N]   # raw moments ⟨xᵏ⟩

m = MaxEntPDF(0.0, 1.0, μ_target)           # fit the distribution

# Evaluate the PDF at any point in [0, 1]
m(0.3)

# Inspect the fit quality
println("Target moments: ", m.μ_target)
println("Fitted moments: ", m.μ_fit)
```

Increasing `N` adds more moment constraints and captures finer distributional
shape. Start with `N = 2` (matches mean and second moment) and increase until
the PDF looks stable.

### Keyword arguments

| Argument | Default | Description |
|---|---|---|
| `n_quad` | `64` | Number of Gauss-Legendre quadrature points |
| `tol` | `1e-12` | Convergence tolerance on the moment residual |
| `maxiter` | `200` | Maximum Newton iterations |

For distributions with sharp peaks or long tails near the boundary, increase
`n_quad` to 200–500.


## API reference

```julia
MaxEntPDF(xmin, xmax, μ_target; n_quad=64, tol=1e-12, maxiter=200)
```

Fits a MaxEnt distribution on `[xmin, xmax]` and returns a `MaxEntPDF` object.
`μ_target[n]` must equal `⟨xⁿ⟩` for `n = 1, …, N` (raw moments, not central
moments). The returned object is callable: `m(x)` evaluates the PDF at `x`
and returns `0` outside `[xmin, xmax]`.

**Struct fields:**

| Field | Type | Description |
|---|---|---|
| `xmin`, `xmax` | `T` | Interval boundaries |
| `λ` | `Vector{T}` | Fitted Lagrange multipliers (length N) |
| `Z` | `T` | Partition function (normalization constant) |
| `μ_target` | `Vector{T}` | Moments supplied by the caller |
| `μ_fit` | `Vector{T}` | Moments achieved by the fit |


## References

- Jaynes, E. T. (1957). Information theory and statistical mechanics.
  *Physical Review*, 106(4), 620–630.
- Mead, L. R., & Papanicolaou, N. (1984). Maximum entropy in the problem of
  moments. *Journal of Mathematical Physics*, 25(8), 2404–2417.
