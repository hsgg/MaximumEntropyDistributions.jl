using MaximumEntropyDistributions
using Test
using Printf
using Statistics
using UnicodePlots
using Distributions
using StableRNGs
using LinearAlgebra

@testset "MaximumEntropyDistributions.jl" begin
    n_samples = 2000
    n_quad    = 500
    n_max     = 6

    test_cases = [
        #  name               dist                  xmin   xmax   seed
        ("Uniform(0,1)",      Uniform(0, 1),         0.0,   1.0,  1),
        ("Uniform(0,1) on [-1,1.5]", Uniform(0, 1), -1.0,   1.5,  1),
        ("Beta(2,5)",         Beta(2, 5),            0.0,   1.0,  2),
        ("Beta(0.5,0.5)",     Beta(0.5, 0.5),        0.0,   1.0,  3),
        ("Normal(0,1)",       Normal(0, 1),          -5.0,  5.0,  4),
        ("Exponential(1)",    Exponential(1),         0.0,  8.0,  2),
        ("LogNormal(1,2)",  LogNormal(0, 0.5),      0.0,   6.0, 10),
    ]

    @testset "$name" for (name, dist, xmin, xmax, seed) in test_cases
        rng  = StableRNG(seed)
        data = rand(rng, dist, n_samples)
        all_moments(n) = [mean(data .^ k) for k in 1:n]

        println("\n=== $name on [$xmin, $xmax] (seed=$seed) ===")

        for N in 1:n_max
            @show N
            m = MaxEntPDF(xmin, xmax, all_moments(N); n_quad)
            @test length(m.λ) == N
            @test length(m.μ_target) == N
            @test length(m.μ_fit) == N
            @test m.xmin == xmin
            @test m.xmax == xmax
            @test isfinite(m.Z)
            @test m.μ_fit ≈ m.μ_target
        end

        # Plot true PDF alongside MaxEnt fits for all N
        xs = range(xmin, xmax, length=300)
        plt = lineplot(collect(xs), pdf.(dist, xs);
            name   = "True PDF",
            title  = "MaxEnt vs $name",
            color  = :black,
            xlabel = "x",
            ylabel = "p(x)",
            width  = 80,
            height = 20,
        )
        for N in 1:n_max
            mN = MaxEntPDF(xmin, xmax, all_moments(N); n_quad)
            lineplot!(plt, collect(xs), mN.(xs); name = "MaxEnt N=$N")
        end
        println(plt)
    end

    # These seeds reliably expose numerical issues in the Newton-Raphson solver:
    # the Hessian H = Cov(x^n, x^m) becomes ill-conditioned for Exponential(1)
    # samples with N≥5 raw moments.  Two distinct failure modes are tested:
    #
    #   bad_cases  — solver completes but moment residual >> convergence tolerance
    #   error_cases — H was exactly singular, causing H\g to throw SingularException
    #                 in the unfixed solver; now caught and handled via pinv(H)*g
    @testset "Robustness (seeds that trigger singularity)" begin
        # Seeds discovered by scanning 1:200 for (Exponential(1), N in 5:8) with
        # raw moments and checking norm(μ_fit - μ) > 1e-4.
        bad_cases = [
            # (seed, dist,          xmin, xmax, N)
            (1,  Exponential(1),  0.0,  8.0,  5),  # resid ≈ 16
            (5,  Exponential(1),  0.0,  8.0,  5),  # resid ≈ 19
            (7,  Exponential(1),  0.0,  8.0,  6),  # resid ≈ 306
        ]
        for (seed, dist, xmin, xmax, N) in bad_cases
            rng  = StableRNG(seed)
            data = rand(rng, dist, n_samples)
            μ    = [mean(data .^ k) for k in 1:N]
            m    = MaxEntPDF(xmin, xmax, μ; n_quad)
            @test isfinite(m.Z)                      # solver doesn't crash ...
            @test_broken norm(m.μ_fit .- μ) < 1e-4  # ... but moment matching fails
        end

        # Seeds that triggered SingularException (H\g threw) in the unfixed solver.
        # Discovered by scanning seeds 1:200 for (Exponential(1), N=8).
        error_cases = [
            # (seed, N)
            (41, 8),   # resid ≈ 1.45e5
            (93, 8),   # resid ≈ 1.22e5
        ]
        for (seed, N) in error_cases
            rng  = StableRNG(seed)
            data = rand(rng, Exponential(1), n_samples)
            μ    = [mean(data .^ k) for k in 1:N]
            m    = MaxEntPDF(0.0, 8.0, μ; n_quad)
            @test isfinite(m.Z)                      # must not throw after fix
            @test_broken norm(m.μ_fit .- μ) < 1e-4
        end
    end

    # ── Cumulant utilities ────────────────────────────────────────────────────

    @testset "moments_to_cumulants / cumulants_to_moments roundtrip" begin
        for _ in 1:20
            N = rand(1:8)
            μ = rand(N) .* 10 .+ 0.1   # positive moments
            κ = moments_to_cumulants(μ)
            @test cumulants_to_moments(κ) ≈ μ atol = 1e-4
            @test moments_to_cumulants(cumulants_to_moments(κ)) ≈ κ atol = 1e-4
        end
    end

    @testset "moment_cumulant_jacobian vs finite differences" begin
        for _ in 1:10
            N = rand(2:6)
            μ = rand(N) .* 5 .+ 0.5
            κ = moments_to_cumulants(μ)
            M = MaximumEntropyDistributions.moment_cumulant_jacobian(μ, κ)

            # Compare against finite differences
            ε = 1e-7
            M_fd = zeros(N, N)
            for k in 1:N
                μ_p = copy(μ); μ_p[k] += ε
                μ_m = copy(μ); μ_m[k] -= ε
                κ_p = moments_to_cumulants(μ_p)
                κ_m = moments_to_cumulants(μ_m)
                M_fd[:, k] .= (κ_p .- κ_m) ./ (2ε)
            end
            @test M ≈ M_fd atol=0.05
        end
    end

    # ── MaxEntPDF_cumulants fitting ───────────────────────────────────────────

    @testset "MaxEntPDF_cumulants: $name" for (name, dist, xmin, xmax, seed) in test_cases
        rng  = StableRNG(seed)
        data = rand(rng, dist, n_samples)
        all_moments(n) = [mean(data .^ k) for k in 1:n]

        for N in 1:n_max
            μ = all_moments(N)
            κ = moments_to_cumulants(μ)

            m_cum = MaxEntPDF_cumulants(xmin, xmax, κ; n_quad)
            @test length(m_cum.λ) == N
            @test isfinite(m_cum.Z)

            # Cumulant matching should be accurate
            κ_fit = moments_to_cumulants(m_cum.μ_fit)
            @test κ_fit ≈ κ atol = 1e-6
        end
    end

    @testset "MaxEntPDF_cumulants agrees with MaxEntPDF" begin
        for (name, dist, xmin, xmax, seed) in test_cases
            rng  = StableRNG(seed)
            data = rand(rng, dist, n_samples)
            for N in 1:n_max
                μ = [mean(data .^ k) for k in 1:N]
                κ = moments_to_cumulants(μ)

                m_moments = MaxEntPDF(xmin, xmax, μ; n_quad)
                m_cum     = MaxEntPDF_cumulants(xmin, xmax, κ; n_quad)

                @test m_moments.λ ≈ m_cum.λ atol = 1e-6
                @test m_moments.Z ≈ m_cum.Z atol = 1e-6
                @test m_moments.μ_fit ≈ m_cum.μ_fit atol = 1e-6
            end
        end
    end
end
