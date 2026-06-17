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
        ("Beta(2,5)",         Beta(2, 5),            0.0,   1.0,  2),
        ("Beta(0.5,0.5)",     Beta(0.5, 0.5),        0.0,   1.0,  3),
        ("Normal(0,1)",       Normal(0, 1),          -5.0,  5.0,  4),
        ("Exponential(1)",    Exponential(1),         0.0,  8.0,  2),
    ]

    @testset "$name" for (name, dist, xmin, xmax, seed) in test_cases
        rng  = StableRNG(seed)
        data = rand(rng, dist, n_samples)
        all_moments(n) = [mean(data .^ k) for k in 1:n]

        println("\n=== $name on [$xmin, $xmax] (seed=$seed) ===")

        for N in 1:n_max
            @show N
            m = MaxEntPDF(xmin, xmax, all_moments(N); n_quad)
            @test m.xmin == xmin
            @test m.xmax == xmax
            @test isfinite(m.Z)
            if dist isa Normal && N >= 2
                @test m.μ_fit ≈ m.μ_target
            end
        end

        # Plot true PDF alongside MaxEnt fits for all N
        xs = range(xmin, xmax, length=300)
        plt = lineplot(collect(xs), pdf.(dist, xs);
            name   = "True PDF",
            title  = "MaxEnt vs $name",
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
end
