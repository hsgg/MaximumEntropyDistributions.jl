using MaximumEntropyDistributions
using Test
using Printf
using Statistics
using UnicodePlots
using Distributions

@testset "MaximumEntropyDistributions.jl" begin
    n_samples = 2000
    n_quad    = 500
    n_max     = 6

    test_cases = [
        #  name               dist                  xmin   xmax
        ("Uniform(0,1)",      Uniform(0, 1),         0.0,   1.0),
        ("Beta(2,5)",         Beta(2, 5),            0.0,   1.0),
        ("Beta(0.5,0.5)",     Beta(0.5, 0.5),        0.0,   1.0),
        ("Normal(0,1)",       Normal(0, 1),          -5.0,  5.0),
        ("Exponential(1)",    Exponential(1),         0.0,  8.0),
    ]

    @testset "$name" for (name, dist, xmin, xmax) in test_cases
        data = rand(dist, n_samples)
        all_moments(n) = [moment(data, k) for k in 1:n]

        println("\n=== $name on [$xmin, $xmax] ===")

        for N in 1:n_max
            @show N
            m = MaxEntPDF(xmin, xmax, all_moments(N); n_quad)
            @test m.xmin == xmin
            @test m.xmax == xmax
            @test isfinite(m.Z)
            if N >= 2
                #@test m.μ_fit ≈ m.μ_target
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
            #mN = MaxEntPDF(xmin, xmax, all_moments(N); n_quad)
            #lineplot!(plt, collect(xs), mN.(xs); name = "MaxEnt N=$N")
        end
        println(plt)
    end
end
