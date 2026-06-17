using MaximumEntropyDistributions
using Test
using Printf
using StatsBase
using UnicodePlots

@testset "MaximumEntropyDistributions.jl" begin
    xmin = -5.0
    xmax = 5.0

    data = randn(1000)
    #data = rand(1000)

    moments(n) = [moment(data, k) for k in 1:n]


    println("=== MaxEnt PDF for x ∈ [$(xmin), $(xmax)] ===\n")

    for N in 1:10
        m = MaxEntPDF(xmin, xmax, moments(N); n_quad=1000)
        println("N = $N moments")
        println("  λ         = ", round.(m.λ, sigdigits=6))
        println("  μ_target  = ", round.(m.μ_target, sigdigits=8))
        println("  μ_fit     = ", round.(m.μ_fit,    sigdigits=8))
        println("  max |Δμ|  = ", maximum(abs, m.μ_fit .- m.μ_target))

        # basic tests
        @test m.xmin == xmin
        @test m.xmax == xmax
        @test isfinite(m.Z)
        @test length(m.λ) == N
        @test length(m.μ_target) == N
        @test length(m.μ_fit) == N

        @test m.μ_target == moments(N)

        if N >= 2
            @test m.μ_fit ≈ m.μ_target
        end

        println()
    end


    # UnicodePlots comparison: true Gaussian vs MaxEnt fits
    xs_plot = range(xmin, xmax, length=300)
    gaussian(x) = exp(-x^2 / 2) / sqrt(2π)

    plt = lineplot(collect(xs_plot), gaussian.(xs_plot);
        name   = "Gaussian",
        title  = "MaxEnt vs Gaussian N(0,1)",
        xlabel = "x",
        ylabel = "p(x)",
        width  = 80,
        height = 20,
    )
    for N in 1:10
        mN = MaxEntPDF(xmin, xmax, moments(N); n_quad=1000)
        lineplot!(plt, collect(xs_plot), mN.(xs_plot); name = "MaxEnt N=$N")
    end
    println(plt)
end
