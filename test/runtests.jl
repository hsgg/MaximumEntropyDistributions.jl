using MaximumEntropyDistributions
using Test
using Printf
using StatsBase

@testset "MaximumEntropyDistributions.jl" begin
    xmin = -5.0
    xmax = 5.0

    data = randn(1000)
    #data = rand(1000)

    moments(n) = [moment(data, k) for k in 1:n]


    println("=== MaxEnt PDF for x ∈ [$(xmin), $(xmax)] ===\n")

    for N in 1:5
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

    m  = MaxEntPDF(xmin, xmax, moments(5))
    xs = range(xmin, xmax, length=20)
    ps = m.(xs)
    ps_norm = ps ./ maximum(ps)
    @test all(isfinite.(ps))

    println("PDF shape (N=$(length(m.λ)), normalised to peak):")
    for (x, p) in zip(xs, ps_norm)
        bar = "█" ^ round(Int, 40 * p)
        @printf "  x = %7.4f  %s\n" x bar
    end
end
