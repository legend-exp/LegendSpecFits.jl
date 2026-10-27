# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test
using Distributions
using ForwardDiff
using StatsBase
using Random

@testset "hist_loglike" begin
    rng = Random.MersenneTwister(1234)
    edges = range(-3, 3, 41)
    h = Histogram(edges, rand(rng, 0:20, 40))
    h.weights[[3, 10, 11, 30]] .= 0

    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    widths = diff(edges)
    reference(f) = sum(logpdf(Poisson(bw * f(x)), k) for (x, bw, k) in zip(centers, widths, h.weights))

    # the model is exactly zero for x > 2, where some of the bins contain counts
    f(x, v) = x > 2 ? zero(v.n) : v.n * pdf(Normal(v.μ, v.σ), x) + v.b
    f_zero_on_empty(x, v) = (x < -2.5 || x > 2.5) ? zero(v.n) : f(x, v)
    v = (n = 60.0, μ = 0.2, σ = 0.9, b = 1.5)

    ll = HistLogLike(h)
    @test ll(x -> f(x, v)) ≈ hist_loglike(x -> f(x, v), h) rtol = 1e-12
    @test ll(x -> f_zero_on_empty(x, v)) ≈ reference(x -> f_zero_on_empty(x, v)) rtol = 1e-12
    @test hist_loglike(x -> f_zero_on_empty(x, v), h) ≈ reference(x -> f_zero_on_empty(x, v)) rtol = 1e-12
    @test ll(x -> v.b) ≈ reference(x -> v.b) rtol = 1e-12

    # zero expected counts in bins with nonzero counts give -Inf
    @test ll(x -> f(x, v)) == -Inf
    @test reference(x -> f(x, v)) == -Inf

    @testset "ForwardDiff" begin
        # strictly positive model, so that the gradient is finite in every bin
        f_pos(x, u) = u[1] * pdf(Normal(u[2], u[3]), x) + u[4]
        u0 = [60.0, 0.2, 0.9, 1.5]
        ref(u) = reference(x -> f_pos(x, u))
        @test ForwardDiff.gradient(u -> ll(x -> f_pos(x, u)), u0) ≈ ForwardDiff.gradient(ref, u0) rtol = 1e-12
        @test ForwardDiff.gradient(u -> hist_loglike(x -> f_pos(x, u), h), u0) ≈ ForwardDiff.gradient(ref, u0) rtol = 1e-12
    end

    @testset "invalid input" begin
        @test_throws DomainError ll(x -> -1.0)
        @test_throws "λ >= zero(λ)" ll(x -> -1.0)
        @test_throws DomainError ForwardDiff.derivative(u -> ll(x -> -u), 1.0)
        @test_throws DomainError Poisson(-1.0)

        h_float = Histogram(edges, fill(1.5, 40))
        @test HistLogLike(h_float)(x -> 1.0) == -Inf
        @test hist_loglike(x -> 1.0, h_float) == sum(logpdf(Poisson(w * 1.0), 1.5) for w in widths)
    end
end
