# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using Distributions
using StatsBase
using Random

# Gaussian peak on a flat background; the parameters are treated as known, so the only
# freedom of the test is the Poisson fluctuation of the counts.
const gof_model = (x, v) -> v.n * pdf(Normal(v.μ, v.σ), x) + v.b
const gof_v = (μ = 100.0, σ = 1.0, n = 2000.0, b = 5.0)
gof_edges = 80.0:0.2:120.0
gof_λ = step(gof_edges) .* gof_model.((gof_edges[1:end-1] .+ gof_edges[2:end]) ./ 2, Ref(gof_v))
gof_hist(rng, λ) = Histogram(gof_edges, [rand(rng, Poisson(l)) for l in λ])

@testset "poisson deviance moments" begin
    rng = Random.MersenneTwister(4711)
    h = gof_hist(rng, gof_λ)
    D, E, V = LegendSpecFits.poisson_deviance(gof_model, h, gof_v)
    @test D > 0 && E > 0 && V > 0
    @test D ≈ LegendSpecFits.likelihood_ratio(x -> gof_model(x, gof_v), h)

    # the analytic moments agree with the sampled distribution of the deviance
    n_mc = 3000
    Ds = [LegendSpecFits.poisson_deviance(gof_model, gof_hist(rng, gof_λ), gof_v)[1] for _ in 1:n_mc]
    @test isapprox(mean(Ds), E; rtol = 0.02)
    @test isapprox(var(Ds), V; rtol = 0.15)

    # bins with a vanishing expectation contribute nothing when empty and everything when not
    λ0 = copy(gof_λ); λ0[1] = 0.0
    model0 = (x, v) -> x < gof_edges[2] ? 0.0 : gof_model(x, v)
    h0 = gof_hist(rng, λ0)
    D0, E0, V0 = LegendSpecFits.poisson_deviance(model0, h0, gof_v)
    @test isfinite(D0) && isfinite(E0) && isfinite(V0)
    h1 = Histogram(gof_edges, copy(h0.weights)); h1.weights[1] = 3
    @test LegendSpecFits.poisson_deviance(model0, h1, gof_v)[1] == Inf
end

@testset "p_value_poissonll is uniform for sparse histograms" begin
    rng = Random.MersenneTwister(1234)
    n_mc = 2000
    # the parameters are known here, so the model takes an empty parameter tuple and the
    # expectation of the deviance gets no parameter correction
    known = (x, _) -> gof_model(x, gof_v)
    # a sparse spectrum: at most a few counts per bin, most bins below one expected count
    λ_sparse = 0.02 .* gof_λ
    known_sparse = (x, _) -> 0.02 * gof_model(x, gof_v)
    p_sparse = [p_value_poissonll(known_sparse, gof_hist(rng, λ_sparse), NamedTuple())[1] for _ in 1:n_mc]
    @test 0.03 < mean(p_sparse .< 0.05) < 0.07
    @test 0.45 < mean(p_sparse) < 0.55
    p_dense = [p_value_poissonll(known, gof_hist(rng, gof_λ), NamedTuple())[1] for _ in 1:n_mc]
    @test 0.03 < mean(p_dense .< 0.05) < 0.07

    # fitted parameters reduce the expectation of the deviance by their number; dof counts every bin
    h = gof_hist(rng, gof_λ)
    D, E, V = LegendSpecFits.poisson_deviance(gof_model, h, gof_v)
    pval, chi2, dof = p_value_poissonll(gof_model, h, gof_v)
    @test chi2 == D
    @test pval ≈ ccdf(Normal(), (D - (E - length(gof_v))) / sqrt(V))
    @test dof == length(gof_λ) - length(gof_v)

    # a shifted model is rejected
    v_off = merge(gof_v, (μ = 100.5,))
    p_off = [p_value_poissonll(gof_model, gof_hist(rng, gof_λ), v_off)[1] for _ in 1:200]
    @test mean(p_off .< 0.01) > 0.9
    # a bin with counts where the model vanishes gives p = 0
    model0 = (x, v) -> x < gof_edges[2] ? 0.0 : gof_model(x, v)
    h1 = gof_hist(rng, gof_λ); h1.weights[1] = 3
    @test p_value_poissonll(model0, h1, gof_v)[1] == 0.0
end
