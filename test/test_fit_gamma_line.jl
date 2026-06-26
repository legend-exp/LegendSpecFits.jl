# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using Measurements: value as mvalue
using Distributions
using StatsBase
using Random

@testset "fit_gamma_line" begin
    # Synthetic spectrum: K40-style line at 1460.8 keV on a flat background.
    Random.seed!(2026)
    μ_true   = 1460.8
    σ_true   = 1.2
    n_true   = 800
    bkg_rate = 50.0           # cts / keV
    win_keV  = 30.0
    binw     = 0.5
    edges    = (μ_true - win_keV):binw:(μ_true + win_keV)

    energies = vcat(
        rand(Normal(μ_true, σ_true), n_true),
        rand(Uniform(μ_true - win_keV, μ_true + win_keV), round(Int, bkg_rate * 2 * win_keV)),
    )
    h  = fit(Histogram, energies, edges)
    ps = estimate_single_peak_stats(h)

    @testset "gauss_flat / MLE" begin
        result, report = fit_gamma_line(h, ps; fit_func = :gauss_flat, method = :mle,
                                          uncertainty = true)
        @test mvalue(result.gof.converged) === true || result.gof.converged
        @test isapprox(mvalue(result.μ), μ_true; atol = 0.2)
        @test isapprox(mvalue(result.σ), σ_true; rtol = 0.25)
        @test isapprox(mvalue(result.n), n_true; rtol = 0.20)
        # `background` is the polynomial-background rate density in cts/keV
        # (model evaluates `n·𝒩(x) + background`, so `background` matches the
        # generation rate of the uniform component directly).
        @test isapprox(mvalue(result.background), bkg_rate; rtol = 0.15)
        # Report shape matches the existing LegendMakie recipe dispatch.
        @test keys(report) == (:v, :h, :f_fit, :f_components, :gof)
        @test report.f_fit(μ_true) > 0
    end

    @testset "gauss_lin / MLE" begin
        result, report = fit_gamma_line(h, ps; fit_func = :gauss_lin, method = :mle,
                                          uncertainty = true)
        @test isapprox(mvalue(result.μ), μ_true; atol = 0.2)
        @test isapprox(mvalue(result.n), n_true; rtol = 0.25)
        @test haskey(result, :background_slope)
        @test keys(report) == (:v, :h, :f_fit, :f_components, :gof)
    end

    @testset "gauss_quad / MLE" begin
        result, report = fit_gamma_line(h, ps; fit_func = :gauss_quad, method = :mle,
                                          uncertainty = true)
        @test isapprox(mvalue(result.μ), μ_true; atol = 0.3)
        @test isapprox(mvalue(result.n), n_true; rtol = 0.35)
        @test haskey(result, :background_curv)
        @test keys(report) == (:v, :h, :f_fit, :f_components, :gof)
    end

    @testset "gauss_flat / BAT" begin
        # Small nsamples so the test stays fast; this is a smoke test on the
        # BAT path, not a precision benchmark.
        result, report = fit_gamma_line(h, ps; fit_func = :gauss_flat, method = :bat,
                                          nsamples = 5_000, nchains = 2, n_thin = 500)
        @test isapprox(mvalue(result.μ), μ_true; atol = 0.5)
        @test isapprox(mvalue(result.n), n_true; rtol = 0.30)
        @test length(result.S_samples) >= 100
        @test isfinite(result.ul_90)
        @test result.detection in (:detected, :upper_limit)
        @test keys(report) == (:v, :h, :f_fit, :f_components, :gof)
    end

    @testset "pseudo_prior override" begin
        # Pin μ via a Dirac-ish narrow prior and check the fit honours it.
        prior_override = LegendSpecFits.ValueShapes.NamedTupleDist(
            μ = Normal(μ_true, 1e-3),
        )
        result, _ = fit_gamma_line(h, ps; fit_func = :gauss_flat, method = :mle,
                                     pseudo_prior = prior_override, uncertainty = true)
        @test isapprox(mvalue(result.μ), μ_true; atol = 0.01)
    end
end
