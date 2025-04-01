# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using LegendDataTypes: fast_flatten
using Measurements: value as mvalue
using Distributions
using StatsBase
using Interpolations
using Unitful 
include("test_utils.jl")

@testset "specfit" begin
    # load data, simple calibration 
    energy_test, th228_lines = generate_mc_spectrum(200000)
    
    # simple calibration fit 
    window_sizes =  vcat([(25.0u"keV",25.0u"keV") for _ in 1:6], (30.0u"keV",30.0u"keV"))
    result_simple, report_simple = simple_calibration(ustrip.(energy_test), th228_lines, window_sizes; calib_type=:th228, quantile_perc=0.995)

    # fit 
    result, report = fit_peaks(result_simple.peakhists, result_simple.peakstats, ustrip.(th228_lines),; uncertainty=true,calib_type = :th228, fit_func = [:gamma_def, :gamma_tails, :gamma_bckExp, :gamma_tails_bckFlat, :gamma_bckSlope, :gamma_minimal, :gamma_bckFlat]);
    @test all([isapprox(mvalue(result[peak].µ), ustrip(th228_lines[i]), atol = 0.2*ustrip(th228_lines[i])) for (i, peak) in enumerate(sort!(Float64.(keys(result))))])
end

@testset "fit_subpeaks_th228" begin

    # Compose a dummy energy peak (E)
    n = 100000
    b = 20000
    μ = 1000.0
    σ = 20.0
    E = vcat(μ .+ σ .* randn(n), rand(Uniform(μ - 10*σ, μ + 10*σ), b))

    # Fit the initial histogram and check that the determined quantities match the given ones
    bins = range(μ - 10*σ, stop = μ + 10*σ, length = round(Int, sqrt(b)))
    h = fit(Histogram, E, bins)
    h_result, h_report = fit_single_peak_th228(h, estimate_single_peak_stats(h), low_e_tail = false)
    @testset "fit single_peak_th228" begin
        @test isapprox(mvalue(h_result.μ), μ, rtol = 0.01)
        @test isapprox(mvalue(h_result.σ), σ, rtol = 0.01)
        @test isapprox(mvalue(h_result.n), n, rtol = 0.01)
        @test isapprox(mvalue(h_result.background), b / (20σ), rtol = 0.1)
    end

    # Introduce a dummy quantity (Q) which has two peaks (around 0 and around 1)
    # that should have 90% of the signal and 50% of the background around 0
    # and 10% of the signal and 50% of the background around 1
    sf = 0.9
    bsf = 0.5
    Q = vcat(0.2 * randn(round(Int, n*sf)), 0.2 * randn(round(Int, n*(1-sf))) .+ 1, 0.2 * randn(round(Int, b*bsf)), 0.2 * randn(round(Int, b*(1-bsf))) .+ 1)

    # Perform the cut on this dummy Q by selecting events with Q < 0.5 (the peak around 0) 
    # and check that the survival fractions for signal and background agree with what was initially given
    cut_value = 0.5
    h_survived = fit(Histogram, E[findall(Q .<= cut_value)], bins)
    h_cut = fit(Histogram, E[findall(Q .> cut_value)], bins)
    result, report = LegendSpecFits.fit_subpeaks_th228(h_survived, h_cut, h_result, uncertainty = true)
    @testset "fit_subpeaks_th228" begin
        @test isapprox(mvalue(result.sf), sf, rtol = 0.02)
        @test isapprox(mvalue(result.bsf), bsf, rtol = 0.02)
    end

end
