# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test

using LegendDataManagement
using TypedTables
using Unitful
using UnitfulAtomic
using StatsBase

@testset "SiPM calibration routine" begin 

    # generate SiPM thresholds as normal distribution with standard deviation 1
    result_thres, report_thres = LegendSpecFits.fit_sipm_threshold(randn(100_000), -5.0, 5.0)
    @test result_thres isa NamedTuple{(:μ, :σ, :n, :gof, :μ_simple, :σ_simple)}
    @test isapprox(result_thres.μ, 0, atol = 0.01)
    @test isapprox(result_thres.σ, 1, rtol = 0.02)
    @test isapprox(result_thres.n, 100_000, rtol = 0.01)

    # generate fake SiPM amplitude spectrum with 4 P.E. peaks
    N = 4
    peakpos = 1.4
    peakσ = 0.3
    e_uncal = vcat([randn(round(Int, exp10(7-0.5*i))).* peakσ .+ i*peakpos for i in Base.OneTo(N)]...)
    result_simple, report_simple = LegendSpecFits.sipm_simple_calibration(e_uncal, peakfinder_rtol=0.005, min_pe_peak=1, max_pe_peak=N, n_fwhm_noise_cut=0.0)
    # check output format
    @test result_simple isa NamedTuple{(:pe_simple_cal, :peakpos, :f_simple_calib, :f_simple_uncal, :c, :offset, :noisepeakpos, :noisepeakwidth)}
    @test length(result_simple.pe_simple_cal) == length(e_uncal)
    # check the correctness of the noisepeak
    @test isapprox(result_simple.noisepeakpos, peakpos, rtol = 0.05)   
    @test isapprox(result_simple.noisepeakwidth, peakσ * 2sqrt(2log(2)), rtol = 0.05)
    # reproduce the peak positions within ~5%
    @test all(isapprox.(result_simple.peakpos, Base.OneTo(N) .* peakpos, rtol = 0.05))
    # check that the calibration was done correctly
    @test isapprox.(result_simple.c, inv(peakpos), rtol = 0.05)
    @test all(isapprox.(result_simple.pe_simple_cal, result_simple.f_simple_calib.(e_uncal)))
    # @test_nowarn LegendMakie.lplot(report_simple, title = "Test")

    while true; try
    global result_fit, report_fit = LegendSpecFits.fit_sipm_spectrum(result_simple.pe_simple_cal, 0.5, N + 0.5, n_mixtures = N)
    break; catch; end; end
    idx = sortperm(result_fit.μ)
    @test result_fit isa NamedTuple{(:μ, :σ, :w, :n_pos_mixtures, :n_mixtures, :peaks, :positions_cal, :positions, :resolutions_cal, :resolutions, :gof)}
    @test all(isapprox.(result_fit.μ[idx], 1:N, rtol = 0.05))
    @test all(isapprox.(result_fit.σ[idx], peakσ / peakpos, rtol = 0.05))
    @test all(isapprox.(result_fit.resolutions[idx], peakσ / peakpos * 2sqrt(2log(2)), rtol = 0.05))
    # @test_nowarn LegendMakie.lplot(report_fit, xerrscaling = 5, title = "Test")

    result_calib, report_calib = LegendSpecFits.fit_calibration(1, result_fit.positions, collect(result_fit.peaks) * u"e_au", e_expression="e_simple")
    e_cal = @test_nowarn ljl_propfunc(result_calib.func)(Table(e_simple = result_simple.pe_simple_cal))
    h = fit(Histogram, ustrip.(u"e_au", e_cal), 0.5:0.2:N+0.5)
    @test result_calib.peaks == Base.OneTo(N)u"e_au"
    @test all(isapprox.(result_calib.μ, result_fit.μ[idx]))
    @test all(isapprox.(result_calib.par, (0u"e_au", 1u"e_au"), atol = 0.05u"e_au"))
    @test midpoints(first(h.edges))[findall(i -> get(h.weights, i-1, Inf) < h.weights[i] && h.weights[i] > get(h.weights, i+1, Inf), eachindex(h.weights))] == 1:N
    # @test_nowarn LegendMakie.lplot(report_calib, xerrscaling = 5, title = "Test")

end