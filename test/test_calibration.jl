# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test
using Measurements: value as mvalue
using Distributions
using Unitful
using Logging

logger = ConsoleLogger(stderr, Logging.Info)

@testset "Energy calibration" begin
    ecal = filter(x -> x <= 265_000, vcat(rand(Distributions.Exponential(70_000),97_000), 261_450 .+ 200 .* randn(2_000), 210_350 .+ 185 .* randn(500), 159_300 .+ 170 .* randn(500)))
    lines = [:Tl208DEP, :Tl208SEP, :Tl208FEP]
    energies = [1592.513, 2103.512, 2614.511]u"keV"
    result_autocal, report_autocal = LegendSpecFits.autocal_energy(ecal, energies, α = 0.01, rtol = 2)
    @test isapprox(result_autocal.c, 0.01, rtol = 0.01)
    result_simple, report_simple = LegendSpecFits.simple_calibration(ecal, energies, [25, 25, 35]u"keV", [25, 25, 30]u"keV", calib_type = :th228)
    @test isapprox(result_simple.c, 0.01 * u"keV", rtol = 0.01)
    m_cal_simple = result_simple.c
    with_logger(logger) do
        @test_nowarn result_ctc, report_ctc = LegendSpecFits.ctc_energy(ecal .* m_cal_simple, rand(length(ecal)), 2614.5u"keV", (5u"keV", 5u"keV"), m_cal_simple)
    end
    result_fit, report_fit = LegendSpecFits.fit_peaks(result_simple.peakhists, result_simple.peakstats, lines; e_unit=result_simple.unit, calib_type=:th228, m_cal_simple=m_cal_simple)
    @test result_fit isa AbstractDict || report_fit isa AbstractDict
    @test length(lines) == length(report_fit)
    μ_fit = getfield.(getindex.(Ref(result_fit), lines), :centroid)
    result_calib, report_calib = LegendSpecFits.fit_calibration(1, μ_fit, energies)
    @test all(isapprox.(sort(mvalue.(result_calib.μ)), [159_300, 210_350, 261_450], rtol = 0.001))
    @test result_calib.peaks == energies
    @test isapprox(result_calib.par[2] , 0.01u"keV", rtol = 9.005)
    f_cal_widths(x) = report_calib.f_fit(x) .* report_calib.e_unit .- first(report_calib.par)
    fwhm_fit = f_cal_widths.(getfield.(getindex.(Ref(result_fit), lines), :fwhm))
    result_fwhm, report_fwhm = LegendSpecFits.fit_fwhm(1, energies, fwhm_fit, uncertainty=true)
    @test result_fwhm.peaks == energies
    @test unit(result_fwhm.qbb) == u"keV"
end