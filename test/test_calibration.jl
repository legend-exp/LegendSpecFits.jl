# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test
using LegendDataManagement: ljl_propfunc
using Measurements: value as mvalue, uncertainty as muncert
using Distributions
using TypedTables
using Unitful
using Logging
using Random

logger = ConsoleLogger(stderr, Logging.Info)

@testset "Energy calibration" begin
    # Seeded, since peak identification fails for some random data sets:
    rng = Xoshiro(1234)
    ecal = filter(x -> x <= 265_000, vcat(rand(rng, Distributions.Exponential(70_000),97_000), 261_450 .+ 200 .* randn(rng, 2_000), 210_350 .+ 185 .* randn(rng, 500), 159_300 .+ 170 .* randn(rng, 500)))
    lines = [:Tl208DEP, :Tl208SEP, :Tl208FEP]
    energies = [1592.513, 2103.512, 2614.511]u"keV"
    result_autocal, report_autocal = LegendSpecFits.autocal_energy(ecal, energies, α = 0.01, rtol = 2)
    @test isapprox(result_autocal.c, 0.01, rtol = 0.01)
    result_simple, report_simple = LegendSpecFits.simple_calibration(ecal, energies, [25, 25, 35]u"keV", [25, 25, 30]u"keV", calib_type = :th228)
    @test isapprox(result_simple.c, 0.01 * u"keV", rtol = 0.01)
    m_cal_simple = result_simple.c
    with_logger(logger) do
        @test_nowarn result_ctc, report_ctc = LegendSpecFits.ctc_energy(ecal .* m_cal_simple, rand(rng, length(ecal)), 2614.5u"keV", (5u"keV", 5u"keV"), m_cal_simple)
    end
    result_fit, report_fit = LegendSpecFits.fit_peaks(result_simple.peakhists, result_simple.peakstats, lines; e_unit=result_simple.unit, calib_type=:th228, m_cal_simple=m_cal_simple)
    @test result_fit isa AbstractDict || report_fit isa AbstractDict
    @test length(lines) == length(report_fit)
    μ_fit = getfield.(getindex.(Ref(result_fit), lines), :centroid)
    result_calib, report_calib = LegendSpecFits.fit_calibration(1, μ_fit, energies; e_expression = "e_raw")
    @test all(isapprox.(sort(mvalue.(result_calib.μ)), [159_300, 210_350, 261_450], rtol = 0.001))
    @test result_calib.peaks == energies
    @test isapprox(result_calib.par[2] , 0.01u"keV", rtol = 9.005)
    # the error function reproduces the uncertainty of the calibration curve, which includes the
    # correlations between the fit parameters
    e_raw_test = [150_000.0, 200_000.0, 250_000.0]
    cal_err = ustrip.(u"keV", ljl_propfunc(result_calib.func_err).(Table(e_raw = e_raw_test)))
    @test mvalue.(cal_err) ≈ mvalue.(report_calib.f_fit.(e_raw_test))
    @test muncert.(cal_err) ≈ muncert.(report_calib.f_fit.(e_raw_test))
    f_cal_widths(x) = report_calib.f_fit(x) .* report_calib.e_unit .- first(report_calib.par)
    fwhm_fit = f_cal_widths.(getfield.(getindex.(Ref(result_fit), lines), :fwhm))
    result_fwhm, report_fwhm = LegendSpecFits.fit_fwhm(1, energies, fwhm_fit, uncertainty=true)
    @test result_fwhm.peaks == energies
    @test unit(result_fwhm.qbb) == u"keV"
    # three nearly flat points put the pre-fit of the Fano term far from its pull term in the
    # quadratic fit; the fit still has to converge from there
    result_fwhm, report_fwhm = LegendSpecFits.fit_fwhm(2, energies, fwhm_fit, uncertainty=true)
    @test result_fwhm.gof.converged

    # resolution curve with known parameters: fwhm² = enc + fano·E + ct·E² with enc = 1 keV², the
    # Fano term of germanium and ct = 1e-7, at the energies of the Th-228 lines
    e_lines = [583.191, 727.330, 860.564, 1592.513, 2103.512, 2614.511]u"keV"
    par_true = [1.0, mvalue(ustrip(u"keV", LegendSpecFits.fwhm_fano_term_ge)), 1e-7]
    fwhm_syn = measurement.(sqrt.(par_true[1] .+ par_true[2] .* ustrip.(u"keV", e_lines) .+ par_true[3] .* ustrip.(u"keV", e_lines) .^ 2), 0.02) .* u"keV"
    # the resolution curve is fitted on an energy scale of its own, so the exported functions and the
    # parameters have to come back on the keV scale the callers work on
    e_test = [500.0, 1500.0, 2039.061, 2614.5]
    test_tbl = Table(e_cal = e_test .* u"keV", e_raw = e_test)
    for pol_order in (1, 2), correlated in (true, false)
        result, report = LegendSpecFits.fit_fwhm(pol_order, e_lines, fwhm_syn; e_type_cal = :e_cal, e_expression = "e_raw", correlated)
        @test result.gof.converged
        @test all(mvalue.(ustrip.(u"keV", ljl_propfunc(result.func_cal).(test_tbl))) .≈ mvalue.(report.f_fit.(e_test)))
        @test all(ljl_propfunc(result.func).(test_tbl) .≈ ljl_propfunc(result.func_cal).(test_tbl))
        @test mvalue(result.qbb) ≈ mvalue(report.f_fit(2039.061)) * u"keV"
        # the ENC term is reported as the width √enc in keV, the linear term as the Fano factor
        @test keys(result.par) == (pol_order == 1 ? (:enc, :fano) : (:enc, :fano, :ct))
        @test unit(result.par.enc) == u"keV" && unit(result.par.fano) == NoUnits
        @test mvalue(result.par.enc) ≈ mvalue(report.f_fit(0.0)) * u"keV"
        if pol_order == 2
            @test isapprox(mvalue(result.par.enc), sqrt(par_true[1]) * u"keV", rtol = 0.05)
            @test isapprox(mvalue(result.par.fano), mvalue(LegendSpecFits.fano_factor_ge), rtol = 0.05)
            @test isapprox(mvalue(result.par.ct), par_true[3], rtol = 0.05)
        end
        # the error functions reproduce the uncertainty of the fit function, which includes the
        # correlations between the fit parameters
        for func_err in (result.func_err, result.func_cal_err)
            fwhm_err = ustrip.(u"keV", ljl_propfunc(func_err).(test_tbl))
            @test mvalue.(fwhm_err) ≈ mvalue.(report.f_fit.(e_test))
            @test muncert.(fwhm_err) ≈ muncert.(report.f_fit.(e_test))
        end
    end
end