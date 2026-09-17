# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Measurements: value as mvalue
using Random
using StatsBase
using Test
using Unitful

@testset "Peak survival-fraction selections" begin
    Random.seed!(42)
    peak = 2614.5u"keV"
    window = (20.0u"keV", 20.0u"keV")
    e = peak .+ 1.5u"keV" .* randn(20_000)
    aoe = randn(20_000)
    lq = -aoe
    low_cut = -0.5
    high_cut = 1.0
    preselection = trues(length(e))
    preselection[1:3:end] .= false

    selection = (low_cut .< aoe .< high_cut) .&& preselection
    Random.seed!(7)
    generic_result, generic_report = get_peak_survival_fraction(e, peak, window, selection; uncertainty=false)
    bin_width = LegendSpecFits.get_friedman_diaconis_bin_width(e[peak - 2.0u"keV" .< e .< peak + 2.0u"keV"])
    binning = ustrip(peak - first(window)):ustrip(bin_width):ustrip(peak + last(window))
    peakhist = fit(Histogram, ustrip.(e), binning)
    survived_hist = fit(Histogram, ustrip.(e[selection]), binning)
    cut_hist = fit(Histogram, ustrip.(e[.!selection]), binning)
    Random.seed!(7)
    histogram_result, histogram_report = get_peak_survival_fraction(peakhist, survived_hist, cut_hist; uncertainty=false)
    @test propertynames(histogram_result) == (:fit_func, :n_before, :n_after, :sf, :gof)
    @test propertynames(histogram_report) == (:n_before, :n_after, :sf, :before, :after)
    for field in (:n_before, :n_after, :sf)
        @test isapprox(mvalue(histogram_result[field]), mvalue(generic_result[field]))
        @test isapprox(mvalue(histogram_report[field]), mvalue(generic_report[field]))
    end
    @test generic_result.peak == generic_report.peak == peak
    @test histogram_report.before.h.weights == peakhist.weights
    @test histogram_report.after.survived.h.weights == survived_hist.weights
    @test histogram_report.after.cut.h.weights == cut_hist.weights

    direct_result, direct_report = get_peak_survival_fraction(e, selection; uncertainty=false)
    @test propertynames(direct_result) == propertynames(histogram_result)
    @test sum(direct_report.before.h.weights) == length(e)
    @test sum(direct_report.after.survived.h.weights) == count(selection)
    @test direct_report.before.h.weights == direct_report.after.survived.h.weights + direct_report.after.cut.h.weights
    @test_throws ArgumentError get_peak_survival_fraction(e, selection[1:end-1]; uncertainty=false)
    @test_throws MethodError get_peak_survival_fraction(e, peak, collect(window), selection; uncertainty=false)

    Random.seed!(7)
    threshold_result, threshold_report = get_peak_survival_fraction(aoe, e, peak, window; low_cut, high_cut, selection=preselection, uncertainty=false)
    @test mvalue(generic_result.sf) == mvalue(threshold_result.sf)
    @test propertynames(generic_result) == (:peak, :fit_func, :n_before, :n_after, :sf, :gof)
    @test propertynames(threshold_report) == (:peak, :n_before, :n_after, :sf, :before, :after)
    @test mvalue(threshold_result.n_after) == mvalue(threshold_report.n_after)
    @test mvalue(threshold_result.sf) == mvalue(threshold_report.sf)
    @test isapprox(mvalue(threshold_result.n_after), mvalue(threshold_result.n_before) * mvalue(ustrip(threshold_result.sf)) / 100)
    @test generic_report.before.h.weights == threshold_report.before.h.weights
    @test generic_report.after.survived.h.weights == threshold_report.after.survived.h.weights

    Random.seed!(7)
    threshold_lq_result, _ = get_peak_survival_fraction(lq, e, peak, window; high_cut=-low_cut, uncertainty=false)
    Random.seed!(7)
    generic_lq_result, _ = get_peak_survival_fraction(e, peak, window, lq .< -low_cut; uncertainty=false)
    @test mvalue(threshold_lq_result.sf) == mvalue(generic_lq_result.sf)

    Random.seed!(7)
    generic_peaks, _ = get_peaks_survival_fractions(e, [peak], [:Tl208FEP], [first(window)], [last(window)], selection; uncertainty=false)
    Random.seed!(7)
    threshold_peaks, threshold_peak_reports = get_peaks_survival_fractions(aoe, e, [peak], [:Tl208FEP], [first(window)], [last(window)]; low_cut, high_cut, selection=preselection, uncertainty=false)
    @test mvalue(generic_peaks[:Tl208FEP].sf) == mvalue(threshold_peaks[:Tl208FEP].sf)
    @test mvalue(threshold_peaks[:Tl208FEP].n_after) == mvalue(threshold_peak_reports[:Tl208FEP].n_after)

    Random.seed!(7)
    unitful_result, _ = get_peak_survival_fraction(aoe .* u"keV^-1", e, peak, window; low_cut=low_cut*u"keV^-1", uncertainty=false)
    Random.seed!(7)
    low_result, _ = get_peak_survival_fraction(e, peak, window, aoe .> low_cut; uncertainty=false)
    @test mvalue(unitful_result.sf) == mvalue(low_result.sf)
end

@testset "Continuum survival-fraction reports" begin
    Random.seed!(24)
    center, window = 2039.0u"keV", 20.0u"keV"
    e = center .+ 4.0u"keV" .* randn(2_000)
    parameter = randn(length(e))
    selection = rand(length(e)) .> 0.2
    low_cut, high_cut = -0.5, 1.0
    in_window = center - window .< e .< center + window
    after_low = parameter .> low_cut
    survived = (low_cut .< parameter .< high_cut) .& selection

    result, report = get_continuum_survival_fraction(parameter, e, center, window; low_cut, high_cut, selection)
    direct_result, direct_report = get_continuum_survival_fraction(e, center, window, survived)
    @test propertynames(result) == (:window, :n_before, :n_after, :sf)
    @test propertynames(report) == (:h_before, :h_after_low, :h_after_ds, :window, :n_before, :n_after, :sf, :e_unit, :bin_width)
    @test mvalue(result.n_before) == count(in_window) == sum(report.h_before.weights)
    @test mvalue(result.n_after) == count(in_window .& survived) == sum(report.h_after_ds.weights)
    @test isapprox(mvalue(ustrip(result.sf)), 100 * mvalue(result.n_after) / mvalue(result.n_before))
    @test sum(report.h_after_low.weights) == count(in_window .& after_low)
    @test report.n_after == result.n_after && report.sf == result.sf
    @test direct_result == result && direct_report.h_after_ds.weights == report.h_after_ds.weights

    lq_result, lq_report = get_continuum_survival_fraction(parameter, e, center, window; high_cut)
    @test mvalue(lq_result.n_after) == count(in_window .& (parameter .< high_cut))
    @test lq_report.h_after_low.weights == lq_report.h_before.weights
    @test sum(lq_report.h_after_ds.weights) == mvalue(lq_result.n_after)

    parameter[1:3] .= [NaN, Inf, -Inf]
    all_result, all_report = get_continuum_survival_fraction(parameter, e, center, window)
    finite_in_window = in_window .& isfinite.(parameter)
    @test mvalue(all_result.n_after) == count(finite_in_window)
    @test sum(all_report.h_after_ds.weights) == count(finite_in_window)
end
