# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using LegendDataTypes: fast_flatten
using Measurements: value as mvalue
using Distributions
using StatsBase
using Interpolations
using Unitful 
using IntervalSets
# include("test_utils.jl")

@testset "simplecal" begin
    # Th228 
    # generate fake data - very simplified
    energy_test, th228_lines = generate_mc_spectrum(200000)
    m_cal_simple = 0.123u"keV"
    e_uncal = energy_test ./ m_cal_simple

    # binning and peak-finder settings that should work for this fake data set 
    window_sizes =  vcat([(25.0u"keV",25.0u"keV") for _ in 1:6], (30.0u"keV",30.0u"keV"))
    quantile_perc = 0.995
  
    # simple calibration
    result_simple, report_simple = simple_calibration(e_uncal, th228_lines, window_sizes,; calib_type= :th228,  quantile_perc = quantile_perc);
    @test isapprox(result_simple.c, m_cal_simple, atol = 0.05.*(m_cal_simple))


    # Co60 
    # generate fake data - very simplified
    energy_test, co60_lines = generate_mc_spectrum_co60(200000)
    m_cal_simple = 0.123u"keV"
    e_uncal = energy_test ./ m_cal_simple

    # binning and peak-finder settings that should work for this fake data set 
    window_sizes =  vcat((25.0u"keV",25.0u"keV"), (30.0u"keV",30.0u"keV"))
    quantile_perc=NaN
    peakfinder_σ = 2.0
    peakfinder_threshold = 50.0
    peak_quantile = 0.0..1.0
    bin_quantile =  0.0..0.5
    quantile_perc = NaN 
    kwargs = (peaksearch_type = :most_prominent, peakfinder_σ=peakfinder_σ, peakfinder_threshold=peakfinder_threshold, peak_quantile=peak_quantile, bin_quantile=bin_quantile, quantile_perc = quantile_perc)
    
    # simple calibration
    result_simple, report_simple = simple_calibration(e_uncal, co60_lines, window_sizes,; calib_type= :gamma, kwargs...);
    @test isapprox(result_simple.c, m_cal_simple, atol = 0.05.*(m_cal_simple))
end