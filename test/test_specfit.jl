# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using LegendDataTypes: fast_flatten
using Interpolations

@testset "specfit" begin
    # load data, simple calibration 
    include("test_utils.jl")

    energy_test, th228_lines = generate_mc_spectrum(200000)
    
    # simple calibration fit 
    window_sizes =  vcat([(25.0,25.0) for _ in 1:6], (30.0,30.0))
    result_simple, report_simple = simple_calibration(energy_test, th228_lines, window_sizes, n_bins=10000,; calib_type=:th228, quantile_perc=0.995)

    # fit 
    result, report = fit_peaks(result_simple.peakhists, result_simple.peakstats, th228_lines,; uncertainty=true);
end

