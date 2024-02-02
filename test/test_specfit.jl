# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test
using LegendHDF5IO, HDF5
using LegendDataTypes: readdata

@testset "specfit" begin
    # load data, simple calibration 
    filename = "/remote/ceph/group/legendex/data/l200/julia/current/generated/tier/jlhitch/cal/p03/r000/l200-p03-r000-cal-ch1080005-tier_jlhit.lh5"
    data = h5open(x -> readdata(x, "ch1080005/dataQC"), filename)
    th228_lines =  [583.191,  727.330,  860.564,  1592.53,    1620.50,    2103.53,    2614.51]
    window_sizes =  vcat([(25.0,25.0) for _ in 1:6], (30.0,30.0))
    result_simple, report_simple = simple_calibration(data.e_cusp, th228_lines, window_sizes, n_bins=10000,; calib_type=:th228, quantile_perc=0.995)

    # fit a th228 peak
    Idx = 5 
    result_fit, report_fit = fit_single_peak_th228(result_simple.peakhists[Idx], result_simple.peakstats[Idx] ; uncertainty=true);

end