

"""
    qc_sg_optimization(dsp_dep, dsp_sep, optimization_config)

Perform simple QC cuts on the DEP and SEP data and return the data for the optimization of the SG window length.
"""
function qc_sg_optimization(dsp_dep::NamedTuple{(:aoe, :e, :blmean, :blslope, :t0)}, dsp_sep::NamedTuple{(:aoe, :e, :blmean, :blslope, :t0)}, optimization_config::PropDict)
    ### DEP
    # Load DEP data and prepare Pile-up cut
    blslope_dep, t0_dep = dsp_dep.blslope[isfinite.(dsp_dep.e)], dsp_dep.t0[isfinite.(dsp_dep.e)]
    aoe_dep, e_dep = dsp_dep.aoe[:, isfinite.(dsp_dep.e)], dsp_dep.e[isfinite.(dsp_dep.e)]
    # get half truncated centered cut on blslope for pile-up rejection
    result_dep_slope_cut, report_dep_slope_cut = get_centered_gaussian_window_cut(blslope_dep, -0.1u"ns^-1", 0.1u"ns^-1", 2, ; n_bins_cut=optimization_config.sg.cuts.dep.nbins_blslope_cut, relative_cut=optimization_config.sg.cuts.dep.rel_cut_blslope_cut)
    # Cut on blslope, energy and t0 for simple QC
    qc_cut_dep = blslope_dep .> result_dep_slope_cut.low_cut .&& blslope_dep .< result_dep_slope_cut.high_cut .&& e_dep .> optimization_config.sg.cuts.dep.min_e .&& quantile(e_dep, first(optimization_config.sg.cuts.dep.e_quantile)) .< e_dep .< quantile(e_dep, last(optimization_config.sg.cuts.dep.e_quantile)) .&& first(optimization_config.sg.cuts.dep.t0)u"µs" .< t0_dep .< last(optimization_config.sg.cuts.dep.t0)u"µs"
    aoe_dep, e_dep = aoe_dep[:, qc_cut_dep], e_dep[qc_cut_dep]

    ### SEP
    # Load SEP data and prepare Pile-up cut
    blslope_sep, t0_sep = dsp_sep.blslope[isfinite.(dsp_sep.e)], dsp_sep.t0[isfinite.(dsp_sep.e)]
    aoe_sep, e_sep = dsp_sep.aoe[:, isfinite.(dsp_sep.e)], dsp_sep.e[isfinite.(dsp_sep.e)]

    # get half truncated centered cut on blslope for pile-up rejection
    result_sep_slope_cut, report_sep_slope_cut = get_centered_gaussian_window_cut(blslope_sep, -0.1u"ns^-1", 0.1u"ns^-1", 2, ; n_bins_cut=optimization_config.sg.cuts.sep.nbins_blslope_cut, relative_cut=optimization_config.sg.cuts.sep.rel_cut_blslope_cut)

    # Cut on blslope, energy and t0 for simple QC
    qc_cut_sep = blslope_sep .> result_sep_slope_cut.low_cut .&& blslope_sep .< result_sep_slope_cut.high_cut .&& e_sep .> optimization_config.sg.cuts.sep.min_e .&& quantile(e_sep, first(optimization_config.sg.cuts.sep.e_quantile)) .< e_sep .< quantile(e_sep, last(optimization_config.sg.cuts.sep.e_quantile)) .&& first(optimization_config.sg.cuts.sep.t0)u"µs" .< t0_sep .< last(optimization_config.sg.cuts.sep.t0)u"µs"
    aoe_sep, e_sep = aoe_sep[:, qc_cut_sep], e_sep[qc_cut_sep]

    return (dep=(aoe=aoe_dep, e=e_dep), sep=(aoe=aoe_sep, e=e_sep))
end
export qc_sg_optimization


"""
    qc_cal_energy(data, qc_config)

Perform simple QC cuts on the data and return the data for energy calibration.
"""
function qc_cal_energy(data::Q, qc_config::PropDict) where Q<:Table
    # get bl mean cut
    result_blmean, _ = get_centered_gaussian_window_cut(data.blmean, qc_config.blmean.min, qc_config.blmean.max, qc_config.blmean.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blmean.n_bins_fraction)), relative_cut=qc_config.blmean.relative_cut, fixed_center=false, left=true)
    blmean_qc = result_blmean.low_cut .< data.blmean .< result_blmean.high_cut
    @debug format("Baseline Mean cut surrival fraction {:.2f}%", count(blmean_qc) / length(data) * 100)
    # get bl slope cut
    result_blslope, _ = get_centered_gaussian_window_cut(data.blslope, qc_config.blslope.min*u"ns^-1", qc_config.blslope.max*u"ns^-1", qc_config.blslope.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blslope.n_bins_fraction)), relative_cut=qc_config.blslope.relative_cut, fixed_center=true, left=false, center=zero(data.blslope[1]))
    blslope_qc = result_blslope.low_cut .< data.blslope .< result_blslope.high_cut
    @debug format("Baseline Slope cut surrival fraction {:.2f}%", count(blslope_qc) / length(data) * 100)
    # get blsigma cut
    result_blsigma, _ = get_centered_gaussian_window_cut(data.blsigma, qc_config.blsigma.min, qc_config.blsigma.max, qc_config.blsigma.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blsigma.n_bins_fraction)), relative_cut=qc_config.blsigma.relative_cut, fixed_center=false, left=true)
    blsigma_qc = result_blsigma.low_cut .< data.blsigma .< result_blsigma.high_cut
    @debug format("Baseline Sigma cut surrival fraction {:.2f}%", count(blsigma_qc) / length(data) * 100)
    # get t0 cut
    t0_qc = qc_config.t0.min*u"µs" .< data.t0 .< qc_config.t0.max*u"µs"
    @debug format("t0 cut surrival fraction {:.2f}%", count(t0_qc) / length(data) * 100)
    # get intrace pile-up cut
    inTrace_qc = .!(data.inTrace_intersect .> data.t0 .+ 2 .* data.drift_time .&& data.inTrace_n .> 1)
    @debug format("Intrace pile-up cut surrival fraction {:.2f}%", count(inTrace_qc) / length(data) * 100)
    # get energy cut
    energy_qc = qc_config.e_trap.min .< data.e_trap .&& .!isnan.(data.e_trap)
    @debug format("Energy cut surrival fraction {:.2f}%", count(energy_qc) / length(data) * 100)

    qc = blmean_qc .&& blslope_qc .&& blsigma_qc .&& t0_qc .&& inTrace_qc .&& energy_qc
    @debug format("Total QC cut surrival fraction {:.2f}%", count(qc) / length(data) * 100)
    return qc
end
export qc_cal_energy