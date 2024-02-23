

"""
    qc_sg_optimization(dsp_dep, dsp_sep, optimization_config)

Perform simple QC cuts on the DEP and SEP data and return the data for the optimization of the SG window length.
"""
function qc_sg_optimization(dsp_dep::NamedTuple{(:aoe, :e, :blmean, :blslope, :t50)}, dsp_sep::NamedTuple{(:aoe, :e, :blmean, :blslope, :t50)}, optimization_config::PropDict)
    ### DEP
    # Load DEP data and prepare Pile-up cut
    blslope_dep, t50_dep = dsp_dep.blslope[isfinite.(dsp_dep.e)], dsp_dep.t50[isfinite.(dsp_dep.e)]
    aoe_dep, e_dep = dsp_dep.aoe[:, isfinite.(dsp_dep.e)], dsp_dep.e[isfinite.(dsp_dep.e)]
    # get half truncated centered cut on blslope for pile-up rejection
    result_dep_slope_cut, report_dep_slope_cut = get_centered_gaussian_window_cut(blslope_dep, -0.1u"ns^-1", 0.1u"ns^-1", optimization_config.cuts.dep.blslope_sigma, ; n_bins_cut=optimization_config.cuts.dep.nbins_blslope_cut, relative_cut=optimization_config.cuts.dep.rel_cut_blslope_cut)
    # Cut on blslope, energy and t0 for simple QC
    qc_cut_dep = blslope_dep .> result_dep_slope_cut.low_cut .&& blslope_dep .< result_dep_slope_cut.high_cut .&& e_dep .> optimization_config.cuts.dep.min_e .&& quantile(e_dep, first(optimization_config.cuts.dep.e_quantile)) .< e_dep .< quantile(e_dep, last(optimization_config.cuts.dep.e_quantile)) .&& first(optimization_config.cuts.dep.t50) .< t50_dep .< last(optimization_config.cuts.dep.t50)
    aoe_dep, e_dep = aoe_dep[:, qc_cut_dep], e_dep[qc_cut_dep]

    ### SEP
    # Load SEP data and prepare Pile-up cut
    blslope_sep, t50_sep = dsp_sep.blslope[isfinite.(dsp_sep.e)], dsp_sep.t50[isfinite.(dsp_sep.e)]
    aoe_sep, e_sep = dsp_sep.aoe[:, isfinite.(dsp_sep.e)], dsp_sep.e[isfinite.(dsp_sep.e)]

    # get half truncated centered cut on blslope for pile-up rejection
    result_sep_slope_cut, report_sep_slope_cut = get_centered_gaussian_window_cut(blslope_sep, -0.1u"ns^-1", 0.1u"ns^-1", optimization_config.cuts.sep.blslope_sigma, ; n_bins_cut=optimization_config.cuts.sep.nbins_blslope_cut, relative_cut=optimization_config.cuts.sep.rel_cut_blslope_cut)

    # Cut on blslope, energy and t0 for simple QC
    qc_cut_sep = blslope_sep .> result_sep_slope_cut.low_cut .&& blslope_sep .< result_sep_slope_cut.high_cut .&& e_sep .> optimization_config.cuts.sep.min_e .&& quantile(e_sep, first(optimization_config.cuts.sep.e_quantile)) .< e_sep .< quantile(e_sep, last(optimization_config.cuts.sep.e_quantile)) .&& first(optimization_config.cuts.sep.t50) .< t50_sep .< last(optimization_config.cuts.sep.t50)
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
    result_blslope, _ = get_centered_gaussian_window_cut(data.blslope, qc_config.blslope.min, qc_config.blslope.max, qc_config.blslope.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blslope.n_bins_fraction)), relative_cut=qc_config.blslope.relative_cut, fixed_center=true, left=false, center=zero(data.blslope[1]))
    blslope_qc = result_blslope.low_cut .< data.blslope .< result_blslope.high_cut
    @debug format("Baseline Slope cut surrival fraction {:.2f}%", count(blslope_qc) / length(data) * 100)
    # get blsigma cut
    result_blsigma, _ = get_centered_gaussian_window_cut(data.blsigma, qc_config.blsigma.min, qc_config.blsigma.max, qc_config.blsigma.sigma, ; n_bins_cut=convert(Int64, round(length(data) * qc_config.blsigma.n_bins_fraction)), relative_cut=qc_config.blsigma.relative_cut, fixed_center=false, left=true)
    blsigma_qc = result_blsigma.low_cut .< data.blsigma .< result_blsigma.high_cut
    @debug format("Baseline Sigma cut surrival fraction {:.2f}%", count(blsigma_qc) / length(data) * 100)
    # get t0 cut
    t0_qc = qc_config.t0.min .< data.t0 .< qc_config.t0.max
    @debug format("t0 cut surrival fraction {:.2f}%", count(t0_qc) / length(data) * 100)
    # get intrace pile-up cut
    inTrace_qc = .!(data.inTrace_intersect .> data.t0 .+ 2 .* data.drift_time .&& data.inTrace_n .> 1)
    @debug format("Intrace pile-up cut surrival fraction {:.2f}%", count(inTrace_qc) / length(data) * 100)
    # get energy cut
    energy_qc = qc_config.e_trap.min .< data.e_trap .&& isfinite.(data.e_trap) .&& isfinite.(data.e_zac) .&& isfinite.(data.e_cusp)
    @debug format("Energy cut surrival fraction {:.2f}%", count(energy_qc) / length(data) * 100)

    # combine all cuts
    qc_tab = TypedTables.Table(blmean = blmean_qc, blslope = blslope_qc, blsigma = blsigma_qc, t0 = t0_qc, inTrace = inTrace_qc, energy = energy_qc, qc = blmean_qc .&& blslope_qc .&& blsigma_qc .&& t0_qc .&& inTrace_qc .&& energy_qc)
    @debug format("Total QC cut surrival fraction {:.2f}%", count(qc) / length(data) * 100)
    return qc_tab, (blmean = result_blmean, blslope = result_blslope, blsigma = result_blsigma)
end
export qc_cal_energy


function get_pulser_identified_idx(data::Table, key::Symbol, σ_search::Vector{Int}, pulser_config::PropDict; min_identified::Int=10)
    # extract config
    f = pulser_config.frequency
    T = upreferred(1/f)
    #get data and unit
    key_unit = unit(getproperty(data, key)[1])
    data_key = getproperty(data, key)
    # get t50 distribution
    h = fit(Histogram, ustrip.(key_unit, data_key[pulser_config[key].min .< data_key .< pulser_config[key].max]), ustrip.(key_unit, pulser_config[key].min:pulser_config[key].bin_width:pulser_config[key].max))
    # create empty arrays for identified pulser events
    pulser_identified_idx, time_idx = Int64[], Int64[]
    for σ in σ_search
        threshold = pulser_config[key].threshold
        while threshold > 0 && length(pulser_identified_idx) < min_identified
            peakhist, peakpos = RadiationSpectra.peakfinder(h, σ=σ, backgroundRemove=true, threshold=threshold)
            # if length(peakpos) < 2 
            #     threshold -= 5
            #     continue
            # end
            # select peak with second highest prominence in background removed histogram
            pulser_peak_candidates = peakpos[sortperm([maximum(peakhist.weights[pp-ustrip.(key_unit, pulser_config[key].peak_width) .< first(peakhist.edges)[2:end] .< pp+ustrip.(key_unit, pulser_config[key].peak_width)]) for pp in peakpos])] .* key_unit
            # pulser_peak_candidates = peakpos[sortperm([maximum(peakhist.weights[pp-ustrip.(key_unit, pulser_config[key].peak_width) .< first(peakhist.edges)[2:end] .< pp+ustrip.(key_unit, pulser_config[key].peak_width)]) for pp in peakpos])][1:end-1] .* key_unit
            
            for pulser_peak in pulser_peak_candidates
                # get idx in peak
                time_idx = findall(x -> pulser_peak - pulser_config[key].peak_width < x < pulser_peak + pulser_config[key].peak_width, data_key)
                # get timestamps in peak which are possible pulser events
                ts = data.timestamp[time_idx]
                pulser_identified_idx = findall(x -> x .== T, diff(ts))
                if isempty(pulser_identified_idx)
                    pulser_identified_idx = findall(x -> T - pulser_config.max_period_err < x < T + pulser_config.max_period_err, diff(ts))
                end
                if length(pulser_identified_idx) > min_identified
                    @info "Found pulser peak in $key distribution at $(pulser_peak)"
                    break
                end
            end
            if length(pulser_identified_idx) > min_identified
                break
            else
                threshold -= 5
            end
        end
        if length(pulser_identified_idx) > min_identified
            break
        end
    end
    return pulser_identified_idx, time_idx
end

"""
    pulser_cal_qc(data, pulser_config; n_pulser_identified=100)

Perform simple QC cuts on the data and return the data for energy calibration.
# Returns 
    - pulser_idx: indices of the pulser events
"""
function pulser_cal_qc(data::Q, pulser_config::PropDict; n_pulser_identified::Int=100) where Q<:Table
    # extract config
    f = pulser_config.frequency
    T = upreferred(1/f)
    # empty arrays for identified idx
    pulser_identified_idx, time_idx = Int64[], Int64[]
    for key in Symbol.(pulser_config.search_keys)
        if !hasproperty(data, key)
            @error "Data does not contain key $(key) for pulser calibration."
            continue
        end
        @debug "Search for pulser events in $key distribution"
        pulser_identified_idx, time_idx = get_pulser_identified_idx(data, key, pulser_config.search_sigmas, pulser_config; min_identified=pulser_config.min_identified)
        # if empty, no pulser tags found
        if isempty(pulser_identified_idx)
            @warn "No Pulser peak could be identified in $key distribution."
        else
            break
        end
    end

    # if empty, no pulser tags found
    if isempty(pulser_identified_idx)
        @warn "No Pulser events found"
        return Int64[]
    end

    # iterate through different pulser options and return unique idxs
    pulser_idx = Int64[]
    for idx in rand(pulser_identified_idx, n_pulser_identified)
        p_evt = data[time_idx[idx]]
        append!(pulser_idx, findall(pulser_config.pulser_diff.min .< (data.timestamp .- p_evt.timestamp .+ (T/4)) .% (T/2) .- (T/4) .< pulser_config.pulser_diff.max))
    end
    unique!(pulser_idx)
end
export pulser_cal_qc