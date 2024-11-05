

"""
    prepare_sep_peakhist(e::Array{T}, dep::T,; relative_cut::T=0.5, n_bins_cut::Int=500) where T<:Real

Prepare an array of uncalibrated SEP energies for parameter extraction and calibration.
# Returns
- `result`: Result of the initial fit
- `report`: Report of the initial fit
"""
function prepare_sep_peakhist(e::Vector{<:T}; sep::Unitful.Energy{<:Real}=2103.53u"keV", window::Vector{<:Unitful.Energy{<:Real}}=[25.0, 25.0]u"keV", relative_cut::T=0.5, n_bins_cut::Int=-1, fit_func::Symbol=:gamma_def, uncertainty::Bool=true) where T<:Real
    # get cut window around peak
    cuts = cut_single_peak(e, minimum(e), maximum(e); n_bins=n_bins_cut, relative_cut=relative_cut)
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> cuts.low .&& e .< cuts.high])
    # get simple calib constant to reject outliers
    m_cal_simple = sep / cuts.max
    # create histogram
    sephist = fit(Histogram, e, (sep - first(window))/m_cal_simple:bin_width:(sep + last(window))/m_cal_simple)
    # get peakstats
    sepstats = estimate_single_peak_stats(sephist)
    # initial fit for calibration and parameter extraction
    result, report = fit_single_peak_th228(sephist, sepstats,; uncertainty=uncertainty, fit_func=fit_func)
    # get calibration estimate from peak postion
    result = merge(result, (m_calib = sep / result.centroid, ))
    return result, report
end

"""
    fit_sf_wl(dep_sep_data, a_grid_wl_sg, optimization_config)

Fit a A/E filter window length for the SEP data and return the optimal window length and the corresponding survival fraction.

# Arguments
- `dep_sep_data`: NamedTuple with the DEP and SEP data
- `a_grid_wl_sg`: range of window lengths to sweep through
- `optimization_config`: configuration dictionary

# Returns
- `result`: optimal window length and corresponding survival fraction
- `report`: report with all window lengths and survival fractions
"""
function fit_sf_wl(e_dep::Vector{<:Real}, aoe_dep::ArrayOfSimilarArrays{<:Real}, e_sep::Vector{<:Real}, aoe_sep::ArrayOfSimilarArrays{<:Real}, a_grid_wl_sg::StepRangeLen; 
                    dep::T=1592.53u"keV", dep_window::Vector{<:T}=[12.0, 10.0]u"keV", 
                    sep::T=2103.53u"keV", sep_window::Vector{<:T}=[25.0, 25.0]u"keV", sep_rel_cut::Real=0.5,
                    min_aoe_quantile::Real=0.1, max_aoe_quantile::Real=0.99, 
                    min_aoe_offset::Quantity{<:Real}=0.0u"keV^-1", max_aoe_offset::Quantity{<:Real}=0.05u"keV^-1",
                    dep_cut_search_fit_func::Symbol=:gamma_def, sep_cut_search_fit_func::Symbol=:gamma_def,
                    uncertainty::Bool = false) where T<:Unitful.Energy{<:Real}
    
    # prepare peakhist
    result_sep, _ = prepare_sep_peakhist(e_sep; sep=sep, window=sep_window, relative_cut=sep_rel_cut, fit_func=sep_cut_search_fit_func, uncertainty=uncertainty)
    
    yield()
    
    # get calib constant from fit on DEP peak
    e_dep_calib = e_dep .* mvalue(result_sep.m_calib)
    e_sep_calib = e_sep .* mvalue(result_sep.m_calib)

    # create empty arrays for sf and sf_err
    fts_success = Bool.(zeros(length(a_grid_wl_sg)))
    sep_sfs = Vector{Quantity}(undef, length(a_grid_wl_sg))
    wls = Vector{eltype(a_grid_wl_sg)}(undef, length(a_grid_wl_sg))

    # for each window lenght, calculate the survival fraction in the SEP
    Threads.@threads for i_aoe in eachindex(a_grid_wl_sg)
        # get window length
        wl = a_grid_wl_sg[i_aoe]
        # get AoE for DEP
        aoe_dep_i = flatview(aoe_dep)[i_aoe, :][isfinite.(flatview(aoe_dep)[i_aoe, :])] ./ mvalue(result_sep.m_calib)
        e_dep_i   = e_dep_calib[isfinite.(flatview(aoe_dep)[i_aoe, :])]

        # prepare AoE
        max_aoe_dep_i = quantile(aoe_dep_i, max_aoe_quantile) + max_aoe_offset
        min_aoe_dep_i = quantile(aoe_dep_i, min_aoe_quantile) + min_aoe_offset
        
        aoe_dep_i_hist = fit(Histogram, ustrip.(aoe_dep_i), ustrip(min_aoe_dep_i):ustrip(get_friedman_diaconis_bin_width(aoe_dep_i[min_aoe_dep_i .< aoe_dep_i .< max_aoe_dep_i])):ustrip(max_aoe_dep_i))
        max_aoe_dep_i = first(aoe_dep_i_hist.edges)[min(end, argmax(aoe_dep_i_hist.weights)+1)] * unit(max_aoe_dep_i)

        try
            psd_cut, _ = get_low_aoe_cut(aoe_dep_i, e_dep_i; dep=dep, window=dep_window, cut_search_interval=(min_aoe_dep_i, max_aoe_dep_i), uncertainty=uncertainty, fit_func=dep_cut_search_fit_func)

            aoe_sep_i = flatview(aoe_sep)[i_aoe, :][isfinite.(flatview(aoe_sep)[i_aoe, :])] ./ result_sep.m_calib
            e_sep_i   = e_sep_calib[isfinite.(flatview(aoe_sep)[i_aoe, :])]

            result_sep_sf, _ = get_peak_surrival_fraction(aoe_sep_i, e_sep_i, sep, sep_window, psd_cut.lowcut; uncertainty=uncertainty, fit_func=sep_cut_search_fit_func)

            sep_sfs[i_aoe] = result_sep_sf.sf
            wls[i_aoe] = wl
            fts_success[i_aoe] = true
        catch e
            @warn "Couldn't process window length $wl"
        end
        yield()
    end

    # get all successful fits
    sep_sfs = sep_sfs[fts_success]
    wls = wls[fts_success]

    # get minimal surrival fraction and window length
    sep_sfs_cut = 1.0u"percent" .< sep_sfs .< 100u"percent"
    if isempty(sep_sfs[sep_sfs_cut])
        @error "No valid SEP SF found"
        throw(ErrorException("No valid SF found, could not determine optimal window length"))
    end
    min_sf       = minimum(sep_sfs[sep_sfs_cut])
    wl_sg_min_sf = wls[sep_sfs_cut][findmin(sep_sfs[sep_sfs_cut])[2]]

    # generate result and report
    result = (
        wl = measurement(wl_sg_min_sf, step(a_grid_wl_sg)),
        sf = min_sf,
        n_dep = length(e_dep),
        n_sep = length(e_sep)
    )
    report = (
        wl = result.wl,
        min_sf = result.sf,
        a_grid_wl_sg = wls,
        sfs = sep_sfs
    )
    return result, report
end
export fit_sf_wl