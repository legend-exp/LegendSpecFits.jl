"""
    fit_enc_sigmas(enc_grid::Matrix{T}, enc_grid_rt::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}, min_enc::T, max_enc::T, nbins::Int64, rel_cut_fit::T) where T<:Real

Fit the ENC values in `enc_grid` for each RT in `enc_grid_rt` with a Gaussian and return the optimal RT and the corresponding ENC value.

# Arguments
- `enc_grid`: 2D array of ENC values for each RT in `enc_grid_rt`
- `enc_grid_rt`: 1D array of RT values for which the ENC values in `enc_grid` are calculated
- `min_enc`: minimum ENC value to consider for the fit
- `max_enc`: maximum ENC value to consider for the fit
- `nbins`: number of bins to use for the histogram of ENC values
- `rel_cut_fit`: relative cut value to use for the fit

# Returns
- `rt`: optimal RT value
- `min_enc`: corresponding ENC value
"""
function fit_enc_sigmas(enc_grid::Matrix{T}, enc_grid_rt::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}, min_enc::T, max_enc::T, nbins::Int64, rel_cut_fit::T) where T<:Real
    @assert size(enc_grid, 1) == length(enc_grid_rt) "enc_grid and enc_grid_rt must have the same number of columns"
    
    # create empty array for results
    enc        = zeros(length(enc_grid_rt))
    enc_err    = zeros(length(enc_grid_rt))
    
    for (r, rt) in enumerate(enc_grid_rt)
        # get enc for this rt
        enc_rt = flatview(enc_grid)[r, :]
        # get cut value
        cuts = cut_single_peak(enc_rt, min_enc, max_enc,; n_bins=nbins, relative_cut=rel_cut_fit)

        # fit gaussian
        result, _ = fit_single_trunc_gauss(enc_rt, cuts)

        # get sigma
        enc[r] = result.σ
        enc_err[r] = result.σ_err
    end

    # get minimal enc and rt
    min_enc     = minimum(enc[enc .> 0])
    rt_min_enc  = enc_grid_rt[enc .> 0][findmin(enc[enc .> 0])[2]]
    
    # generate result and report
    result = (
        rt = rt_min_enc, 
        min_enc = min_enc
    )
    report = (
        rt = result.rt, 
        min_enc = result.min_enc,
        enc_grid_rt = collect(enc_grid_rt),
        enc = enc,
        enc_err = enc_err
    )
    return result, report

end
export fit_enc_sigmas

"""
    fit_fwhm_ft_fep(e_grid::Matrix{T}, e_grid_ft::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}) where T <:Real

Fit the FWHM values in `e_grid` for each FT in `e_grid_ft` with a Gamma Peakshape and return the optimal FT and the corresponding FWHM value.

# Arguments
- `e_grid`: 2D array of energy values for each FT in `e_grid_ft`
- `e_grid_ft`: 1D array of FT values for which the FWHM values in `e_grid` are calculated

# Returns
- `ft`: optimal FT value
- `min_fwhm`: corresponding FWHM value
"""
function fit_fwhm_ft_fep(e_grid::Matrix, e_grid_ft::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64},; default_rt::Quantity{T}=2.0u"µs") where {T <:Real}
    @assert size(e_grid, 1) == length(e_grid_ft) "e_grid and e_grid_rt must have the same number of columns"
    
    # create empty array for results
    fwhm        = zeros(length(e_grid_ft))
    fwhm_err    = zeros(length(e_grid_ft))
    
    for (r, rt) in enumerate(e_grid_ft)
        # get e values for this rt
        e_ft = Array{Float64}(flatview(e_grid)[r, :])
        e_ft = e_ft[isfinite.(e_ft)]
        # create histogram from it
        bin_width = 2 * (quantile(e_ft, 0.75) - quantile(e_ft, 0.25)) / ∛(length(e_ft))
        h = fit(Histogram, e_ft, median(e_ft) - 100:bin_width:median(e_ft) + 100)
        # create peakstats
        ps = estimate_single_peak_stats_th228(h)
        # check if ps guess is valid
        if any(tuple_to_array(ps) .<= 0)
            @debug "Invalid guess for peakstats, skipping"
            fwhm[r]     = NaN
            fwhm_err[r] = NaN
            continue
        end
        # fit peak 
        result, report = fit_single_peak_th228(h, ps,; uncertainty=false)
        # get fwhm
        fwhm[r]     = result.fwhm
        # fwhm_err[r] = result.fwhm_err
    end

    # calibration constant from last fit to get rough calibration for better plotting
    c = 2614.5 ./ result.μ 
    fwhm = fwhm .* c

    # get minimal fwhm and rt
    if isempty(fwhm[fwhm .> 0])
        @warn "No valid FWHM found, setting to NaN"
        min_fwhm = NaN
        @warn "No valid FT found, setting to default"
        ft_min_fwhm = default_rt
    else
        min_fwhm    = minimum(fwhm[fwhm .> 0])
        ft_min_fwhm = e_grid_ft[fwhm .> 0][findmin(fwhm[fwhm .> 0])[2]]
    end
    # generate result and report
    result = (
        ft = ft_min_fwhm, 
        min_fwhm = min_fwhm
    )
    report = (
        ft = result.ft, 
        min_fwhm = result.min_fwhm,
        e_grid_ft = collect(e_grid_ft),
        fwhm = fwhm,
        # fwhm_err = fwhm_err
    )
    return result, report

end
export fit_fwhm_ft_fep


function fit_sg_wl(dep_sep_data::NamedTuple{(:dep, :sep)}, a_grid_wl_sg::StepRangeLen, optimization_config::PropDict)
    # unpack config
    dep, dep_window = optimization_config.sg.dep, Float64.(optimization_config.sg.dep_window)
    sep, sep_window = optimization_config.sg.sep, Float64.(optimization_config.sg.sep_window)

    # unpack data
    e_dep, e_sep = dep_sep_data.dep.e, dep_sep_data.sep.e
    aoe_dep, aoe_sep = dep_sep_data.dep.aoe, dep_sep_data.sep.aoe


    # prepare peakhist
    result_dep, _ = prepare_dep_peakhist(e_dep, dep; n_bins_cut=500, relative_cut=0.5)

    # get calib constant from fit on DEP peak
    e_dep_calib = e_dep .* result_dep.m_calib
    e_sep_calib = e_sep .* result_dep.m_calib

    # create empty arrays for sf and sf_err
    sep_sfs     = ones(length(a_grid_wl_sg)) .* 100
    sep_sfs_err = zeros(length(a_grid_wl_sg))


    # for each window lenght, calculate the survival fraction in the SEP
    for (i_aoe, wl) in enumerate(a_grid_wl_sg)

        aoe_dep_i = aoe_dep[i_aoe, :][isfinite.(aoe_dep[i_aoe, :])] ./ result_dep.m_calib
        e_dep_i   = e_dep_calib[isfinite.(aoe_dep[i_aoe, :])]

        # prepare AoE
        max_aoe_dep_i = quantile(aoe_dep_i, optimization_config.sg.max_aoe_quantile) + optimization_config.sg.max_aoe_offset
        min_aoe_dep_i = quantile(aoe_dep_i, optimization_config.sg.min_aoe_quantile)

        try
            psd_cut = get_psd_cut(aoe_dep_i, e_dep_i; window=dep_window, cut_search_interval=(min_aoe_dep_i, max_aoe_dep_i))

            aoe_sep_i = aoe_sep[i_aoe, :][isfinite.(aoe_sep[i_aoe, :])] ./ result_dep.m_calib
            e_sep_i   = e_sep_calib[isfinite.(aoe_sep[i_aoe, :])]

            result_sep, _ = get_peak_surrival_fraction(aoe_sep_i, e_sep_i, sep, sep_window, psd_cut.cut; uncertainty=true, low_e_tail=false)
            sep_sfs[i_aoe]     = result_sep.sf * 100
            sep_sfs_err[i_aoe] = result_sep.err.sf * 100
        catch
            @warn "Couldn't process window length $wl"
        end
    end
    # get minimal surrival fraction and window length
    if isempty(sep_sfs[sep_sfs .< 100])
        @warn "No valid SEP SF found, setting to NaN"
        min_sf = NaN
        min_sf_err = NaN
        @warn "No valid window length found, setting to default"
        wl_sg_min_sf = 100u"ns"
    else
        min_sf     = minimum(sep_sfs[sep_sfs .< 100])
        min_sf_err = sep_sfs_err[sep_sfs .== min_sf][1]
        wl_sg_min_sf = a_grid_wl_sg[sep_sfs .< 100][findmin(sep_sfs[sep_sfs .< 100])[2]]
    end

    # generate result and report
    result = (
        wl = wl_sg_min_sf,
        sf = min_sf,
        sf_err = min_sf_err
    )
    report = (
        wl = result.wl,
        min_sf = result.sf,
        min_sf_err = result.sf_err,
        a_grid_wl_sg = collect(a_grid_wl_sg),
        sfs = sep_sfs,
        sfs_err = sep_sfs_err
    )
    return result, report
end
export fit_sg_wl