"""
    fit_enc_sigmas(enc_grid::Matrix{T}, enc_grid_rt::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int}, min_enc::T, max_enc::T, nbins::Int, rel_cut_fit::T) where T<:Real

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
function fit_enc_sigmas(enc_grid::Matrix{T}, enc_grid_rt::StepRangeLen{<:Quantity{<:T}, <:Base.TwicePrecision{<:Quantity{<:T}}, <:Base.TwicePrecision{<:Quantity{<:T}}, Int}, min_enc::T, max_enc::T, nbins::Int, rel_cut_fit::T) where T<:Real
    @assert size(enc_grid, 1) == length(enc_grid_rt) "enc_grid and enc_grid_rt must have the same number of columns"
    
    # create empty array for results
    rts = Vector{eltype(enc_grid_rt)}(undef, length(enc_grid_rt))
    enc = Vector{Measurement}(undef, length(enc_grid_rt))
    rts_success  = Bool.(zeros(length(enc_grid_rt)))
    
    Threads.@threads for r in eachindex(enc_grid_rt)
        # get rt
        rt = enc_grid_rt[r]
        
        # get enc for this rt
        enc_rt = flatview(enc_grid)[r, :]
        # sanity check
        if all(enc_rt .== 0.0) || all(!isfinite, enc_rt)
            continue
        end
        # get cut value
        cuts = cut_single_peak(enc_rt, min_enc, max_enc,; n_bins=nbins, relative_cut=rel_cut_fit)

        # fit gaussian
        result, _ = fit_single_trunc_gauss(enc_rt, cuts)

        # get sigma
        enc[r] =  result.σ
        rts[r] = rt
        rts_success[r] = true
    end

    # get only successful fits
    enc = enc[rts_success]
    rts = rts[rts_success]

    # get minimal enc and rt
    if isempty(enc)
        @error "No valid ENC fit found"
        throw(ErrorException("No valid ENC value found, could not determine optimal RT"))
    end
    min_enc     = minimum(enc)
    rt_min_enc  = rts[findmin(enc)[2]]
    
    # generate result and report
    result = (
        rt = measurement(rt_min_enc, step(enc_grid_rt)),
        min_enc = min_enc
    )
    report = (
        rt = result.rt,
        min_enc = result.min_enc,
        enc_grid_rt = rts,
        enc = enc
    )
    return result, report

end
export fit_enc_sigmas

"""
    fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, rel_cut_fit::T, apply_ctc::Bool=true; kwargs...)
    fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen, rt::Unitful.RealOrRealQuantity, min_e, max_e, rel_cut_fit; kwargs...)
    fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen, qdrift::Vector{<:Real}, rt::Unitful.RealOrRealQuantity, min_e, max_e, rel_cut_fit; kwargs...)

Fit the FWHM values in `e_grid` for each FT in `e_grid_ft` with a Gamma Peakshape and return the optimal FT and the corresponding FWHM value. The cut values cut for each flat-top time a window for better histogramming.
If the `apply_ctc` flag is set to `true`, the CTC correction is applied to the energy values. 
Othwise, if a `qdrift` vector is provided, the CTC correction is applied to the energy values.

# Arguments
- `e_grid`: 2D array of energy values for each FT in `e_grid_ft`
- `e_grid_ft`: 1D array of FT values for which the FWHM values in `e_grid` are calculated
- `rt`: RT value for which the FWHM values in `e_grid` are calculated
- `min_e`: minimum energy value to consider for the fit
- `max_e`: maximum energy value to consider for the fit
- `rel_cut_fit`: relative cut value to use for the fit

# Returns
- `ft`: optimal FT value
- `min_fwhm`: corresponding FWHM value
"""
function fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen, qdrift::Vector{<:Real}, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, rel_cut_fit::T, apply_ctc::Bool=true; kwargs...) where {T <:Real}
    if apply_ctc
        return _fit_fwhm_ft_ctc(e_grid, e_grid_ft, qdrift, rt, min_e, max_e, rel_cut_fit; kwargs...)
    else
        return _fit_fwhm_ft(e_grid, e_grid_ft, rt, min_e, max_e, rel_cut_fit; kwargs...)
    end
end
fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen, rt::Unitful.RealOrRealQuantity, min_e, max_e, rel_cut_fit; kwargs...) = _fit_fwhm_ft(e_grid, e_grid_ft, rt, min_e, max_e, rel_cut_fit; kwargs...)
fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen, qdrift::Vector{<:Real}, rt::Unitful.RealOrRealQuantity, min_e, max_e, rel_cut_fit; kwargs...) = _fit_fwhm_ft_ctc(e_grid, e_grid_ft, qdrift, rt, min_e, max_e, rel_cut_fit; kwargs...)
export fit_fwhm_ft


"""
    _fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int}, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, nbins::Int, rel_cut_fit::T; default_ft::Quantity{T}=3.0u"µs", ft_fwhm_tol::Unitful.Energy{<:Real} = 0.1u"keV") where {T <:Real}

Fit the FWHM values in `e_grid` for each FT in `e_grid_ft` with a Gamma Peakshape and return the optimal FT and the corresponding FWHM value. The cut values cut for each flat-top time a window for better histogramming.

# Arguments
- `e_grid`: 2D array of energy values for each FT in `e_grid_ft`
- `e_grid_ft`: 1D array of FT values for which the FWHM values in `e_grid` are calculated
- `rt`: RT value for which the FWHM values in `e_grid` are calculated
- `min_e`: minimum energy value to consider for the fit
- `max_e`: maximum energy value to consider for the fit
- `nbins`: number of bins to use for the histogram of energy values
- `rel_cut_fit`: relative cut value to use for the fit
- `ft_fwhm_tol`: search for lowest "optimal" ft within `minimum(fwhm) + ft_fwhm_tol` to avoid artificially large ft 
# Returns
- `ft`: optimal FT value
- `min_fwhm`: corresponding FWHM value
"""
function _fit_fwhm_ft(e_grid::Matrix, e_grid_ft::StepRangeLen, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, rel_cut_fit::T; n_bins::Int=-1, peak::Unitful.Energy{<:Real}=2614.5u"keV", window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}}=(35.0u"keV", 25.0u"keV"), ft_fwhm_tol::Unitful.Energy{<:Real} = 0.1u"keV") where {T <:Real}
    @assert size(e_grid, 1) == length(e_grid_ft) "e_grid and e_grid_rt must have the same number of columns"
    
    # create empty array for results
    fts = Vector{eltype(e_grid_ft)}(undef, length(e_grid_ft))
    fwhm = Vector{Measurement}(undef, length(e_grid_ft))
    modes = Vector{Float64}(undef, length(e_grid_ft))
    fts_success  = Bool.(zeros(length(e_grid_ft)))
    
    Threads.@threads for f in eachindex(e_grid_ft)
        # get ft
        ft = e_grid_ft[f]
        
        # get e values for this ft
        e_ft = Array{Float64}(flatview(e_grid)[f, :])
        e_ft = e_ft[isfinite.(e_ft)]

        # sanity check
        if count(min_e .< e_ft .< max_e) < 100
            @debug "Not enough data points for FT $ft, skipping"
            continue
        end
        # cut around peak to increase performance
		fit_cut = cut_single_peak(e_ft, min_e, max_e,; n_bins=n_bins, relative_cut=rel_cut_fit)
		e_peak_cut = fit_cut.max - 15*(fit_cut.max - fit_cut.low) .< e_ft .< fit_cut.max + 15*(fit_cut.max - fit_cut.low)
		e_ft = e_ft[e_peak_cut]
        
        # create histogram from it
        if isempty(e_ft)
            @debug "Invalid energy vector, skipping"
            continue
        end
        bin_width = 2 * (quantile(e_ft, 0.75) - quantile(e_ft, 0.25)) / ∛(length(e_ft))
        h = fit(Histogram, e_ft, minimum(e_ft):bin_width:maximum(e_ft))

        # create peakstats
        ps = estimate_single_peak_stats_th228(h)
        # check if ps guess is valid
        if any(tuple_to_array(ps) .<= 0)
            @debug "Invalid guess for peakstats, skipping"
            continue
        end
        # fit peak 
        result, _ = fit_single_peak_th228(h, ps,; uncertainty=false)
        # get fwhm
        fwhm[f] = result.fwhm
        fts[f] = ft
        modes[f] = fit_cut.max
        fts_success[f] = true
    end

    # get all successful fits
    fts = fts[fts_success]
    fwhm = fwhm[fts_success]
    modes = modes[fts_success]

    # get minimal fwhm and rt
    if isempty(fwhm)
        @error "No valid FWHM found."
        throw(ErrorException("No valid FWHM found, could not determine optimal FT"))
    else
        # calibration constant from mean of modes
        c = peak ./ mean(modes)
        fwhm = fwhm .* c
        # get minimal fwhm and ft
        sel = findall(mvalue.(fwhm .- minimum(fwhm)) .< ft_fwhm_tol)
        ft_min_fwhm, min_i = findmin(e_grid_ft[sel])
        min_fwhm = fwhm[sel][min_i]
    end
    # generate result and report
    result = (
        ft = measurement(ft_min_fwhm, step(e_grid_ft)),
        min_fwhm = min_fwhm
    )
    report = (
        ft = result.ft, 
        min_fwhm = result.min_fwhm,
        e_grid_ft = fts,
        fwhm = fwhm,
    )
    return result, report

end


"""
    _fit_fwhm_ft_ctc(e_grid::Matrix, e_grid_ft::StepRangeLen, qdrift::Vector{<:Real}, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, nbins::Int, rel_cut_fit::T; default_ft::Quantity{T}=3.0u"µs", peak::Unitful.Energy{<:Real}=2614.5u"keV", window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}}=(35.0u"keV", 25.0u"keV"), ft_fwhm_tol::Unitful.Energy{<:Real} = 0.1u"keV") where {T <:Real}

Fit the FWHM values in `e_grid` for each FT in `e_grid_ft` with a Gamma Peakshape and return the optimal FT and the corresponding FWHM value. The cut values cut for each flat-top time a window for better histogramming.

# Arguments
- `e_grid`: 2D array of energy values for each FT in `e_grid_ft`
- `e_grid_ft`: 1D array of FT values for which the FWHM values in `e_grid` are calculated
- `qdrift`: drift time values for each energy value in `e_grid`
- `rt`: RT value for which the FWHM values in `e_grid` are calculated
- `min_e`: minimum energy value to consider for the fit
- `max_e`: maximum energy value to consider for the fit
- `nbins`: number of bins to use for the histogram of energy values
- `rel_cut_fit`: relative cut value to use for the fit
- `ft_fwhm_tol`: search for lowest "optimal" ft within `minimum(fwhm) + ft_fwhm_tol` to avoid artificially large ft 

# Returns
- `ft`: optimal FT value
- `min_fwhm`: corresponding FWHM value
"""
function _fit_fwhm_ft_ctc(e_grid::Matrix, e_grid_ft::StepRangeLen, qdrift::Vector{<:Real}, rt::Unitful.RealOrRealQuantity, min_e::T, max_e::T, rel_cut_fit::T; n_bins::Int=-1, peak::Unitful.Energy{<:Real}=2614.5u"keV", window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}}=(35.0u"keV", 25.0u"keV"), ft_fwhm_tol::Unitful.Energy{<:Real} = 0.1u"keV") where {T <:Real}
    @assert size(e_grid, 1) == length(e_grid_ft) "e_grid and e_grid_rt must have the same number of columns"
    
    # create empty array for results
    fts = Vector{eltype(e_grid_ft)}(undef, length(e_grid_ft))
    fwhm = Vector{Measurement}(undef, length(e_grid_ft))
    modes = Vector{Float64}(undef, length(e_grid_ft))
    fts_success  = Bool.(zeros(length(e_grid_ft)))
    
    Threads.@threads for f in eachindex(e_grid_ft)
        # get ft
        ft = e_grid_ft[f]

        # get e values for this ft
        e_ft = Array{Float64}(flatview(e_grid)[f, :])
        e_isfinite_cut = isfinite.(e_ft) .&& isfinite.(qdrift) .&& e_ft .> 0 .&& qdrift .> 0
        qdrift_ft = qdrift[e_isfinite_cut]
        e_ft = e_ft[e_isfinite_cut]

        # sanity check
        if count(min_e .< e_ft .< max_e) < 100
            @debug "Not enough data points for FT $ft, skipping"
            continue
        end
        # cut around peak to increase performance
		fit_cut = cut_single_peak(e_ft, min_e, max_e,; n_bins=n_bins, relative_cut=rel_cut_fit)
		e_peak_cut = fit_cut.max - 15*(fit_cut.max - fit_cut.low) .< e_ft .< fit_cut.max + 15*(fit_cut.max - fit_cut.low)
		e_ft = e_ft[e_peak_cut]
        qdrift_ft = qdrift_ft[e_peak_cut]

        # create histogram from it
        bin_width = get_friedman_diaconis_bin_width(e_ft)
        h = fit(Histogram, e_ft, minimum(e_ft):bin_width:maximum(e_ft))

        # create peakstats
        ps = estimate_single_peak_stats_th228(h)
        # check if ps guess is valid
        if any(tuple_to_array(ps) .<= 0)
            @debug "Invalid guess for peakstats, skipping"
            continue
        end
        # fit peak
        m_cal_simple = peak / fit_cut.max
        result, _ = ctc_energy(e_ft .* m_cal_simple, qdrift_ft, peak, window)
        
        # get fwhm
        fwhm[f] = result.fwhm_after / m_cal_simple
        fts_success[f] = true
        modes[f] = fit_cut.max
    end

    # get all successful fits
    fts = collect(e_grid_ft)[fts_success]
    fwhm = fwhm[fts_success]
    modes = modes[fts_success]
    
    # get minimal fwhm and rt
    if isempty(fwhm)
        @error "No valid FWHM found."
        throw(ErrorException("No valid FWHM found, could not determine optimal FT"))
    else
        # calibration constant from mean of modes
        c = peak ./ mean(modes)
        fwhm = fwhm .* c
        # get minimal fwhm and ft
        sel = findall(mvalue.(fwhm .- minimum(fwhm)) .< ft_fwhm_tol)
        ft_min_fwhm, min_i = findmin(e_grid_ft[sel])
        min_fwhm = fwhm[sel][min_i]
    end
    # generate result and report
    result = (
        ft = measurement(ft_min_fwhm, step(e_grid_ft)),
        min_fwhm = min_fwhm
    )
    report = (
        ft = result.ft, 
        min_fwhm = result.min_fwhm,
        e_grid_ft = fts,
        fwhm = fwhm,
    )
    return result, report

end