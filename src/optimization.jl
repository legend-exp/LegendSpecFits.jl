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
        cuts = cut_single_peak(enc_rt, min_enc, max_enc, nbins, rel_cut_fit)

        # fit gaussian
        result, report = fit_single_trunc_gauss(enc_rt, cuts)

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
function fit_fwhm_ft_fep(e_grid::Matrix, e_grid_ft::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}) where {T <:Real}
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
        result, report = fit_single_peak_th228(h, ps, false)
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
        @warn "No valid FT found, setting to maximum"
        ft_min_fwhm = e_grid_ft[end]
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
