# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

# helper fucntions for fitting peakshapes
th228_fit_functions = (
    f_fit = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background),
    f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
    f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
    f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background),
    f_sigWithTail = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width)
)
export th228_fit_functions
"""
    estimate_single_peak_stats(h::Histogram, calib_type::Symbol=:th228)

Estimate statistics/parameters for a single peak in the given histogram `h`.

`h` must only contain a single peak. The peak should have a Gaussian-like
shape.
`calib_type` specifies the calibration type. Currently only `:th228` is implemented.
If you want get the peak statistics for a PSD calibration, use `:psd`.

# Returns 
`NamedTuple` with the fields
    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
"""
function estimate_single_peak_stats(h::Histogram,; calib_type::Symbol=:th228)
    if calib_type == :th228
        return estimate_single_peak_stats_th228(h)
    elseif calib_type == :psd
        return estimate_single_peak_stats_psd(h)
    else
        error("Calibration type not supported")
    end
end
export estimate_single_peak_stats


function estimate_single_peak_stats_th228(h::Histogram{T}) where T<:Real
    W = h.weights
    E = first(h.edges)
    peak_amplitude, peak_idx = findmax(W)
    fwhm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 2, W)
    fwhm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 2, W)
    peak_max_pos = (E[peak_idx] + E[peak_idx+1]) / 2
    peak_mid_pos = (E[fwhm_idx_right] + E[fwhm_idx_left]) / 2
    peak_pos = (peak_max_pos + peak_mid_pos) / 2.0
    peak_fwhm = (E[fwhm_idx_right] - E[fwhm_idx_left]) / 1.0
    peak_sigma = peak_fwhm * inv(2*√(2log(2)))
    peak_fwqm = NaN
    # make sure that peakstats have non-zero sigma and fwhm values to prevent fit priors from being zero
    if peak_fwhm == 0
        fwqm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 4, W)
        fwqm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 4, W)
        peak_fwqm = (E[fwqm_idx_right] - E[fwqm_idx_left]) / 1.0
        peak_sigma = peak_fwqm * inv(2*√(2log(4)))
        peak_fwhm  = peak_sigma * 2*√(2log(2))
    end
    if peak_sigma == 0
        peak_sigma = 1.0
        peak_fwhm = 2.0
    end
    #peak_area = peak_amplitude * peak_sigma * sqrt(2*π)
    mean_background = (first(W) + last(W)) / 2
    mean_background = ifelse(mean_background == 0, 0.01, mean_background)
    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)
    peak_counts = ifelse(peak_counts < 0.0, inv(0.761) * sum(view(W,fwhm_idx_left:fwhm_idx_right)), peak_counts)
    if !isnan(peak_fwqm)
        peak_counts = inv(0.904) * (sum(view(W,fwqm_idx_left:fwqm_idx_right)) - mean_background * peak_fwqm)
        peak_counts = ifelse(peak_counts < 0.0, inv(0.904) * sum(view(W,fwqm_idx_left:fwqm_idx_right)), peak_counts)
    end
    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts, 
        mean_background = mean_background
    )
end

"""
    fit_peaks(peakhists::Array, peakstats::StructArray, th228_lines::Array,; calib_type::Symbol=:th228, uncertainty::Bool=true, low_e_tail::Bool=true)
Perform a fit of the peakshape to the data in `peakhists` using the initial values in `peakstats` to the calibration lines in `th228_lines`. 
# Returns
    * `peak_fit_plots`: array of plots of the peak fits
    * `return_vals`: dictionary of the fit results
"""
function fit_peaks(peakhists::Array, peakstats::StructArray, th228_lines::Array,; calib_type::Symbol=:th228, uncertainty::Bool=true, low_e_tail::Bool=true)
    if calib_type == :th228
        return fit_peaks_th228(peakhists, peakstats, th228_lines,; uncertainty=uncertainty, low_e_tail=low_e_tail)
    else
        error("Calibration type not supported")
    end
end
export fit_peaks

function fit_peaks_th228(peakhists::Array, peakstats::StructArray, th228_lines::Array{T},; uncertainty::Bool=true, low_e_tail::Bool=true) where T<:Any
    # create return and result dicts
    result = Dict{T, NamedTuple}()
    report = Dict{T, NamedTuple}()
    # iterate throuh all peaks
    for (i, peak) in enumerate(th228_lines)
        # get histogram and peakstats
        h  = peakhists[i]
        ps = peakstats[i]
        # fit peak
        result_peak, report_peak = fit_single_peak_th228(h, ps, ; uncertainty=uncertainty, low_e_tail=low_e_tail)

        # check covariance matrix for being semi positive definite (no negative uncertainties)
        if !isposdef(result_peak.err.covmat)
            @warn "Covariance matrix not positive definite for peak $peak - repeat fit without low energy tail"
            pval_save = result_peak.pval
            result_peak, report_peak = fit_single_peak_th228(h, ps, ; uncertainty=uncertainty, low_e_tail=false)
            @info "New covariance matrix is positive definite: $(isposdef(result_peak.err.covmat))"
            @info "p-val with low-energy tail  p=$(round(pval_save,digits=5)) , without low-energy tail: p=$(round((result_peak.pval),digits=5))"
        end
        # save results
        result[peak] = result_peak
        report[peak] = report_peak
    end
    return result, report
end


"""
    fit_single_peak_th228(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background), NTuple{5, T}};, uncertainty::Bool=true, fixed_position::Bool=false, low_e_tail::Bool=true) where T<:Real
Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `gamma_peakshape` with low-E tail.
Also, FWHM is calculated from the fitted peakshape with MC error propagation. The peak position can be fixed to the value in `ps` by setting `fixed_position=true`. If the low-E tail should not be fitted, it can be disabled by setting `low_e_tail=false`.
# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_single_peak_th228(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background), NTuple{5, T}}; uncertainty::Bool=true, low_e_tail::Bool=true, fixed_position::Bool=false, pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true)) where T<:Real
    # create standard pseudo priors
    standard_pseudo_prior = NamedTupleDist(
        μ = ifelse(fixed_position, ConstValueDist(ps.peak_pos), Uniform(ps.peak_pos-10, ps.peak_pos+10)),
        σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
        n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
        step_amplitude = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
        skew_fraction = ifelse(low_e_tail, Uniform(0.005, 0.25), ConstValueDist(0.0)),
        skew_width = ifelse(low_e_tail, LogUniform(0.001, 0.1), ConstValueDist(1.0)),
        background = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
    )
    # use standard priors in case of no overwrites given
    if !(:empty in keys(pseudo_prior))
        # check if input overwrite prior has the same fields as the standard prior set
        @assert all(f -> f in keys(standard_pseudo_prior), keys(standard_pseudo_prior)) "Pseudo priors can only have $(keys(standard_pseudo_prior)) as fields."
        # replace standard priors with overwrites
        pseudo_prior = merge(standard_pseudo_prior, pseudo_prior)
    else
        # take standard priors as pseudo priors with overwrites
        pseudo_prior = standard_pseudo_prior    
    end
    
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=th228_fit_functions.f_fit, h=h
        v -> hist_loglike(Base.Fix2(f_fit, v), h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    f_loglike_array = let f_fit=gamma_peakshape, h=h
        v -> - hist_loglike(x -> f_fit(x, v...), h)
    end

    if uncertainty
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance = inv(H)

        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        # calculate p-value
        pval, chi2, dof = p_value(th228_fit_functions.f_fit, h, v_ml)
        
        # get fwhm of peak
        fwhm, fwhm_err = get_peak_fwhm_th228(v_ml, v_ml_err)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "p: $pval , chi2 = $(chi2) with $(dof) dof"
        @debug "FWHM: $(fwhm) ± $(fwhm_err)"

        result = merge(v_ml, (pval = pval,chi2 = chi2, dof = dof, fwhm = fwhm, err = merge(v_ml_err, (fwhm = fwhm_err, covmat = param_covariance))))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(th228_fit_functions.f_fit, v_ml)(x),
            f_sig = x -> Base.Fix2(th228_fit_functions.f_sig, v_ml)(x),
            f_lowEtail = x -> Base.Fix2(th228_fit_functions.f_lowEtail, v_ml)(x),
            f_bck = x -> Base.Fix2(th228_fit_functions.f_bck, v_ml)(x)
        )
    else
        # get fwhm of peak
        fwhm, fwhm_err = get_peak_fwhm_th228(v_ml, v_ml, false)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "FWHM: $(fwhm)"

        result = merge(v_ml, (fwhm = fwhm, ))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(th228_fit_functions.f_fit, v_ml)(x),
            f_sig = x -> Base.Fix2(th228_fit_functions.f_sig, v_ml)(x),
            f_lowEtail = x -> Base.Fix2(th228_fit_functions.f_lowEtail, v_ml)(x),
            f_bck = x -> Base.Fix2(th228_fit_functions.f_bck, v_ml)(x)
        )
    end
    return result, report
end
export fit_single_peak_th228




"""
    estimate_fwhm(v::NamedTuple, v_err::NamedTuple)
Get the FWHM of a peak from the fit parameters.

# Returns
    * `fwhm`: the FWHM of the peak
"""
function estimate_fwhm(v::NamedTuple)
    # get FWHM
    try
        half_max_sig = maximum(Base.Fix2(th228_fit_functions.f_sigWithTail, v).(v.μ - v.σ:0.001:v.μ + v.σ))/2
        roots_low = find_zero(x -> Base.Fix2(th228_fit_functions.f_sigWithTail, v)(x) - half_max_sig, v.μ - v.σ, maxiter=100)
        roots_high = find_zero(x -> Base.Fix2(th228_fit_functions.f_sigWithTail,v)(x) - half_max_sig, v.μ + v.σ, maxiter=100)
        return roots_high - roots_low
    catch e
        return NaN
    end
end


"""
    get_peak_fwhm_th228(v_ml::NamedTuple, v_ml_err::NamedTuple)
Get the FWHM of a peak from the fit parameters while performing a MC error propagation.

# Returns
    * `fwhm`: the FWHM of the peak
    * `fwhm_err`: the uncertainty of the FWHM of the peak
"""
function get_peak_fwhm_th228(v_ml::NamedTuple, v_ml_err::NamedTuple, uncertainty::Bool=true)
    # get fwhm for peak fit
    fwhm = estimate_fwhm(v_ml)
    if !uncertainty
        return fwhm, NaN
    end
    # get MC for FWHM err
    v_mc = get_mc_value_shapes(v_ml, v_ml_err, 1000)
    fwhm_mc = estimate_fwhm.(v_mc)
    fwhm_err = std(fwhm_mc[isfinite.(fwhm_mc)])
    return fwhm, fwhm_err
end


"""
    fitCalibration
Fit the calibration lines to a linear function.
# Returns
    * `slope`: the slope of the linear fit
    * `intercept`: the intercept of the linear fit
"""
function fit_calibration(peaks::Array, μ::Array)
    @assert length(peaks) == length(μ)
    @debug "Fit calibration curve with linear function"
    calib_fit_result = linregress(peaks, μ)
    return LinearRegression.slope(calib_fit_result)[1], LinearRegression.bias(calib_fit_result)[1]
end
export fit_calibration