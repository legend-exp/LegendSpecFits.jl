# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    estimate_single_peak_stats(h::Histogram, calib_type::Symbol=:th228)

Estimate statistics/parameters for a single peak in the given histogram `h`.

`h` must only contain a single peak. The peak should have a Gaussian-like
shape.
`calib_type` specifies the calibration type. Currently only `:th228` is implemented.

# Returns 
`NamedTuple` with the fields
    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
"""
function estimate_single_peak_stats(h::Histogram, calib_type::Symbol=:th228)
    if calib_type == :th228
        return estimate_single_peak_stats_th228(h)
    else
        error("Calibration type not supported")
    end
end
export estimate_single_peak_stats


function estimate_single_peak_stats_th228(h::Histogram{T}) where T<:Real
    W = h.weights
    E = first(h.edges)
    peak_amplitude, peak_idx = findmax(W)
    fwhm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) /2, W)
    fwhm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) /2, W)
    peak_max_pos = (E[peak_idx] + E[peak_idx+1]) / 2
    peak_mid_pos = (E[fwhm_idx_right] + E[fwhm_idx_left]) / 2
    peak_pos = (peak_max_pos + peak_mid_pos) / 2.0
    peak_fwhm = (E[fwhm_idx_right] - E[fwhm_idx_left]) / 1.0
    peak_sigma = peak_fwhm * inv(2*√(2log(2)))
    #peak_area = peak_amplitude * peak_sigma * sqrt(2*π)
    mean_background = (first(W) + last(W)) / 2
    mean_background = ifelse(mean_background == 0, 0.01, mean_background)
    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)

    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts, 
        mean_background = mean_background
    )
end


# helper fucntions for fitting peakshapes
th228_fit_functions = (
    f_fit = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background),
    f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
    f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
    f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background),
    f_sigWithTail = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width)
)
# f_fit(x, v) = gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background)
# f_sig(x, v) = signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction)
# f_lowEtail(x, v) = lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width)
# f_bck(x, v) = background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background)
# f_sigWithTail(x, v) = signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width) 

"""
    fitPeaks
Perform a fit of the peakshape to the data in `peakhists` using the initial values in `peakstats` to the calibration lines in `th228_lines`.
# Returns
    * `peak_fit_plots`: array of plots of the peak fits
    * `return_vals`: dictionary of the fit results
"""
function fit_peaks(peakhists::Array, peakstats::StructArray, th228_lines::Array, calib_type::Symbol=:th228)
    if calib_type == :th228
        return fit_peaks_th228(peakhists, peakstats, th228_lines)
    else
        error("Calibration type not supported")
    end
end
export fit_peaks

function fit_peaks_th228(peakhists::Array, peakstats::StructArray, th228_lines::Array)
    # create return and result dicts
    result = Dict{Float64, NamedTuple}()
    report = Dict{Float64, NamedTuple}()
    # iterate throuh all peaks
    for (i, peak) in enumerate(th228_lines)
        # get histogram and peakstats
        h  = peakhists[i]
        ps = peakstats[i]
        # fit peak
        result_peak, report_peak = fit_single_peak_th228(h, ps, true)
        # save results
        result[peak] = result_peak
        report[peak] = report_peak
    end
    return result, report
end


"""
    fitSinglePeak(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background), NTuple{5, T}}) where T<:Real
Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `gamma_peakshape` with low-E tail.
Also, FWHM is calculated from the fitted peakshape with MC error propagation.
# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_single_peak_th228(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background), NTuple{5, T}}, uncertainty::Bool=true) where T<:Real
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μ = Uniform(ps.peak_pos-10, ps.peak_pos+10),
        σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
        n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
        step_amplitude = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
        skew_fraction = Uniform(0.01, 0.25),
        skew_width = LogUniform(0.001, 0.1),
        background = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
    )

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

    # loglikelihood of real data
    likelihood = f_loglike(v_ml)

    sim = rand(Product(Poisson.(h.weights)), 10000)


    f_loglike_sim = let f_fit=th228_fit_functions.f_fit, v=v_ml
        w -> hist_loglike.(Base.Fix2(f_fit, v), fit.(Histogram, Ref(midpoints(h.edges[1])), weights.(w), Ref(h.edges[1])))
    end

    # calculate loglikelihood of the simulated histograms
    likelihood_sim = f_loglike_sim(eachcol(sim))

    diff = likelihood_sim .- likelihood

    # calculate p-value
    p_value = count(diff .> 0) / length(diff)


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

        # get fwhm of peak
        fwhm, fwhm_err = get_peak_fwhm_th228(v_ml, v_ml_err)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "FWHM: $(fwhm) ± $(fwhm_err)"

        result = merge(v_ml, (fwhm = fwhm, err = merge(v_ml_err, (fwhm = fwhm_err,))))
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
    return result, report, p_value
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
