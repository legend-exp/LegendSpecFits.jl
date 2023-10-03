# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

th228_combined_shared_functions = (
    μ = (v) -> v.μ_slope * v.e + v.μ_intercept,
    skew_fraction = (v) -> v.skew_fraction_slope * v.e + v.skew_fraction_intercept
)

th228_combined_fit_functions = (
    f_fit = (x, v) -> gamma_peakshape(x, th228_combined_shared_functions.μ(v), v.σ, v.n, v.step_amplitude, th228_combined_shared_functions.skew_fraction(v), v.skew_width, v.background),
    f_sig = (x, v) -> signal_peakshape(x, th228_combined_shared_functions.μ(v), v.σ, v.n, th228_combined_shared_functions.skew_fraction(v)),
    f_lowEtail = (x, v) -> lowEtail_peakshape(x, th228_combined_shared_functions.μ(v), v.σ, v.n, th228_combined_shared_functions.skew_fraction(v), v.skew_width),
    f_bck = (x, v) -> background_peakshape(x, th228_combined_shared_functions.μ(v), v.σ, v.step_amplitude, v.background),
    f_sigWithTail = (x, v) -> signal_peakshape(x, th228_combined_shared_functions.μ(v), v.σ, v.n, th228_combined_shared_functions.skew_fraction(v)) + lowEtail_peakshape(x, th228_combined_shared_functions.μ(v), v.σ, v.n, th228_combined_shared_functions.skew_fraction(v), v.skew_width)
)

""" 
    estimate_combined_peak_stats(peakstats::StructArray,; calib_type::Symbol=:th228)

Estimate the peak position, FWHM, sigma, counts and background of a peak from a histogram.
"""
function estimate_combined_peak_stats(peakstats::StructArray,; calib_type::Symbol=:th228)
    if calib_type == :th228
        return estimate_single_peak_stats_th228(peakstats)
    elseif calib_type == :psd
        error("Calibration type not supported")
    else
        error("Calibration type not supported")
    end
end
export estimate_single_peak_stats


function estimate_combined_peak_stats_th228(peakstats::StructArray) where T<:Real
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


""" 
    fit_peaks_combined(peakhists::Array, peakstats::StructArray, th228_lines::Array{T},; calib_type::Symbol=:th228, uncertainty::Bool=true, fixed_position::Bool=false) where T<:Real

Fit the peaks of a histogram to a combined peakshape function while sharring parameters between peaks.
"""
function fit_peaks_combined(peakhists::Array, peakstats::StructArray, th228_lines::Array,; calib_type::Symbol=:th228, uncertainty::Bool=true, fixed_position::Bool=false)
    if calib_type == :th228
        return fit_peaks_combined_th228(peakhists, peakstats, th228_lines,; uncertainty=uncertainty, fixed_position=fixed_position)
    else
        error("Calibration type not supported")
    end
end
export fit_peaks_combined


function fit_peaks_combined_th228(peakhists::Array, peakstats::StructArray, th228_lines::Array{T},; uncertainty::Bool=true, fixed_position::Bool=false) where T<:Real
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        e = BAT.ConstValueDist(th228_lines),
        # centroid parameter share
        μ_slope         = Uniform(0.99, 1.01),
        μ_intercept     = Uniform(-2.0, 2.0),
        # low-E tail parameter share
        skew_fraction_intercept = LogUniform(1e-4, 1e-1),
        skew_fraction_slope = Biweight(0, 1e-4),
        # single parameters
        σ              = product_distribution(weibull_from_mx.(peakstats.peak_sigma, 2*peakstats.peak_sigma)),
        n              = product_distribution(weibull_from_mx.(peakstats.peak_counts, 2*peakstats.peak_counts)),
        step_amplitude = product_distribution(weibull_from_mx.(peakstats.mean_background, 2*peakstats.mean_background)),
        skew_width     = product_distribution(fill(LogUniform(0.001, 0.1), length(peakstats))),
        background     = product_distribution(weibull_from_mx.(peakstats.mean_background, 2*peakstats.mean_background)),
    )
    if fixed_position
        pseudo_prior = NamedTupleDist(
        μ = ConstValueDist(ps.peak_pos),
        σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
        n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
        step_amplitude = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
        skew_fraction = Uniform(0.01, 0.25),
        skew_width = LogUniform(0.001, 0.1),
        background = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
        )
    end
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=th228_combined_fit_functions.f_fit, hists=peakhists
        v -> sum(hist_loglike.(Base.Fix2.(f_fit, expand_vars(v)), hists))
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init), Optim.Options(iterations=100000))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))



    return v_ml, opt_r
end