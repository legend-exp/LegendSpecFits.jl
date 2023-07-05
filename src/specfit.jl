# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    estimate_single_peak_stats(h::Histogram)

Estimate statistics/parameters for a single peak in the given histogram `h`.

`h` must only contain a single peak. The peak should have a Gaussian-like
shape.

Returns a `NamedTuple` with the fields

    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
"""
function estimate_single_peak_stats(h::Histogram)
    W = h.weights
    E = first(h.edges)
    peak_amplitude, peak_idx = findmax(W)
    fwhm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) /2, W)
    fwhm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) /2, W)
    peak_max_pos = (E[peak_idx] + E[peak_idx+1]) / 2
    peak_mid_pos = (E[fwhm_idx_right] + E[fwhm_idx_left]) / 2
    peak_pos = (peak_max_pos + peak_mid_pos) / 2
    peak_fwhm = E[fwhm_idx_right] - E[fwhm_idx_left]
    peak_sigma = peak_fwhm * inv(2*√(2log(2)))
    #peak_area = peak_amplitude * peak_sigma * sqrt(2*π)
    mean_background = (first(W) + last(W)) / 2
    mean_background = ifelse(mean_background == 0, 0.01, mean_background)
    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)

    (
        peak_pos = peak_pos, peak_fwhm = peak_fwhm,
        peak_sigma, peak_counts, mean_background
    )
end
export estimate_single_peak_stats


# helper fumctions for fitting peakshapes
f_fit(x, v) = gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background)
f_sig(x, v) = signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction)
f_lowEtail(x, v) = lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width)
f_bck(x, v) = background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background)
f_sigWithTail(x, v) = signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width) 

"""
    fitPeaks
Perform a fit of the peakshape to the data in `peakhists` using the initial values in `peakstats` to the calibration lines in `th228_lines`.
Returns
    * `peak_fit_plots`: array of plots of the peak fits
    * `return_vals`: dictionary of the fit results
"""
function fitPeaks(peakhists::Array, peakstats::StructArray, th228_lines::Array)

    peak_fit_plots = Plots.Plot[]
    return_vals = Dict{Float64, NamedTuple}()

    for i in eachindex(peakhists)
        h = peakhists[i]
        ps = peakstats[i]

        pseudo_prior = NamedTupleDist(
            μ = Uniform(ps.peak_pos-10, ps.peak_pos+10),
            σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
            n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
            step_amplitude = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
            skew_fraction = Uniform(0.01, 0.25),
            skew_width = LogUniform(0.001, 0.1),
            background = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
        )
    
        f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    
        v_init = mean(pseudo_prior)
    
        f_loglike = let f_fit=f_fit, h=h
            v -> hist_loglike(Base.Fix2(f_fit, v), h)
        end
    
        opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))
        v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

        # plot peak fits
        showlegend = ifelse(i == 1, true, false)
        plt = plot(LinearAlgebra.normalize(h, mode = :density), st = :stepbins, yscale = :log10, label="Data", showlegend=showlegend, Layout=(width=800, height=800))
        plot!(minimum(h.edges[1]):0.1:maximum(h.edges[1]), Base.Fix2(f_fit, v_ml), label="Best Fit", showlegend=showlegend)
        xlims!(xlims())
        ylims!(ylims())
        plot!(minimum(h.edges[1]):0.1:maximum(h.edges[1]), Base.Fix2(f_sig, v_ml), label="Signal", showlegend=showlegend)
        plot!(minimum(h.edges[1]):0.1:maximum(h.edges[1]), Base.Fix2(f_lowEtail, v_ml), label="Low-E tail", showlegend=showlegend)
        plot!(minimum(h.edges[1]):0.1:maximum(h.edges[1]), Base.Fix2(f_bck, v_ml), label="Background", showlegend=showlegend)
        push!(peak_fit_plots, plt)

        # get FWHM
        half_max_sig = maximum(Base.Fix2(f_sigWithTail, v_ml).(v_ml.μ - v_ml.σ:0.001:v_ml.μ + v_ml.σ))/2
        roots_low = find_zero(x -> Base.Fix2(f_sigWithTail, v_ml)(x) - half_max_sig, v_ml.μ - v_ml.σ)
        roots_high = find_zero(x -> Base.Fix2(f_sigWithTail, v_ml)(x) - half_max_sig, v_ml.μ + v_ml.σ)
        fwhm = roots_high - roots_low

        
        return_vals[th228_lines[i]] = (
            μ = v_ml.μ,
            σ = v_ml.σ,
            n = v_ml.n,
            skew_fraction = v_ml.skew_fraction,
            fwhm = fwhm
        )
    end
    return peak_fit_plots, return_vals
end
export fitPeaks


"""
    fitCalibration
Fit the calibration lines to a linear function.
Returns
    * `slope`: the slope of the linear fit
    * `intercept`: the intercept of the linear fit
"""
function fitCalibration(calib_vals::Dict)
    calib_fit_result = linregress(collect(keys(calib_vals)), collect(values(calib_vals)))
    return LinearRegression.slope(calib_fit_result)[1], LinearRegression.bias(calib_fit_result)[1]
end
export fitCalibration
