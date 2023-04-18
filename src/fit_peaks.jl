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