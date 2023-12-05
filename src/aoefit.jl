# aoe compton region peakshapes
f_aoe_compton(x, v) = aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ)
f_aoe_sig(x, v)     = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)
f_aoe_bkg(x, v)     = aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ)

# aoe compton centroids energy depencence
f_aoe_μ(x::Real, v::Array{T}) where T<:Real = linear_function(x, -v[1], v[2])
f_aoe_μ(x::Array{<:Real}, v::Array{T}) where T<:Real = linear_function.(x, -v[1], v[2])
f_aoe_μ(x::T, v::NamedTuple) where T<:Real = f_aoe_μ(x, [v.μ_scs_slope, v.μ_scs_intercept])

# aoe compton sigma energy depencence
f_aoe_σ(x::Real, v::Array{T}) where T<:Real = exponential_decay(x, v[1], v[2], v[3])
f_aoe_σ(x::Array{<:Real}, v::Array{T}) where T<:Real = exponential_decay.(x, v[1], v[2], v[3])
f_aoe_σ(x::T, v::NamedTuple) where T<:Real = f_aoe_σ(x, [v.σ_scs_amplitude, v.σ_scs_decay, v.σ_scs_offset])

"""
    estimate_single_peak_stats_psd(h::Histogram{T}) where T<:Real

Estimate peak parameters for a single peak in a histogram using the maximum, the FWHM and the area of the peak.

# Returns
    * `peak_pos`: Position of the peak
    * `peak_fwhm`: Full width at half maximum of the peak
    * `peak_sigma`: Standard deviation of the peak
    * `peak_counts`: Counts of the peak
    * `mean_background`: Mean background of the peak
"""
function estimate_single_peak_stats_psd(h::Histogram{T}) where T<:Real
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
    # mean_background = (first(W) + last(W)) / 2
    background_max = findfirst(e -> e >= peak_pos - 3*peak_sigma, E)
    mean_background = convert(typeof(peak_pos), (sum(view(W, 1:background_max))))
    # mean_background = ifelse(mean_background == 0, 0.01, mean_background)
    # peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)
    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)))

    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts, 
        mean_background = mean_background
    )
end

"""
    generate_aoe_compton_bands(aoe::Array{<:Real}, e::Array{<:Real}, compton_bands::Array{<:Real}, compton_window::T) where T<:Real

Generate histograms for the A/E Compton bands and estimate peak parameters. 
The compton bands are cutted out of the A/E spectrum and then binned using the Freedman-Diaconis Rule. For better performance
the binning is only done in the area around the peak. The peak parameters are estimated using the `estimate_single_peak_stats_psd` function.

# Returns
    * `peakhists`: Array of histograms for each compton band
    * `peakstats`: StructArray of peak parameters for each compton band
    * `min_aoe`: Array of minimum A/E values for each compton band
    * `max_aoe`: Array of maximum A/E values for each compton band
"""
function generate_aoe_compton_bands(aoe::Array{<:Real}, e::Array{<:Real}, compton_bands::Array{<:Real}, compton_window::T) where T<:Real
    # get aoe values in compton bands
    aoe_compton_bands = [aoe[(e .> c) .&& (e .< c + compton_window) .&& (aoe .> 0.0)] for c in compton_bands]

    # can constrain data to the area around the peak
    max_aoe              = [quantile(aoe_c, 0.99) + 0.05 for aoe_c in aoe_compton_bands]
    min_aoe              = [quantile(aoe_c, 0.1)         for aoe_c in aoe_compton_bands]
    half_quantile_aoe    = [quantile(aoe_c, 0.5)         for aoe_c in aoe_compton_bands]

    # Freedman-Diaconis Rule for binning only in the area aroung the peak
    # bin_width   = [2 * (quantile(aoe_c[aoe_c .> half_quantile_aoe[i] .&& aoe_c .< max_aoe[i]], 0.75) - quantile(aoe_c[aoe_c .> half_quantile_aoe[i] .&& aoe_c .< max_aoe[i]], 0.25)) / ∛(length(aoe_c[aoe_c .> half_quantile_aoe[i] .&& aoe_c .< max_aoe[i]])) for (i, aoe_c) in enumerate(aoe_compton_bands)]
    bin_width   = [get_friedman_diaconis_bin_width(aoe_c[aoe_c .> half_quantile_aoe[i] .&& aoe_c .< max_aoe[i]]) for (i, aoe_c) in enumerate(aoe_compton_bands)]

    # generate histograms
    peakhists = [fit(Histogram, aoe_compton_bands[i], min_aoe[i]:bin_width[i]:max_aoe[i]) for i in eachindex(aoe_compton_bands)]

    # estimate peak parameters
    peakstats = StructArray(estimate_single_peak_stats_psd.(peakhists))

    # make sure that peakstats have non-zero sigma and fwhm values to prevent fit priors from being zero
    median_fwhm   = median(peakstats.peak_fwhm[peakstats.peak_fwhm .!= 0])
    median_sigma  = median(peakstats.peak_sigma[peakstats.peak_sigma .!= 0])
    peakstats.peak_fwhm[:]  = ifelse.(peakstats.peak_fwhm .== 0, median_fwhm, peakstats.peak_fwhm)
    peakstats.peak_sigma[:] = ifelse.(peakstats.peak_sigma .== 0, median_sigma, peakstats.peak_sigma)

    # estimate peak positions energy depencence 
    peak_pos = peakstats.peak_pos
    mean_peak_pos, std_peak_pos = mean(peak_pos), std(peak_pos)
    peak_pos_cut = peak_pos .< mean_peak_pos + 3*std_peak_pos .&& peak_pos .> mean_peak_pos - 3*std_peak_pos
    # simple curve fit for parameter extraction
    simple_fit_aoe_μ        = curve_fit(f_aoe_μ, compton_bands[peak_pos_cut], peak_pos[peak_pos_cut], [0.0, mean_peak_pos])
    simple_pars_aoe_μ       = simple_fit_aoe_μ.param
    simple_pars_error_aoe_μ = standard_errors(simple_fit_aoe_μ)

    # estimate peak sigmas energy depencence
    peak_sigma = peakstats.peak_sigma
    mean_peak_sigma_end, std_peak_sigma_end = mean(peak_sigma[20:end]), std(peak_sigma[20:end])
    # simple curve fit for parameter extraction
    simple_fit_aoe_σ        = curve_fit(f_aoe_σ, compton_bands, peak_sigma, [0.0, 0.0, mean_peak_sigma_end])
    simple_pars_aoe_σ       = simple_fit_aoe_σ.param
    simple_pars_error_aoe_σ = standard_errors(simple_fit_aoe_σ)

    (
        aoe_compton_bands = aoe_compton_bands,
        peakhists = peakhists,
        peakstats = peakstats,
        min_aoe = min_aoe,
        max_aoe = max_aoe,
        mean_peak_pos = mean_peak_pos,
        std_peak_pos = std_peak_pos,
        simple_pars_aoe_μ = simple_pars_aoe_μ,
        simple_pars_error_aoe_μ = simple_pars_error_aoe_μ,
        mean_peak_sigma = mean_peak_sigma_end,
        std_peak_sigma = std_peak_sigma_end,
        simple_pars_aoe_σ = simple_pars_aoe_σ,
        simple_pars_error_aoe_σ = simple_pars_error_aoe_σ
    )
end
export generate_aoe_compton_bands



"""
    fit_aoe_compton(peakhists::Array, peakstats::StructArray, compton_bands::Array{T}) where T<:Real

Fit the A/E Compton bands using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Returns
    * `result`: Dict of NamedTuples of the fit results containing values and errors for each compton band
    * `report`: Dict of NamedTuples of the fit report which can be plotted for each compton band
"""
function fit_aoe_compton(peakhists::Array, peakstats::StructArray, compton_bands::Array{T},; pars_aoe::NamedTuple{(:μ, :μ_err, :σ, :σ_err)}=NamedTuple{(:μ, :μ_err, :σ, :σ_err)}(nothing, nothing, nothing, nothing)) where T<:Real
    if isnothing(pars_aoe.μ)
        # create return and result dicts
        result = Dict{T, NamedTuple}()
        report = Dict{T, NamedTuple}()
        # iterate throuh all peaks
        for (i, band) in enumerate(compton_bands)
            # get histogram and peakstats
            h  = peakhists[i]
            ps = peakstats[i]
            # fit peak
            result_band, report_band = fit_single_aoe_compton(h, ps, ; uncertainty=false)
            # save results
            result[band] = result_band
            report[band] = report_band
        end
        return result, report
    else
        # create return and result dicts
        result = Dict{T, NamedTuple}()
        report = Dict{T, NamedTuple}()
        # iterate throuh all peaks
        for (i, band) in enumerate(compton_bands)
            # get histogram and peakstats
            h  = peakhists[i]
            ps = merge(peakstats[i], (μ = f_aoe_μ(band, pars_aoe.μ), σ = f_aoe_σ(band, pars_aoe.σ)))
            # fit peak
            result_band, report_band = fit_single_aoe_compton(h, ps, ; uncertainty=false)
            # save results
            result[band] = result_band
            report[band] = report_band
        end
        return result, report
    end
end
export fit_aoe_compton


"""
    fit_single_aoe_compton(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real

Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_single_aoe_compton(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background), NTuple{5, T}}; uncertainty::Bool=true) where T<:Real
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                μ = Uniform(ps.peak_pos-3*ps.peak_sigma, ps.peak_pos+3*ps.peak_sigma),
                σ = weibull_from_mx(ps.peak_sigma, 5*ps.peak_sigma),
                n = weibull_from_mx(ps.peak_counts, 7*ps.peak_counts),
                B = weibull_from_mx(ps.mean_background, 5*ps.mean_background),
                δ = weibull_from_mx(0.1, 0.5)
            )
    
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=f_aoe_compton, h=h
        v -> hist_loglike(Base.Fix2(f_fit, v), h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    f_loglike_array = let f_fit=f_aoe_compton, h=h
        v -> - hist_loglike(x -> f_fit(x, v...), h)
    end

    if uncertainty
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance = inv(H)

        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)


        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "B: $(v_ml.B) ± $(v_ml_err.B)"

        result = merge(v_ml, (err = v_ml_err, ))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "B: $(v_ml.B)"

        result = v_ml
    end
    report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(f_aoe_compton, v_ml)(x),
            f_sig = x -> Base.Fix2(f_aoe_sig, v_ml)(x),
            f_bck = x -> Base.Fix2(f_aoe_bkg, v_ml)(x)
        )
    return result, report
end


function fit_single_aoe_compton(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                μ = weibull_from_mx(ps.μ, 2*ps.μ),
                σ = weibull_from_mx(ps.σ, 2*ps.σ),
                n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
                B = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
                δ = weibull_from_mx(0.1, 0.8)
            )
    
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=f_aoe_compton, h=h
        v -> hist_loglike(Base.Fix2(f_fit, v), h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    f_loglike_array = let f_fit=f_aoe_compton, h=h
        v -> - hist_loglike(x -> f_fit(x, v...), h)
    end

    # get p-value
    mle = f_loglike(v_ml)

    weights_rand = rand(Product(Poisson.(h.weights)), 10000)

    f_loglike_h = let f_fit=f_aoe_compton, v=v_ml
        w -> hist_loglike.(Base.Fix2(f_fit, v), fit.(Histogram, Ref(midpoints(h.edges[1])), weights.(w), Ref(h.edges[1])))
    end

    mle_rand = f_loglike_h(eachcol(weights_rand))

    p_value = count(mle_rand .> mle) / length(mle_rand)

    if uncertainty
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance = inv(H)

        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)


        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "B: $(v_ml.B) ± $(v_ml_err.B)"

        result = merge(merge(v_ml, (p_value = p_value, )), (err = v_ml_err, ))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "B: $(v_ml.B)"

        result = merge(v_ml, (p_value = p_value, ))
    end
    report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(f_aoe_compton, v_ml)(x),
            f_sig = x -> Base.Fix2(f_aoe_sig, v_ml)(x),
            f_bck = x -> Base.Fix2(f_aoe_bkg, v_ml)(x)
        )
    return result, report
end
