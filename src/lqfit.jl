# lq compton region peakshapes
f_lq_compton(x, v) = aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ)
f_lq_sig(x, v)     = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)
f_lq_bkg(x, v)     = aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ)

#######
f_lq_mu(x, p) = p[1] .+ p[2].*x
@. f_lq_sigma(x, p) = sqrt(abs(p[1]) + abs(p[2])/x^2)
#######

# lq compton centroids energy depencence
MaybeWithEnergyUnits = Union{Real, Unitful.Energy{<:Real}}
f_lq_μ(x::T, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = -v[1] * x + v[2]
f_lq_μ(x::Array{<:T}, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = f_lq_μ.(x, v)
f_lq_μ(x, v::NamedTuple) = f_lq_μ(x, [v.μ_scs_slope, v.μ_scs_intercept])

# lq compton sigma energy depencence
f_lq_σ(x::T, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = exponential_decay(x, v[1], v[2], v[3])
f_lq_σ(x::Array{<:T}, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = exponential_decay.(x, v[1], v[2], v[3])
f_lq_σ(x, v::NamedTuple) = f_lq_σ(x, [v.σ_scs_amplitude, v.σ_scs_decay, v.σ_scs_offset])

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
function estimate_single_peak_stats_psd_lq(h::Histogram{T}) where T<:Real
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
    # make sure that peakstats have non-zero sigma and fwhm values to prevent fit priors from being zero
    if peak_fwhm == 0
        fwqm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 4, W)
        fwqm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 4, W)
        peak_fwqm = (E[fwqm_idx_right] - E[fwqm_idx_left]) / 1.0
        peak_sigma = peak_fwqm * inv(2*√(2log(4)))
        peak_fwhm  = peak_sigma * 2*√(2log(2))
    end
    #peak_area = peak_amplitude * peak_sigma * sqrt(2*π)
    # mean_background = (first(W) + last(W)) / 2
    # five_sigma_idx_left = findfirst(e -> e >= peak_pos - 5*peak_sigma, E)
    three_sigma_idx_left = findfirst(e -> e >= peak_pos - 3*peak_sigma, E)
    mean_background = convert(typeof(peak_pos), (sum(view(W, 1:three_sigma_idx_left))))
    mean_background = ifelse(mean_background == 0.0, 100.0, mean_background)
    # peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)
    # peak_counts = sum(view(W,three_sigma_idx_left:lastindex(W))) / (1 - exp(-3))
    peak_counts = 2*sum(view(W,peak_idx:lastindex(W)))

    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts, 
        mean_background = mean_background
    )
end

"""
    generate_lq_compton_bands(lq::Vector{<:Real}, e::Vector{<:T}, compton_bands::Vector{<:T}, compton_window::T) where T<:Unitful.Energy{<:Real}

Generate histograms for the A/E Compton bands and estimate peak parameters. 
The compton bands are cutted out of the A/E spectrum and then binned using the Freedman-Diaconis Rule. For better performance
the binning is only done in the area around the peak. The peak parameters are estimated using the `estimate_single_peak_stats_psd` function.

# Returns
    * `peakhists`: Array of histograms for each compton band
    * `peakstats`: StructArray of peak parameters for each compton band
    * `min_lq`: Array of minimum A/E values for each compton band
    * `max_lq`: Array of maximum A/E values for each compton band
    * `mean_peak_pos`: Mean peak position of all compton bands
    * `std_peak_pos`: Standard deviation of the peak position of all compton bands
    * `simple_pars_lq_μ`: Simple curve fit parameters for the peak position energy depencence
    * `simple_pars_error_lq_μ`: Simple curve fit parameter errors for the peak position energy depencence
    * `simple_pars_lq_σ`: Simple curve fit parameters for the peak sigma energy depencence
    * `simple_pars_error_lq_σ`: Simple curve fit parameter errors for the peak sigma energy depencence
"""
function generate_lq_compton_bands(lq::Vector{<:Real}, e::Vector{<:T}, compton_bands::Vector{<:T}, compton_window::T) where T<:Unitful.Energy{<:Real}
    @assert length(lq) == length(e) "A/E and Energy arrays must have the same length"
    e_unit = u"keV"
    # get lq values in compton bands
    lq_compton_bands = [lq[c .< e .< c + compton_window .&& lq .> 0.0] for c in compton_bands]

    # can constrain data to the area around the peak
    max_lq              = [quantile(lq_c, 0.99) + 0.05 for lq_c in lq_compton_bands]
    min_lq              = [quantile(lq_c, 0.2)         for lq_c in lq_compton_bands]
    half_quantile_lq    = [quantile(lq_c, 0.5)         for lq_c in lq_compton_bands]

    # Freedman-Diaconis Rule for binning only in the area aroung the peak
    # bin_width   = [2 * (quantile(lq_c[lq_c .> half_quantile_lq[i] .&& lq_c .< max_lq[i]], 0.75) - quantile(lq_c[lq_c .> half_quantile_lq[i] .&& lq_c .< max_lq[i]], 0.25)) / ∛(length(lq_c[lq_c .> half_quantile_lq[i] .&& lq_c .< max_lq[i]])) for (i, lq_c) in enumerate(lq_compton_bands)]
    bin_width   = [get_friedman_diaconis_bin_width(lq_c[half_quantile_lq[i] .< lq_c .< max_lq[i]])/2 for (i, lq_c) in enumerate(lq_compton_bands)]
    # n_bins   = [round(Int, (max_lq[i] - half_quantile_lq[i]) / get_friedman_diaconis_bin_width(lq_c[lq_c .> half_quantile_lq[i] .&& lq_c .< max_lq[i]])) for (i, lq_c) in enumerate(lq_compton_bands)]

    # cuts = [cut_single_peak(lq_c, min_lq[i], max_lq[i]; n_bins=n_bins[i], relative_cut=0.5) for (i, lq_c) in enumerate(lq_compton_bands)]
    # cuts = [cut_single_peak(lq_c, min_lq[i], max_lq[i]; n_bins=-1, relative_cut=0.5) for (i, lq_c) in enumerate(lq_compton_bands)]
    # bin_width = [get_friedman_diaconis_bin_width(lq_c[cuts[i].low .< lq_c .< cuts[i].high]) for (i, lq_c) in enumerate(lq_compton_bands)]
    # generate histograms
    peakhists = [fit(Histogram, lq_compton_bands[i], min_lq[i]:bin_width[i]/2:max_lq[i]) for i in eachindex(lq_compton_bands)]

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
    peak_pos_cut = mean_peak_pos - 3*std_peak_pos .< peak_pos .< mean_peak_pos + 3*std_peak_pos
    # simple curve fit for parameter extraction
    simple_fit_lq_μ        = curve_fit(f_lq_mu, ustrip.(e_unit, compton_bands[peak_pos_cut]), peak_pos[peak_pos_cut], [mean_peak_pos, 0.0])
    simple_pars_lq_μ       = simple_fit_lq_μ.param
    simple_pars_error_lq_μ = zeros(length(simple_pars_lq_μ))
    try
        simple_pars_error_lq_μ = standard_errors(simple_fit_lq_μ)
    catch e
        @warn "Error calculating standard errors for simple fitted μ: $e"
    end

    # estimate peak sigmas energy depencence
    peak_sigma = peakstats.peak_sigma
    mean_peak_sigma, std_peak_sigma = mean(peak_sigma[20:end]), std(peak_sigma[20:end])
    # simple curve fit for parameter extraction
    simple_fit_lq_σ        = curve_fit(f_lq_sigma, ustrip.(e_unit, compton_bands), peak_sigma, [mean_peak_sigma^2, 1])
    simple_pars_lq_σ       = simple_fit_lq_σ.param
    simple_pars_error_lq_σ = zeros(length(simple_pars_lq_σ))
    try
        simple_pars_error_lq_σ = standard_errors(simple_fit_lq_σ)
    catch e
        @warn "Error calculating standard errors for simple fitted σ: $e"
    end


    # Recalculate max_lq to get rid out high-A/E outliers
    max_lq  = peakstats.peak_pos .+ 4 .* abs.(peakstats.peak_sigma)
    # Recalculate min_lq to focus on main peak
    min_lq = peakstats.peak_pos .- 20 .* abs.(peakstats.peak_sigma)
    min_3sigma_lq = peakstats.peak_pos .- 3 .* abs.(peakstats.peak_sigma)
    # Freedman-Diaconis Rule for binning only in the area aroung the peak
    # bin_width   = [get_friedman_diaconis_bin_width(lq_c[lq_c .> half_quantile_lq[i] .&& lq_c .< max_lq[i]])/4 for (i, lq_c) in enumerate(lq_compton_bands)]
    bin_width   = [get_friedman_diaconis_bin_width(lq_c[lq_c .> min_3sigma_lq[i] .&& lq_c .< max_lq[i]])/4 for (i, lq_c) in enumerate(lq_compton_bands)]

    # regenerate histograms
    peakhists = [fit(Histogram, lq_compton_bands[i], min_lq[i]:bin_width[i]:max_lq[i]) for i in eachindex(lq_compton_bands)]

    # reestimate peak parameters
    peakstats = StructArray(estimate_single_peak_stats_psd.(peakhists))

    (
        ;
        peakhists,
        peakstats,
        min_lq,
        e_unit,
        max_lq,
        mean_peak_pos,
        std_peak_pos,
        simple_pars_lq_μ,
        simple_pars_error_lq_μ,
        mean_peak_sigma,
        std_peak_sigma,
        simple_pars_lq_σ,
        simple_pars_error_lq_σ
    )
end
export generate_lq_compton_bands



"""
    fit_lq_compton(peakhists::Array, peakstats::StructArray, compton_bands::Array{T}) where T<:Real

Fit the A/E Compton bands using the `f_lq_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Returns
    * `result`: Dict of NamedTuples of the fit results containing values and errors for each compton band
    * `report`: Dict of NamedTuples of the fit report which can be plotted for each compton band
"""
function fit_lq_compton(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T},; pars_lq::NamedTuple{(:μ, :μ_err, :σ, :σ_err)}=NamedTuple{(:μ, :μ_err, :σ, :σ_err)}(nothing, nothing, nothing, nothing), uncertainty::Bool=false) where T<:Unitful.Energy{<:Real}
    # create return and result dicts
    result = Dict{T, NamedTuple}()
    report = Dict{T, NamedTuple}()
    # iterate throuh all peaks
    for (i, band) in enumerate(compton_bands)
        # get histogram and peakstats
        h  = peakhists[i]
        ps = peakstats[i]
        if !isnothing(pars_lq.μ)
            ps = merge(peakstats[i], (μ = f_lq_μ(band, pars_lq.μ), σ = f_lq_σ(band, pars_lq.σ)))
        end
        # fit peak
        result_band, report_band = nothing, nothing
        try
            result_band, report_band = fit_single_lq_compton(h, ps, ; uncertainty=uncertainty)
        catch e
            @warn "Error fitting band $band: $e"
            continue
        end
        # save results
        result[band] = result_band
        report[band] = report_band
    end
    return result, report
end
export fit_lq_compton


"""
    fit_single_lq_compton(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real

Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_lq_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_single_lq_compton(h::Histogram, ps::NamedTuple; uncertainty::Bool=true)
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                μ = Uniform(ps.peak_pos-0.5*ps.peak_sigma, ps.peak_pos+0.5*ps.peak_sigma),
                # σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
                σ = Uniform(0.95*ps.peak_sigma, 1.05*ps.peak_sigma),
                # σ = Normal(ps.peak_sigma, 0.01*ps.peak_sigma),
                # n = weibull_from_mx(ps.peak_counts, 1.1*ps.peak_counts),
                # n = Normal(ps.peak_counts, 0.5*ps.peak_counts),
                # n = Normal(0.9*ps.peak_counts, 0.5*ps.peak_counts),
                n = LogUniform(0.01*ps.peak_counts, 5*ps.peak_counts),
                # n = Uniform(0.8*ps.peak_counts, 1.2*ps.peak_counts),
                # B = weibull_from_mx(ps.mean_background, 1.2*ps.mean_background),
                # B = Normal(ps.mean_background, 0.8*ps.mean_background),
                B = Uniform(0.1*ps.mean_background, 100*ps.mean_background),
                # B = Uniform(0.8*ps.mean_background, 1.2*ps.mean_background),
                # B = Uniform(0.8*ps.mean_background, 1.2*ps.mean_background),
                # δ = weibull_from_mx(0.1, 0.8)
                δ = Uniform(0.01, 1e7)
            )
    if haskey(ps, :μ)
        # create pseudo priors
        pseudo_prior = NamedTupleDist(
                    μ = weibull_from_mx(ps.μ, 2*ps.μ),
                    σ = weibull_from_mx(ps.σ, 2*ps.σ),
                    n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
                    B = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
                    δ = weibull_from_mx(0.1, 0.8)
                )
    end
    
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=f_lq_compton, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    f_loglike_array = let f_fit=aoe_compton_peakshape, h=h
        v -> - hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v...) : 0, h)
    end

    if uncertainty
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        param_covariance = nothing
        if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        # get p-value 
        pval, chi2, dof = p_value(f_lq_compton, h, v_ml)
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_lq_compton, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "B: $(v_ml.B) ± $(v_ml_err.B)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                  (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                  residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "B: $(v_ml.B)"

        result = merge(v_ml, )
    end
    report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(f_lq_compton, v_ml)(x),
            f_sig = x -> Base.Fix2(f_lq_sig, v_ml)(x),
            f_bck = x -> Base.Fix2(f_lq_bkg, v_ml)(x)
        )
    return result, report
end
