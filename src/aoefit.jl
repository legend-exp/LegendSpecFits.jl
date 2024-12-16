

"""
    generate_aoe_compton_bands(aoe::Vector{<:Real}, e::Vector{<:T}, compton_bands::Vector{<:T}, compton_window::T) where T<:Unitful.Energy{<:Real}

Generate histograms for the A/E Compton bands and estimate peak parameters. 
The compton bands are cut out of the A/E spectrum and then binned using the Freedman-Diaconis Rule. For better performance
the binning is only done in the area around the peak. The peak parameters are estimated using the `estimate_single_peak_stats_psd` function.

# Arguments
    * `aoe`: A/E
    * `e`: Calibrated energies
    * `compton_bands`: Compton bands
    * `compton_window`: Data window with Compton Scattering events 

# Returns
    * `peakhists`: Array of histograms for each compton band
    * `peakstats`: StructArray of peak parameters for each compton band
    * `min_aoe`: Array of minimum A/E values for each compton band
    * `e_unit`: Energy units
    * `max_aoe`: Array of maximum A/E values for each compton band
    * `mean_peak_pos`: Mean peak position of all compton bands
    * `std_peak_pos`: Standard deviation of the peak position of all compton bands
    * `simple_pars_aoe_μ`: Simple curve fit parameters for the peak position energy depencence
    * `simple_pars_error_aoe_μ`: Simple curve fit parameter errors for the peak position energy depencence
    * `mean_peak_sigma`: Mean peak sigma value
    * `std_peak_sigma`: Standard deviation of the peak sigma of all compton bands
    * `simple_pars_aoe_σ`: Simple curve fit parameters for the peak sigma energy depencence
    * `simple_pars_error_aoe_σ`: Simple curve fit parameter errors for the peak sigma energy depencence
"""
function generate_aoe_compton_bands(aoe::Vector{<:Real}, e::Vector{<:T}, compton_bands::Vector{<:T}, compton_window::T) where T<:Unitful.Energy{<:Real}
    @assert length(aoe) == length(e) "A/E and Energy arrays must have the same length"
    e_unit = u"keV"
    # get aoe values in compton bands
    aoe_compton_bands = [aoe[c .< e .< c + compton_window .&& aoe .> 0.0] for c in compton_bands]

    # can constrain data to the area around the peak
    max_aoe              = [quantile(aoe_c, 0.99) + 0.05 for aoe_c in aoe_compton_bands]
    min_aoe              = [quantile(aoe_c, 0.2)         for aoe_c in aoe_compton_bands]
    half_quantile_aoe    = [quantile(aoe_c, 0.5)         for aoe_c in aoe_compton_bands]

    # Freedman-Diaconis Rule for binning only in the area aroung the peak
    bin_width   = [get_friedman_diaconis_bin_width(aoe_c[half_quantile_aoe[i] .< aoe_c .< max_aoe[i]])/2 for (i, aoe_c) in enumerate(aoe_compton_bands)]

    # generate histograms
    peakhists = [fit(Histogram, aoe_compton_bands[i], min_aoe[i]:bin_width[i]/2:max_aoe[i]) for i in eachindex(aoe_compton_bands)]

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
    simple_fit_aoe_μ        = curve_fit(f_aoe_mu, ustrip.(e_unit, compton_bands[peak_pos_cut]), peak_pos[peak_pos_cut], [mean_peak_pos, 0.0])
    simple_pars_aoe_μ       = simple_fit_aoe_μ.param
    simple_pars_error_aoe_μ = zeros(length(simple_pars_aoe_μ))
    try
        simple_pars_error_aoe_μ = standard_errors(simple_fit_aoe_μ)
    catch e
        @warn "Error calculating standard errors for simple fitted μ: $e"
    end

    # estimate peak sigmas energy depencence
    peak_sigma = peakstats.peak_sigma
    mean_peak_sigma, std_peak_sigma = mean(peak_sigma[20:end]), std(peak_sigma[20:end])
    # simple curve fit for parameter extraction
    simple_fit_aoe_σ        = curve_fit(f_aoe_sigma, ustrip.(e_unit, compton_bands), peak_sigma, [mean_peak_sigma^2, 1])
    simple_pars_aoe_σ       = simple_fit_aoe_σ.param
    simple_pars_error_aoe_σ = zeros(length(simple_pars_aoe_σ))
    try
        simple_pars_error_aoe_σ = standard_errors(simple_fit_aoe_σ)
    catch e
        @warn "Error calculating standard errors for simple fitted σ: $e"
    end


    # Recalculate max_aoe to get rid out high-A/E outliers
    max_aoe  = peakstats.peak_pos .+ 3 .* abs.(peakstats.peak_sigma)
    # Recalculate min_aoe to focus on main peak
    min_aoe = peakstats.peak_pos .- 15 .* abs.(peakstats.peak_sigma)
    min_3sigma_aoe = peakstats.peak_pos .- 3 .* abs.(peakstats.peak_sigma)
    # Freedman-Diaconis Rule for binning only in the area aroung the peak
    bin_width   = [get_friedman_diaconis_bin_width(aoe_c[aoe_c .> min_3sigma_aoe[i] .&& aoe_c .< max_aoe[i]])/4 for (i, aoe_c) in enumerate(aoe_compton_bands)]

    # regenerate histograms
    peakhists = [fit(Histogram, aoe_compton_bands[i], min_aoe[i]:bin_width[i]:max_aoe[i]) for i in eachindex(aoe_compton_bands)]

    # reestimate peak parameters
    peakstats = StructArray(estimate_single_peak_stats_psd.(peakhists))

    (
        ;
        peakhists,
        peakstats,
        min_aoe,
        e_unit,
        max_aoe,
        mean_peak_pos,
        std_peak_pos,
        simple_pars_aoe_μ,
        simple_pars_error_aoe_μ,
        mean_peak_sigma,
        std_peak_sigma,
        simple_pars_aoe_σ,
        simple_pars_error_aoe_σ
    )
end
export generate_aoe_compton_bands



"""
    fit_aoe_compton(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T},; uncertainty::Bool=false, fit_func::Symbol=:aoe_one_bck) where T<:Unitful.Energy{<:Real}

Fit the A/E Compton bands using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Arguments
    * `peakhists`: Data range of the histogram peaks
    * `peakstats`: Peak statistics
    * `compton_bands`: Array of Compton bands
    
# Keywords
    * `uncertainty`: Fit uncertainty
    * `fit_func`: Fitted function

# Returns
    * `result`: Dict of NamedTuples of the fit results containing values and errors for each compton band
    * `report`: Dict of NamedTuples of the fit report which can be plotted for each compton band

"""
function fit_aoe_compton(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T},; uncertainty::Bool=false, fit_func::Symbol=:aoe_one_bck) where T<:Unitful.Energy{<:Real}    
    # create return and result dicts
    v_result = Vector{NamedTuple}(undef, length(compton_bands))
    v_report = Vector{NamedTuple}(undef, length(compton_bands))

    # iterate throuh all peaks
    Threads.@threads for i in eachindex(compton_bands)
        band = compton_bands[i]
        # get histogram and peakstats
        h  = peakhists[i]
        ps = peakstats[i]
        # fit peak
        result_band, report_band = nothing, nothing
        try
            result_band, report_band = fit_single_aoe_compton(h, ps, ; uncertainty=uncertainty, fit_func=fit_func)
        catch e
            @warn "Error fitting band $band: $e"
            continue
        end
        # save results
        v_result[i] = result_band
        v_report[i] = report_band
    end

    # create return and result dicts
    result = Dict{T, NamedTuple}(compton_bands .=> v_result)
    report = Dict{T, NamedTuple}(compton_bands .=> v_report)

    return result, report
end
export fit_aoe_compton


"""
    fit_single_aoe_compton(h::Histogram, ps::NamedTuple; uncertainty::Bool=true, pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), fit_func::Symbol=:aoe_one_bck, background_center::Union{Real,Nothing} = ps.peak_pos, fixed_position::Bool=false)

Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Arguments
    * `h`: Histogram data
    * `ps`: peak statistics 

# Keywords
    * `uncertainty`: Fit uncertainty
    * `pseudo_prior`: Initial guess for histogram parameters 
    * `fit_func`: Fit function symbol
    * `background_center`: Center of background fit curve
    * `fixed_position`: Boolean that decides if the position of fit should be fixed or not

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted

"""
function fit_single_aoe_compton(h::Histogram, ps::NamedTuple; uncertainty::Bool=true, pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), fit_func::Symbol=:aoe_one_bck, background_center::Union{Real,Nothing} = ps.peak_pos, fixed_position::Bool=false)
    # create pseudo priors
    pseudo_prior = get_aoe_pseudo_prior(h, ps, fit_func; pseudo_prior = pseudo_prior, fixed_position = fixed_position)
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = Vector(mean(f_trafo.target_dist))

    # get fit function with background center
    fit_function = get_aoe_fit_functions(; )[fit_func]

    # create loglikehood function
    f_loglike = let f_fit=fit_function, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    optf = OptimizationFunction((u, p) -> ((-) ∘ f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, v_init, [])
    res = solve(optpro, Optimization.LBFGS(), maxiters = 3000) #, maxtime=optim_time_limit)

    converged = (res.retcode == ReturnCode.Success)
    if !converged @warn "Fit did not converge" end

    # best fit results
    v_ml = inverse(f_trafo)(res.u)
        
    if uncertainty && converged
    
        f_loglike_array(v) = - f_loglike(array_to_tuple(v, v_ml))
        
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
        pval, chi2, dof = p_value(fit_function, h, v_ml)
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(fit_function, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "B: $(v_ml.B) ± $(v_ml_err.B)"
        @debug "p: $pval , chi2 = $(chi2) with $(dof) dof"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, residuals = residuals, residuals_norm = residuals_norm, converged = converged),))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(fit_function, v_ml)(x),
            f_components = aoe_compton_peakshape_components(fit_func, v_ml),
            gof = merge(result.gof, (residuals = residuals, residuals_norm = residuals_norm,))        )
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "B: $(v_ml.B)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], NaN) for k in keys(v_ml)]...),
                    (gof = (converged = converged,) ,))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(fit_function, v_ml)(x),
            f_components = aoe_compton_peakshape_components(fit_func, v_ml; background_center = background_center),
            gof = NamedTuple()
        )
    end
    
    return result, report
end