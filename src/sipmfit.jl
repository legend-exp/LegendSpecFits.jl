
"""
    fit_sipm_spectrum(
        pe_cal::Vector{<:Real}, min_pe::Real = 0.5, max_pe::Real = 3.5;
        n_mixtures::Int = ceil(Int, (max_pe - min_pe) * 4), nIter::Int = 25, nInit::Int = 50,
        method::Symbol = :kmeans, kind = :diag, Δpe_peak_assignment::Real = 0.3,
        f_uncal::Function = identity, uncertainty::Bool = true
    )

Fits a Gaussian mixture model to the calibrated SiPM amplitudes `pe_cal` (in
photoelectron units) between `min_pe` and `max_pe`, and derives the position and
resolution of each photoelectron peak.

The mixture is fitted with the expectation-maximization algorithm of
`GaussianMixtures.jl`. The position of the peak at integer PE value `pe` is
the weighted mean of the Gaussian centers within `pe ± Δpe_peak_assignment`; its
resolution is the FWHM of the sum of these Gaussians. Uncertainties are determined from the
Hessian of the binned likelihood.

## Arguments
* `pe_cal`: Calibrated amplitudes in PE units.
* `min_pe`, `max_pe`: Fit range.

## Keywords
* `n_mixtures::Int`: Number of Gaussians.
* `nIter::Int = 25`, `nInit::Int = 50`: Number of EM iterations and of initializations.
* `method::Symbol = :kmeans`: Initialization method of the mixture.
* `kind = :diag`: Covariance type of the mixture.
* `Δpe_peak_assignment::Real = 0.3`: Distance in PE within which a Gaussian is assigned to a peak.
* `f_uncal::Function = identity`: Function that converts calibrated values to uncalibrated ones, to express positions and resolutions in uncalibrated units.
* `uncertainty::Bool = true`: Calculate uncertainties and goodness of fit.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `μ`, `σ`, `w`: means, standard deviations and weights of the Gaussians (`Measurement`s).
* `n_mixtures`, `n_pos_mixtures`: number of Gaussians in total and assigned to each peak.
* `peaks`: integer PE values of the peaks.
* `positions_cal`, `resolutions_cal`: peak positions and FWHM in PE units.
* `positions`, `resolutions`: the same in uncalibrated units (via `f_uncal`).
* `gof`: `pvalue`, `chi2`, `dof` (from [`p_value_poissonll`](@ref)), `covmat` and mean, median and standard deviation of the normalized residuals (empty if `uncertainty` is `false`).

`report` contains the histogram `h_cal`, the fit function `f_fit`, the components `f_fit_components(x, i)`,
`min_pe`, `max_pe`, `bin_width`, `n_mixtures`, `n_pos_mixtures`, `peaks`, `positions`, `μ` and `gof`,
for plotting with `LegendMakie.lplot`.
"""
function fit_sipm_spectrum(pe_cal::Vector{<:Real}, min_pe::Real=0.5, max_pe::Real=3.5;
    n_mixtures::Int=ceil(Int, (max_pe - min_pe) * 4), nIter::Int=25, nInit::Int=50, 
    method::Symbol=:kmeans, kind=:diag, Δpe_peak_assignment::Real=0.3, f_uncal::Function=identity, uncertainty::Bool=true)
    # first filter peak positions out of amplitude vector
    amps_fit = filter(in(min_pe..max_pe), pe_cal)
    
    # reshape necessary to deal with the GMM
    dmat = reshape(amps_fit, length(amps_fit), 1)

    # set up mixture model with given number of mixtures
    gmm = GMM(n_mixtures, dmat; method=method, nInit=nInit, nIter=nIter, kind=kind, parallel=false)
    
    # get mixture model out of EM best fit estimate
    gmm_dist = MixtureModel(gmm)

    # get Gauss center and weights vector out of gmm
    μ_ml = reshape(gmm.μ, n_mixtures)
    σ_ml = sqrt.(reshape(gmm.Σ, n_mixtures))
    w_ml = gmm.w

    # PE positions to be determined are all integers up to the max_pe
    pes = ceil(Int, min_pe):1:floor(Int, max_pe)

    # calculate bin width for histogram
    # explicit endpoints: broadcasting `(-Δ..Δ) .+ pe` over an interval is not supported by IntervalSets
    bin_width = get_friedman_diaconis_bin_width(filter(in((first(pes) - Δpe_peak_assignment)..(first(pes) + Δpe_peak_assignment)), pe_cal))

    # create gof NamedTuple
    gof, gof_report = NamedTuple(), NamedTuple()
    
    if uncertainty
        # define loglikelihood function for binned data to enhance speed
        h = fit(Histogram, pe_cal, minimum(amps_fit):bin_width:maximum(amps_fit))

        # create vector of all parameters
        μσw_ml = vcat(μ_ml, σ_ml, w_ml)

        # define loglikelihood function
        f_loglike = let n=n_mixtures, bin_edges=only(h.edges), bin_counts=h.weights
            μσw -> -_gmm_binned_loglike_func(μσw[1:n], μσw[n+1:2*n], μσw[2*n+1:end], bin_edges)(bin_counts)
        end
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike, μσw_ml)
    
        # Calculate the parameter covariance matrix
        param_covariance_raw = inv(H)
        param_covariance = nearestSPD(param_covariance_raw)

        # Extract the parameter uncertainties
        μσw_ml_err = sqrt.(abs.(diag(param_covariance)))
        μ_err, σ_err, w_err = μσw_ml_err[1:n_mixtures], μσw_ml_err[n_mixtures+1:2*n_mixtures], μσw_ml_err[2*n_mixtures+1:end]

        # create fit function
        fit_function = let n=n_mixtures, total_counts=sum(h.weights)
            (x, μσw)  -> sum(h.weights) .* sum(μσw[2*n+1:end] .* pdf.(Normal.(μσw[1:n], μσw[n+1:2*n]), x))
        end

        # calculate p-value
        pval, chi2, dof = p_value_poissonll(fit_function, h, μσw_ml) # based on likelihood ratio 

        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(fit_function, h, μσw_ml)

        gof = (pvalue = pval, 
                chi2 = chi2, 
                dof = dof, 
                covmat = param_covariance, 
                mean_residuals = mean(residuals_norm),
                median_residuals = median(residuals_norm),
                std_residuals = std(residuals_norm))
        gof_report = merge(gof, (residuals = residuals, 
                                residuals_norm = residuals_norm,
                                bin_centers = bin_centers))

        μ, σ, w = measurement.(μ_ml, μ_err), measurement.(σ_ml, σ_err), measurement.(w_ml, w_err)
    else
        μ, σ, w = measurement.(μ_ml, Ref(NaN)), measurement.(σ_ml, Ref(NaN)), measurement.(w_ml,Ref(NaN))
    end

    # get pe_pos
    get_pe_pos = pe -> let sel = in.(μ, Ref((pe - Δpe_peak_assignment)..(pe + Δpe_peak_assignment)))
        dot(view(μ, sel), view(w, sel)) / sum(view(w, sel))
    end

    # get pe resolution (FWHM)
    get_pe_res = pe -> let sel = in.(μ, Ref((pe - Δpe_peak_assignment)..(pe + Δpe_peak_assignment)))
        model_parameters = [
            view(μ_ml, sel),
            view(σ_ml, sel),
            view(w_ml, sel)
        ]
        
        # Find maximum of PDF in the expected range, step size depends on the maximum variance of the single gaussians
        range_steps = maximum(getfield.(view(μ, sel), :err)) / 100
        x_max = 0.0
        f_max = 0.0
        # Search range depends on the selection of gaussians
        for x_i in minimum(view(μ_ml, sel)):range_steps:maximum(view(μ_ml, sel))
            if _gmm_pdf(x_i, model_parameters) > f_max
                x_max = x_i
                f_max = _gmm_pdf(x_i, model_parameters)
            end
        end
        f_half_max = f_max / 2

        # Find all the points where the half max is reached
        roots = find_zeros(x -> f_half_max - _gmm_pdf(x, model_parameters), 0, 5)
        return maximum(roots) - minimum(roots)
    end


    n_pos_mixtures = [count(in((pe - Δpe_peak_assignment)..(pe + Δpe_peak_assignment)), μ) for pe in pes]

    pe_pos = get_pe_pos.(pes)
    pe_res = get_pe_res.(pes)

    # create return histogram for report
    h_cal = fit(Histogram, pe_cal, ifelse(min_pe >= 0.5, min_pe-0.5, min_pe):bin_width:max_pe+0.5)

    result = (
        μ = μ,
        σ = σ,
        w = w,
        n_pos_mixtures = n_pos_mixtures,
        n_mixtures = n_mixtures,
        peaks = pes,
        positions_cal = pe_pos,
        positions = f_uncal.(pe_pos),
        resolutions_cal = pe_res,
        resolutions = f_uncal.(pe_res) .- f_uncal(0.0),
        gof = gof
    )
    report = (
        h_cal = h_cal,
        f_fit = x -> pdf(gmm_dist, x) * length(amps_fit) * bin_width,
        f_fit_components = (x, i) -> length(amps_fit) * bin_width * w_ml[i] * pdf(Normal(μ_ml[i], σ_ml[i]), x),
        min_pe = min_pe,
        max_pe = max_pe,
        bin_width = bin_width,
        n_mixtures = result.n_mixtures,
        n_pos_mixtures = result.n_pos_mixtures,
        peaks = result.peaks,
        positions = result.positions_cal,
        μ = result.μ,
        gof = gof_report
    )
    return result, report
end
export fit_sipm_spectrum


function _gmm_calc_p_bin(
    mix_μ::AbstractVector{<:Real}, mix_σ::AbstractVector{<:Real}, mix_w::AbstractVector{<:Real},
    bin_edges::AbstractVector{<:Real}
)
    edge_cdf = vec(sum(mix_w .* cdf.(Normal.(mix_μ, mix_σ), bin_edges'), dims = 1))
    renorm_edge_cdf = (edge_cdf .- edge_cdf[begin]) .* inv(edge_cdf[end] - edge_cdf[begin])
    diff(renorm_edge_cdf)
end

function _gmm_binned_loglike_func(
    mix_μ::AbstractVector{<:Real}, mix_σ::AbstractVector{<:Real},
    mix_w::AbstractVector{<:Real}, bin_edges::AbstractVector{<:Real}
)
    p_bin = _gmm_calc_p_bin(mix_μ, mix_σ, mix_w, bin_edges)
    bin_widths = diff(bin_edges)
    binned_density = p_bin ./ bin_widths
    # Without permutation correction to get values similar to unbinned:
    f_loglike(bin_counts) = sum(
        xlogy.(bin_counts, binned_density)
    )
    return f_loglike
end

function _gmm_pdf(x::Real, μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    return sum(@. w * exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π)))
end

_gmm_pdf(x::Real, V::Vector{<:AbstractVector{<:Real}}) = _gmm_pdf(x, V...)