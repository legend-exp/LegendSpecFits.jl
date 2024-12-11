
"""
    fit_sipm_spectrum(pe_cal::Vector{<:Real}, min_pe::Real=0.5, max_pe::Real=3.5; 
        n_mixtures::Int=ceil(Int, (max_pe - min_pe) * 4), nIter::Int=50, nInit::Int=50, 
        method::Symbol=:kmeans, kind=:diag, Δpe_peak_assignment::Real=0.3, f_uncal::Function=identity, uncertainty::Bool=true)

Fit a Gaussian Mixture Model to the given pe calibration data and return the fit parameters.

# Arguments
- `pe_cal::Vector{<:Real}`: the pe calibration data
- `min_pe::Real=0.5`: the minimum pe to consider
- `max_pe::Real=3.5`: the maximum pe to consider
- `n_mixtures::Int=ceil(Int, (max_pe - min_pe) * 4)`: the number of mixtures to fit
- `nIter::Int=50`: the number of iterations for the EM algorithm
- `nInit::Int=50`: the number of initializations for the EM algorithm
- `method::Symbol=:kmeans`: the method to use for initialization
- `kind::Symbol=:diag`: the kind of covariance matrix to use
- `Δpe_peak_assignment::Real=0.3`: the range to consider for peak assignment
- `f_uncal::Function=identity`: the function to use for uncalibration
- `uncertainty::Bool=true`: whether to calculate the uncertainty

# Returns
- `result`: a tuple with the fit parameters
- `report`: a tuple with the fit report which can be plotted via a recipe
"""
function fit_sipm_spectrum(pe_cal::Vector{<:Real}, min_pe::Real=0.5, max_pe::Real=3.5;
    n_mixtures::Int=ceil(Int, (max_pe - min_pe) * 4), nIter::Int=50, nInit::Int=50, 
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
    bin_width = get_friedman_diaconis_bin_width(filter(in((-Δpe_peak_assignment..Δpe_peak_assignment) .+ first(pes)), pe_cal))

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
    get_pe_pos = pe -> let sel = in.(μ, (-Δpe_peak_assignment..Δpe_peak_assignment) .+ pe)
        dot(view(μ,sel), view(w,sel)) / sum(view(w,sel))
    end
    get_pe_res = pe -> let sel = in.(μ, (-Δpe_peak_assignment..Δpe_peak_assignment) .+ pe)
        dot(view(σ,sel), view(w,sel)) / sum(view(w,sel))
    end 
    n_pos_mixtures = [count(in.(μ, (-Δpe_peak_assignment..Δpe_peak_assignment) .+ pe)) for pe in pes]

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