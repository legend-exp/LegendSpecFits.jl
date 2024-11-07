
"""
    fit_sipm_spectrum(pe_cal::Vector{<:Real}, min_pe::Real=0.5, max_pe::Real=3.5; 
        n_mixtures::Int=ceil(Int, (max_pe - min_pe) * 4), nIter::Int=50, nInit::Int=50, 
        method::Symbol=:kmeans, kind=:diag, Δpe_peak_assignment::Real=0.3)

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

# Returns
- `result`: a tuple with the fit parameters
- `report`: a tuple with the fit report which can be plotted via a recipe
"""
function fit_sipm_spectrum(pe_cal::Vector{<:Real}, min_pe::Real=0.5, max_pe::Real=3.5;
    n_mixtures::Int=ceil(Int, (max_pe - min_pe) * 4), nIter::Int=50, nInit::Int=50, 
    method::Symbol=:kmeans, kind=:diag, Δpe_peak_assignment::Real=0.3)
    
    # first filter peak positions out of amplitude vector
    amps_fit = filter(in(min_pe..max_pe), pe_cal)
    
    # reshape necessary to deal with the GMM
    dmat = reshape(amps_fit, length(amps_fit), 1)

    # set up mixture model with given number of mixtures
    gmm = GMM(n_mixtures, dmat; method=method, nInit=nInit, nIter=nIter, kind=kind)
    
    # get mixture model out of EM best fit estimate
    gmm_dist = MixtureModel(gmm)

    # get Gauss center and weights vector out of gmm
    μ = reshape(gmm.μ, n_mixtures)
    σ = reshape(gmm.Σ, n_mixtures)
    w = gmm.w
    
    # PE positions to be determined are all integers up to the max_pe
    pes = ceil(Int, min_pe):1:floor(Int, max_pe)

    # get pe_pos
    get_pe_pos = pe -> dot(μ[in.(μ, (-Δpe_peak_assignment..Δpe_peak_assignment) .+ pe)], w[in.(μ, (-Δpe_peak_assignment..Δpe_peak_assignment) .+ pe)] ) / sum(w[in.(μ, (-Δpe_peak_assignment..Δpe_peak_assignment) .+ pe)])
    n_mixtures = [count(in.(μ, (-Δpe_peak_assignment..Δpe_peak_assignment) .+ pe)) for pe in pes]

    pe_pos = get_pe_pos.(pes)

    # create return histogram for report
    bin_width = get_friedman_diaconis_bin_width(filter(in((-Δpe_peak_assignment..Δpe_peak_assignment) .+ first(pes)), pe_cal))
    h_cal = fit(Histogram, pe_cal, ifelse(min_pe >= 0.5, min_pe-0.5, min_pe):bin_width:max_pe+0.5)

    result = (
        μ = μ,
        σ = σ,
        w = w,
        n_mixtures = n_mixtures,
        peaks = pes,
        positions = pe_pos,
        gmm = gmm_dist
    )
    report = (
        h_cal = h_cal,
        f_fit = x -> pdf(gmm_dist, x) * length(amps_fit) * bin_width,
        min_pe = min_pe,
        max_pe = max_pe,
        bin_width = bin_width,
        n_mixtures = result.n_mixtures,
        peaks = result.peaks,
        positions = result.positions
    )
    return result, report
end
export fit_sipm_spectrum