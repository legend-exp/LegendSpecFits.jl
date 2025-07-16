
"""
    cut_single_peak(x::Array, min_x::Float64, max_x::Float64,; n_bins::Int=-1, relative_cut::Float64=0.5)

Cut out a single peak from the array `x` between `min_x` and `max_x`.
The number of bins is the number of bins to use for the histogram.
The relative cut is the fraction of the maximum counts to use for the cut.
# Returns 
    * `max`: maximum position of the peak
    * `low`: lower edge of the cut peak
    * `high`: upper edge of the cut peak
"""
function cut_single_peak(x::Vector{<:Unitful.RealOrRealQuantity}, min_x::T, max_x::T,; n_bins::Int=-1, relative_cut::Float64=0.5, n_tries::Int=5) where T<:Unitful.RealOrRealQuantity
    x_unit = unit(min_x)
    x, min_x, max_x = ustrip.(x_unit, x), ustrip(x_unit, min_x), ustrip(x_unit, max_x)

    # cut out window of interest
    x = x[(x .> min_x) .&& (x .< max_x)]

    # fit histogram
    if n_bins < 0
        bin_width = get_friedman_diaconis_bin_width(x)
        h = fit(Histogram, x, minimum(x):bin_width:maximum(x))
    else
        h = fit(Histogram, x, nbins=n_bins)
    end

    # initialize cut window
    cut_low, cut_high, cut_max = Array(h.edges[1])[1] * x_unit, Array(h.edges[1])[1] * x_unit, Array(h.edges[1])[1] * x_unit
    for i in 1:n_tries
        # fit histogram
        if n_bins < 0
            h = fit(Histogram, x, minimum(x):bin_width/i:maximum(x))
        else
            h = fit(Histogram, x, nbins=n_bins*i)
        end
        # determine cut
        if !(cut_low < cut_max < cut_high)
            i > 1 && @warn "Cut window not found, trying again"
            
            # find peak
            cts_argmax = mapslices(argmax, h.weights, dims=1)[1]
            cts_max    = h.weights[cts_argmax]

            # find left and right edge of peak
            cut_low_arg  = findfirst(w -> w >= relative_cut*cts_max, h.weights[1:cts_argmax])
            cut_high_arg = findfirst(w -> w <= relative_cut*cts_max, h.weights[cts_argmax:end]) + cts_argmax - 1
            cut_low, cut_high, cut_max = Array(h.edges[1])[cut_low_arg] * x_unit, Array(h.edges[1])[cut_high_arg] * x_unit, Array(h.edges[1])[cts_argmax] * x_unit
        else
            @debug "Cut window: [$cut_low, $cut_high]"
        end
    end
    return (low = cut_low, high = cut_high, max = cut_max)
end
export cut_single_peak


"""
    get_centered_gaussian_window_cut(x::Array, min_x::Float64, max_x::Float64, n_σ::Real, center::Float64=0.0, n_bins::Int=500, relative_cut::Float64=0.2, left::Bool=false)

Cut out a single peak from the array `x` between `min_x` and `max_x` by fitting a truncated one-sided Gaussian and extrapolating a window cut with `n_σ` standard deviations.
The `center` and side of the fit can be specified with `left` and `center` variable.
# Returns
    * `low_cut`: lower edge of the cut peak
    * `high_cut`: upper edge of the cut peak
    * `µ`: center of the peak
    * `σ`: standard deviation of the Gaussian
    * `gof`: goodness of fit
    * `low_cut_fit`: lower edge of the cut peak from the fit
    * `high_cut_fit`: upper edge of the cut peak from the fit
    * `max_cut_fit`: maximum position of the peak
"""
function get_centered_gaussian_window_cut(x::Vector{<:Unitful.RealOrRealQuantity}, min_x::T, max_x::T, n_σ::Real,; center::T=zero(min_x), n_bins::Int=500, relative_cut::Float64=0.2, left::Bool=false, fixed_center::Bool=true) where T<:Unitful.RealOrRealQuantity
    # prepare data
    x_unit = unit(min_x)
    x, min_x, max_x, center = ustrip.(x_unit, x), ustrip(x_unit, min_x), ustrip(x_unit, max_x), ustrip(x_unit, center)

    # get cut window around peak
    cuts = cut_single_peak(x, min_x, max_x,; n_bins=n_bins, relative_cut=relative_cut)

    # fit half centered gaussian to define sigma width
    result_fit, report_fit = if !fixed_center
        fit_half_trunc_gauss(x, cuts,; left=left)
    else
        fit_half_centered_trunc_gauss(x, center, cuts,; left=left)
    end

    result = (
        low_cut  = (result_fit.μ - n_σ*result_fit.σ)*x_unit,
        high_cut = (result_fit.μ + n_σ*result_fit.σ)*x_unit,
        μ = result_fit.μ*x_unit,
        σ = result_fit.σ*x_unit,
        gof = result_fit.gof,
        low_cut_fit = ifelse(left, cuts.low, result_fit.μ), 
        high_cut_fit = ifelse(left, result_fit.μ, cuts.high),
        max_cut_fit = cuts.max
    )
    report = (
        f_fit = t -> report_fit.f_fit(t),
        h = report_fit.h,
        μ = result.μ,
        σ = result.σ,
        gof = result.gof,
        low_cut = result.low_cut,
        high_cut = result.high_cut,
    )
    return result, report
end
export get_centered_gaussian_window_cut
