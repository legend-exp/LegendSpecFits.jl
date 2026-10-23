
"""
    cut_single_peak(
        x::AbstractVector{<:Unitful.RealOrRealQuantity}, min_x::T, max_x::T;
        n_bins::Int = -1, relative_cut::Float64 = 0.5, n_tries::Int = 5
    ) where T<:Unitful.RealOrRealQuantity

Determines a window around the single peak in the values `x` that lie
strictly between `min_x` and `max_x`.

The values are histogrammed and the window is the range around the highest bin
where the counts exceed `relative_cut` times the maximum count. If the lower edge
of the window does not lie below the peak maximum, the histogram is rebuilt with
`i`-times finer bins for `i = 2, ..., n_tries`, stopping at the first valid
window. If none is found, a warning is logged and the window of the last
histogram is returned. The output has the unit of `min_x`.

## Keywords
* `n_bins::Int = -1`: Number of histogram bins. A negative value selects the
  Freedman-Diaconis bin width.
* `relative_cut::Float64 = 0.5`: Fraction of the maximum bin content that
  defines the window edges.
* `n_tries::Int = 5`: Maximum number of histograms.

## Returns
A `NamedTuple` `(low, high, max)` with the left bin edge where the counts first
reach the cut level on the rising side, the left bin edge where they fall to
the cut level on the falling side, and the left edge of the maximum bin.
"""
function cut_single_peak(x::AbstractVector{<:Unitful.RealOrRealQuantity}, min_x::T, max_x::T,; n_bins::Int=-1, relative_cut::Float64=0.5, n_tries::Int=5) where T<:Unitful.RealOrRealQuantity
    x_unit = unit(min_x)
    min_x, max_x = ustrip(x_unit, min_x), ustrip(x_unit, max_x)

    # cut out window of interest
    x = ustrip.(x_unit, filter(xi -> min_x < ustrip(x_unit, xi) < max_x, x))
    x_min, x_max = extrema(x)
    bin_width = n_bins < 0 ? get_friedman_diaconis_bin_width(x) : zero(x_min)

    # retry with finer binning until the lower edge of the cut window lies below the peak maximum
    cut_low, cut_high, cut_max = x_min * x_unit, x_min * x_unit, x_min * x_unit
    for i in 1:n_tries
        # fit histogram
        if n_bins < 0
            h = fit(Histogram, x, x_min:bin_width/i:x_max)
        else
            h = fit(Histogram, x, nbins=n_bins*i)
        end
        edges = first(h.edges)

        # find peak
        cts_argmax = argmax(h.weights)
        cts_max    = h.weights[cts_argmax]

        # find left and right edge of peak
        cut_low_arg  = findfirst(w -> w >= relative_cut*cts_max, h.weights[1:cts_argmax])
        cut_high_arg = findfirst(w -> w <= relative_cut*cts_max, h.weights[cts_argmax:end]) + cts_argmax - 1
        cut_low, cut_high, cut_max = edges[cut_low_arg] * x_unit, edges[cut_high_arg] * x_unit, edges[cts_argmax] * x_unit
        if cut_low < cut_max < cut_high
            @debug "Cut window: [$cut_low, $cut_high]"
            break
        end
        @debug "Cut window not found with histogram $i of $n_tries"
    end
    cut_low < cut_max < cut_high || @warn "Cut window not found after $n_tries histograms with finer binning, returning [$cut_low, $cut_high] around $cut_max"
    return (low = cut_low, high = cut_high, max = cut_max)
end
export cut_single_peak


"""
    get_centered_gaussian_window_cut(
        x::Vector{<:Unitful.RealOrRealQuantity}, min_x::T, max_x::T, n_σ::Real;
        center::T = zero(min_x), n_bins::Int = 500, relative_cut::Float64 = 0.2,
        left::Bool = false, fixed_center::Bool = true
    ) where T<:Unitful.RealOrRealQuantity

Determines a symmetric cut window of `±n_σ` standard deviations around a peak
in the values `x` between `min_x` and `max_x`.

The peak region is found with [`cut_single_peak`](@ref). A one-sided truncated
Gaussian is fitted to that region, with [`fit_half_centered_trunc_gauss`](@ref)
(mean fixed to `center`) if `fixed_center` is `true` and with
[`fit_half_trunc_gauss`](@ref) (free mean) otherwise. The fitted mean and
standard deviation define the window.

## Keywords
* `center::T = zero(min_x)`: Mean of the Gaussian if `fixed_center` is `true`.
* `n_bins::Int = 500`: Number of histogram bins for [`cut_single_peak`](@ref).
* `relative_cut::Float64 = 0.2`: Relative cut level for [`cut_single_peak`](@ref).
* `left::Bool = false`: Fit the left side of the peak instead of the right side.
* `fixed_center::Bool = true`: Fix the mean of the Gaussian to `center`.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `low_cut`, `high_cut`: lower and upper edge of the window, `μ ∓ n_σ * σ`.
* `μ`, `σ`: fitted mean and standard deviation.
* `gof`: goodness-of-fit result of the Gaussian fit.
* `low_cut_fit`, `high_cut_fit`: edges of the region used for the fit.
* `max_cut_fit`: position of the maximum bin.

All quantities carry the unit of `min_x`. `report` holds the fit function
`f_fit`, the histogram `h` and the values `μ`, `σ`, `gof`, `low_cut` and
`high_cut`, for plotting with `LegendMakie.lplot`.
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
        low_cut_fit = ifelse(left, cuts.low, result_fit.μ*x_unit),
        high_cut_fit = ifelse(left, result_fit.μ*x_unit, cuts.high),
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
