
"""
    cut_single_peak(x::Array, min_x::Float64, max_x::Float64, n_bins::Int=15000, relative_cut::Float64=0.5)

Cut out a single peak from the array `x` between `min_x` and `max_x`.
The number of bins is the number of bins to use for the histogram.
The relative cut is the fraction of the maximum counts to use for the cut.
Returns 
    * `h`: histogram of the cut peak
    * `cut_low`: lower edge of the cut peak
    * `cut_high`: upper edge of the cut peak
"""
function cut_single_peak(x::Vector{T}, min_x::T, max_x::T, n_bins::Int=1000, relative_cut::Float64=0.5) where T<:Unitful.RealOrRealQuantity
    @assert unit(min_x) == unit(max_x) == unit(x[1]) "Units of min_x, max_x and x must be the same"
    x_unit = unit(x[1])
    x, min_x, max_x = ustrip.(x), ustrip(min_x), ustrip(max_x)

    # cut out window of interest
    x = x[(x .> min_x) .&& (x .< max_x)]
    # fit histogram
    h = fit(Histogram, x, nbins=n_bins)
    # find peak
    cts_argmax = mapslices(argmax, h.weights, dims=1)[1]
    cts_max    = h.weights[cts_argmax]

    # find left and right edge of peak
    cut_low_arg  = findfirst(w -> w >= relative_cut*cts_max, h.weights[1:cts_argmax])
    cut_high_arg = findfirst(w -> w <= relative_cut*cts_max, h.weights[cts_argmax:end]) + cts_argmax - 1
    cut_low, cut_high, cut_max = Array(h.edges[1])[cut_low_arg] * x_unit, Array(h.edges[1])[cut_high_arg] * x_unit, Array(h.edges[1])[cts_argmax] * x_unit
    @debug "Cut window: [$cut_low, $cut_high]"
    return (low = cut_low, high = cut_high, max = cut_max)
end
export cut_single_peak

