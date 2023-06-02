# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    estimate_single_peak_stats(h::Histogram)

Estimate statistics/parameters for a single peak in the given histogram `h`.

`h` must only contain a single peak. The peak should have a Gaussian-like
shape.

Returns a `NamedTuple` with the fields

    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
"""
function estimate_single_peak_stats(h::Histogram)
    W = h.weights
    E = first(h.edges)
    peak_amplitude, peak_idx = findmax(W)
    fwhm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) /2, W)
    fwhm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) /2, W)
    peak_max_pos = (E[peak_idx] + E[peak_idx+1]) / 2
    peak_mid_pos = (E[fwhm_idx_right] + E[fwhm_idx_left]) / 2
    peak_pos = (peak_max_pos + peak_mid_pos) / 2
    peak_fwhm = E[fwhm_idx_right] - E[fwhm_idx_left]
    peak_sigma = peak_fwhm * inv(2*√(2log(2)))
    #peak_area = peak_amplitude * peak_sigma * sqrt(2*π)
    mean_background = (first(W) + last(W)) / 2
    mean_background = ifelse(mean_background == 0, 0.01, mean_background)
    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)

    (
        peak_pos = peak_pos, peak_fwhm = peak_fwhm,
        peak_sigma, peak_counts, mean_background
    )
end
export estimate_single_peak_stats

