"""
    sipm_simple_calibration(pe_uncal::AbstractVector{<:Real})
    sipm_simple_calibration(pe_uncal_vov::VectorOfVectors{<:Real})

Perform a simple calibration for the uncalibrated p.e. spectrum array `pe_uncal`
using the 1 p.e. and 2 p.e. peak positions estimated by a peakfinder.

The noise-1PE threshold is auto-detected as the first local minimum (valley) of
the amplitude histogram between the noise peak and the upper search bound.
`n_fwhm_noise_cut` caps the search window in units of the noise-peak
half-width above its center; if no valley is found within this window, the
window cap itself is returned (legacy FWHM formula). A value of `0.0` skips
valley detection and uses `initial_min_amp` directly. Negative values force the
legacy FWHM formula `cuts_1pe.max + n_fwhm_noise_cut * fwhm_noise` (kept for
backward-compatibility with configs that intentionally dip into the noise peak).

When called with a `VectorOfVectors{<:Real}` (per-waveform trigger amplitudes),
an optional second-stage QC keeps only waveforms with exactly one trigger above
the detected noise threshold (`single_trigger_only=true`, default). Set to
`false` to use all triggers above threshold without per-waveform multiplicity
filtering.

Inputs:
    * `pe_uncal`: vector of uncalibrated peak amplitudes, or VoV per-waveform
kwargs:
    * `initial_min_amp`, `initial_max_amp`: initial histogram bounds for the noise/peak search
    * `relative_cut_noise_cut`: passed to `cut_single_peak` for the noise peak
    * `n_fwhm_noise_cut`: see above
    * `single_trigger_only` (VoV only): if `true`, keep only single-trigger-above-threshold waveforms
    * `min_pe_peak`, `max_pe_peak`, `peakfinder_*`: peakfinder controls

Returns
    * `result`: NamedTuple with `pe_simple_cal`, `peakpos`, `f_simple_calib`,
      `f_simple_uncal`, `c`, `offset`, `noisepeakpos`, `noisepeakwidth`,
      `noise_threshold`, `noise_threshold_cal`
    * `report`: NamedTuple with `peakpos`, `peakpos_cal`, `h_uncal`,
      `h_calsimple`, `noise_threshold`, `noise_threshold_cal`, `valley_found`
"""
function sipm_simple_calibration end
export sipm_simple_calibration

# Auto-detect the *first* valley between the noise peak and the 1 PE peak.
# Returns `(threshold, valley_found::Bool)`. Walks the smoothed histogram from
# left to right and exits at the first local minimum that lies sufficiently
# below the start of the search window — so multi-channel SiPMs with split 1PE
# sub-peaks return the noise/sub-peak valley, not the deeper 1PE/2PE valley.
function _find_noise_threshold(pe_data, cuts_1pe, n_fwhm_noise_cut, initial_min_amp, initial_max_amp)
    if n_fwhm_noise_cut == 0.0
        return initial_min_amp, false
    elseif n_fwhm_noise_cut < 0.0
        # Legacy: explicit FWHM formula, no valley search.
        return cuts_1pe.max + n_fwhm_noise_cut * (cuts_1pe.high - cuts_1pe.max), false
    end
    fwhm_noise = max(cuts_1pe.high - cuts_1pe.max, 0.1)
    search_end = min(cuts_1pe.max + n_fwhm_noise_cut * fwhm_noise, initial_max_amp)
    if search_end ≤ cuts_1pe.high
        return cuts_1pe.high, false
    end
    # ~40 bins across the search window — decoupled from the noise-peak width.
    nbins = 40
    edges = range(cuts_1pe.high, search_end; length=nbins + 1)
    h = fit(Histogram, filter(x -> cuts_1pe.high ≤ x ≤ search_end, pe_data), edges)
    w = h.weights
    sg_window = 5  # odd, ≥ degree + 1 for quadratic Savitzky–Golay
    if length(w) < sg_window + 2
        return search_end, false
    end
    # Smooth to suppress Poisson fluctuations on the falling noise tail.
    w_s = savitzky_golay(w, sg_window, 2).y
    # Depth gate: only accept a valley once we've descended to at most 50% of
    # the start of the search window. The reference is the max of the first
    # few bins to avoid being thrown by a single low first bin.
    descent_threshold = 0.5 * maximum(@view w_s[1:min(3, length(w_s))])
    for i in 2:length(w_s)-1
        if w_s[i] ≤ descent_threshold && w_s[i] ≤ w_s[i-1] && w_s[i] < w_s[i+1]
            return edges[i+1], true
        end
    end
    return search_end, false
end

function sipm_simple_calibration(pe_uncal_vov::VectorOfVectors{<:Real};
    initial_min_amp::Real=0.0, initial_max_amp::Real=50.0,
    relative_cut_noise_cut::Real=0.5, n_fwhm_noise_cut::Real=5.0,
    single_trigger_only::Bool=true, kwargs...
)
    # Detect the noise threshold globally on the flat trigger distribution.
    pe_flat = filter(isfinite, reduce(vcat, pe_uncal_vov))
    cuts_1pe = cut_single_peak(pe_flat, initial_min_amp, initial_max_amp, relative_cut=relative_cut_noise_cut)
    noise_threshold, valley_found = _find_noise_threshold(pe_flat, cuts_1pe, n_fwhm_noise_cut, initial_min_amp, initial_max_amp)

    # Optional second-stage QC: keep only above-threshold triggers from waveforms
    # with exactly one trigger above the noise threshold.
    pe_uncal = if single_trigger_only
        [t for trigs in pe_uncal_vov
              if count(t -> isfinite(t) && t > noise_threshold, trigs) == 1
              for t in trigs if isfinite(t) && t > noise_threshold]
    else
        filter(x -> x > noise_threshold, pe_flat)
    end

    # Threshold already applied → tell the Vector method to skip its own valley detection.
    result, report = sipm_simple_calibration(pe_uncal;
        initial_min_amp=noise_threshold, initial_max_amp=initial_max_amp,
        relative_cut_noise_cut=relative_cut_noise_cut, n_fwhm_noise_cut=0.0,
        kwargs...)

    # Override the threshold/valley fields with the values detected at the VoV level.
    noise_threshold_cal = result.f_simple_calib(noise_threshold)
    result = merge(result, (noise_threshold = noise_threshold,
                            noise_threshold_cal = noise_threshold_cal))
    report = merge(report, (noise_threshold = noise_threshold,
                            noise_threshold_cal = noise_threshold_cal,
                            valley_found = valley_found))
    return result, report
end

function sipm_simple_calibration(pe_uncal::Vector{<:Real};
    min_pe_peak::Int=1, max_pe_peak::Int=5, relative_cut_noise_cut::Real=0.5, n_fwhm_noise_cut::Real=5.0,
    initial_min_amp::Real=0.0, initial_max_amp::Real=50.0, initial_max_bin_width_quantile::Real=0.9,
    peakfinder_σ::Real=-1.0, peakfinder_threshold::Real=10.0, peakfinder_rtol::Real=0.1, peakfinder_α::Real=0.05
)
    # Initial peak search
    cuts_1pe = cut_single_peak(pe_uncal, initial_min_amp, initial_max_amp, relative_cut=relative_cut_noise_cut)

    bin_width_cut_min, valley_found = _find_noise_threshold(pe_uncal, cuts_1pe, n_fwhm_noise_cut, initial_min_amp, initial_max_amp)
    bin_width_cut = get_friedman_diaconis_bin_width(filter(in(bin_width_cut_min..quantile(pe_uncal, initial_max_bin_width_quantile)), pe_uncal))
    peakpos = []
    for bin_width_scale in exp10.(range(0, stop=-3, length=10))
        bin_width_cut_scaled = bin_width_cut * bin_width_scale
        @debug "Using bin width: $(bin_width_cut_scaled)"
        h_uncal_cut = fit(Histogram, pe_uncal, bin_width_cut_min:bin_width_cut_scaled:initial_max_amp)
        peakfinder_σ_scaled = if peakfinder_σ <= 0.0
            round(Int, 2*(cuts_1pe.high - cuts_1pe.max) / bin_width_cut_scaled / (2 * sqrt(2 * log(2))) )
        else
            isinteger(peakfinder_σ) || throw(ArgumentError("Expected `peakfinder_σ` to be an integer, but received $peakfinder_σ."))
            round(Int, peakfinder_σ)
        end
        @debug "Peakfinder σ: $(peakfinder_σ_scaled)"
        try
            # use SavitzkyGolay filter to smooth the histogram
            sg_uncal_cut = savitzky_golay(h_uncal_cut.weights, ifelse(isodd(peakfinder_σ_scaled), peakfinder_σ_scaled, peakfinder_σ_scaled + 1), 3)
            h_uncal_cut_sg = Histogram(h_uncal_cut.edges[1], sg_uncal_cut.y)
            _, _, peakpos, _ = RadiationSpectra.determine_calibration_constant_through_peak_ratios(h_uncal_cut_sg, collect(range(min_pe_peak, max_pe_peak, step=1)),
                min_n_peaks = 2, max_n_peaks = max_pe_peak, threshold=peakfinder_threshold, rtol=peakfinder_rtol, α=peakfinder_α, σ=peakfinder_σ_scaled)
        catch e
            @warn "Failed to find peaks with bin width scale $(bin_width_scale): $(e)"
            continue
        else
            @debug "Found peaks with bin width scale $(bin_width_scale)"
            if !isempty(peakpos) && length(peakpos) >= 2
                break
            end
        end
    end

    if length(peakpos) < 2
        @warn "Failed to find peaks with peakfinder method, use alternative"
        bin_width_cut_scaled = bin_width_cut * 0.5
        @debug "Using bin width: $(bin_width_cut_scaled)"

        h_uncal_cut = fit(Histogram, pe_uncal, bin_width_cut_min:bin_width_cut_scaled:initial_max_amp)
        peakfinder_σ_scaled = if peakfinder_σ <= 0.0
            round(Int, 2*(cuts_1pe.high - cuts_1pe.max) / bin_width_cut_scaled / (2 * sqrt(2 * log(2))) )
        else
            isinteger(peakfinder_σ) || throw(ArgumentError("Expected `peakfinder_σ` to be an integer, but received $peakfinder_σ."))
            round(Int, peakfinder_σ)
        end

        # use SavitzkyGolay filter to smooth the histogram
        sg_uncal_cut = savitzky_golay(h_uncal_cut.weights, ifelse(isodd(peakfinder_σ_scaled), peakfinder_σ_scaled, peakfinder_σ_scaled + 1), 3)
        edges = StatsBase.midpoints(first(h_uncal_cut.edges))  # use midpoints as x values
        counts_sg = sg_uncal_cut.y
        # get local maxima
        min_i_prominence = round(Int, peakfinder_σ_scaled / 2)
        is_local_maximum(i, y) = i > min_i_prominence && i < length(y) - min_i_prominence &&
                all(y[i] .> y[i-min_i_prominence:i-1]) && all(y[i] .> y[i+1:i+min_i_prominence])
        peakpos = edges[findall(is_local_maximum.(eachindex(counts_sg), Ref(counts_sg)))]
    end

    if length(peakpos) < 2
        throw(ErrorException("Failed to find peaks"))
    end

    # simple calibration
    sort!(peakpos)

    # Merge close peaks for SiPM arrays with split PE peaks (e.g. multi-channel arrays):
    # if adjacent peaks are closer than 60% of the estimated gain, fold them into one.
    if length(peakpos) >= 3
        diffs = diff(peakpos)
        gain_est = maximum(diffs)
        merged = [peakpos[1]]
        for i in 2:length(peakpos)
            if peakpos[i] - merged[end] < 0.6 * gain_est
                merged[end] = (merged[end] + peakpos[i]) / 2
            else
                push!(merged, peakpos[i])
            end
        end
        if length(merged) >= 2
            peakpos = merged
        end
    end

    @debug "Found $(min_pe_peak) PE Peak positions: $(peakpos[1])"
    @debug "Found $(min_pe_peak+1) PE Peak positions: $(peakpos[2])"
    gain = peakpos[2] - peakpos[1]
    @debug "Calculated gain: $(round(gain, digits=2))"
    c = 1/gain
    offset = - (peakpos[1] * c - min_pe_peak)
    @debug "Calculated offset: $(round(offset, digits=2))"

    f_simple_calib = x -> x .* c .+ offset
    f_simple_uncal = x -> (x .- offset) ./ c

    pe_simple_cal = f_simple_calib.(pe_uncal)
    peakpos_cal = f_simple_calib.(peakpos)

    bin_width_cal = get_friedman_diaconis_bin_width(filter(in(0.5..min_pe_peak), pe_simple_cal))
    bin_width_uncal = f_simple_uncal(bin_width_cal) - f_simple_uncal(0.0)

    h_calsimple = fit(Histogram, pe_simple_cal, 0.0:bin_width_cal:max_pe_peak + 1)
    h_uncal = fit(Histogram, pe_uncal, 0.0:bin_width_uncal:f_simple_uncal(max_pe_peak + 1))

    noise_threshold_cal = f_simple_calib(bin_width_cut_min)
    result = (
        pe_simple_cal = pe_simple_cal,
        peakpos = peakpos,
        f_simple_calib = f_simple_calib,
        f_simple_uncal = f_simple_uncal,
        c = c,
        offset = offset,
        noisepeakpos = cuts_1pe.max,
        noisepeakwidth = cuts_1pe.high - cuts_1pe.low,
        noise_threshold = bin_width_cut_min,
        noise_threshold_cal = noise_threshold_cal,
    )
    report = (
        peakpos = peakpos,
        peakpos_cal = peakpos_cal,
        h_uncal = h_uncal,
        h_calsimple = h_calsimple,
        noise_threshold = bin_width_cut_min,
        noise_threshold_cal = noise_threshold_cal,
        valley_found = valley_found,
    )
    return result, report
end
