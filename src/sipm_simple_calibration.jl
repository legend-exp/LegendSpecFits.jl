"""
    sipm_simple_calibration(pe_uncal::AbstractVector{<:Real})
    sipm_simple_calibration(pe_uncal_vov::VectorOfVectors{<:Real})

Simple SiPM calibration from the 1 PE and 2 PE peak positions found by a
peakfinder. The noise/1PE cut is auto-detected at the centroid of the first
valley after the noise peak.

`n_fwhm_noise_cut` caps the valley search window in units of the noise-peak
half-width above its center. `0.0` skips valley detection and uses
`initial_min_amp`. Negative values force the legacy formula
`cuts_1pe.max + n_fwhm_noise_cut * fwhm_noise` (kept for backward-compatibility
with configs that dip into the noise peak).

For VoV input, an optional second-stage QC (`single_trigger_only=true`) keeps
only waveforms with exactly one trigger above the detected threshold.

kwargs:
    * `initial_min_amp`, `initial_max_amp`: histogram bounds for the noise/peak search
    * `relative_cut_noise_cut`: passed to `cut_single_peak`
    * `n_fwhm_noise_cut`: see above
    * `single_trigger_only` (VoV only)
    * `min_pe_peak`, `max_pe_peak`, `peakfinder_*`: peakfinder controls

Returns `(result, report)`. `result` carries the calibration (`f_simple_calib`,
`c`, `offset`, `peakpos`, `noisepeakpos`, `noisepeakwidth`, `noise_threshold`,
`noise_threshold_cal`). `report` carries plotting data (`peakpos*`, `h_uncal`,
`h_calsimple`, `h_*_full` for the unfiltered all-trigger spectra,
`noise_threshold*`, `valley_found`).
"""
function sipm_simple_calibration end
export sipm_simple_calibration

# Detects the noise/1PE valley by walking a smoothed 40-bin histogram,
# tracking the running minimum, and exiting once the curve clearly rises
# again. The cut is placed at the centroid of the plateau at the minimum, so
# wide empty valleys land mid-region rather than at the leftmost zero bin.
function _find_noise_threshold(pe_data, cuts_1pe, n_fwhm_noise_cut, initial_min_amp, initial_max_amp)
    n_fwhm_noise_cut == 0.0 && return initial_min_amp, false
    n_fwhm_noise_cut < 0.0 && return cuts_1pe.max + n_fwhm_noise_cut * (cuts_1pe.high - cuts_1pe.max), false

    fwhm_noise = max(cuts_1pe.high - cuts_1pe.max, 0.1)
    search_end = min(cuts_1pe.max + n_fwhm_noise_cut * fwhm_noise, initial_max_amp)
    search_end ≤ cuts_1pe.high && return cuts_1pe.high, false

    edges = range(cuts_1pe.high, search_end; length=41)
    h = fit(Histogram, filter(x -> cuts_1pe.high ≤ x ≤ search_end, pe_data), edges)
    length(h.weights) < 7 && return search_end, false
    w_s = savitzky_golay(h.weights, 5, 2).y

    rise_factor = 1.5
    descent_threshold = 0.5 * maximum(@view w_s[1:min(3, length(w_s))])
    running_min, running_min_idx, exit_idx = w_s[1], 1, 0
    for i in 2:length(w_s)
        if w_s[i] < running_min
            running_min, running_min_idx = w_s[i], i
        # `+ 2.0` floor handles near-empty valleys where `rise_factor * 0 == 0`.
        elseif w_s[i] > max(rise_factor * running_min, running_min + 2.0) && running_min ≤ descent_threshold
            exit_idx = i
            break
        end
    end
    exit_idx == 0 && return search_end, false

    # Centroid the cut over the bins from where `running_min` was first reached
    # up to the rise — i.e. the actual flat region. Bins still on the descent
    # before reaching `running_min` are excluded so the cut isn't biased left.
    plateau_floor = max(rise_factor * running_min, running_min + 1.0)
    plateau_idxs = findall(j -> w_s[j] ≤ plateau_floor, running_min_idx:exit_idx-1)
    # Plateau too short → no real valley (e.g. noise tail blends into the 1PE
    # rising flank). Fall back to N×FWHM above the noise peak.
    length(plateau_idxs) < 4 && return cuts_1pe.max + 3.0 * fwhm_noise, false
    idx = running_min_idx - 1 + (first(plateau_idxs) + last(plateau_idxs)) ÷ 2
    return edges[idx + 1], true
end

function sipm_simple_calibration(pe_uncal_vov::VectorOfVectors{<:Real};
    initial_min_amp::Real=0.0, initial_max_amp::Real=50.0,
    relative_cut_noise_cut::Real=0.5, n_fwhm_noise_cut::Real=5.0,
    single_trigger_only::Bool=true, cut_pool_max_mult::Int=3, kwargs...
)
    # Cut-detection pool: low-multiplicity waveforms (≤ `cut_pool_max_mult`
    # triggers). Excludes heavily contaminated multi-trig waveforms while
    # keeping enough statistics for the noise peak and valley to be visible.
    cut_pool = if single_trigger_only
        [t for trigs in pe_uncal_vov if length(trigs) ≤ cut_pool_max_mult
              for t in trigs if isfinite(t)]
    else
        # flatview is the flat backing vector of the VoV - linear, no pairwise vcat copies
        filter(isfinite, flatview(pe_uncal_vov))
    end
    cuts_1pe = cut_single_peak(cut_pool, initial_min_amp, initial_max_amp, relative_cut=relative_cut_noise_cut)
    noise_threshold, valley_found = _find_noise_threshold(cut_pool, cuts_1pe, n_fwhm_noise_cut, initial_min_amp, initial_max_amp)

    # Calibration pool: triggers from waveforms with exactly one trigger above
    # the cut. Looser than 1-trig-total → more statistics for the peakfinder.
    pe_uncal = if single_trigger_only
        [t for trigs in pe_uncal_vov
              if count(t -> isfinite(t) && t > noise_threshold, trigs) == 1
              for t in trigs if isfinite(t) && t > noise_threshold]
    else
        filter(x -> x > noise_threshold, cut_pool)
    end

    # Threshold already applied → skip the Vector method's own valley detection.
    result, report = sipm_simple_calibration(pe_uncal;
        initial_min_amp=noise_threshold, initial_max_amp=initial_max_amp,
        relative_cut_noise_cut=relative_cut_noise_cut, n_fwhm_noise_cut=0.0,
        kwargs...)

    noise_threshold_cal = result.f_simple_calib(noise_threshold)
    h_uncal_full = fit(Histogram, cut_pool, first(report.h_uncal.edges))
    h_calsimple_full = fit(Histogram, result.f_simple_calib.(cut_pool), first(report.h_calsimple.edges))
    result = merge(result, (; noise_threshold, noise_threshold_cal))
    report = merge(report, (; noise_threshold, noise_threshold_cal, valley_found,
                              h_uncal_full, h_calsimple_full))
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

    noise_threshold = bin_width_cut_min
    noise_threshold_cal = f_simple_calib(noise_threshold)
    noisepeakpos, noisepeakwidth = cuts_1pe.max, cuts_1pe.high - cuts_1pe.low
    result = (; pe_simple_cal, peakpos, f_simple_calib, f_simple_uncal, c, offset,
                noisepeakpos, noisepeakwidth, noise_threshold, noise_threshold_cal)
    # The VoV method overwrites `h_*_full` with the unfiltered all-trigger spectra.
    report = (; peakpos, peakpos_cal, h_uncal, h_calsimple,
                h_uncal_full = h_uncal, h_calsimple_full = h_calsimple,
                noise_threshold, noise_threshold_cal, valley_found)
    return result, report
end
