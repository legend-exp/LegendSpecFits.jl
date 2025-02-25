"""
    sipm_simple_calibration(pe_uncal::Array)

Perform a simple calibration for the uncalibrated p.e. spectrum array `pe_uncal`
using just the 1 p.e. and 2 p.e. peak positions estimated by a peakfinder.

Inputs:
    * `pe_uncal`: array of uncalibrated peak amplitudes
kwargs:
    * `initial_min_amp`: uncalibrated amplitude value as a left boundary to build the uncalibrated histogram where the peak search is performed on.
                        For the peak search with noise peak, this value is consecutively increased i.o.t exclude the noise peak from the histogram.
    * `initial_max_quantile`: quantile of the uncalibrated amplitude array to used as right boundary to build the uncalibrated histogram
    * `peakfinder_σ`: sigma value in number of bins for peakfinder
    * `peakfinder_threshold`: threshold value for peakfinder

Returns 
    * `pe_simple_cal`: array of the calibrated pe array with the simple calibration
    * `func`: function to use for the calibration (`pe_simple_cal = pe_uncal .* c .+ offset`)
    * `c`: calibration factor
    * `offset`: calibration offset 
    * `peakpos`: 1 p.e. and 2 p.e. peak positions in uncalibrated amplitude
    * `peakpos_cal`: 1 p.e. and 2 p.e. peak positions in calibrated amplitude
    * `h_uncal`: histogram of the uncalibrated pe array
    * `h_calsimple`: histogram of the calibrated pe array with the simple calibration
"""
function sipm_simple_calibration end
export sipm_simple_calibration

function sipm_simple_calibration(pe_uncal::Vector{<:Real};
    min_pe_peak::Int=1, max_pe_peak::Int=5, relative_cut_noise_cut::Real=0.5, n_fwhm_noise_cut::Real=5.0,
    initial_min_amp::Real=0.0, initial_max_amp::Real=50.0, initial_max_bin_width_quantile::Real=0.9, 
    peakfinder_σ::Real=-1.0, peakfinder_threshold::Real=10.0, peakfinder_rtol::Real=0.1, peakfinder_α::Real=0.05
)
    
    # Initial peak search
    cuts_1pe = cut_single_peak(pe_uncal, initial_min_amp, initial_max_amp, relative_cut=relative_cut_noise_cut)
    
    bin_width_cut_min = if n_fwhm_noise_cut == 0.0
        initial_min_amp
    else
        cuts_1pe.max+n_fwhm_noise_cut*(cuts_1pe.high - cuts_1pe.max)
    end
    bin_width_cut = get_friedman_diaconis_bin_width(filter(in(bin_width_cut_min..quantile(pe_uncal, initial_max_bin_width_quantile)), pe_uncal))
    peakpos = []
    for bin_width_scale in exp10.(range(0, stop=-3, length=50))
        bin_width_cut_scaled = bin_width_cut * bin_width_scale
        @debug "Using bin width: $(bin_width_cut_scaled)"
        h_uncal_cut = fit(Histogram, pe_uncal, bin_width_cut_min:bin_width_cut_scaled:initial_max_amp)
        peakfinder_σ_scaled = if peakfinder_σ <= 0.0
            peakfinder_σ_scaled = round(Int, 2*(cuts_1pe.high - cuts_1pe.max) / bin_width_cut_scaled / 2 * sqrt(2 * log(2)))
        else
            peakfinder_σ
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
            peakfinder_σ_scaled = round(Int, 2*(cuts_1pe.high - cuts_1pe.max) / bin_width_cut_scaled / 2 * sqrt(2 * log(2)))
        else
            peakfinder_σ
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

    result = (
        pe_simple_cal = pe_simple_cal,
        peakpos = peakpos,
        f_simple_calib = f_simple_calib,
        f_simple_uncal = f_simple_uncal,
        c = c,
        offset = offset,
        noisepeakpos = cuts_1pe.max,
        noisepeakwidth = cuts_1pe.high - cuts_1pe.low
    )
    report = (
        peakpos = peakpos,
        peakpos_cal = peakpos_cal,
        h_uncal = h_uncal, 
        h_calsimple = h_calsimple
    )
    return result, report
end