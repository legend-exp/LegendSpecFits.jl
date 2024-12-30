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
    min_pe_peak::Int=2, max_pe_peak::Int=5, relative_cut_noise_cut::Real=0.5, n_fwhm_noise_cut::Real=5.0,
    initial_min_amp::Real=1.0, initial_max_amp::Real=25.0, initial_max_bin_width_quantile::Real=0.9, 
    peakfinder_σ::Real=2.0, peakfinder_threshold::Real=10.0, peakfinder_rtol::Real=0.1, peakfinder_α::Real=0.05
)
    
    # Start with a big window where the noise peak is included
    bin_width = get_friedman_diaconis_bin_width(filter(in(initial_min_amp..quantile(pe_uncal, initial_max_bin_width_quantile)), pe_uncal))

    # Initial peak search
    h_uncal = fit(Histogram, pe_uncal, initial_min_amp:bin_width:initial_max_amp)

    cuts_1pe = cut_single_peak(pe_uncal, initial_min_amp, initial_max_amp, relative_cut=relative_cut_noise_cut)

    h_uncal_cut = fit(Histogram, pe_uncal, cuts_1pe.max+n_fwhm_noise_cut*(cuts_1pe.high - cuts_1pe.max):bin_width:initial_max_amp)

    c, h_deconv, peakpos, threshold = RadiationSpectra.determine_calibration_constant_through_peak_ratios(h_uncal_cut, collect(range(min_pe_peak, max_pe_peak, step=1)),
        min_n_peaks = 2, max_n_peaks = 2*max_pe_peak, threshold=peakfinder_threshold, rtol=peakfinder_rtol, α=peakfinder_α, σ=peakfinder_σ)


    # simple calibration
    sort!(peakpos)
    gain = peakpos[min_pe_peak + 1] - peakpos[min_pe_peak]
    c = 1/gain
    offset = - (peakpos[min_pe_peak] * c - min_pe_peak)

    f_simple_calib = x -> x .* c .+ offset
    f_simple_uncal = x -> (x .- offset) ./ c

    pe_simple_cal = f_simple_calib.(pe_uncal)
    peakpos_cal = f_simple_calib.(peakpos)

    bin_width_cal = get_friedman_diaconis_bin_width(filter(in(0.5..1.5), pe_simple_cal))
    bin_width_uncal = f_simple_uncal(bin_width_cal) - f_simple_uncal(0.0)

    h_calsimple = fit(Histogram, pe_simple_cal, 0.0:bin_width_cal:6.0)
    h_uncal = fit(Histogram, pe_uncal, 0.0:bin_width_uncal:(6.0 - offset) / c)

    result = (
        pe_simple_cal = pe_simple_cal,
        peakpos = peakpos,
        f_simple_calib = f_simple_calib,
        f_simple_uncal = f_simple_uncal,
        c = c,
        offset = offset
    )
    report = (
        peakpos = peakpos,
        peakpos_cal = peakpos_cal,
        h_uncal = h_uncal, 
        h_calsimple = h_calsimple
    )
    return result, report
end


function find_peaks(
    amps::Vector{<:Real}; 
    min_pe_peak::Real=1.0, max_pe_peak::Real=5.0, n_fwhm_noise_cut::Real=5.0,
    initial_min_amp::Real=1.0, initial_max_amp::Real=100.0, initial_max_bin_width_quantile::Real=0.9, 
    peakfinder_σ::Real=2.0, peakfinder_threshold::Real=10.0, peakfinder_rtol::Real=0.1, peakfinder_α::Real=0.05
)
    # Start with a big window where the noise peak is included
    bin_width = get_friedman_diaconis_bin_width(filter(in(initial_min_amp..quantile(amps, initial_max_bin_width_quantile)), amps))

    # Initial peak search
    h_uncal = fit(Histogram, amps, initial_min_amp:bin_width:initial_max_amp)

    cuts_1pe = cut_single_peak(amps, initial_min_amp, initial_max_amp, relative_cut=0.5)

    h_uncal_cut = fit(Histogram, trig_max, cuts_1pe.max+n_fwhm_noise_cut*(cuts_1pe.high - cuts_1pe.max):bin_width:initial_max_amp)

    c, h_deconv, peak_positions, threshold = RadiationSpectra.determine_calibration_constant_through_peak_ratios(h_uncal_cut, collect(range(min_pe_peak, max_pe_peak, step=1)),
        min_n_peaks = 2, max_n_peaks = 2*max_pe_peak, threshold=peakfinder_threshold, rtol=peakfinder_rtol, α=peakfinder_α, σ=peakfinder_σ)

    return h_decon, peakpos
end