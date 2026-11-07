# # A/E Tutorial

# The ratio of the maximum current amplitude A to the energy E of a pulse separates single-site events (SSE)
# from multi-site events (MSE) in a germanium detector. Background events near the Q-value of the neutrinoless
# double-beta decay are mostly multi-site, while the decay itself is single-site.
# This tutorial derives an energy-independent A/E classifier from a calibration spectrum and measures the
# fraction of events that survive a cut on it.

# The workflow follows the A/E calibration of the Juleana.jl production and uses a synthetic
# Th-228 calibration spectrum, so that the example runs without any detector data.
# Also have a look at the manual page [A/E Pulse-Shape Discrimination](@ref).

# The example loads the packages and sets the plotting backend first. Plots are produced by `lplot` from
# LegendMakie, with the LEGEND watermarks switched off.

using CairoMakie
using Distributions
using LegendMakie
using LegendSpecFits
using Measurements
using Measurements: value as mvalue
using Random
using StatsBase
using Unitful
using Unitful: keV

CairoMakie.activate!(type = "png")

# ## Generating Synthetic Events

# The synthetic data set contains Compton continuum events and the Th-228 lines at 583.2, 727.3, 860.6,
# 1592.5 (double escape peak, DEP), 1620.5, 2103.5 (single escape peak, SEP) and 2614.5 keV
# (full energy peak, FEP). The energy resolution is Gaussian with a standard deviation of
# 0.75 keV + 0.00022 E.

# Every event has an A/E value. Single-site events follow a Gaussian with an energy-dependent mean
# $\mu(E) = \mu_A + \mu_B E$ and standard deviation $\sigma(E) = \sqrt{\sigma_A^2 + \sigma_B^2 / E^2}$.
# Multi-site events are shifted to lower A/E by an exponentially distributed amount, in units of $\sigma(E)$.
# The multi-site fraction is small in the DEP, whose events are mostly single-site, and large in the SEP.
# The raw A/E scale is arbitrary and is multiplied by a constant, which the first step of the analysis removes.

Random.seed!(2024)

lines = (
    ## energy (keV), number of events, multi-site fraction
    (583.2, 60_000, 0.3), (727.3, 20_000, 0.3), (860.6, 8_000, 0.3),
    (1592.5, 25_000, 0.12), (1620.5, 6_000, 0.4), (2103.5, 30_000, 0.75),
    (2614.5, 120_000, 0.35),
)
n_continuum = 1_200_000
resolution(E) = 0.75 + 0.00022 * E

e_lines = reduce(vcat, [E .+ resolution(E) .* randn(n) for (E, n, _) in lines])
e_continuum = 200 .+ rand(truncated(Exponential(800), 0, 2400), n_continuum)
e_keV = vcat(e_lines, e_continuum)
mse_fraction = vcat(reduce(vcat, [fill(f, n) for (_, n, f) in lines]), fill(0.25, n_continuum))
nothing #hide

# The A/E parameters of the single-site population are the values that the analysis recovers in the following sections.

μA, μB, σA, σB = 1.0, -4.0e-6, 5.0e-3, 12.0
sse_mean(E) = μA + μB * E
sse_sigma(E) = sqrt(σA^2 + σB^2 / E^2)

σ_sse = sse_sigma.(e_keV)
aoe_scale = 55.0
aoe_raw = aoe_scale .* (sse_mean.(e_keV) .+ σ_sse .* randn(length(e_keV)) .-
    (rand(length(e_keV)) .< mse_fraction) .* σ_sse .* rand(Exponential(6), length(e_keV)))
e_cal = e_keV .* u"keV"
nothing #hide

# ## Normalizing the Raw A/E

# The raw A/E has an arbitrary scale. The function [`cut_single_peak`](@ref) locates the single-site peak in
# the distribution of all events, and dividing by the position of its maximum sets the peak to about one.
# The division is stored as a string expression, which the later fits extend to the full correction function.

cuts_aoe = cut_single_peak(aoe_raw, 0.0, quantile(filter(isfinite, aoe_raw), 0.99); n_bins = -1)
aoe_expression = "aoe / $(cuts_aoe.max)"
aoe = aoe_raw ./ cuts_aoe.max
nothing #hide

# The two-dimensional histogram of A/E against energy shows the single-site band, whose position and width
# depend on the energy, and the multi-site events below it.

h_aoe_raw = fit(Histogram, (e_keV, aoe), (0:5.0:3000, 0.5:0.005:1.3))
lhist(h_aoe_raw, figsize = (670, 400), watermark = false, rasterize = true,
    title = "Normalized A/E", xlabel = "Energy (keV)", ylabel = "A/E (a.u.)")

# ## Fitting Compton Bands

# The energy dependence of the single-site band is measured in narrow energy slices of the Compton continuum, called
# Compton bands. The bands avoid the peaks, so that their events are mostly Compton scattered.
# The function [`generate_aoe_compton_bands`](@ref) cuts the A/E values of every band into a histogram
# and estimates the position and width of the single-site peak.
# The band-wise fit needs at least 20 bands.

compton_bands = [500, 540, 620, 660, 760, 800, 900, 940, 1000, 1040, 1100, 1140, 1200, 1240,
    1300, 1340, 1400, 1440, 1500, 1540, 1660, 1700, 1760, 1800, 1860, 1900, 1960, 2000,
    2060, 2140, 2200, 2240, 2300, 2340] .* u"keV"
compton_window = 20.0u"keV"

compton_band_peakhists = generate_aoe_compton_bands(aoe, e_cal, compton_bands, compton_window)
nothing #hide

# The histogram of the first band shows the single-site peak and the tail of multi-site events toward low A/E.

lhist(compton_band_peakhists.peakhists[1], watermark = false,
    title = "A/E Compton band at $(compton_bands[1])", xlabel = "A/E (a.u.)", ylabel = "Counts")

# The function [`fit_aoe_compton`](@ref) fits every band with a Gaussian for the single-site peak and an exponentially
# modified Gaussian for the multi-site background. It returns the results and reports as dictionaries keyed by
# the band energy.

result_fit, report_fit = fit_aoe_compton(compton_band_peakhists.peakhists,
    compton_band_peakhists.peakstats, compton_bands; uncertainty = true)
result_fit[compton_bands[1]]

# The report of a band contains the data and the best fit, which `lplot` draws with the fit residuals.

lplot(report_fit[compton_bands[1]], watermark = false)

# Bands whose fit did not converge are removed before the energy dependence is fitted. The accepted bands provide
# the position `μ` and width `σ` of the single-site peak with uncertainties.

good_bands = findall(b -> haskey(result_fit, b) && result_fit[b].gof.converged, compton_bands)
compton_bands = compton_bands[good_bands]
μ = [result_fit[b].μ for b in compton_bands]
σ = [result_fit[b].σ for b in compton_bands]
length(good_bands)

# ## Fitting the Energy Dependence

# The function [`fit_aoe_corrections`](@ref) fits a straight line to the band positions and
# $\sqrt{\sigma_A^2 + \sigma_B^2 / E^2}$ to the band widths. The result also contains the string `func`, which
# expresses the normalized classifier $(A/E - \mu(E)) / \sigma(E)$ in terms of the variables `aoe` and `e`.

result_fit_single, report_fit_single = fit_aoe_corrections(compton_bands, μ, σ;
    aoe_expression = aoe_expression, e_expression = "e")
result_fit_single.µ_compton.par

# The fitted position of the single-site peak decreases linearly with the energy.

lplot(report_fit_single.report_μ, watermark = false)

# The fitted width decreases with the energy.

lplot(report_fit_single.report_σ, watermark = false)

# The function [`fit_aoe_compton_combined`](@ref) fits all bands at once with the shared position and width
# functions and starts from the band-wise result. This reduces the statistical uncertainty of the parameters.

result_fit_combined, report_fit_combined = fit_aoe_compton_combined(
    compton_band_peakhists.peakhists[good_bands], compton_band_peakhists.peakstats[good_bands],
    compton_bands, result_fit_single; aoe_expression = aoe_expression, e_expression = "e", uncertainty = true)
result_fit_combined

# The single and the combined fit agree with each other. Compared to the synthetic parameters, `μA` and `σA` carry the
# factor of the normalization step (about 1.009), while `σB` and the ratio `μB / μA` equal the synthetic values within their uncertainties.

lplot(report_fit_single.report_μ, report_fit_combined.report_µ, watermark = false,
    legend_position = :lb, title = "A/E μ")

# ## Applying the Correction

# The string `func` of the combined fit is turned into a function of `aoe` and `e` and evaluated for every event.
# In the production dataflow, `ljl_propfunc` from LegendDataManagement evaluates such strings on table rows.
# The string contains the unit `keV`, which the example imports from Unitful, and it includes the normalization
# by the peak position, so that it is applied to the raw A/E.
# The corrected A/E is a classifier with the single-site peak at zero and unit width at all energies.

correct_aoe = Core.eval(@__MODULE__, Meta.parse("(aoe, e) -> " * result_fit_combined.func))
aoe_corr = Base.invokelatest.(correct_aoe, aoe_raw, e_cal)
nothing #hide

# After the correction, the single-site band is flat in energy, and multi-site events are found at negative values.

h_aoe_corr = fit(Histogram, (e_keV, aoe_corr), (0:5.0:3000, -25:0.1:10))
lhist(h_aoe_corr, figsize = (670, 400), watermark = false, rasterize = true,
    title = "Corrected A/E", xlabel = "Energy (keV)", ylabel = "A/E (corrected)")

# ## Low A/E Cut from the DEP

# The function [`get_low_aoe_cut`](@ref) searches the lower A/E cut value at which 90% of the events in
# the DEP survive. The DEP consists mostly of single-site events, so that the cut keeps the single-site
# population. The keyword `sigma_high_sided` sets an upper cut in units of the corrected A/E, which is three standard
# deviations of the single-site peak here.

result_cut, report_cut = get_low_aoe_cut(aoe_corr, e_cal; dep = 1592.5u"keV",
    window = [20.0, 20.0]u"keV", bin_width_window = 3.0u"keV", cut_search_interval = (-25.0, 0.0),
    dep_sf = 0.9, rtol = 1e-5, maxiters = 200, sigma_high_sided = 3.0, fit_func = :gamma_def, uncertainty = true)
result_cut.lowcut

# The report shows the energy spectrum before and after the cut and the DEP region in detail.

lplot(report_cut, figsize = (750, 400), watermark = false)

# ## Peak Survival Fractions on Synthetic Data

# The efficiency of the cut for other energies is the fraction of events that survive in the corresponding
# peak. The function [`get_peaks_survival_fractions`](@ref) fits every peak before and after the cut.
# The first call applies only the lower cut.

peaks = [1592.5, 1620.5, 2103.5, 2614.5] .* u"keV"
peak_names = [:DEP, :Bi212FEP, :SEP, :FEP]
left_windows, right_windows = fill(20.0u"keV", 4), fill(20.0u"keV", 4)

result_peaks_low, report_peaks_low = get_peaks_survival_fractions(aoe_corr, e_cal, peaks, peak_names,
    left_windows, right_windows; low_cut = mvalue(result_cut.lowcut), bin_width_window = 3.0u"keV",
    fit_funcs = [:gamma_def, :gamma_def, :gamma_def, :gamma_def], uncertainty = true)
result_peaks_low[:SEP].sf

# The report of every peak shows the spectrum of the surviving and rejected events with the fit.

lplot(report_peaks_low[:SEP], watermark = false)

# The second call also applies the upper cut (double-sided cut).

result_peaks_ds, _ = get_peaks_survival_fractions(aoe_corr, e_cal, peaks, peak_names,
    left_windows, right_windows; low_cut = mvalue(result_cut.lowcut), high_cut = mvalue(result_cut.highcut),
    bin_width_window = 3.0u"keV", fit_funcs = [:gamma_def, :gamma_def, :gamma_def, :gamma_def], uncertainty = true)
result_peaks_ds[:SEP].sf

# The Q-value region has no peak. The function [`get_continuum_survival_fraction`](@ref) counts the events in a window
# around the Q-value of Ge-76 of 2039.06 keV before and after the cut.

result_qbb, report_qbb = get_continuum_survival_fraction(aoe_corr, e_cal, 2039.06u"keV", 35.0u"keV";
    low_cut = mvalue(result_cut.lowcut), high_cut = mvalue(result_cut.highcut))
result_qbb.sf
