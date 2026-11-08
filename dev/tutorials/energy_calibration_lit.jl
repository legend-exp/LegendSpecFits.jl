# # Energy Calibration Tutorial
#
# This tutorial calibrates the energy scale of a germanium detector from a simulated Th-228 calibration spectrum.
# It follows the steps of the energy calibration in the LEGEND data production:
# a coarse calibration constant is found from the most prominent peak, the charge-trapping correction is fitted,
# every calibration line is fitted, and the fitted line positions and widths are used to fit a calibration curve and a resolution curve.
#
# The data are generated in the tutorial and mimic the output of a pulse-shape analysis:
# an uncalibrated energy in arbitrary units of the analog-to-digital converter (ADC) and a drift-time-like parameter `qdrift`.
# The results are therefore not tied to any real detector.

using LegendSpecFits
using CairoMakie, LegendMakie
using Distributions
using Measurements: value as mvalue, uncertainty as muncert
using Random
using Unitful
using Unitful: keV  # the function strings of the results refer to `keV`

CairoMakie.activate!(type = "png")

# ## Synthetic Data
#
# The spectrum of each calibration line is sampled from the peak shape model that is also used to fit the lines
# (see [Peak Shapes and Likelihoods](@ref)): a Gaussian with a low-energy tail on top of a flat background and a Compton step.
# The sampling inverts the cumulative distribution function of the model on a fine energy grid.
# The tutorial uses a fixed seed, so the numbers shown below are reproducible.

Random.seed!(1234)

f_peak = LegendSpecFits.get_th228_fit_functions().gamma_def

## Peak shape parameters of every line; σ² grows linearly with the energy
th228_lines = [583.191, 727.330, 860.564, 1592.513, 1620.50, 2103.512, 2614.511]u"keV"
shape_pars = [
    (σ = 0.73, n = 1865.5, step_amplitude = 3.0,  skew_fraction = 0.02, skew_width = 0.003, background = 3),
    (σ = 0.78, n = 452.9,  step_amplitude = 1.0,  skew_fraction = 0.01, skew_width = 0.003, background = 3),
    (σ = 0.83, n = 355.8,  step_amplitude = 0.5,  skew_fraction = 0.01, skew_width = 0.003, background = 3),
    (σ = 1.02, n = 206.8,  step_amplitude = 0.1,  skew_fraction = 0.01, skew_width = 0.003, background = 2),
    (σ = 1.04, n = 130.3,  step_amplitude = 0.1,  skew_fraction = 0.01, skew_width = 0.003, background = 2),
    (σ = 1.14, n = 385.1,  step_amplitude = 0.1,  skew_fraction = 0.01, skew_width = 0.003, background = 2),
    (σ = 1.25, n = 3130.4, step_amplitude = 0.1,  skew_fraction = 0.02, skew_width = 0.003, background = 1),
]

function sample_line(μ, pars; n_events, half_width = 60.0, n_grid = 10_000)
    x = range(μ - half_width, μ + half_width, length = n_grid)
    cdf = cumsum(f_peak.(x, Ref((; μ, pars...))))
    cdf ./= last(cdf)
    u = rand(n_events)
    ## linear interpolation of the inverse cumulative distribution function
    map(u) do ui
        i = clamp(searchsortedfirst(cdf, ui), 2, n_grid)
        x[i-1] + (ui - cdf[i-1]) / (cdf[i] - cdf[i-1]) * step(x)
    end
end

## The relative peak heights follow the model heights; each line gets its share of 200_000 events
peak_heights = map(zip(th228_lines, shape_pars)) do (μ, pars)
    maximum(f_peak.(range(ustrip(μ) - 60, ustrip(μ) + 60, length = 2000), Ref((; μ = ustrip(μ), pars...))))
end
n_events = round.(Int, 200_000 .* peak_heights ./ sum(peak_heights))
e_lines = reduce(vcat, [sample_line(ustrip(μ), pars; n_events = n) for (μ, pars, n) in zip(th228_lines, shape_pars, n_events)])
length(e_lines)

# A flat continuum below 2614.5 keV and the drift-time-like parameter complete the data set.
# The energy reconstruction loses a part of the charge that depends on `qdrift`.
# The simulated loss is independent of the energy, so the correction fitted at one peak applies to all peaks.
# The conversion factor from keV to ADC units is arbitrary.

c_true = 0.01u"keV"            # keV per ADC unit
n_continuum = 150_000
e_continuum = rand(truncated(Exponential(700.0), 100.0, 2614.5), n_continuum)

e_true = vcat(e_lines, e_continuum)
qdrift = rand(Normal(1000.0, 300.0), length(e_true))
e_uncal = @. e_true / ustrip(c_true) - 0.5 * (qdrift - 1000.0)
length(e_uncal)

# ## Configuration
#
# The calibration is controlled by a few settings per detector.
# They are the energies of the calibration lines, the sizes of the energy window around each line that is cut out for the fits,
# the peak shape used for each line, and the orders of the polynomials of the calibration and resolution curves.
# The values below are the defaults for a Th-228 source.
# The names `Tl208DEP` and `Tl208SEP` denote the double escape peak (DEP) and single escape peak (SEP) of the 2614.5 keV gamma line of Tl-208,
# and `Bi212FEP` denotes a full-energy peak of Bi-212.

th228_names = [:Tl208a, :Bi212a, :Tl208b, :Tl208DEP, :Bi212FEP, :Tl208SEP, :Tl208FEP]
left_window_sizes  = [25, 25, 25, 25, 15, 25, 35]u"keV"
right_window_sizes = [25, 25, 25, 15, 25, 25, 30]u"keV"
fit_funcs = fill(:gamma_def, length(th228_names))
binning_peak_window = 4u"keV"      # energy range around a peak used to choose the histogram bin width
quantile_perc = 0.91               # fraction of events below the 2614.5 keV peak
cal_pol_order = 1
fwhm_pol_order = 1
excluded_peaks = [:Tl208DEP, :Tl208SEP]  # escape peaks are not used for the calibration and resolution curves

th228_lines_dict = Dict(th228_names .=> th228_lines)

# ## First Calibration Constant
#
# [`simple_calibration`](@ref) finds the 2614.5 keV full-energy peak in the uncalibrated spectrum.
# It returns a calibration constant `c` that converts ADC units to keV and, for each calibration line, a histogram and start values for the peak fit.
# The peak finder does not work reliably for a spectrum with few events per bin, so the position of the peak is
# given as the quantile `quantile_perc` of the energy distribution.
# The constant is only an approximation, since it ignores any offset.

result_simple, report_simple = simple_calibration(
    e_uncal, th228_lines, left_window_sizes, right_window_sizes;
    calib_type = :th228, binning_peak_window, quantile_perc
)
m_cal_simple = result_simple.c

# The report is plotted with `lplot`, which is provided by LegendMakie.
# The keyword `watermark = false` removes the logo and date that LegendMakie adds by default.

lplot(report_simple, cal = true, title = "Simple Calibration", watermark = false)

# ## Correcting Charge Trapping
#
# [`ctc_energy`](@ref) fits a correction of the form `e + fct * qdrift` at the 2614.5 keV peak.
# The coefficient `fct` is chosen such that the peak becomes narrow and high.
# The function takes the energy in keV, so the uncalibrated energy is multiplied by the simple calibration constant.
# The expression `e_expression` names the uncalibrated energy in the function string of the result.
# See [Charge-Trapping Correction](@ref) for details.

result_ctc, report_ctc = ctc_energy(
    e_uncal .* m_cal_simple, qdrift, 2614.511u"keV", (35u"keV", 30u"keV"), m_cal_simple;
    e_expression = "e_uncal", pol_order = 1
)
result_ctc.converged || error("The charge-trapping correction did not converge")
result_ctc.func

# The correction factor is given in ADC units per unit of `qdrift`.
# The simulated loss of 0.5 ADC units per unit of `qdrift` is recovered, and the width of the peak decreases.

(fct = only(result_ctc.fct), fwhm_before = mvalue(result_ctc.fwhm_before), fwhm_after = mvalue(result_ctc.fwhm_after))

#

lplot(report_ctc, figsize = (600, 600), title = "Charge-Trapping Correction", watermark = false)

# The function string of the result describes how to apply the correction to a table of outputs of the digital signal processing (DSP).
# In the production dataflow, it is evaluated for every row of the table with `ljl_propfunc` from LegendDataManagement.
# Here the correction is applied directly.

e_uncal_ctc = e_uncal .+ only(result_ctc.fct) .* qdrift;

# ## Simple Calibration After the Correction
#
# The corrected energies are calibrated again.
# The result contains the histograms and the start values of the peak fits for all calibration lines.

result_simple, report_simple = simple_calibration(
    e_uncal_ctc, th228_lines, left_window_sizes, right_window_sizes;
    calib_type = :th228, binning_peak_window, quantile_perc
)
m_cal_simple = result_simple.c

lplot(report_simple, cal = true, title = "Simple Calibration After Correction", watermark = false)

# ## Fitting the Calibration Lines
#
# [`fit_peaks`](@ref) fits the peak shape of each calibration line to the histograms.
# The default shape `:gamma_def` is a Gaussian with a low-energy tail on top of a flat background and a Compton step.
# The shapes `:gamma_tails` and `:gamma_tails_bckFlat` add a high-energy tail and remove the step, respectively.
# See [Peak Shapes and Likelihoods](@ref) for all shapes.
#
# With `e_unit` and `m_cal_simple`, the fitted positions and widths are returned in ADC units, because the calibration curve is fitted against ADC values.
# The results are dictionaries with the line names as keys.

result_fit, report_fit = fit_peaks(
    result_simple.peakhists, result_simple.peakstats, th228_names;
    e_unit = result_simple.unit, calib_type = :th228, fit_func = fit_funcs, m_cal_simple
)
all(r -> r.gof.converged, values(result_fit)) || error("A peak fit did not converge")

# Each entry holds the fitted parameters with uncertainties, for example the position (`centroid`) and the width (`fwhm`) of the 2614.5 keV peak:

result_fit[:Tl208FEP].centroid, result_fit[:Tl208FEP].fwhm

#

lplot(report_fit, figsize = (600, 400 * length(report_fit)), title = "Peak Fits", watermark = false)

# ## Fitting the Calibration Curve
#
# [`fit_calibration`](@ref) fits a polynomial that maps the fitted peak positions in ADC units to the literature energies.
# The escape peaks are excluded, because their positions are not tied to a line of the source in the same way.
# The polynomial order `cal_pol_order = 1` gives a linear function.
# `e_expression` is the expression of the uncalibrated energy that appears in the function string of the result.

cal_names = [p for p in th228_names if p ∉ excluded_peaks]
result_calib, report_calib = fit_calibration(
    cal_pol_order,
    [result_fit[p].centroid for p in cal_names],
    [th228_lines_dict[p] for p in cal_names];
    e_expression = "e_uncal + $(only(result_ctc.fct)) * qdrift"
)
result_calib.gof.converged || error("The calibration curve fit did not converge")
result_calib.par

# The function string of the result converts the uncalibrated energy and `qdrift` of a table directly to keV.

result_calib.func

# The escape peaks are shown as additional points. They are not part of the fit.

lplot(
    report_calib, xerrscaling = 100, title = "Calibration Curve", watermark = false,
    additional_pts = (
        μ = [result_fit[p].centroid for p in excluded_peaks],
        peaks = [th228_lines_dict[p] for p in excluded_peaks]
    )
)

# ## Fitting the Resolution Curve
#
# The width of the peaks as a function of energy follows the square root of a polynomial in energy.
# [`fit_fwhm`](@ref) fits it to the widths of the peaks in keV.
# The fitted widths are in ADC units, so they are converted to keV with the calibration curve.
# The constant term of the calibration curve is subtracted, because a width is a difference of energies.

f_cal_widths(x) = report_calib.f_fit(x) .* report_calib.e_unit .- first(report_calib.par)
fwhm_names = [p for p in th228_names if p ∉ excluded_peaks]

result_fwhm, report_fwhm = fit_fwhm(
    fwhm_pol_order,
    [th228_lines_dict[p] for p in fwhm_names],
    f_cal_widths.([result_fit[p].fwhm for p in fwhm_names]);
    e_type_cal = :e_cal, uncertainty = true
)
result_fwhm.gof.converged || error("The resolution curve fit did not converge")

# The result contains the parameters of the curve and its value at Qββ = 2039.06 keV, the Q-value of the neutrinoless double-beta decay of Ge-76.

result_fwhm.qbb

# The curve is expressed in the calibrated energy `e_cal`:

result_fwhm.func_cal

#

lplot(
    report_fwhm, title = "Resolution Curve", watermark = false,
    additional_pts = (
        peaks = [th228_lines_dict[p] for p in excluded_peaks],
        fwhm = f_cal_widths.([result_fit[p].fwhm for p in excluded_peaks])
    )
)

# ## Calibrated Spectrum
#
# The function string `result_calib.func` converts the uncalibrated energy and `qdrift` to keV.
# In the production dataflow, `ljl_propfunc` from LegendDataManagement evaluates such strings on a table of DSP outputs.
# Here the string is turned into a function directly.

calibrate = Core.eval(@__MODULE__, Meta.parse("(e_uncal, qdrift) -> " * result_calib.func))
e_cal = ustrip.(u"keV", Base.invokelatest.(calibrate, e_uncal, qdrift))

fig = Figure(size = (700, 400))
ax = Axis(
    fig[1, 1], xlabel = "Energy (keV)", ylabel = "Counts / 1 keV", yscale = log10,
    limits = ((0, 3000), (1, 2e4)), xticks = 0:500:3000, title = "Calibrated Spectrum"
)
stephist!(ax, e_cal, bins = 0:1:3000)
vlines!(ax, ustrip.(u"keV", th228_lines), color = (:red, 0.3), linestyle = :dash, label = "Literature energies")
axislegend(ax, position = :rt)
fig
