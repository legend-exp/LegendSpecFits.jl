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
# The spectrum is built from the processes that shape a real calibration spectrum.
# Every gamma line of the source contributes
# * a full-energy peak (FEP): a Gaussian whose variance grows linearly with the energy, plus a small low-energy tail from incomplete charge collection,
# * a Compton continuum: the gamma scatters once and leaves the detector, depositing an energy between zero and the Compton edge with the Klein-Nishina distribution,
# * a multiple-scattering continuum: the gamma scatters several times or loses energy outside the detector before being absorbed.
#   It fills the gap between the Compton edge and the line and forms the step below each peak.
#
# Besides the seven calibration lines, the source emits weaker lines (510.8, 785.4 and 1078.6 keV) that are not used for the calibration.
# The double escape peak (DEP) and single escape peak (SEP) of the 2614.5 keV line are peaks without a continuum of their own.
# The counts of the lines are rough values for a Th-228 calibration of one detector; they are not tied to a real detector.
# The tutorial uses a fixed seed, so the numbers shown below are reproducible.

Random.seed!(1234)

th228_lines = [583.191, 727.330, 860.564, 1592.513, 1620.50, 2103.512, 2614.511]u"keV"

## energy (keV), counts in the FEP, Compton events per FEP event; escape peaks have no own continuum
gamma_lines = [
    (510.77,  6_000, 2.8), (583.191, 25_000, 2.9), (727.330, 4_500, 3.1), (785.37,  700, 3.2),
    (860.564, 3_000, 3.3), (1078.62, 400, 3.6),    (1592.513, 2_500, 0.0), (1620.50, 800, 4.4),
    (2103.512, 4_000, 0.0), (2614.511, 20_000, 5.9),
]
resolution(E) = sqrt(0.42 + 2.7e-4 * E)   # Gaussian σ in keV
tail_fraction, tail_scale = 0.05, 0.001     # low-energy tail: fraction of the FEP, exponential scale relative to E
multiscatter_fraction = 0.03                # fraction of the continuum events between the Compton edge and the line

## Energy deposited by a single Compton scattering, sampled from the Klein-Nishina cross section
function sample_compton(E; m_e = 510.999)
    ε = E / m_e
    T_edge = E * 2ε / (1 + 2ε)
    dσdT(T) = (s = T / E; 2 + s^2 / (ε^2 * (1 - s)^2) + s / (1 - s) * (s - 2 / ε))
    f_max = maximum(dσdT, range(0, T_edge, length = 100))
    while true
        T = rand() * T_edge
        rand() * f_max < dσdT(T) && return T
    end
end
compton_edge(E) = E * (1 - 1 / (1 + 2E / 510.999))

e_true = Float64[]
for (E, n_fep, compton_per_fep) in gamma_lines
    tail = rand(n_fep) .< tail_fraction
    append!(e_true, E .- tail .* rand(Exponential(tail_scale * E), n_fep))
    n_compton = round(Int, compton_per_fep * n_fep)
    multiscatter = rand(n_compton) .< multiscatter_fraction
    append!(e_true, [m ? rand(Uniform(compton_edge(E), E)) : sample_compton(E) for m in multiscatter])
end

## gammas scattered in the materials around the detector add a continuum that rises towards low energies;
## random coincidences (pile-up) add a small flat continuum, also above 2614.5 keV
append!(e_true, rand(Exponential(250.0), 150_000))
append!(e_true, rand(Uniform(0, 3300), 8_000))
e_true .+= resolution.(e_true) .* randn(length(e_true))

## the detector trigger threshold removes the lowest energies
filter!(>(100.0), e_true)
length(e_true)

# The data also contain a drift-time-like parameter `qdrift`.
# The energy reconstruction loses a part of the charge that depends on `qdrift`.
# The simulated loss is independent of the energy, so the correction fitted at one peak applies to all peaks.
# The conversion factor from keV to ADC units is arbitrary.

c_true = 0.01u"keV"            # keV per ADC unit
qdrift = rand(Normal(1000.0, 300.0), length(e_true))
e_uncal = @. e_true / ustrip(c_true) - 0.5 * (qdrift - 1000.0)
length(e_uncal)

# The true spectrum before the charge loss:

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "Energy (keV)", ylabel = "Counts / 1 keV", yscale = log10, limits = ((0, 3000), (1, 2e4)))
stephist!(ax, e_true, bins = 0:1:3000)
fig

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
cal_pol_order = 1
fwhm_pol_order = 1
excluded_peaks = [:Tl208DEP, :Tl208SEP]  # escape peaks are not used for the calibration and resolution curves

th228_lines_dict = Dict(th228_names .=> th228_lines)

# ## First Calibration Constant
#
# [`simple_calibration`](@ref) finds the 2614.5 keV full-energy peak in the uncalibrated spectrum.
# It returns a calibration constant `c` that converts ADC units to keV and, for each calibration line, a histogram and start values for the peak fit.
# For `calib_type = :th228`, the peak finder searches the highest 10 % of the energies, where the 2614.5 keV peak is the most prominent one.
# For spectra in which the peak finder fails, the keyword `quantile_perc` sets the peak position to a quantile of the energy distribution instead.
# The constant is only an approximation, since it ignores any offset.

result_simple, report_simple = simple_calibration(
    e_uncal, th228_lines, left_window_sizes, right_window_sizes;
    calib_type = :th228, binning_peak_window
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
# The optimization fits the peak for many trial values of `fct`; fits at trial values far from the optimum may not converge.
# Their warnings are suppressed here.

using Logging #hide
disable_logging(Logging.Warn) #hide
result_ctc, report_ctc = ctc_energy(
    e_uncal .* m_cal_simple, qdrift, 2614.511u"keV", (35u"keV", 30u"keV"), m_cal_simple;
    e_expression = "e_uncal", pol_order = 1
)
disable_logging(Logging.BelowMinLevel) #hide
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
    calib_type = :th228, binning_peak_window
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
