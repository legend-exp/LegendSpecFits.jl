# Peak Shapes and Likelihoods

LegendSpecFits.jl describes a gamma line in an energy spectrum with a parametric peak shape: a Gaussian signal, optional exponential tails and a smooth background. The shape is fitted to a histogram by maximizing a binned Poisson likelihood. This page lists the shape functions, the available `fit_func` options, the likelihood, the priors that set the start values, and the generic single-peak fits that do not use the gamma peak shape.

## Gamma Peak Shape

[`gamma_peakshape`](@ref) evaluates the sum of four components, each available as a separate function:

```math
f(x) = \underbrace{n\,(1 - f_\mathrm{L} - f_\mathrm{H})\,\mathcal{N}(x;\mu,\sigma)}_{\text{signal}}
     + \underbrace{n\,f_\mathrm{L}\,\mathcal{E}(-x;-\mu,\sigma,\theta_\mathrm{L})}_{\text{low-energy tail}}
     + \underbrace{n\,f_\mathrm{H}\,\mathcal{E}(x;\mu,\sigma,\theta_\mathrm{H})}_{\text{high-energy tail}}
     + \underbrace{A\,S(x;\mu,\sigma) + b\,e^{-d\,(x - x_\mathrm{c})} + s\,(x - x_\mathrm{c})}_{\text{background}}\,,
```

where

* $x$ is the energy (or any other horizontal axis of the histogram) and $f(x)$ is a density in counts per unit of $x$,
* $\mathcal{N}(x;\mu,\sigma)$ is the normal density with mean $\mu$ and standard deviation $\sigma$ ([`LegendSpecFits.gauss_pdf`](@ref)),
* $\mathcal{E}(x;\mu,\sigma,\theta)$ is the density of an [exponentially modified Gaussian](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution) with exponential scale $\theta$ ([`LegendSpecFits.ex_gauss_pdf`](@ref)),
  ```math
  \mathcal{E}(x;\mu,\sigma,\theta) = \frac{1}{2\theta}\exp\!\left(\frac{\sigma^2}{2\theta^2} - \frac{x-\mu}{\theta}\right)\mathrm{erfc}\!\left(\frac{1}{\sqrt{2}}\left(\frac{\sigma}{\theta} - \frac{x-\mu}{\sigma}\right)\right)\,.
  ```
  Evaluating it at $-x$ with mean $-\mu$ mirrors the tail toward lower energies,
* $n$ is the number of counts in the signal and the tails together,
* $f_\mathrm{L}$ and $f_\mathrm{H}$ are the fractions of $n$ in the low- and high-energy tail, and $\theta_\mathrm{L} = w_\mathrm{L}\,\mu$ and $\theta_\mathrm{H} = w_\mathrm{H}\,\mu$ are their exponential scales, given by the relative widths $w_\mathrm{L}$ and $w_\mathrm{H}$,
* $S(x;\mu,\sigma) = \tfrac{1}{2}\,\mathrm{erfc}\!\left((x-\mu)/(\sqrt{2}\,\sigma)\right)$ is a step that falls from 1 to 0 across $\mu$ with the resolution $\sigma$ ([`LegendSpecFits.step_gauss`](@ref)), and $A$ is its amplitude. It models the higher background below the peak that comes from Compton-scattered photons of the same line,
* $b$ is the flat background level, $d$ the decay constant of an exponential background, $s$ the slope of a linear background and $x_\mathrm{c}$ the reference position of the last two terms. By default, $x_\mathrm{c} = \mu$.

The parameters carry the following names in function arguments and in fit results:

| Symbol | Name | Meaning |
|:--|:--|:--|
| $\mu$, $\sigma$ | `μ`, `σ` | position and standard deviation of the Gaussian |
| $n$ | `n` | counts in signal and tails |
| $f_\mathrm{L}$, $w_\mathrm{L}$ | `skew_fraction`, `skew_width` | low-energy tail |
| $f_\mathrm{H}$, $w_\mathrm{H}$ | `skew_fraction_highE`, `skew_width_highE` | high-energy tail |
| $A$ | `step_amplitude` | amplitude of the step background |
| $b$ | `background` | flat background |
| $s$ | `background_slope` | slope of a linear background |
| $d$ | `background_exp` | decay constant of an exponential background |

The components are available individually as [`signal_peakshape`](@ref), [`lowEtail_peakshape`](@ref), [`highEtail_peakshape`](@ref) and [`background_peakshape`](@ref). The low-energy tail is the most important one for HPGe detectors: incomplete charge collection moves a small fraction of the events of a line to lower energies.

## Fit Functions

The `fit_func` keyword of [`fit_single_peak_th228`](@ref), [`fit_peaks`](@ref) and the functions built on them selects which terms of $f(x)$ are active. Each option corresponds to a function `f(x, v)` of the energy and a `NamedTuple` `v` of parameters. [`LegendSpecFits.get_th228_fit_functions`](@ref) returns all of them as a `NamedTuple`.

| `fit_func` | Active terms | Fitted parameters |
|:--|:--|:--|
| `:gamma_def` (default) | signal, low-energy tail, flat and step background | `μ σ n skew_fraction skew_width background step_amplitude` |
| `:gamma_tails` | `:gamma_def` plus high-energy tail | `:gamma_def` and `skew_fraction_highE skew_width_highE` |
| `:gamma_bckSlope` | `:gamma_def` plus linear background | `:gamma_def` and `background_slope` |
| `:gamma_bckExp` | `:gamma_def` plus exponential background | `:gamma_def` and `background_exp` |
| `:gamma_bckFlat` | `:gamma_def` without step background | `μ σ n skew_fraction skew_width background` |
| `:gamma_tails_bckFlat` | `:gamma_tails` without step background | `:gamma_bckFlat` and `skew_fraction_highE skew_width_highE` |
| `:gamma_minimal` | Gaussian signal and flat background | `μ σ n background` |

`get_th228_fit_functions` also contains `gamma_sigWithTail` (signal and low-energy tail only). It is used to compute the FWHM of the fitted peak and is not a `fit_func` option.

The FWHM reported by the fit is the width of signal plus low-energy tail at half of its maximum ([`LegendSpecFits.estimate_fwhm`](@ref)), so the tail widens the reported FWHM compared to $2\sqrt{2\ln 2}\,\sigma$. The peak position that follows from the same shape is available as [`peak_centroid`](@ref), the mean of the Gaussian shifted by the tails.

[`LegendSpecFits.peakshape_components`](@ref) returns, for a `fit_func` and a parameter set, the component functions together with labels, colors and line styles. The plot recipes of LegendMakie use it for the component curves of a fit report.

The following code evaluates the `:gamma_def` shape for a set of parameters and draws its components:

```@example peakshapes
using LegendSpecFits, CairoMakie

v = (μ = 2614.5, σ = 1.8, n = 2.0e4, skew_fraction = 0.05, skew_width = 0.002,
     background = 25.0, step_amplitude = 3.0)
f_gamma_def = LegendSpecFits.get_th228_fit_functions().gamma_def
components = LegendSpecFits.peakshape_components(:gamma_def, v)

x = range(2590, 2635, length = 400)
fig = Figure(size = (700, 350))
ax = Axis(fig[1, 1]; xlabel = "Energy (keV)", ylabel = "Counts / keV", yscale = log10)
lines!(ax, x, f_gamma_def.(x, Ref(v)); color = :black, label = "Sum")
for k in keys(components.funcs)
    lines!(ax, x, components.funcs[k].(x); color = components.colors[k],
        linestyle = components.linestyles[k], label = components.labels[k])
end
axislegend(ax; position = :lt)
fig
```

The expected counts in a bin of width $\Delta x$ are $f(x)\,\Delta x$ for $x$ at the bin center.

Besides the gamma peak, the package defines the shapes used for A/E distributions ([`aoe_compton_peakshape`](@ref) and its components, [`two_emg_aoe_compton_peakshape`](@ref)), the sum of two Gaussians [`double_gaussian`](@ref), the energy dependence of the A/E resolution [`ex_step_gauss`](@ref), and the generic [`linear_function`](@ref) and [`exponential_decay`](@ref). The A/E shapes are used in [A/E Pulse-Shape Discrimination](@ref).

## Binned Poisson Likelihood

A histogram `h` with bin centers $x_i$, bin widths $\Delta x_i$ and counts $k_i$ is compared with a model density $f$ by the Poisson log-likelihood

```math
\ln L(\vec{p}) = \sum_i \ln \mathrm{Pois}\!\left(k_i;\, \lambda_i\right), \qquad \lambda_i = \Delta x_i\, f(x_i; \vec{p})\,,
```

where $\vec{p}$ are the model parameters and $\lambda_i$ is the expected count in bin $i$. [`hist_loglike`](@ref) computes this sum for a function `f_fit(x)` of the energy. The model is evaluated at the bin centers only (midpoint rule), so the bin width must be small compared to the peak width $\sigma$.

[`fit_single_peak_th228`](@ref) maximizes $\ln L$ with the L-BFGS-B algorithm. The uncertainties of the parameters are the square roots of the diagonal of the inverse Hessian matrix of $-\ln L$ at the maximum, computed with automatic differentiation. If this covariance matrix is not positive definite, it is replaced by the nearest positive-definite matrix. The goodness of fit uses the Poisson deviance (see [Cuts and Goodness of Fit](@ref)).

## Priors and Start Values

Each fit parameter has a prior distribution. The prior does not enter the objective function: the fit is a maximum-likelihood fit, not a maximum a posteriori fit. The priors serve two purposes:

1. They define a transformation of the parameters to unconstrained, standard-normal coordinates in which the optimizer runs. This keeps parameters inside their allowed ranges, for example $\sigma > 0$.
2. The mean of each prior is the start value of the optimizer. This is why these distributions are called pseudo-priors in the code.

The default pseudo-priors depend on the initial peak statistics `ps`, a `NamedTuple` that [`estimate_single_peak_stats`](@ref) computes from the histogram. For `calib_type = :th228` it contains the peak position, FWHM and standard deviation, the number of counts in the peak, the bin width, and the mean and standard deviation of the background and of the background step. The defaults are:

* `μ`: normal distribution around the estimated position with a standard deviation of 20 % of the estimated `peak_sigma`,
* `σ`, `n`, `background`, `step_amplitude`: Weibull distributions (see below) centered on the estimated values,
* `skew_fraction`, `skew_fraction_highE`: Weibull distributions with a median of 0.002 (before truncation), truncated to at most 0.5 and 0.1,
* `skew_width`, `skew_width_highE`: Weibull distributions with a median of the estimated relative peak width,
* `background_slope`: zero if the mean background is below 5 counts per bin, otherwise a truncated normal distribution of non-positive slopes,
* `background_exp`: Weibull distribution with a median of 0.03.

[`weibull_from_mx`](@ref) builds a Weibull distribution from its median `m` and a second point `x` that is the `p_x`-quantile, 68.27 % by default. Weibull distributions are positive and skewed, which suits quantities that have a natural scale but no sharp upper bound:

```@example peakshapes
using Distributions

d = weibull_from_mx(1.0, 1.5)
(median(d), cdf(d, 1.5))
```

With `low_e_tail = false`, the tail fractions are fixed to zero. With `fixed_position = true`, `μ` is fixed to the estimated position. The `pseudo_prior` keyword replaces individual default priors. Its value is a `NamedTupleDist` from [ValueShapes.jl](https://github.com/oschulz/ValueShapes.jl), and its keys must be parameters of the chosen `fit_func`. A `ConstValueDist` fixes a parameter:

```julia
using Distributions, ValueShapes

pseudo_prior = NamedTupleDist(
    σ = Normal(2.0, 0.2),             # replaces the default prior of σ
    skew_fraction = ConstValueDist(0.0),  # fixes the parameter
)
result, report = fit_single_peak_th228(h, ps; pseudo_prior)
```

## Fitting a Single Peak

[`fit_single_peak_th228`](@ref) takes a histogram and the peak statistics and returns the usual `(result, report)` pair. `result` holds the fitted parameters as `Measurement`s, the `fwhm` and `centroid` of the peak, the used `fit_func` and the goodness of fit `gof`. `report` holds everything needed to plot the fit with `lplot` from LegendMakie. [`fit_peaks`](@ref) runs the same fit for a list of lines in parallel; it is described with the energy calibration.

The example generates a 2614.5 keV line with a low-energy tail on a flat background, estimates the start values and fits `:gamma_def`:

```@example peakshapes
using Random, StatsBase, Measurements

Random.seed!(1)
E = 2614.5
energies = vcat(
    rand(Normal(E, 1.8), 20_000),                         # signal
    E .- rand(Exponential(4.0), 1_500) .+ 1.8 .* randn(1_500),  # low-energy tail
    rand(Uniform(E - 40, E + 40), 4_000),                 # flat background
)
h = fit(Histogram, energies, (E - 30):0.5:(E + 30))
ps = estimate_single_peak_stats(h)
result, report = fit_single_peak_th228(h, ps; fit_func = :gamma_def)
(μ = result.μ, σ = result.σ, fwhm = result.fwhm, pvalue = result.gof.pvalue)
```

The generating values are $\mu = 2614.5$ and $\sigma = 1.8$, which gives a FWHM of 4.24 for the Gaussian alone; the fitted FWHM is larger because it includes the low-energy tail. The report is plotted with LegendMakie:

```@example peakshapes
using LegendMakie

lplot(report; title = "Synthetic 2614.5 keV line")
```

The upper panel shows the histogram, the best fit and its components on a logarithmic scale; the lower panel shows the normalized residuals.

## Generic Single Fits

Several fit functions do not use the gamma peak shape. They fit a Gaussian to a single distribution, for example the distribution of a pulse-shape parameter, a baseline noise or a trigger threshold. They take the data and `cuts`, accept `uncertainty = true` and return `(result, report)`; `fit_half_centered_trunc_gauss` takes the fixed mean as an additional argument before `cuts`. The `cuts` argument is the `NamedTuple` `(low, high, max)` that [`cut_single_peak`](@ref) returns. A `NaN` bound is replaced by the edge of the data. Data with a `Unitful` unit keep their unit in the results.

| Function | Data | Model | Fitted parameters |
|:--|:--|:--|:--|
| [`fit_single_trunc_gauss`](@ref) | values | Gaussian truncated to `[cuts.low, cuts.high]`, unbinned likelihood | `μ σ` |
| [`fit_half_trunc_gauss`](@ref) | values | one half of a Gaussian: right of `cuts.max` or, with `left = true`, left of it | `μ σ` |
| [`fit_half_centered_trunc_gauss`](@ref) | values | one half of a Gaussian with fixed mean `μ` | `σ` |
| [`fit_binned_trunc_gauss`](@ref) | histogram | as `fit_single_trunc_gauss`, binned Poisson likelihood with amplitude | `μ σ n` |
| [`fit_binned_half_trunc_gauss`](@ref) | histogram | as `fit_half_trunc_gauss`, binned Poisson likelihood with amplitude | `μ σ n` |
| [`fit_binned_double_gauss`](@ref) | histogram | sum of two Gaussians ([`double_gaussian`](@ref)) | `μ1 σ1 n1 μ2 σ2 n2` |

The half-Gaussian fits use only the side of a peak that is not distorted by a tail or by a second population. The truncation to `cuts` excludes the background far from the peak.

A typical use is a window cut: the example below finds the peak region of a sample with [`cut_single_peak`](@ref) and fits the truncated Gaussian to it:

```@example peakshapes
Random.seed!(2)
x = rand(Normal(5.0, 0.5), 5_000)
cuts = cut_single_peak(x, 0.0, 10.0)
result_gauss, report_gauss = fit_single_trunc_gauss(x, cuts)
(μ = result_gauss.μ, σ = result_gauss.σ)
```

`report_gauss` contains the fitted density `f_fit` and the histogram `h` normalized to a probability density, and is plotted with `lplot(report_gauss)`.

## Least-Squares Fits

[`chi2fit`](@ref) fits a model `f(x, p...)` to points $(x_i, y_i)$ with uncertainties by minimizing

```math
\chi^2(\vec{p}) = \sum_i \frac{\left(y_i - f(x_i;\vec{p})\right)^2}{\sigma_{y,i}^2 + \left(f'(x_i;\vec{p})\,\sigma_{x,i}\right)^2} + \sum_j \frac{(p_j - m_j)^2}{s_j^2}\,,
```

where $\sigma_{x,i}$ and $\sigma_{y,i}$ are the uncertainties of the points and $f'$ is the derivative of the model with respect to $x$, computed by automatic differentiation. The second sum contains optional Gaussian pull terms with mean $m_j$ and standard deviation $s_j$ for selected parameters, given by the `pull_t` keyword. The covariance matrix of the parameters is $2H^{-1}$ with $H$ the Hessian matrix of $\chi^2$ at the minimum. [`fit_calibration`](@ref) and [`fit_fwhm`](@ref) are built on this function. The input points are `Real`s or `Measurement`s; with `Measurement`s, the uncertainties are taken from the input.

The example fits a straight line with the polynomial shortcut `chi2fit(n_poly, x, y, yerr)`:

```@example peakshapes
Random.seed!(3)
xs = collect(1.0:10.0)
yerr = fill(0.1, 10)
ys = 2 .* xs .+ 1 .+ yerr .* randn(10)
result_chi2, report_chi2 = chi2fit(1, xs, ys, yerr)
(par = result_chi2.par, pvalue = result_chi2.gof.pvalue)
```

The parameters `par` are ordered from the constant term upward, so `par[1]` is the intercept and `par[2]` the slope (see [`PolCalFunc`](@ref)). With `correlated = true` (the default) the parameters share their correlations, so that a derived quantity such as `report_chi2.f_fit(5.0)` has the uncertainty that follows from the full covariance matrix.
