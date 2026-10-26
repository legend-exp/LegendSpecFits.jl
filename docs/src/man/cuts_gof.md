# Cuts and Goodness of Fit

This page describes the helpers that select the region of a distribution around a single peak, the quality-control cuts built on them, and the statistics that quantify how well a model describes a histogram.

## Peak Window Cuts

[`cut_single_peak`](@ref) locates the single peak of a sample `x` between `min_x` and `max_x`. It histograms the values that lie strictly inside this range, finds the highest bin and then the first bin from the left whose counts reach `relative_cut` times the maximum and the first bin right of the maximum whose counts fall to that level. The bin width follows the Freedman-Diaconis rule, $2\,\mathrm{IQR}(x)/\sqrt[3]{n}$, unless `n_bins` is set to a positive number. If the three positions are not ordered, the histogram is rebuilt with finer bins, up to `n_tries` times. The result is a `NamedTuple` `(low, high, max)` with the left edges of the two window bins and of the maximum bin, in the unit of `min_x`.

The result serves as input to the Gaussian fits of [Peak Shapes and Likelihoods](@ref) (`cuts` argument) and as a quick estimate of the peak position and width.

```@example cuts
using LegendSpecFits, Random, Distributions, Measurements

Random.seed!(1)
x = vcat(randn(50_000), 10 .* rand(5_000) .- 5)   # peak at 0 on a flat background
cut_single_peak(x, -5.0, 5.0; relative_cut = 0.5)
```

For a Gaussian, the window between the two half-maximum positions has the width of the FWHM, $2\sqrt{2\ln 2}\,\sigma \approx 2.355\,\sigma$.

## Window Cuts from a Gaussian Fit

[`get_centered_gaussian_window_cut`](@ref) turns a peak into a symmetric window of $\pm n_\sigma$ standard deviations. It determines the peak region with `cut_single_peak` and fits one side of the peak with a half Gaussian: [`fit_half_centered_trunc_gauss`](@ref) with the mean fixed to `center` if `fixed_center = true` (the default), and [`fit_half_trunc_gauss`](@ref) with a free mean otherwise. `left = true` selects the left side. Fitting one side avoids the influence of a tail or a second population on the other side. The window is

```math
\mu - n_\sigma\,\sigma < x < \mu + n_\sigma\,\sigma\,,
```

with $\mu$ and $\sigma$ from the fit. The function returns `(result, report)`. `result` holds `low_cut`, `high_cut`, `μ`, `σ`, `gof` and the edges used for the fit; `report` holds the fit function, the histogram and the cuts for `lplot`.

```@example cuts
result, report = get_centered_gaussian_window_cut(x, -5.0, 5.0, 3.0;
    relative_cut = 0.2, n_bins = 500, fixed_center = false)
(low_cut = result.low_cut, high_cut = result.high_cut, σ = result.σ)
```

## Quality-Control Cuts

[`qc_window_cut`](@ref) applies this window cut to several columns of a table. It is used for quality-control (QC) cuts on parameters such as baseline, trigger position or rise time, where good events form a single peak and outliers lie far away.

The configuration maps each column name (a `Symbol`) to a dictionary with the keys `:min`, `:max`, `:sigma` and `:kwargs`. `:min` and `:max` bound the search for the peak, `:sigma` is $n_\sigma$ and `:kwargs` holds further keyword arguments of `get_centered_gaussian_window_cut`. The columns are processed in parallel on the available threads. The method for a single vector, `qc_window_cut(x, min, max, sigma; col_expression, kwargs...)`, is used for each column.

The result contains one entry per column with the fields of `get_centered_gaussian_window_cut` plus `qc`, a string that formulates the cut, and a top-level field `qc` that joins the strings of all columns with `&&`. The report is an `OrderedDict` with one report per column. The strings can be parsed into a filter on the table, so the cuts can be stored as text together with other parameters and applied later.

```@example cuts
using TypedTables

Random.seed!(2)
tab = Table(baseline = 100 .+ 2 .* randn(20_000), trig_pos = 50 .+ 0.5 .* randn(20_000))
config = Dict(
    :baseline => Dict(:min => 80.0, :max => 120.0, :sigma => 4.0, :kwargs => Dict(:fixed_center => false, :relative_cut => 0.2)),
    :trig_pos => Dict(:min => 40.0, :max => 60.0, :sigma => 4.0, :kwargs => Dict(:fixed_center => false, :relative_cut => 0.2)),
)
result_qc, report_qc = qc_window_cut(tab, config)
result_qc.qc
```

## Goodness-of-Fit Statistics

The functions below compare a model with a histogram `h`. `f_fit(x, v)` is the model density as a function of the position `x` and the parameters `v` (a `NamedTuple` or a vector), and `v_ml` are the best-fit parameters. The expected count in a bin is the bin width times `f_fit` at the bin center, $\lambda_i = \Delta x_i\,f(x_i;\vec{p})$. All functions return the p-value, the test statistic and the number of degrees of freedom.

* [`p_value`](@ref) is the Pearson chi-squared test,
  ```math
  \chi^2 = \sum_{k_i > 0} \frac{(k_i - \lambda_i)^2}{\lambda_i}\,,
  ```
  where $k_i$ is the count in bin $i$. Empty bins are skipped, so the degrees of freedom are the number of non-empty bins minus the number of parameters. The p-value is the upper tail of the $\chi^2$ distribution and is `NaN` if there are fewer non-empty bins than parameters. The test assumes Gaussian bin counts and is not valid for bins with fewer than about five counts.
* [`poisson_deviance`](@ref) returns the Poisson deviance
  ```math
  D = 2 \sum_i \left[\lambda_i - k_i + k_i \ln\frac{k_i}{\lambda_i}\right]
  ```
  together with its mean and variance under the model. The mean and the variance are computed exactly from the Poisson distribution of each bin.
* [`p_value_poissonll`](@ref) uses these moments for a test that is also valid for low counts: the p-value is the upper tail of a normal distribution at $z = (D - (\langle D\rangle - n_\mathrm{par}))/\sqrt{\mathrm{Var}(D)}$, with the number of fitted parameters $n_\mathrm{par}$ ([Kaastra 2017](https://doi.org/10.1051/0004-6361/201629319)). The degrees of freedom are the number of bins minus $n_\mathrm{par}$. The fits of [`fit_single_peak_th228`](@ref), [`fit_sipm_spectrum`](@ref) and related functions use this p-value for their `gof`.
* [`p_value_MC`](@ref) estimates a p-value from Monte Carlo samples: it draws Poisson-distributed histograms from the best-fit model, refits each and compares the maximum likelihood values. It requires one fit per sample and is much slower than the other methods.
* [`LegendSpecFits.get_residuals`](@ref) returns the residuals `model - data`, the normalized residuals `(model - data) / sqrt(model)`, a bin-wise p-value and the bin centers, for the bins with a positive model value. The normalized residuals are what the lower panel of the fit plots shows.

The example computes both p-values for a Gaussian histogram:

```@example cuts
using StatsBase

Random.seed!(3)
data = rand(Normal(0.0, 1.0), 10_000)
h = fit(Histogram, data, -4:0.25:4)
f_fit(x, v) = v.n * pdf(Normal(v.μ, v.σ), x)
v_ml = (μ = mean(data), σ = std(data), n = length(data) * 1.0)

pval, chi2, dof = p_value(f_fit, h, v_ml)
pval_poisson, deviance, dof_poisson = p_value_poissonll(f_fit, h, v_ml)
residuals, residuals_norm, _, bin_centers = LegendSpecFits.get_residuals(f_fit, h, v_ml)
(p_value = pval, p_value_poissonll = pval_poisson, std_residuals = std(residuals_norm))
```

The model counts $\lambda_i$ come from the bin center only, so the p-values are reliable when the bin width is small compared to the scale on which the model changes.

The binned likelihood that these statistics refer to is [`hist_loglike`](@ref).
