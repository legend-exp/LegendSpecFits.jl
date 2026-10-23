# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
"""
    get_th228_fit_functions(; background_center::Union{Real, Nothing} = nothing)

Returns a `NamedTuple` of the gamma peak shape functions `f(x, v)` used in the
calibration fits, where `v` is a `NamedTuple` of the fit parameters. Each is
built from [`gamma_peakshape`](@ref):

* `gamma_def`: Gaussian signal, low-energy tail, flat background and step background.
* `gamma_tails`: `gamma_def` plus a high-energy tail.
* `gamma_sigWithTail`: signal and low-energy tail only, without background.
* `gamma_bckSlope`: `gamma_def` plus a linear background slope.
* `gamma_bckExp`: `gamma_def` plus an exponential background.
* `gamma_bckFlat`: `gamma_def` without the step background.
* `gamma_tails_bckFlat`: `gamma_tails` without the step background.
* `gamma_minimal`: Gaussian signal and flat background only.

`background_center` is the reference position of the slope and exponential
background terms. If `nothing`, the peak position `v.μ` is used.
"""
function get_th228_fit_functions(; background_center::Union{Real,Nothing} = nothing)
    merge( 
        (gamma_def = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background),
        gamma_tails = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width , v.background; skew_fraction_highE = v.skew_fraction_highE, skew_width_highE = v.skew_width_highE),
        gamma_sigWithTail = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
        gamma_bckFlat = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, 0.0, v.skew_fraction, v.skew_width, v.background),
        gamma_tails_bckFlat = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, 0.0, v.skew_fraction, v.skew_width , v.background; skew_fraction_highE = v.skew_fraction_highE, skew_width_highE = v.skew_width_highE),
        gamma_minimal = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, 0.0, 0.0, 0.0, v.background),
        ),
    if isnothing(background_center)
        (gamma_bckSlope = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_slope =  v.background_slope, background_center = v.μ),
        gamma_bckExp = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_exp =  v.background_exp,  background_center = v.μ),
        )
    else
        (gamma_bckSlope = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_slope =  v.background_slope, background_center = background_center),
        gamma_bckExp = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_exp =  v.background_exp, background_center = background_center),
        )
    end   
    )
end

"""
    peakshape_components(fit_func::Symbol; background_center::Union{Real, Nothing} = nothing)
    peakshape_components(fit_func::Symbol, v::NamedTuple; background_center = v.μ)

Returns the component functions of the peak shape `fit_func` (one of the names
in [`get_th228_fit_functions`](@ref) except `gamma_sigWithTail`) as a
`NamedTuple` with the fields

* `funcs`: component functions `f(x, v)` (`f_sig`, `f_lowEtail`, `f_bck` and, for the shapes with a
  high-energy tail, `f_highEtail`). In the method with `v`, these are functions of `x` only,
  with the parameters `v` applied.
* `labels`, `colors`, `linestyles`: plot label, color and line style of each component.

The components are used in fit reports and plot recipes.
"""
function peakshape_components(fit_func::Symbol; background_center::Union{Real,Nothing} = nothing)
    if fit_func == :gamma_def
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background))
    elseif fit_func == :gamma_bckSlope
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background; background_slope =  v.background_slope, background_center = background_center))
    elseif fit_func == :gamma_bckExp
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background; background_exp =  v.background_exp, background_center = background_center))
    elseif fit_func == :gamma_bckFlat
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, 0.0, v.background))
    elseif fit_func == :gamma_tails
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, (v.skew_fraction + v.skew_fraction_highE)),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_highEtail = (x, v) -> highEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction_highE, v.skew_width_highE), 
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background))
    elseif fit_func == :gamma_tails_bckFlat
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, (v.skew_fraction + v.skew_fraction_highE)),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_highEtail = (x, v) -> highEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction_highE, v.skew_width_highE), 
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, 0.0, v.background))
    elseif fit_func == :gamma_minimal
            funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, 0.0),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, 0.0, 0.0),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, 0.0, v.background))
    end 
    labels = (f_sig = "Signal", f_lowEtail = "Low-energy tail", f_bck = "Background", f_highEtail = "High-energy tail")
    colors = (f_sig = :orangered1, f_lowEtail = :orange, f_bck = :dodgerblue2, f_highEtail = :forestgreen)
    linestyles = (f_sig = :solid, f_lowEtail = :dashdot, f_bck = :dash, f_highEtail = :dot)
    return (funcs = funcs, labels = labels, colors = colors, linestyles = linestyles) 
end

function peakshape_components(fit_func::Symbol, v::NamedTuple; background_center::Union{Real,Nothing} = v.μ)
    components  = peakshape_components(fit_func; background_center = background_center)
    out = (; components..., funcs = merge([NamedTuple{Tuple([name])}(Tuple([x -> Base.Fix2(components.funcs[name], v)(x)]))  for name in  keys(components.funcs)]...))
    return out
end