# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
"""
    get_th228_fit_functions(; background_center::Union{Real,Nothing} = nothing)
This function defines the gamma peakshape fit functions used in the calibration specfits.
* gamma_def: "default" gamma peakshape with gaussian signal, low-energy tail, and background (flat + step)
* gamma_tails: default gamma peakshape + high-energy tail
* gamma_bckSlope: default gamma peakshape + linear background slope
* gamma_bckExp: default gamma peakshape + exponential background 
* gamma_bckFlat: default gamma peakshape - step background (only flat component!)
* gamma_tails_bckFlat: default gamma peakshape + high-energy tail - step background (only flat component!)
"""
function get_th228_fit_functions(; background_center::Union{Real,Nothing} = nothing)
    merge( 
        (gamma_def = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background),
        gamma_tails = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width , v.background; skew_fraction_highE = v.skew_fraction_highE, skew_width_highE = v.skew_width_highE),
        gamma_sigWithTail = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
        gamma_bckFlat = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, 0.0, v.skew_fraction, v.skew_width, v.background),
        gamma_tails_bckFlat = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, 0.0, v.skew_fraction, v.skew_width , v.background; skew_fraction_highE = v.skew_fraction_highE, skew_width_highE = v.skew_width_highE),
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
    peakshape_components(fit_func::Symbol; background_center::Union{Real,Nothing} = nothing) 
This function defines the components (signal, low/high-energy tail, backgrounds) of the fit function used in gamma specfits. 
These component functions are used in the fit-report and in plot receipes 

# Arguments
    * `fit_func`: Fitted function

# Keywords
    * `background_center`: Center of background fit curve

# Returns
    * `funcs`: Functions
    * `labels`: Function labels
    * `colors`: Fit color
    * `linestyles`: Fit linestyles

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
    end 
    labels = (f_sig = "Signal", f_lowEtail = "Low-energy tail", f_bck = "Background", f_highEtail = "High-energy tail")
    colors = (f_sig = :orangered1, f_lowEtail = :orange, f_bck = :dodgerblue2, f_highEtail = :forestgreen)
    linestyles = (f_sig = :solid, f_lowEtail = :dashdot, f_bck = :dash, f_highEtail = :dot)
    return (funcs = funcs, labels = labels, colors = colors, linestyles = linestyles) 
end

"""
    peakshape_components(fit_func::Symbol, v::NamedTuple; background_center::Union{Real,Nothing} = v.μ)

This function defines the components (signal, low/high-energy tail, backgrounds) of the fit function used in gamma specfits. 
These component functions are used in the fit-report and in plot receipes.

# Arguments
    * `fit_func`: Fitted function
    * `v`:  namedTuple of fit parameters

# Keywords
    *`background_center`: Center of background fit curve

# Returns
    * `out`: Peakshape components

TO DO: argument descriptions
"""
function peakshape_components(fit_func::Symbol, v::NamedTuple; background_center::Union{Real,Nothing} = v.μ)
    components  = peakshape_components(fit_func; background_center = background_center)
    out = (; components..., funcs = merge([NamedTuple{Tuple([name])}(Tuple([x -> Base.Fix2(components.funcs[name], v)(x)]))  for name in  keys(components.funcs)]...))
    return out
end