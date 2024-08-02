# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

# helper functions for fitting peakshapes, legacy 
th228_fit_functions = (
    f_fit = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background),
    f_fit_bckSlope = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_slope =  v.background_slope),
    f_fit_bckExp = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_exp =  v.background_exp),
    f_fit_tails = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width , v.background; skew_fraction_highE = v.skew_fraction_highE, skew_width_highE = v.skew_width_highE),
    f_sigWithTail = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
)

function get_th228_fit_functions(; background_center::Union{Real,Nothing} = nothing)
    merge( 
        (f_fit = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background),
        f_fit_tails = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width , v.background; skew_fraction_highE = v.skew_fraction_highE, skew_width_highE = v.skew_width_highE),
        f_sigWithTail = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) + lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
        ),
    if isnothing(background_center)
        (f_fit_bckSlope = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_slope =  v.background_slope, background_center = v.μ),
        f_fit_bckExp = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_exp =  v.background_exp,  background_center = v.μ),
        )
    else
        (f_fit_bckSlope = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_slope =  v.background_slope, background_center = background_center),
        f_fit_bckExp = (x, v) -> gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width, v.background; background_exp =  v.background_exp, background_center = background_center),
        )
    end   
    )
end

"""
    peakshape_components(fit_func::Symbol; background_center::Real) 
This function defines the components (signal, low/high-energy tail, backgrounds) of the fit function used in gamma specfits. 
These component functions are used in the fit-report and in plot receipes 
"""
function peakshape_components(fit_func::Symbol; background_center::Real)
    if fit_func == :f_fit
       funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background))
    elseif fit_func == :f_fit_bckSlope
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background; background_slope =  v.background_slope, background_center = background_center))
    elseif fit_func == :f_fit_bckExp
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background; background_exp =  v.background_exp, background_center = background_center))
    elseif fit_func == :f_fit_tails
        funcs = (f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, (v.skew_fraction + v.skew_fraction_highE)),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_highEtail = (x, v) -> highEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction_highE, v.skew_width_highE), 
            f_bck = (x, v) -> background_peakshape(x, v.μ, v.σ, v.step_amplitude, v.background))
    end 
    labels = (f_sig = "Signal", f_lowEtail = "Low-energy tail", f_bck = "Background", f_highEtail = "High-energy tail")
    colors = (f_sig = :orangered1, f_lowEtail = :orange, f_bck = :dodgerblue2, f_highEtail = :forestgreen)
    linestyles = (f_sig = :solid, f_lowEtail = :dashdot, f_bck = :dash, f_highEtail = :dot)
    return (funcs = funcs, labels = labels, colors = colors, linestyles = linestyles) 
end

function peakshape_components(fit_func::Symbol, v::NamedTuple; background_center::Real = v.μ)
     components  = peakshape_components(fit_func; background_center = background_center)
     out = (; components..., funcs = merge([NamedTuple{Tuple([name])}(Tuple([x -> Base.Fix2(components.funcs[name], v)(x)]))  for name in  keys(components.funcs)]...))
    return out
end