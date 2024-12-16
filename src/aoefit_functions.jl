# aoe compton region peakshapes
f_aoe_compton(x, v) = aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ)
f_aoe_sig(x, v)     = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)
f_aoe_bkg(x, v)     = aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ)

MaybeWithEnergyUnits = Union{Real, Unitful.Energy{<:Real}}

"""
    get_aoe_fit_functions(; background_center::Union{Real,Nothing} = nothing)

This function gives the A/E fit functions. 

# Keywords
    * `background_center`: Center of background fit curve
"""

function get_aoe_fit_functions(; background_center::Union{Real,Nothing} = nothing)
    merge( 
        (aoe_one_bck = (x, v) -> aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ),
        aoe_two_bck = (x, v) -> two_emg_aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ, v.μ2, v.σ2, v.B2, v.δ2),
        ),
    if isnothing(background_center)
        NamedTuple()
    else
        NamedTuple()
    end   
    )
end

"""
    aoe_compton_peakshape_components(fit_func::Symbol; background_center::Union{Real,Nothing} = nothing)

This function defines the components (signal, low/high-energy tail, backgrounds) of the fit function used in gamma specfits. 
These component functions are used in the fit-report and in plot recipes.
There are 2 methods for this function.

# Arguments
    * `fit_func`: Fit function of the A/E compton peakshape

# Keywords
    * `background_center`: Center of background fit curve

# Returns
    * A/E compton peakshape components

"""
function aoe_compton_peakshape_components(fit_func::Symbol; background_center::Union{Real,Nothing} = nothing)
    if fit_func == :aoe_one_bck
        funcs = (f_sig = (x, v) -> aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n),
            f_bck = (x, v) -> aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ))
        labels = (f_sig = "Signal", f_bck = "Background")
        colors = (f_sig = :orangered1, f_bck = :dodgerblue2)
        linestyles = (f_sig = :solid, f_bck = :dash)
    elseif fit_func == :aoe_two_bck
        funcs = (f_sig = (x, v) -> aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n),
            f_bck_one = (x, v) -> aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ),
            f_bck_two = (x, v) -> aoe_compton_background_peakshape(x, v.μ2, v.σ2, v.B2, v.δ2)) #maybe use one function only
        labels = (f_sig = "Signal", f_bck_one = "First EMG", f_bck_two = "Second EMG")
        colors = (f_sig = :orangered1, f_bck_one = :dodgerblue2, f_bck_two = :green)
        linestyles = (f_sig = :solid, f_bck_one = :dash, f_bck_two = :dashdot)
    end 
    return (funcs = funcs, labels = labels, colors = colors, linestyles = linestyles)
end

function aoe_compton_peakshape_components(fit_func::Symbol, v::NamedTuple; background_center::Union{Real,Nothing} = v.μ)
    components  = aoe_compton_peakshape_components(fit_func; background_center = background_center)
    out = (; components..., funcs = merge([NamedTuple{Tuple([name])}(Tuple([x -> Base.Fix2(components.funcs[name], v)(x)]))  for name in  keys(components.funcs)]...))
    return out
end