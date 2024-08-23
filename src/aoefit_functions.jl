# aoe compton region peakshapes
f_aoe_compton(x, v) = aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ)
f_aoe_sig(x, v)     = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)
f_aoe_bkg(x, v)     = aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ)

MaybeWithEnergyUnits = Union{Real, Unitful.Energy{<:Real}}


function get_aoe_fit_functions(; background_center::Union{Real,Nothing} = nothing)
    merge( 
        (f_fit = (x, v) -> aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ),),
    if isnothing(background_center)
        NamedTuple()
    else
        NamedTuple()
    end   
    )
end

"""
    aoe_compton_peakshape_components(fit_func::Symbol; background_center::Real) 
This function defines the components (signal, low/high-energy tail, backgrounds) of the fit function used in gamma specfits. 
These component functions are used in the fit-report and in plot receipes 
"""
function aoe_compton_peakshape_components(fit_func::Symbol; background_center::Union{Real,Nothing} = nothing)
    if fit_func == :f_fit
        funcs = (f_sig = (x, v) -> aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n),
            f_bck = (x, v) -> aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ))
    end 
    labels = (f_sig = "Signal", f_bck = "Background")
    colors = (f_sig = :orangered1, f_bck = :dodgerblue2)
    linestyles = (f_sig = :solid, f_bck = :dash)
    return (funcs = funcs, labels = labels, colors = colors, linestyles = linestyles)
end

function aoe_compton_peakshape_components(fit_func::Symbol, v::NamedTuple; background_center::Union{Real,Nothing} = v.μ)
    components  = aoe_compton_peakshape_components(fit_func; background_center = background_center)
    out = (; components..., funcs = merge([NamedTuple{Tuple([name])}(Tuple([x -> Base.Fix2(components.funcs[name], v)(x)]))  for name in  keys(components.funcs)]...))
    return out
end