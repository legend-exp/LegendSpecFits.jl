# aoe compton region peakshapes
f_aoe_compton(x, v) = aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ)
f_aoe_sig(x, v)     = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)
f_aoe_bkg(x, v)     = aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ)

MaybeWithEnergyUnits = Union{Real, Unitful.Energy{<:Real}}


"""
    get_aoe_fit_functions(; background_center = nothing)

Returns a `NamedTuple` of the A/E Compton-region fit functions `f(x, v)`, where
`v` is a `NamedTuple` of the fit parameters:

* `aoe_one_bck`: [`aoe_compton_peakshape`](@ref) with parameters `μ`, `σ`, `n`, `B` (background amplitude), `δ`.
* `aoe_two_bck`: [`two_emg_aoe_compton_peakshape`](@ref) with the additional parameters `μ2`, `σ2`,
  `B2f` (fraction of the background in the second component), `δ2`.

`background_center` is not used.
"""
function get_aoe_fit_functions(; background_center::Union{Real,Nothing} = nothing)
    merge( 
        (aoe_one_bck = (x, v) -> aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ),
        aoe_two_bck = (x, v) -> two_emg_aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ, v.μ2, v.σ2, v.B2f, v.δ2),
        ),
    if isnothing(background_center)
        NamedTuple()
    else
        NamedTuple()
    end   
    )
end

"""
    aoe_compton_peakshape_components(fit_func::Symbol; background_center = nothing)
    aoe_compton_peakshape_components(fit_func::Symbol, v::NamedTuple; background_center = v.μ)

Returns the component functions of the A/E peak shape `fit_func` (`:aoe_one_bck`
or `:aoe_two_bck`) as a `NamedTuple` with the fields

* `funcs`: component functions `f(x, v)`: `f_sig` and `f_bck` for `:aoe_one_bck`; `f_sig`,
  `f_bck_one` and `f_bck_two` for `:aoe_two_bck`. In the method with `v`, these are functions of `x`
  only, with the parameters `v` applied.
* `labels`, `colors`, `linestyles`: plot label, color and line style of each component.

The components are used in fit reports and plot recipes.
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
            f_bck_one = (x, v) -> aoe_compton_background_peakshape(x, v.μ, v.σ, v.B * (1 - v.B2f), v.δ),
            f_bck_two = (x, v) -> aoe_compton_background_peakshape(x, v.μ2, v.σ2, v.B * v.B2f, v.δ2)) #maybe use one function only
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