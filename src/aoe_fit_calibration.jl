@. f_aoe_sigma(x, p) = sqrt(p[1]^2 + p[2]^2/x^2)
f_aoe_mu(x, p) = p[1] .+ p[2].*x
"""
fit_aoe_corrections(e::Array{<:Unitful.Energy{<:Real}}, μ::Array{<:Real}, σ::Array{<:Real})

Fit the corrections for the AoE value of the detector.
# Returns
- `e`: Energy values
- `μ`: Mean values
- `σ`: Sigma values
- `μ_scs`: Fit result for the mean values
- `f_μ_scs`: Fit function for the mean values
- `σ_scs`: Fit result for the sigma values
- `f_σ_scs`: Fit function for the sigma values
"""
function fit_aoe_corrections(e::Array{<:Unitful.Energy{<:Real}}, μ::Array{<:Real}, σ::Array{<:Real}; aoe_expression::Union{String,Symbol}="a / e", e_expression::Union{String,Symbol}="e")
    # fit compton band mus with linear function
    μ_cut = (mean(μ) - 2*std(μ) .< μ .< mean(μ) + 2*std(μ)) .&& muncert.(μ) .> 0.0
    e, μ, σ = e[μ_cut], μ[μ_cut], σ[μ_cut]
    e_unit = unit(first(e))
    e = ustrip.(e_unit, e)
    
    # fit compton band µ with linear function
    result_µ, report_µ = chi2fit(1, e, µ; uncertainty=true)
    func_µ = "$(mvalue(result_µ.par[1])) + ($e_expression) * $(mvalue(result_µ.par[2]))$e_unit^-1"
    func_generic_µ = "p1 + ($e_expression) * p2" #  "p[1] + ($e_expression) * p[2]"
    par_µ = [result_µ.par[i] ./ e_unit^(i-1) for i=1:length(result_µ.par)] # add units
    result_µ = merge(result_µ, (par = par_µ, func = func_µ, func_generic = func_generic_µ, µ = µ)) 
    report_µ = merge(report_µ, (e_unit = e_unit, label_y = "µ", label_fit = "Best Fit: $(mvalue(round(result_µ.par[1], digits=2))) + E * $(mvalue(round(ustrip(result_µ.par[2]) * 1e6, digits=2)))1e-6"))
    @debug "Compton band µ correction: $(result_µ.func)"

    # fit compton band σ with sqrt function 
    σ_cut = (mean(σ) - std(σ) .< σ .< mean(σ) + std(σ)) .&& muncert.(σ) .> 0.0
    f_fit_σ = f_aoe_sigma # fit function 
    result_σ, report_σ = chi2fit((x, p1, p2) -> f_fit_σ(x,[p1,p2]), e[σ_cut], σ[σ_cut]; uncertainty=true)
    par_σ = [result_σ.par[1], result_σ.par[2] * e_unit^2]
    func_σ = nothing
    func_generic_σ = nothing
    if string(f_fit_σ) == "f_aoe_sigma"
        func_σ = "sqrt( ($(mvalue(result_σ.par[1])))^2 + ($(mvalue(result_σ.par[2]))$(e_unit))^2 / ($e_expression)^2 )" 
        func_generic_σ = "sqrt( (p[1])^2 + (p[2])^2 / ($e_expression)^2 )"
    end
    result_σ = merge(result_σ, (par = par_σ, func = func_σ, func_generic = func_generic_σ, σ = σ))
    report_σ = merge(report_σ, (e_unit = e_unit, label_y = "σ", label_fit = "Best fit: sqrt($(round(mvalue(result_σ.par[1])*1e6, digits=1))e-6 + $(round(ustrip(mvalue(result_σ.par[2])), digits=2)) / E^2)"))
    @debug "Compton band σ normalization: $(result_σ.func)"

    # put everything together into A/E correction/normalization function 
    aoe_str = "($aoe_expression)" # get aoe
    func_aoe_corr = "($aoe_str - ($(result_µ.func)) ) / ($(result_σ.func))"
    func_generic_aoe_corr = "(aoe - $(result_µ.func_generic)) / $(result_σ.func_generic)"

    result = (µ_compton = result_µ, σ_compton = result_σ, compton_bands = (e = e,), func = func_aoe_corr, func_generic = func_generic_aoe_corr)
    report = (report_µ = report_µ, report_σ = report_σ)

    return result, report
end
export fit_aoe_corrections