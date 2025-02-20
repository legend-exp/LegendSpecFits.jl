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
    σ_cut = (mean(σ) - std(σ) .< σ .< mean(σ) + std(σ)) .&& muncert.(σ) .> 0.0
    e_unit = unit(first(e))
    e = ustrip.(e_unit, e)
    
    # start values
    p_start = mvalue.([median(μ), 0.0])
    # fit compton band µ with linear function
    result_µ, report_µ = chi2fit(1, e, µ; v_init=p_start, uncertainty=true)
    func_µ = "$(mvalue(result_µ.par[1])) + ($e_expression) * $(mvalue(result_µ.par[2]))$e_unit^-1"
    par_µ = [result_µ.par[i] ./ e_unit^(i-1) for i=1:length(result_µ.par)] # add units
    result_µ = merge(result_µ, (par = par_µ, func = func_µ, µ = µ)) 
    report_µ = merge(report_µ, (e_unit = e_unit, label_y = "µ", label_fit = latexstring("Best Fit: \$ $(mvalue(round(result_µ.par[1], digits=2))) $(mvalue(ustrip(result_µ.par[2])) >= 0 ? "+" : "") $(mvalue(round(ustrip(result_µ.par[2]) * 1e6, digits=2)))\\cdot 10^{-6} \\; E \$")))
    @debug "Compton band µ correction: $(result_µ.func)"

    # fit compton band σ with sqrt function
    result_σ, report_σ = chi2fit((x, p1, p2) -> f_aoe_sigma(x,[p1,p2]), e[σ_cut], σ[σ_cut]; uncertainty=true)
    par_σ = [result_σ.par[1], result_σ.par[2] * e_unit^2]
    func_σ = "sqrt( ($(mvalue(result_σ.par[1])))^2 + ($(mvalue(result_σ.par[2]))$(e_unit))^2 / ($e_expression)^2 )" 

    result_σ = merge(result_σ, (par = par_σ, func = func_σ, σ = σ))
    report_σ = merge(report_σ, (e_unit = e_unit, label_y = "σ", label_fit = latexstring("Best Fit: \$\\sqrt{($(abs(round(mvalue(result_σ.par[1])*1e3, digits=2)))\\cdot10^{-3})^2 + $(abs(round(ustrip(mvalue(result_σ.par[2])), digits=2)))^2 / E^2}\$")))
    #report_σ = merge(report_σ, (e_unit = e_unit, label_y = "σ", label_fit = "Best Fit: sqrt(($(round(mvalue(result_σ.par[1])*1e3, digits=2))e-3)^2 + $(round(ustrip(mvalue(result_σ.par[2])), digits=2))^2 / E^2)"))
    @debug "Compton band σ normalization: $(result_σ.func)"

    # put everything together into A/E correction/normalization function 
    aoe_str = "($aoe_expression)" # get aoe
    func_aoe_corr = "($aoe_str - ($(result_µ.func)) ) / ($(result_σ.func))"

    result = (µ_compton = result_µ, σ_compton = result_σ, compton_bands = (e = e,), func = func_aoe_corr)
    report = (report_µ = report_µ, report_σ = report_σ)

    return result, report
end
export fit_aoe_corrections
