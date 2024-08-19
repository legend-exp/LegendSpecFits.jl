"""
    fit_calibration(n_poly::Int, µ::AbstractVector{<:Union{Real,Measurement{<:Real}}}, peaks::AbstractVector{<:Union{Real,Measurement{<:Real}}}; pull_t::Vector{<:NamedTuple}=fill(NamedTuple(), n_poly+1), v_init::Vector = [], uncertainty::Bool=true )
Fit the calibration lines with polynomial function of n_poly order
    n_poly == 1 -> linear function
    n_poly == 2 -> quadratic function
# Returns
    * `result`: NamedTuple with the following fields
        * `par`: best-fit parameters
        * `gof`: godness of fit
    * `report`: 
"""
function fit_calibration(n_poly::Int, µ::AbstractVector{<:Union{Unitful.RealOrRealQuantity,Measurement{<:Unitful.RealOrRealQuantity}}}, peaks::AbstractVector{<:Quantity}; e_expression::Union{Symbol, String}="e", uncertainty::Bool=true)
    @assert length(peaks) == length(μ)
    e_unit = u"keV"
    if unit(first(μ)) != NoUnits
        @warn "The unit of µ is not $(e_unit), it will be converted to $(e_unit) and stripped."
        µ = ustrip.(e_unit, µ)
    end
    @debug "Fit calibration curve with $(n_poly)-order polynominal function"
    result_fit, report  = chi2fit(n_poly, µ, ustrip.(e_unit, peaks); uncertainty=uncertainty)
    
    par =  result_fit.par
    par_unit = par .* e_unit

    result = merge(result_fit, (par = par_unit,))

    # built function in string 
    func = join(["$(mvalue(par[i]))$e_unit * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")
    func_generic = join(["par[$(i-1)] * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")
    
    result = merge(result, (func = func, func_generic = func_generic, µ = µ, peaks = peaks))
    report = merge(report, (e_unit = e_unit, par = result.par, type = :cal))
    return result, report
end
export fit_calibration