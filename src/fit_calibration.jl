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
function fit_calibration(n_poly::Int, µ::AbstractVector{<:Union{Real,Measurement{<:Real}}}, peaks::AbstractVector{<:Quantity};e_type::Union{Symbol,String}="x")
    @assert length(peaks) == length(μ)
    e_unit = u"keV"
    @debug "Fit calibration curve with $(n_poly)-order polynominal function"
    result_fit, report  = chi2fit(n_poly, µ, ustrip.(e_unit,peaks))
    
    par =  result_fit.par
    par_unit = par .* e_unit 
   
    result = merge(result_fit, (par = par_unit,))

    # built function in string 
    func = join(["$(mvalue(par[i])) * $(e_type)^$(i-1)" for i=1:length(par)], " + ")
    func_generic = join(["p$(i-1) * $(e_type)^$(i-1)" for i=1:length(par)], " + ")

    result = merge(result, (func = func, func_generic = func_generic))
    return result, report
end
export fit_ecal