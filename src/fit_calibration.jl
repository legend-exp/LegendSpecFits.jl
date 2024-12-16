"""
    fit_calibration(pol_order::Int, µ::AbstractVector{<:Union{Unitful.RealOrRealQuantity,Measurement{<:Unitful.RealOrRealQuantity}}}, peaks::AbstractVector{<:Quantity}; e_expression::Union{Symbol, String}="e", m_cal_simple::Union{MaybeWithEnergyUnits, Nothing} = nothing, uncertainty::Bool=true)

Fit the calibration lines with polynomial function of pol_order order
    pol_order == 1 -> linear function
    pol_order == 2 -> quadratic function
    
# Arguments
    * `pol_order`: Polynomial function order
    * `µ`: Mean values 
    * `peaks`: Data peaks

# Keywords
    * `e_expression`: Energy expression
    * `m_cal_simple`: Simple calibration factor
    * `uncertainty`: Boolean that determines if uncertainty is included or not

# Returns
    * Best fit results along with function, function error, mean values and data peaks. 

"""
function fit_calibration(pol_order::Int, µ::AbstractVector{<:Union{Unitful.RealOrRealQuantity,Measurement{<:Unitful.RealOrRealQuantity}}}, peaks::AbstractVector{<:Quantity}; e_expression::Union{Symbol, String}="e", m_cal_simple::Union{MaybeWithEnergyUnits, Nothing} = nothing, uncertainty::Bool=true)
    @assert length(peaks) == length(μ) "Number of calibration points does not match the number of energies"
    @assert pol_order >= 1 "The polynomial order must be greater than 0"

    e_unit = unit(first(peaks))
    # make all inputs unitless with the dimension e_unit
    μ_nounit = if !Unitful.isunitless(unit(first(μ)))
        @warn "µ has a unit, it will be converted to $(e_unit) and stripped."
        ustrip.(e_unit, µ)
    else
        μ
    end
    peaks_nounit = ustrip.(e_unit, peaks)
    c_cal = if isnothing(m_cal_simple)
        c_cal = mvalue(maximum(peaks)/μ_nounit[argmax(peaks)])
        @debug "Use the highest peak as a simple calibration point with slope: $c_cal"
        c_cal
    else
        m_cal_simple
    end
    c = if !Unitful.isunitless(unit(c_cal))
        @debug "m_cal_simple has a unit, it will be converted to $(e_unit) and stripped."
        ustrip.(e_unit, c_cal)
    else
        c_cal
    end
    @debug "Fit calibration curve with $(pol_order)-order polynominal function"
    p_start = append!([0.0, c], fill(0.0, pol_order-1))
    @debug "Initial parameters: $p_start"
    pseudo_prior = get_fit_calibration_pseudo_prior(pol_order, c)
    @debug "Pseudo prior: $pseudo_prior"

    # fit calibration curve
    result_fit, report_fit = chi2fit(pol_order, μ_nounit, peaks_nounit; v_init=p_start, pseudo_prior=pseudo_prior, uncertainty=uncertainty)
    
    # get best fit results
    par =  result_fit.par
    par_unit = par .* e_unit

    result_fit = merge(result_fit, (par = par_unit,))

    # built function in string 
    func = join(["$(mvalue(par[i]))$e_unit * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")
    func_err = join(["($(par[i]))$e_unit * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")
    
    result = merge(result_fit, (func = func, func_err = func_err, µ = μ, peaks = peaks))
    report = merge(report_fit, (e_unit = e_unit, par = result.par, type = :cal))
    return result, report
end
export fit_calibration

"""
    get_fit_calibration_pseudo_prior(pol_order::Int, m_cal_simple::Real)

# Arguments
    * `pol_order`: Polynomial order
    * `m_cal_simple`: Simple calibration factor

    TO DO: function description
"""


function get_fit_calibration_pseudo_prior(pol_order::Int, m_cal_simple::Real)
    unshaped(if pol_order == 0
        NamedTupleDist(
            intercept = Normal(0.0, 0.5/m_cal_simple)
        )
    elseif pol_order == 1
        NamedTupleDist(
            intercept = Normal(0.0, 0.5/m_cal_simple),
            slope = Normal(m_cal_simple, 0.02*m_cal_simple)
        )
    elseif pol_order == 2
        NamedTupleDist(
            intercept = Normal(0.0, 0.5/m_cal_simple),
            slope = Normal(m_cal_simple, 0.02*m_cal_simple),
            quad = Normal(0.0, (0.005*m_cal_simple)^2)
        )
    else
        throw(ArgumentError("Only 0, 1, 2 order polynominal calibration is supported"))
    end)
end