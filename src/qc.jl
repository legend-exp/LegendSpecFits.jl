"""
    qc_window_cut(tab::Table, config::AbstractDict)
    qc_window_cut(x::Vector{<:Unitful.RealOrRealQuantity}, min::T, max::T, sigma::Real; col_expression = "col", kwargs...)

Determines quality-control window cuts on one or more columns, each from a
centered Gaussian fit with [`get_centered_gaussian_window_cut`](@ref).

For a table, `config` maps each column name (a `Symbol`) to a dictionary with
the entries `:min`, `:max` (range of the data used), `:sigma` (half-width of the
window in standard deviations) and `:kwargs` (keyword arguments of
[`get_centered_gaussian_window_cut`](@ref) as a dictionary or `NamedTuple`).
The columns are processed in parallel on the available threads. For a vector
`x`, these values are passed directly and `kwargs` are forwarded.

## Keywords
* `col_expression`: Name of the variable in the returned cut string (vector method).

## Returns
A tuple `(result, report)`.
* Table method: `result` is a `NamedTuple` with the field `qc`, a string of the
  conditions of all columns joined by ` && `, and one entry per column holding its result;
  `report` is an `OrderedDict` keyed by column name.
* Vector method: `result` is the result of [`get_centered_gaussian_window_cut`](@ref) extended by `qc`, the cut condition string
  `<low_cut> < col_expression && col_expression < <high_cut>` (with units if present);
  `report` is its report (`f_fit`, `h`, `μ`, `σ`, `gof`, `low_cut`, `high_cut`).
"""
function qc_window_cut end
export qc_window_cut

function qc_window_cut(tab::Table, config::AbstractDict)
    cols = collect(keys(config))

    @assert all([typeof(col) == Symbol && col in propertynames(tab) for col in cols]) "All columns specified in the config must be present in the table"

    # create return and result dicts
    v_result = Vector{NamedTuple}(undef, length(cols))
    v_report = Vector{NamedTuple}(undef, length(cols))

    v_qc_strings = Vector{String}(undef, length(cols))

    Threads.@threads for i in eachindex(cols)
        col = cols[i]
        col_config = get(config, col, nothing)
        result_col, report_col = qc_window_cut(getproperty(tab, col), col_config[:min], col_config[:max], col_config[:sigma], ; col_expression=Symbol(col), NamedTuple(col_config[:kwargs])...)
        v_result[i] = result_col
        v_report[i] = report_col
        @debug "$col cut surrival fraction: $(round(mean(result_col.low_cut .< getproperty(tab, col) .< result_col.high_cut) * 100, digits=2))%"
    end

    result = merge((qc = join(getproperty.(v_result, :qc), " && "),), NamedTuple{Tuple(cols)}(v_result))
    report = OrderedDict{Symbol, NamedTuple}(cols .=> v_report)

    return result, report
end

function qc_window_cut(x::Vector{<:Unitful.RealOrRealQuantity}, min::T, max::T, sigma::Real; col_expression::Union{Symbol, String}="col", kwargs...) where T<:Unitful.RealOrRealQuantity
    # generate window cut
    result, report = get_centered_gaussian_window_cut(x, min, max, sigma; kwargs...)
    
    @assert unit(result.low_cut) == unit(result.high_cut) "The units of the low and high cuts must be the same"
    
    # QC string to be used as a PropertyFunction
    low_cut_value = mvalue(ustrip(result.low_cut))
    low_cut_unit = unit(result.low_cut) == NoUnits ? "" : string(unit(result.low_cut))
    high_cut_value = mvalue(ustrip(result.high_cut))
    high_cut_unit = unit(result.high_cut) == NoUnits ? "" : string(unit(result.high_cut))

    qc_string = "$low_cut_value$low_cut_unit < $col_expression && $col_expression < $high_cut_value$high_cut_unit"
    
    return merge((qc = qc_string,), result), report
end