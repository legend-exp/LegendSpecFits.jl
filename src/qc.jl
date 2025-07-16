"""
    qc_window_cut(tab::Table, config::PropDict, cols::NTuple{<:Any, Symbol})

Perform quality control on the data using a window cut based on a centered Gaussian fit.
The `config` is a `PropDict` containing the configuration for each column to be checked. Each column should have a `min`, `max`, and `sigma` value, along with optional keyword arguments for the cut.
# Arguments
    * `data`: a `Table` containing the data to be checked
    * `config`: a `PropDict` containing the configuration for each column to be checked
    * `cols`: a tuple of column names to be checked
# Returns
    * `result`: a `NamedTuple` containing the results of the quality control, with a `qc` field that is a string describing the cut condition for each column
    * `report`: an `OrderedDict` containing the reports for each column which can be plotted, with fields like `f_fit`, `h`, `μ`, `σ`, `gof`, `low_cut`, and `high_cut`
"""
function qc_window_cut end
export qc_window_cut

function qc_window_cut(tab::Table, config::PropDict, cols::NTuple{<:Any, Symbol})
    @assert all(hasproperty.(Ref(tab), cols)) "Not all columns found in the data"
    @assert all(hasproperty.(Ref(config), cols)) "Not all columns found in the config"

    # create return and result dicts
    v_result = Vector{NamedTuple}(undef, length(cols))
    v_report = Vector{NamedTuple}(undef, length(cols))

    v_qc_strings = Vector{String}(undef, length(cols))

    Threads.@threads for i in eachindex(cols)
        col = cols[i]
        result_col, report_col = qc_window_cut(getproperty(tab, col), config[col].min, config[col].max, config[col].sigma, ; col_expression=col, NamedTuple(config[col].kwargs)...)
        v_result[i] = result_col
        v_report[i] = report_col
        @debug "$col cut surrival fraction: $(round(mean(result_col.low_cut .< getproperty(tab, col) .< result_col.high_cut) * 100, digits=2))%"
    end

    result = merge((qc = join(getproperty.(v_result, :qc), " && "),), NamedTuple{cols}(v_result))
    report = OrderedDict{Symbol, NamedTuple}(cols .=> v_report)

    return result, report
end

function qc_window_cut(x::Vector{<:Unitful.RealOrRealQuantity}, min::T, max::T, sigma::Real; col_expression::Union{Symbol, String}="col", kwargs...) where T<:Unitful.RealOrRealQuantity
    # generate window cut
    result, report = get_centered_gaussian_window_cut(x, min, max, sigma; kwargs...)
    
    @assert unit(result.low_cut) == unit(result.high_cut) "The units of the low and high cuts must be the same"
    
    # QC string to be used as a PropertyFunction
    qc_string = "$(mvalue(ustrip(result.low_cut)))$(unit(result.low_cut) == NoUnits ? "" : string(unit(result.low_cut))) < $col_expression && $col_expression < $(mvalue(ustrip(result.high_cut)))$(unit(result.high_cut) == NoUnits ? "" : string(unit(result.high_cut)))"
    
    return merge((qc = qc_string,), result), report
end