# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    expand_vars(v::NamedTuple)::StructArray

Expand all fields in `v` (scalars or arrays) to same array size and return
a `StructArray`.
"""
function expand_vars(v::NamedTuple)
    sz = Base.Broadcast.broadcast_shape((1,), map(size, values(v))...)
    _expand(x::Real) = Fill(x, sz)
    _expand(x::AbstractArray) = broadcast((a,b) -> b, Fill(nothing, sz), x)
    StructArray(map(_expand, v))
end
export expand_vars

"""
    subhist(h::Histogram, r::Tuple{<:Real,<:Real})

Return a new `Histogram` with the bins in the range `r`.
"""
function subhist(h::Histogram{<:Any, 1}, r::Tuple{<:Real,<:Real})
    first_bin, last_bin = (StatsBase.binindex(h, r[1]), StatsBase.binindex(h, r[2]))
    if first_bin < 1 first_bin = 1 end
    if (last_bin > length(h.weights)) last_bin = length(h.weights) end
    Histogram(h.edges[1][first_bin:last_bin+1], h.weights[first_bin:last_bin])
end
function subhist(x::Array{<:Real})
    bin_width = 2 * (quantile(x, 0.75) - quantile(x, 0.25)) / ∛(length(x))
    fit(Histogram, x, minimum(x):bin_width:maximum(x))
end
subhist(x::Array, r::Tuple{<:Real,<:Real}) = subhist(x[r[1] .< x .< r[2]])
subhist(h, i::Interval) = subhist(h, (i.left, i.right))
export subhist


"""
    get_distribution_transform(d::Distribution, pprior::Prior)

Return a `DistributionTransform` for the given `Distribution` and `Prior`.
"""
function get_distribution_transform end


"""
    tuple_to_array(nt::NamedTuple, fields::Vector{Symbol})

Return an array with the values of the fields in `nt` in the order given by
`fields`.
"""
function tuple_to_array(nt::NamedTuple)
    [nt[f] for f in fieldnames(nt)]
end


"""
    array_to_tuple(a::AbstractArray, as_nt::NamedTuple)

Return a `NamedTuple` with the values of `a` in the order given by
`fieldnames(as_nt)`.
"""
function array_to_tuple(a::AbstractArray, as_nt::NamedTuple)
    NamedTuple{fieldnames(as_nt)}(a)
end


""" 
    get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Int64)

Return a `NamedTuple` with the same fields as `v` and `v_err` but with
`Normal` distributions for each field.
"""
function get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Int64)
    vs = BAT.distprod(map(Normal, v, v_err))
    NamedTuple.(rand(vs, n))
end