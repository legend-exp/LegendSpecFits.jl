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
subhist(h, i::Interval) = subhist(h, (i.left, i.right))
export subhist
