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
