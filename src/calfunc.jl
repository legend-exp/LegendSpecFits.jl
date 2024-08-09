# This file is a part of LegendDataManagement.jl, licensed under the MIT License (MIT).


export PolCalFunc

struct PolCalFunc{N,T<:Number} <:Function
    params::NTuple{N,T}

    PolCalFunc(params::T...) where T = new{length(params),T}(params)
end


function (f::PolCalFunc{N,T})(x::U) where {N,T,U}
    R = promote_type(T, U)
    y = zero(R)
    xn = one(U)
    @inbounds for p in f.params
        y = R(fma(p, xn, y))
        xn *= x
    end
    y
end


function Base.convert(::Type{PolCalFunc{N,T}}, p::PropDict) where {N,T}
    funcstr = p.func
    if !(funcstr in ("pol1", "[0]+[1]*x", "pol2"))
        error("Unsupported function \"$funcstr\" for $(PolCalFunc{N,T})")
    end

    coeffs = zeros(T, N)
    for (i,v) in p.params
        coeffs[i + 1] = v
    end
    PolCalFunc(coeffs...)
end


function Base.convert(::Type{Dict{Int, PolCalFunc{N,T}}}, p::PropDict) where {N,T}
    calfuncs = Dict{Int, PolCalFunc{N,T}}()
    for (ch, caldict) in p
        calfuncs[ch] = convert(PolCalFunc{2,Float64}, caldict)
    end
    calfuncs
end


const CalFuncDict = Dict{Int,PolCalFunc{2,Float64}}
export CalFuncDict
