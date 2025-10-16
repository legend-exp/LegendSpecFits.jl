# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using ArraysOfArrays

"""
    struct ParGridBinnedMCResult{M,N,parnames}

Represents the result of a binned Monte Carlo simulation on a parameter grid.

Example:

```julia
data = ParGridBinnedMCResult(
    (a = 1:0.5:5, b = -3:1:3),
    (0:0.1:100, 0:0.25:200),
    ArrayOfSimilarArrays{Int,2,2}(rand(0:999, 1001, 801, 9, 7))
)

p = (a = 2.3, b = 0.7)
interpolate_counts(data, p)
````
"""
struct ParGridBinnedMCResult{
	M,N,parnames,
	PG<:NamedTuple{parnames,<:Tuple{Vararg{AbstractVector{<:Real},N}}},
	BE<:Tuple{Vararg{AbstractVector{<:Real},M}},
	Cts<:AbstractArray{<:AbstractArray{<:Real,M},N}
}
    par_grid::PG
	bin_edges::BE
	bin_counts::Cts
end
export ParGridBinnedMCResult


# ToDo: Add ctor from Histograms.


"""
    LegendSpecFits.interpolate_counts(data::ParGridBinnedMCResult{parnames}))
    LegendSpecFits.interpolate_counts(data::ParGridBinnedMCResult{parnames}, par_point::NamedTuple{parnames})

Interpolate the binned counts in `data` at the given parameter point
`par_point` via (multi-)linear interpolation.

Curries when called with only `data`:

```julia
f = interpolate_counts(data)
f(par_point) == interpolate_counts(data, par_point)
```
"""
function interpolate_counts end

interpolate_counts(data::ParGridBinnedMCResult{parnames}) = Base.Fix1(interpolate_counts, data)

function interpolate_counts(data::ParGridBinnedMCResult{parnames}, par_point::NamedTuple{parnames}) where parnames
    # ToDo: Handle out-of-bounds points (throw error, no extrapolation).

	(;par_grid, bin_counts) = data
	grid = values(par_grid)>
	p = values(par_point)
	idxs_lower = map(searchsortedlast, grid, p)
	idxs_upper = map(searchsortedfirst, grid, p)
	p_lower = map(getindex, grid, idxs_lower)
	p_upper = map(getindex, grid, idxs_upper)
	w_lower = map(_interpolation_weight_l, p, p_lower, p_upper)
	w_upper = one(eltype(w_lower)) .- w_lower

    # ToDo ...
end

function _interpolation_weight_l(x, l, h)
	@assert l <= x <= h
	tmp_w = (x - l) / (h - l)
	R = typeof(tmp_w)
	w = ifelse(l ≈ h, one(R), tmp_w)
	@assert zero(R) <= w <= one(R)
	return w
end
