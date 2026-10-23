
optim_max_mem::Float64 = 10.0 # in GB for the whole current process ignoring if things could run in parallel
optim_time_limit::Float64 = 20.0 # in seconds a single Optim.jl optimization step

"""
    set_memlimit(gig::Float64)

Sets the module-level memory limit, in GB, for the whole current Julia process,
regardless of how many fits run in parallel. The default is 10 GB. The value
is stored in `LegendSpecFits.optim_max_mem`.
"""
function set_memlimit(gig::Float64)
    @info "Setting `Optimization.jl` memory limit to $gig GB"
    global optim_max_mem=gig
end

"""
    set_timelimit(sec::Float64)

Sets the time limit, in seconds, for a single optimization step that uses
`LegendSpecFits.optim_time_limit` as `maxtime` (the NLopt-based fits in
[`ctc_energy`](@ref) and [`ctc_aoe`](@ref)). The default is 20 s.
"""
function set_timelimit(sec::Float64)
    @info "Setting `Optimization.jl` time limit to $sec seconds"
    global optim_time_limit=sec
end

