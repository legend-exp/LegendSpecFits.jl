
optim_max_mem::Float64 = 10.0 # in GB for the whole current process ignoring if things could run in parallel
optim_time_limit::Float64 = 20.0 # in seconds a single Optim.jl optimization step

"""
    set_memlimit(gig::Float64)
    Set memory limit in GB for the whole current process ignoring if things could run in parallel
"""
function set_memlimit(gig::Float64)
    @info "Setting `Optimization.jl` memory limit to $gig GB"
    global optim_max_mem=gig
end

"""
    set_timelimit(sec::Float64)
    Set time limit in seconds a single Optim.jl optimization step
"""
function set_timelimit(sec::Float64)
    @info "Setting `Optimization.jl` time limit to $sec seconds"
    global optim_time_limit=sec
end

