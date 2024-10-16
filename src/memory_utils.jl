
optim_max_mem=10.0 # in GB for the whole current process ignoring if things could run in parallel
optim_time_limit=20.0 # in seconds a single Optim.jl optimization step

"""
    set_memlimit(gig::Real)
Set memory limit in GB for the whole current process ignoring if things could run in parallel

# Arguments
    * 'gig': Gigabyte
"""
function set_memlimit(gig::Real)
    @info "Setting `Optimization.jl` memory limit to $gig GB"
    global optim_max_mem=gig
end

"""
    set_timelimit(sec::Real)
Set time limit in seconds a single Optim.jl optimization step
"""
function set_timelimit(sec::Real)
    @info "Setting `Optimization.jl` time limit to $sec seconds"
    global optim_time_limit=sec
end

"""
    advanced_time_and_memory_control( ; start_time::Float64=time(), start_mem::Float64=Sys.maxrss()/2^30, time_to_setup::Vector{<:Real}=zeros(1), time_limit::Real=-1, mem_limit::Real=-1)

Control function to stop optimization based on time and memory usage.

# Keyword Arguments
    * `start_time`: start time
    * 'start_mem': starting memory storage
    * `time_to_setup`: time to setup
    * `time_limit`: time limit
    * `mem_limit`: memory limit

# Return
- `Bool`: true if optimization should stop

"""
function advanced_time_and_memory_control( ; start_time::Float64=time(), start_mem::Float64=Sys.maxrss()/2^30, time_to_setup::Vector{<:Real}=zeros(1), time_limit::Real=-1, mem_limit::Real=-1)
    if time_limit < 0
        time_limit = optim_time_limit
    end
    if mem_limit < 0
        mem_limit = optim_max_mem
    end
    function callback(x::Optimization.OptimizationState)
        # @debug " * Iteration:       $(x.iteration)"
        so_far =  time() - start_time
        # @debug " * Time so far:     $so_far"
        if x.iter == 0
            time_to_setup .= time() - start_time
            return false
        elseif Sys.maxrss()/2^30 - start_mem > mem_limit
            @warn " * Memory limit reached"
            return true
        else
            expected_next_time = so_far + (time() - start_time - time_to_setup[1])/(x.iter)
            # @debug " * Next iteration ≈ $expected_next_time"
            # @debug " * Time limit:      $time_limit"
            # @debug " * Start limit:     $start_time"
            # @debug " * Time to setup:   $time_to_setup"
            if expected_next_time > time_limit 
                @warn " * Time limit reached"
                return true
            else
                return false
            end
        end
    end
end