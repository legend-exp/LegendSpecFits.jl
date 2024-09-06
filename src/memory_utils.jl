# """
#     advanced_time_and_memory_control(x::Optim.OptimizationState, start_time::Float64, time_to_setup::Float64; time_limit::Float64=60.0, mem_limit::Float64=30.0)

#     Control function to stop optimization based on time and memory usage.
# # Arguments
# - `x::Optim.OptimizationState`: optimization state
# - `start_time::Float64`: start time
# - `time_to_setup::Float64`: time to setup
# - `time_limit::Float64`: time limit
# - `mem_limit::Float64`: memory limit

# # Return
# - `Bool`: true if optimization should stop
# """
# function advanced_time_and_memory_control(x::Optim.OptimizationState, start_time::Float64, time_to_setup::Vector{<:Real}; time_limit::Real=60.0, mem_limit::Real=30.0)
#     # @debug " * Iteration:       $(x.iteration)"
#     so_far =  time()-start_time
#     # @debug " * Time so far:     $so_far"
#     if x.iteration == 0
#         time_to_setup .= time() - start_time
#     elseif Sys.maxrss()/2^30 > mem_limit
#         @warn " * Memory limit reached"
#         return true
#     else
#         expected_next_time = so_far + (time() - start_time - time_to_setup[1])/(x.iteration)
#         # @debug " * Next iteration ≈ $expected_next_time"
#         if expected_next_time > time_limit @warn " * Time limit reached" end
#         return expected_next_time < time_limit ? false : true
#     end
#     return false
# end


"""
    advanced_time_and_memory_control(x::Optim.OptimizationState, start_time::Float64, time_to_setup::Float64; time_limit::Float64=60.0, mem_limit::Float64=30.0)

    Control function to stop optimization based on time and memory usage.
# Arguments
- `x::Optim.OptimizationState`: optimization state
- `start_time::Float64`: start time
- `time_to_setup::Float64`: time to setup
- `time_limit::Float64`: time limit
- `mem_limit::Float64`: memory limit

# Return
- `Bool`: true if optimization should stop
"""
function advanced_time_and_memory_control( ; start_time::Float64=time(), time_to_setup::Vector{<:Real}=zeros(1), time_limit::Real=60.0, mem_limit::Real=30.0)
    function callback(x::Optim.OptimizationState)
        @debug " * Iteration:       $(x.iteration)"
        so_far =  time() - start_time
        @debug " * Time so far:     $so_far"
        if x.iteration == 0
            time_to_setup .= time() - start_time
        elseif Sys.maxrss()/2^30 > mem_limit
            @warn " * Memory limit reached"
            return true
        else
            expected_next_time = so_far + (time() - start_time - time_to_setup[1])/(x.iteration)
            @debug " * Next iteration ≈ $expected_next_time"
            @debug " * Time limit:      $time_limit"
            @debug " * Time to setup:   $time_to_setup"
            if expected_next_time > time_limit @warn " * Time limit reached" end
            return expected_next_time < time_limit ? false : true
        end
        return false
    end
end