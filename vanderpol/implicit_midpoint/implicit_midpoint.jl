# define imports
using ArgParse;
using DelimitedFiles;
using LinearAlgebra;
using Plots;
using Quadmath;
using TimerOutputs;

# define arguments and usage
const ValidDatatypes = Dict("Float128" => Float128, "Float64" => Float64, "Float32" => Float32);
const NBS = '\ua0';

function isValidDatatype(datatype::String)
    return haskey(ValidDatatypes, datatype)
end

function ArgParse.parse_item(::Type{Union{Int, Array{Int}}}, x::AbstractString)
    return reshape(readdlm(IOBuffer(x), ',', Int), :)
end

s = ArgParseSettings(description = "example$(NBS)usage:$(NBS)julia$(NBS)$PROGRAM_FILE$(NBS)Float64$(NBS)Float32$(NBS)100000$(NBS)1.0")
@add_arg_table s begin
    "FullType"
        help = "The data type used for full precision calculations."
        arg_type = String
        range_tester = isValidDatatype
        required = true
    "ReduType"
        help = "The data type used for lower (mixed) precision calculations."
        arg_type = String
        range_tester = isValidDatatype
        required = true
    "num_steps"
        help = "The number of time steps to evaluate at between the starting and ending time. This can be a scalar or an integer array delimited by commas without any spaces (i.e. '1,10,100')."
        arg_type = Union{Int, Array{Int}}
        required = true
    "alpha"
        help = "Value of alpha to be used in the vanderpol equation."
        arg_type = Float64
        required = true
    "--time", "-t"
        help = "The amount of time to calculate."
        arg_type = Float64
        default = 1.0
    "--output", "-o"
        help = "The file name to save output values to."
        arg_type = String
    "--plot", "-p"
        help = "The file name to save output plots to."
        arg_type = String
end

# parse the arguments
parsed_args = parse_args(ARGS, s)

# define float constants
const FullFloat = get(ValidDatatypes, get(parsed_args, "FullType", "Float64"), Float64);
const ReduFloat = get(ValidDatatypes, get(parsed_args, "ReduType", "Float32"), Float32);


# function definitions

"""
    updateF!(F, y, u, dt, alpha)

Calculate and update the values in F.
"""
function updateF!(F::Array{ReduFloat}, y::Array{ReduFloat}, u::Array{ReduFloat}, dt::ReduFloat, alpha::ReduFloat)::Array{ReduFloat}
    # calculate values of F
    F[1] = u[1] + dt * (y[2]) - y[1]
    F[2] = u[2] + dt * (alpha * (ReduFloat(1.0) - (y[1]^2)) * y[2] - y[1]) - y[2]
    
    # return F for completeness
    return F
end

"""
    updateJ!(J, y, dt, alpha)

Calculate and update the values in J, the Jacobian matrix.
"""
function updateJ!(J::Matrix{ReduFloat}, y::Array{ReduFloat}, dt::ReduFloat, alpha::ReduFloat)::Matrix{ReduFloat}
    # calculate values of J
    J[1, 1] = ReduFloat(-1.0)
    J[1, 2] = dt
    J[2, 1] = -dt * (2 * alpha * y[1] * y[2] + ReduFloat(1.0))
    J[2, 2] = dt * alpha * (ReduFloat(1.0) - (y[1]^2)) - ReduFloat(1.0)
    
    # return J for completeness
    return J
end

"""
    newtonsMethod!(F!, J!, x_0, F, J, [tol, max_iters])

Performs the iterative Newton's Method to solve the system of
equations defined by y[1] and y[2], with F! and J! as the
update functions for F and J, respectively.
"""
function newtonsMethod!(F!::Function, J!::Function, x_0::Array{ReduFloat}, F::Array{ReduFloat}, J::Matrix{ReduFloat}, tol::ReduFloat = ReduFloat(5.0) * eps(ReduFloat), max_iters::Integer = 20)::Array{ReduFloat}
    # TODO: Why is the tolerance being set to 5 * epsilon? Was this supposed to be smaller or is 5 correct?

    # loop until max_iters has been reached
    for i = 1:max_iters
        # update the values of F
        F!(F, x_0)

        # break if tolerance reached
        if norm(F) < tol
            break
        end

        # update the values of J
        J!(J, x_0)

        # update guess for x_0, with J \ F more efficient than J inverse x F
        x_0 = x_0 - (J \ F)
    end

    # return x_0 for completeness
    return x_0
end

"""
    driver(F, J, u, num_steps, time_start, time_end, alpha)

Performs the main calculations over num_steps time steps,
using the provided input and initial values.
"""
function driver(F::Array{ReduFloat}, J::Matrix{ReduFloat}, u::Array{FullFloat}, num_steps::Integer, time_start::FullFloat, time_end::FullFloat, alpha::FullFloat)::Array{FullFloat}
    # calculate dt
    dt = (time_end - time_start) / num_steps

    # track total time
    time_total = time_start

    # define Jacobian now since it doesn't change with each time step
    J!(J::Matrix{ReduFloat}, y::Array{ReduFloat}) = updateJ!(J, y, ReduFloat(0.5) * ReduFloat(dt), ReduFloat(alpha))

    # store u values for plotting
    u_vals = zeros(FullFloat, num_steps, 2)

    # iterate over each time step
    for i = 1:num_steps
        # ensure that the final time is reached
        if i == num_steps
            dt = time_end - time_total
        end

        # implicit step
        # convert u to reduced precision
        redu_u = convert(Array{ReduFloat}, u)

        # define F each time step since it changes
        F!(F::Array{ReduFloat}, y::Array{ReduFloat}) = updateF!(F, y, redu_u, ReduFloat(0.5) * ReduFloat(dt), ReduFloat(alpha))

        # solve the system with newtons method and convert to full precision
        redu_y = newtonsMethod!(F!, J!, redu_u, F, J)
        full_y = convert(Array{FullFloat}, redu_y)

        # explicit step
        # calculate u
        u[1] = u[1] + dt * full_y[2]
        u[2] = u[2] + dt * (alpha * (FullFloat(1.0) - (full_y[1]^2)) * full_y[2] - full_y[1])

        # increment total time
        time_total += dt

        # TODO: save values of u for plotting
        u_vals[i, :] = u
    end

    # add values to plot if plotting is enabled
    plot!(u_vals[:, 1], u_vals[:, 2])

    # return u
    return u
end

# main code
# constants
const num_steps_arr = convert(Array{Int}, get(parsed_args, "num_steps", [100000]))
const alpha = FullFloat(get(parsed_args, "alpha", 1.0))
const time_start = FullFloat(0.0)
const time_end = FullFloat(get(parsed_args, "time", 1.0))

# store all final u values
u_finals = zeros(FullFloat, length(num_steps_arr), 2)

# initialize plot if enabled
plot_file = get(parsed_args, "plot", Nothing)
if !isnothing(plot_file) plot() end

# loop through each num_steps
for (i, num_steps) in enumerate(num_steps_arr)
    # variables
    F = zeros(ReduFloat, 2)
    J = zeros(ReduFloat, 2, 2)
    init_u = [FullFloat(2.0), FullFloat(0.0)]

    # call the driver function
    u = driver(F, J, init_u, num_steps, time_start, time_end, alpha);

    # store final value of u
    u_finals[i, :] = u

    # print final value of u
    println(if length(num_steps_arr) > 1 "$num_steps: " else "" end * string(u[1]) * "," * string(u[2]))
end

# save u final values with corresponding num_steps to a file if option has been used
output_file = get(parsed_args, "output", Nothing)
if !isnothing(output_file) writedlm(output_file, cat(num_steps_arr, u_finals; dims = 2)) end

# save plot to file if enabled
if !isnothing(plot_file) savefig(plot_file) end