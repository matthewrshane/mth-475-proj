# define imports
using ArgParse;
using DelimitedFiles;
using FFTW;
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

function ArgParse.parse_item(::Type{Expr}, x::AbstractString)
    return Meta.parse(x)
end

function ArgParse.parse_item(::Type{Float64}, x::AbstractString)
    return Float64(eval(Meta.parse(x)))
end

s = ArgParseSettings(description = "example$(NBS)usage:$(NBS)julia$(NBS)$PROGRAM_FILE$(NBS)Float64$(NBS)Float32$(NBS)100$(NBS)181$(NBS)-i$(NBS)0$(NBS)-a$(NBS)2pi$(NBS)-u$(NBS)\"sin.(x)\"")
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
    "num_x_pts"
        help = "The number of x-values in space to discretize the problem into. Must be odd."
        arg_type = Int
        range_tester = isodd
        required = true
    "--xmin", "-i"
        help = "Minimum x-value in space."
        arg_type = Float64
        default = 0.0
    "--xmax", "-a"
        help = "Maximum x-value in space."
        arg_type = Float64
        default = 2pi
    "--ufunc", "-u"
        help = "The expression used to calculate the initial u-values, written as a broadcastable function of x. Must evaluate to a valid Julia expression, wrapped in quotes."
        arg_type = Expr
        default = :(sin.(x))
    "--time", "-t"
        help = "The amount of time to calculate."
        arg_type = Float64
        default = 1.75
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
    calcF(y, u, Dx, dt)

Calculate the values of F as a function of y.
"""
function calculateF(y::Array{ReduFloat}, u::Array{ReduFloat}, Dx::Matrix{ReduFloat}, dt::ReduFloat)::Array{ReduFloat}
    # calculate value of u
    return y - u + (dt * Dx * (0.5 * y.^2))
end

"""
    calcJ(J, y, dt, alpha)

Calculate the values of J, the Jacobian matrix.
"""
function calculateJ(u::Array{ReduFloat}, Dx::Matrix{ReduFloat}, dt::ReduFloat)::Matrix{ReduFloat}
    # calculate value of J
    return I + (dt * Dx * diagm(u))
end

"""
    newtonsMethod!(calcF, calcJ, u, [tol, max_iters])

Performs the iterative Newton's Method to solve the system of
equations defined by calcF with Jacobian calcJ, using an initial
guess of u.
"""
function newtonsMethod!(calcF::Function, calcJ::Function, u::Array{ReduFloat}, tol::ReduFloat = ReduFloat(5.0) * eps(ReduFloat), max_iters::Integer = 20)::Array{ReduFloat}
    # loop until max_iters has been reached
    for i = 1:max_iters
        # calculate the values of F
        F = calcF(u)

        # break if tolerance reached
        if norm(F, Inf) < tol
            break
        end

        # calculate the values of J
        J = calcJ(u)

        # update guess for u
        u = u - (J \ F)
    end

    # return u for completeness
    return u
end

"""
    driver(u, Dx, num_steps, x, time_start, time_end)

Performs the main calculations over num_steps time steps,
using the provided input and initial values.
"""
function driver(u::Array{FullFloat}, Dx::Matrix{ReduFloat}, num_steps::Integer, x::Array{FullFloat}, time_start::FullFloat, time_end::FullFloat)::Array{FullFloat}
    # calculate dt
    dt = (time_end - time_start) / num_steps

    # track total time
    time_total = time_start

    # define Jacobian now since it doesn't change with each time step
    calcJ(u::Array{ReduFloat}) = calculateJ(u, Dx, ReduFloat(dt))

    # iterate over each time step
    for i = 1:num_steps
        # ensure that the final time is reached
        if i == num_steps
            dt = time_end - time_total
        end

        # implicit step
        # convert u to reduced precision
        redu_u = convert(Array{ReduFloat}, u)

        # define calcU each time step since it changes
        calcF(y::Array{ReduFloat}) = calculateF(y, redu_u, Dx, ReduFloat(dt))

        # solve the system with newtons method and convert to full precision
        redu_y = newtonsMethod!(calcF, calcJ, redu_u, ReduFloat(1.e-14), 1)
        full_y = convert(Array{FullFloat}, redu_y)

        # TODO: full precision broyden's style update to correct Newton's iter
        u = full_y

        # increment total time
        time_total += dt
    end

    # TODO: just plot the final time step
    plot!(x, u, label="u (num_steps=$num_steps)")

    # return u
    return u
end

# main code
# constants
const num_steps_arr = convert(Array{Int}, get(parsed_args, "num_steps", [100]))
const num_x_pts = get(parsed_args, "num_x_pts", 181)
const xmin = FullFloat(get(parsed_args, "xmin", 0.0))
const xmax = FullFloat(get(parsed_args, "xmax", 2pi))
const ufunc = get(parsed_args, "ufunc", :(sin.(x)))
const time_start = FullFloat(0.0)
const time_end = FullFloat(get(parsed_args, "time", 1.75))

# calculate x points and initial u values
x = collect(range(xmin, xmax, length=num_x_pts + 1))
pop!(x)
u_init = eval(ufunc)

# calculate fourier matrices
K_size = (num_x_pts - 1) / 2
K = diagm(-K_size:K_size)
Dx = ifft(ifftshift(im * K * fftshift(fft(I(num_x_pts)))))
Dx = real.(Dx)
Dx = convert(Matrix{ReduFloat}, Dx)

# store all final u values
u_finals = zeros(FullFloat, length(num_steps_arr), num_x_pts)

# initialize plot if enabled
plot_file = get(parsed_args, "plot", Nothing)
if !isnothing(plot_file)
    plot(x, u_init, title="Burger's Equation w/ Newton's (t = $time_end)", label=string(ufunc))
end

# loop through each num_steps
for (i, num_steps) in enumerate(num_steps_arr)
    # call the driver function
    u = driver(u_init, Dx, num_steps, x, time_start, time_end);

    # store final value of u
    u_finals[i, :] = u

    # print final value of u
    println(if length(num_steps_arr) > 1 "$num_steps: " else "" end * string(u))
end

# save u final values with corresponding num_steps to a file if option has been used
output_file = get(parsed_args, "output", Nothing)
if !isnothing(output_file) writedlm(output_file, cat(num_steps_arr, u_finals; dims = 2)) end

# save plot to file if enabled
if !isnothing(plot_file) savefig(plot_file) end