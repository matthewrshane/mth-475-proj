# define imports
using ArgParse;
using DelimitedFiles;
using FFTW;
using LinearAlgebra;
using Plots;
using Printf;
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
function calculateF!(F::Array{ReduFloat}, y::Array{ReduFloat}, u::Array{ReduFloat}, Dx::Matrix{ReduFloat}, dt::ReduFloat, y2::Array{ReduFloat}, dtDx::Matrix{ReduFloat})
    # calculate value of F
    y2 .= y
    y2 .^= 2
    y2 .*= ReduFloat(0.5)
    # mul!(dtDx, dt, Dx)
    mul!(F, Dx, y2)
    F .*= -dt
    F .+= u
    F .-= y

    # F .= u - dt * Dx * (0.5 * y.^2) - y
end

"""
    calcJ(J, y, dt, alpha)

Calculate the values of J, the Jacobian matrix.
"""
function calculateJ!(J::Matrix{ReduFloat}, u::Array{ReduFloat}, Dx::Matrix{ReduFloat}, dt::ReduFloat)
    # calculate value of J
    mul!(J, Dx, Diagonal(u))
    J .*= -dt
    J .-= I(length(u))

    # J .= -(dt * Dx * Diagonal(u)) - I
end

"""
    newtonsMethod!(calcF, calcJ, u, [tol, max_iters])

Performs the iterative Newton's Method to solve the system of
equations defined by calcF with Jacobian calcJ, using an initial
guess of u.
"""
function newtonsMethod!(calcF!::Function, calcJ!::Function, F::Array{ReduFloat}, J::Matrix{ReduFloat}, u::Array{ReduFloat}, ut::Array{ReduFloat}; tol::ReduFloat = ReduFloat(5.0) * eps(ReduFloat), max_iters::Integer = 20)::Array{ReduFloat}
    # loop until max_iters has been reached
    for i = 1:max_iters
        # calculate the values of F
        @timeit to "F" calcF!(F, u)

        # break if tolerance reached
        if norm(F, Inf) < tol
            break
        end

        # calculate the values of J
        @timeit to "J" calcJ!(J, u)
        
        # update guess for u
        # A \ b => x = A'b
        @timeit to "newtons" begin
            # TODO: BLAS has no functions for Float128, but they might be more efficient for lower precisions?
            # BLAS.gemv!('T', ReduFloat(1.0), J, F, ReduFloat(0.0), ut)

            # mul!(ut, transpose(J), F)
            # u = u - ut

            # TODO: try to find a way to not allocate but still solve J \ F (maybe use in place ldiv! with lu! but i think these don't support Float128)
            u = u - (J \ F)
        end
    end

    # return u for completeness
    return u
end

"""
    driver(u, Dx, num_steps, x, time_start, time_end)

Performs the main calculations over num_steps time steps,
using the provided input and initial values.
"""
function driver(F::Array{ReduFloat}, J::Matrix{ReduFloat}, u::Array{FullFloat}, Dx::Matrix{FullFloat}, num_steps::Integer, x::Array{FullFloat}, time_start::FullFloat, time_end::FullFloat)::Array{FullFloat}
    # calculate dt
    dt = (time_end - time_start) / num_steps

    # track total time
    time_total = time_start

    # store reduced precision Dx
    redu_Dx = convert(Matrix{ReduFloat}, Dx)

    # define Jacobian now since it doesn't change with each time step
    calcJ!(J::Matrix{ReduFloat}, u::Array{ReduFloat}) = calculateJ!(J, u, redu_Dx, ReduFloat(0.5 * dt))

    # initialize variables
    ut = zeros(ReduFloat, length(u))::Array{ReduFloat}
    y2 = zeros(ReduFloat, length(u))::Array{ReduFloat}
    dtDx = zeros(ReduFloat, size(Dx))::Matrix{ReduFloat}
    redu_u = zeros(ReduFloat, length(u))::Array{ReduFloat}
    redu_y = zeros(ReduFloat, length(u))::Array{ReduFloat}

    full_y2 = zeros(ReduFloat, length(u))::Array{FullFloat}
    full_ut = zeros(ReduFloat, length(u))::Array{FullFloat}

    # iterate over each time step
    for i = 1:num_steps
        # ensure that the final time is reached
        if i == num_steps
            dt = time_end - time_total
        end

        # implicit step
        # convert u to reduced precision
        # redu_u = convert(Array{ReduFloat}, u)
        redu_u .= ReduFloat.(u)

        # define calcU each time step since it changes
        calcF!(F::Array{ReduFloat}, y::Array{ReduFloat}) = calculateF!(F, y, redu_u, redu_Dx, ReduFloat(0.5 * dt), y2, dtDx)

        # solve the system with newtons method and convert to full precision
        redu_y .= newtonsMethod!(calcF!, calcJ!, F, J, redu_u, ut)
        full_y = convert(Array{FullFloat}, redu_y)

        # full precision explicit step
        # TODO: !!! we want to make this in place? !!!
        full_y2 .= full_y
        full_y2 .^= 2
        full_y2 .*= FullFloat(0.5)
        mul!(full_ut, Dx, y2)
        full_ut .*= -dt
        u .+= full_ut
        # u = u - (dt * (Dx * (0.5 * full_y.^2)))

        # increment total time
        time_total += dt
    end

    # plot the final time step
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
Dx = convert(Matrix{FullFloat}, Dx)

# store all final u values
u_finals = zeros(FullFloat, length(num_steps_arr), num_x_pts)

# initialize plot if enabled
plot_file = get(parsed_args, "plot", Nothing)
if !isnothing(plot_file)
    plot(x, u_init, title=@sprintf("Burger's Equation w/ Newton's (t = %.2f)", time_end), label=string(ufunc))
end

const to = TimerOutput()

# loop through each num_steps
for (i, num_steps) in enumerate(num_steps_arr)
    F = zeros(ReduFloat, length(u_init))
    J = zeros(ReduFloat, size(Dx))

    # call the driver function
    @timeit to "t=$num_steps" u = driver(F, J, u_init, Dx, num_steps, x, time_start, time_end);

    # store final value of u
    u_finals[i, :] = u

    # print final value of u
    # TODO: println(if length(num_steps_arr) > 1 "$num_steps: " else "" end * string(u))
end

show(to)

# save u final values with corresponding dt values to a file if option has been used
output_file = get(parsed_args, "output", Nothing)
if !isnothing(output_file) writedlm(output_file, cat((time_end - time_start) ./ num_steps_arr, u_finals; dims = 2)) end

# save plot to file if enabled
if !isnothing(plot_file) savefig(plot_file) end