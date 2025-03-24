# define imports
using DelimitedFiles;
using LinearAlgebra;
using Quadmath;
using TimerOutputs;

#=
    Usage:
    julia implicit_midpoint_old.jl FullType ReducedType num_steps alpha

    Example:
    julia implicit_midpoint_old.jl Float64 Float32 100000 1.0
=#

# define float constants
"""
    parseDatatype(datatype)

Attempts to parse the provided datatype.
"""
function parseDatatype(datatype::String)
    if datatype == "Float32"
        return Float32
    elseif datatype == "Float64"
        return Float64
    elseif datatype == "Float128"
        return Float128
    else
        throw("Unknown data type: " * datatype)
    end
end

const FullFloat = parseDatatype(ARGS[1]);
const ReduFloat = parseDatatype(ARGS[2]);


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

    # TODO: [DEBUG] Save every u
    # u_vals = zeros(FullFloat, num_steps, 2)

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

        # TODO: [DEBUG] Save every u
        # u_vals[i, :] = u
    end

    # TODO: [DEBUG] Save every u
    # writedlm("output.txt", u_vals)

    # return u
    return u
end

# main code
# constants
const num_steps = parse(Int64, ARGS[3])
const alpha = parse(FullFloat, ARGS[4])

const time_start = FullFloat(0.0)
const time_end = FullFloat(1.0)

# variables
F = zeros(ReduFloat, 2)
J = zeros(ReduFloat, 2, 2)
init_u = zeros(FullFloat, 2)
init_u[1] = FullFloat(2.0)
init_u[2] = FullFloat(0.0)

# call the driver function
u = driver(F, J, init_u, num_steps, time_start, time_end, alpha);

# TODO: [DEBUG] Print final u
print(string(u[1]) * "," * string(u[2]))