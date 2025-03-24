# define imports
using ArgParse;
using DelimitedFiles;
using LinearAlgebra;
using Quadmath;
using TimerOutputs;

const ValidDatatypes = Dict("Float128" => Float128, "Float64" => Float64, "Float32" => Float32)

function isValidDatatype(datatype::String)
    return haskey(ValidDatatypes, datatype)
end

function ArgParse.parse_item(::Type{Union{Int, Array{Int}}}, x::AbstractString)
    return reshape(readdlm(IOBuffer(x), ',', Int), :)
end

function ArgParse.parse_item(::Type{Array{Float64}}, x::AbstractString)
    return reshape(readdlm(IOBuffer(x), ',', Float64), :)
end

s = ArgParseSettings(description = "example usage: julia $PROGRAM_FILE Float64 Float32 100000 1.0")
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
    "--time_init", "-i"
        help = "The initial time to begin time stepping."
        arg_type = Float64
        default = 0.0
    "--time_final", "-f"
        help = "The final time to end time stepping."
        arg_type = Float64
        default = 1.0
    "--u_init", "-u"
        help = "The initial values of u, as an array of floats delimited by commas without any spaces (i.e. '2.0,0.0')."
        arg_type = Array{Float64}
        default = [2.0, 0.0]
    "--output", "-o"
        help = "The file name to save output values to."
        default = Nothing
    "--plot", "-p"
        help = "Plots output values."
        action = :store_true
end

parsed_args = parse_args(ARGS, s)
print(parsed_args)