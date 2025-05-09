using ArgParse;
using DelimitedFiles;
using Plots;
using Quadmath;

# define l2 error calculation function
l2(A, d=1) = sqrt.(sum(abs2, A, dims=d)) / sqrt.(length(A))

# define arguments and usage
const NBS = '\ua0';

s = ArgParseSettings(description = "example$(NBS)usage:$(NBS)julia$(NBS)$PROGRAM_FILE$(NBS)outputs/quad_solution.txt$(NBS)outputs/full_solution.txt$(NBS)outputs/mixed_solution.txt")
@add_arg_table s begin
    "ref_sol"
        help = "The file name to the reference solution to base errors upon."
        arg_type = String
        required = true
    "solution"
        help = "The file name(s) to the calculated solution(s) which to calculate the error(s) of."
        arg_type = String
        nargs = '+'
        required = true
    "--plot", "-p"
        help = "The file name to save output plots to."
        arg_type = String
end

# parse the arguments
parsed_args = parse_args(ARGS, s)

# get file names from the parsed arguments
ref_sol_file = get(parsed_args, "ref_sol", Nothing)
sol_files = get(parsed_args, "solution", Nothing)

# load reference solution from files as a string first, then parsed into an array of Float128
ref_sol_str = readdlm(ref_sol_file, '\t', String, '\n')[2:end]
ref_sol = [parse(Float128, x) for x in ref_sol_str]

# load other solutions from files
sols = [readdlm(sol_file, '\t', '\n') for sol_file in sol_files]

# calculate errors of other solutions (ignoring the dt column)
errors = [l2(sol[:, 2:end] .- ref_sol', 2) for sol in sols]

# get dt column
dts = sols[1][:, 1]

# plot errors
p = plot(dts, errors, label=reshape(sol_files, (1, :)), dpi=300, legend=:bottomright)
plot!(xscale=:log10, yscale=:log10, yticks=10.0 .^ (-16:1))
xlims!(10^floor(log10(minimum(dts))), 10^ceil(log10(maximum(dts))))
ylims!(1e-16, 1e+1)
title!("L₂ Errors")
xlabel!("log₁₀(Δt)")
ylabel!("log₁₀(error)")

vline!(dts, label="dt values", ls=:dash, lc=:gray)

# check if the plot file name was specified
plot_file = get(parsed_args, "plot", Nothing)
if !isnothing(plot_file) savefig(plot_file) else savefig("plots/l2_errors.png") end