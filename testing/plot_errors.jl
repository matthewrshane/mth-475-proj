using DelimitedFiles
using Plots;
using Quadmath;

# time deltas
# 1.0 / [num_steps]
const time_deltas = [1.0e-0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]

# file paths
const mixed_sol_filename = "../vanderpol/implicit_midpoint/64_32_sol.txt"
const mixed2_sol_filename = "../vanderpol/implicit_midpoint/64_32_sol2.txt"
const full_sol_filename = "../vanderpol/implicit_midpoint/64_64_sol.txt"
const correct_sol_filename = "../vanderpol/implicit_midpoint/128_128_sol.txt"

# load from files
mixed_sol = readdlm(mixed_sol_filename, ',', '\n')
mixed2_sol = readdlm(mixed2_sol_filename, ',', '\n')
full_sol = readdlm(full_sol_filename, ',', '\n')
correct_sol_str = readdlm(correct_sol_filename, ',', String, '\n')
correct_sol = [parse(Float128, correct_sol_str[1]), parse(Float128, correct_sol_str[2])]

# calculate errors
l2(A, d=1) = sqrt.(sum(abs2, A, dims=d))
mixed_errors = l2(mixed_sol .- correct_sol', 2)
mixed2_errors = l2(mixed2_sol .- correct_sol', 2)
full_errors = l2(full_sol .- correct_sol', 2)

# plot errors
p = plot(time_deltas, [mixed_errors mixed2_errors full_errors], label=["mixed" "mixed2" "full"])
plot!(xscale=:log10, yscale=:log10, yticks=10.0 .^ (-16:0))
xlims!(1e-6, 1e+0)
ylims!(1e-16, 1e+0)
title!("L2 Errors")
xlabel!("log(Δt)")
ylabel!("log(error)")

Plots.pdf(p, "l2_errors")
# png("l2_errors")