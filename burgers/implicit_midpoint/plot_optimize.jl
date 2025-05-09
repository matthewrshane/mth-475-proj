using Plots;

nts = [1; 10; 100; 1000; 10000]
speedups = [
    0.638418079	549	53.7826087	9.15503876	3.105960265;
    0.682496608	37	12.86138614	2.353584447	0.656365884;
    0.575433912	6.592592593	4.199004975	0.933962264	0.674698795
]

# plot errors
p = scatter(nts, transpose(speedups .* 100), label=["091" "181" "361"], dpi=300, legend=:topright)
scatter!(xscale=:log10, yscale=:log10, yticks=10.0 .^ (0:5))
xlims!(1, 10000)
ylims!(1, 100000)
title!("Implicit Midpoint Optimizations (Double Precision)")
xlabel!("# time steps")
ylabel!("% speedup")

savefig("plots/optimize.pdf")