#=
# Example 9: Regularisation

This example shows how to use regularisation penalties to improve the robustness of optimised portfolios. Both regularisation penalties work in different ways
=#
using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
mipresfmt = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
mmtfmt = (v, i, j) -> begin
    if i == j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
hmmtfmt = (v, i, j) -> begin
    if i == j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100*1e4, digits=2))e-4 %" : v
    end
end;