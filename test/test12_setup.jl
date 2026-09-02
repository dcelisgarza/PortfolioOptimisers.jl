# Shared fixtures for the test_12 prior split files.
# Not a test file (no `test_` prefix); excluded from discovery,
# included by each split file. See ADR 0003.
using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, StatsBase, Clarabel
rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end],
                       TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end])
sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                                "group2" => rd.nx[2:2:end],
                                "clusters1" =>
                                    [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                     3, 3],
                                "clusters2" =>
                                    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                                     1, 2]))
fsets = UniverseSets(; dict = Dict("nx" => rd.nf))
# The post-#224 shape for a factor-flavoured consumer: both axes declared, factors under
# `fkey`. `fsets` keeps the pre-migration shape for the consumers still reading `xkey`.
xfsets = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf))
# The dual-axis shape `AugmentedBlackLittermanPrior` needs: `sets`' asset groups, which its
# `a_views` resolve against, *plus* the declared factor axis its `f_views` land on.
afsets = UniverseSets(; dict = merge(sets.dict, Dict("nf" => rd.nf)))
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = "verbose" => false),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = ["verbose" => false, "max_step_fraction" => 0.95]),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
       Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
       Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
       Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                              "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                              "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                              "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                              "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                              "reduced_tol_gap_rel" => 1e-4, "reduced_tol_ktratio" => 1e-3,
                              "reduced_tol_feas" => 1e-4, "reduced_tol_infeas_abs" => 1e-4,
                              "reduced_tol_infeas_rel" => 1e-4))]
# A value at risk view binds the posterior mass of the tail at its target to `alpha`, and an
# entropy pooling solve meets that constraint to about `1e-8`. The posterior value at risk is a
# sample order statistic: `ValueatRisk` reads the first observation whose cumulative weight
# reaches `alpha`, so a mass short of `alpha` by that much reads the next observation down the
# tail. `var_view_floor` returns that observation, which is the largest loss the target excludes.
# A posterior at or above it meets the view to the resolution the sample has, and an assertion
# written against it does not turn on the sign of a solver residual. See issues #573 and #695.
function var_view_floor(x::AbstractVector, target::Real)
    losses = sort(x)
    k = count(<=(-target), x)
    return -losses[min(k + 1, length(losses))]
end
T = size(rd.X, 1)
iT = inv(T)
w = StatsBase.pweights(range(iT, iT; length = T))
