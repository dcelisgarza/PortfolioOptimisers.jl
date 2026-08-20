# Standalone twin of the listing executed inside `main.typ`.
#
# `main.typ` now carries the executed copy of this workflow in a jlyfish cell, so
# the paper cannot drift away from the library. Keep the two in sync, or delete
# this file and let the paper be the only copy. Run it from `docs/paper/`.
#
# Import packages.
using PortfolioOptimisers, CSV, TimeSeries, Clarabel, StatsPlots, GraphRecipes
# Load prices and turn them into returns.
X = TimeArray(CSV.File(joinpath(@__DIR__, "../../examples/SP500.csv.gz"));
              timestamp = :Date)
rd = prices_to_returns(X)
rd_train, rd_test = train_test_split(rd; test_size = 0.2)
# Mean risk optimisation, which minimises risk by default.
mr = MR(;
        # Variance, written directly as a quadratic expression.
        r = Variance(; alg = QuadRiskExpr()),
        opt = JuMPOpt(;
                      # Solvers are tried in order until one of them succeeds.
                      slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                                   settings = Dict("verbose" => false)),
                      # An estimator, so the bounds are built from the data: every asset
                      # is capped at 1 and floored at 0, except AAPL, capped at 0.2.
                      wb = WBE(; ub = "AAPL" => 0.2),
                      # Maps asset names to their columns, and names to sets of assets.
                      # Constraint estimators and some priors are built against this.
                      sets = UniverseSets(; dict = Dict("nx" => rd.nx)),
                      # Squared L2 penalty on the weights, lambda = 1e-4.
                      l2 = L2Reg(; val = 0.0001, alg = QuadRiskExpr()),
                      # Sweep 100 return levels: one solve each, one efficient frontier.
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 100)))) # opt
        ) # mr
# Fit on the training set, then score both sets.
res = optimise(mr, rd_train)
pred_train = predict(res, rd_train)
pred_test = predict(res, rd_test)
# Scenario based standard deviation, as a second order cone expression.
r = SCM(; alg = SOCRiskExpr())
plt = plot_measures(pred_train; x = r, label = "Training", zcolor = nothing)
plt = plot_measures(pred_test; x = r, plt = plt, label = "Test", zcolor = nothing,
                    markercolor = :red, ylabel = "Mean Return",
                    xlabel = "Standard Deviation")
savefig(plt, joinpath(@__DIR__, "fig1.svg"))
