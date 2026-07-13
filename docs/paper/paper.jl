# Import packages.
using PortfolioOptimisers, CSV, TimeSeries, PrettyTables, Clarabel, StatsPlots, GraphRecipes
resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end
# Load data.
X = TimeArray(CSV.File(joinpath(@__DIR__, "../../examples/SP500.csv.gz"));
              timestamp = :Date)
# Compute returns.
rd = prices_to_returns(X)
# Split into training and test sets.
rd_train, rd_test = train_test_split(rd; test_size = 0.2)
# Define solver. Use clarabel with no verbose output and default settings.
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false))
# Define assets sets used to match constraints to the asset universe.
sets = AssetSets(; dict = Dict("nx" => rd.nx))
# Lower bounds of 0 and upper bounds of 20 % on AAPL, 1 for the rest.
wb = WBE(; lb = 0, ub = "AAPL" => 0.2)
# L2 norm constraint.
l2 = L2Reg(; val = 0.0001, alg = SquaredSOCRiskExpr())
# Define the return with 100 points in the pareto front (efficient frontier in this case) of its lower bounds.
ret = ArithmeticReturn(; lb = Frontier(; N = 100))
# Define the optimiser with all the above settings.
opt = JuMPOpt(; slv = slv, wb = wb, sets = sets, l2 = l2, ret = ret)
# Markowitz model.
r = Variance()
# Mean Risk optimisation with all the above settings.
mr = MR(; r = r, opt = opt)
# Perform optimisation on the training set.
res = optimise(mr, rd_train)
# Predict on training data.
pred_train = predict(res, rd_train)
# Predict on test data.
pred_test = predict(res, rd_test)
# Scenario based standard deviation.
r = SCM(; alg = SOCRiskExpr())
plt = plot_measures(pred_train; x = r, label = "Training", zcolor = nothing)
plt = plot_measures(pred_test; x = r, plt = plt, label = "Test", zcolor = nothing,
                    markercolor = :red, ylabel = "Mean Return",
                    xlabel = "Standard Deviation")
savefig(plt, joinpath(@__DIR__, "fig1.svg"))
