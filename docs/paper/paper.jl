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
# Mean Risk optimisation. Defaults to minimising risk.
mr = MR(;
        # Variance (default)
        r = Variance(),
        # Configured optimiser.
        opt = JuMPOpt(;
                      # Using Clarabel as the solver. It's possible to provide fallbacks in the form of a vector, `PortfolioOptimisers.jl` will iterate until it finds one that works or they all fail.
                      slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                                   settings = Dict("verbose" => false)),
                      # Weight bounds using an estimator (it builds the constraint based on the data). Lower bound for all assets is 0, upper bound for AAPL is 0.2, for the rest it's 1.
                      wb = WBE(; lb = 0, ub = "AAPL" => 0.2),
                      # This maps asset names to their indices in the data, as well as sets to which they belong. It is needed to build constraints from estimators and is used by other components such as some prior statistics.
                      sets = AssetSets(; dict = Dict("nx" => rd.nx)),
                      # L2 regularisation using a squared L2 norm with a scalar value of 0.0001. This is used to prevent weight concentration and thus reduce overfitting and improve generalisation.
                      l2 = L2Reg(; val = 0.01, alg = SquaredSOCRiskExpr()),
                      # Arithmetic returns with 100 evenly distributed points between the minimum and maximum returns in the training set. This way we can compute the efficient frontier, which is a subset of pareto fronts.
                      ret = ArithmeticReturn(; lb = Frontier(; N = 100))))
# Perform optimisation on the training set.
res = optimise(mr, rd_train)
# Predict on training data.
pred_train = predict(res, rd_train)
# Predict on test data.
pred_test = predict(res, rd_test)
# Scenario based standard deviation.
r = SCM(; alg = SOCRiskExpr())
# Plot the results.
plt = plot_measures(pred_train; x = r, label = "Training", zcolor = nothing)
plt = plot_measures(pred_test; x = r, plt = plt, label = "Test", zcolor = nothing,
                    markercolor = :red, ylabel = "Mean Return",
                    xlabel = "Standard Deviation")
savefig(plt, joinpath(@__DIR__, "fig1.svg"))
