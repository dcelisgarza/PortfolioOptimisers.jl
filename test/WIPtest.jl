# Import module and plotting extension.
using PortfolioOptimisers, StatsPlots, GraphRecipes
# Import optimisers.
using Clarabel, HiGHS, JuMP, Pajarito
# Download data and pretty printing
using YFinance, PrettyTables, TimeSeries, DataFrames

function stock_price_to_time_array(x)
    # Only get the keys that are not ticker or datetime.
    coln = collect(keys(x))[3:end]
    # Convert the dictionary into a matrix.
    m = hcat([x[k] for k in coln]...)
    return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
end

# Tickers to download. These are popular meme stocks, use something better.
assets = sort!(["JCI", "TGT", "CMCSA", "CPB", "MO", "APA", "MMC", "JPM", "ZION", "PSA",
                "BAX", "BMY", "LUV", "PCAR", "TXT", "TMO", "DE", "MSFT", "HPQ", "SEE", "VZ",
                "CNP", "NI", "T", "BA"])

# Prices date range.
Date_0 = "2016-01-01"
Date_1 = "2019-12-30"

# Download the price data using YFinance.
prices = get_prices.(assets; startdt = Date_0, enddt = Date_1)
prices = stock_price_to_time_array.(prices)
prices = hcat(prices...)
cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
prices = prices[cidx]
TimeSeries.rename!(prices, Symbol.(assets))

rd = prices_to_returns(prices)
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
mip_slv = [Solver(; name = :mip1,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip2,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false,
                                                                                                 "max_step_fraction" => 0.95)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip3,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false,
                                                                                                 "max_step_fraction" => 0.90)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip4,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false,
                                                                                                 "max_step_fraction" => 0.85)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip5,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false,
                                                                                                 "max_step_fraction" => 0.80)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip6,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false,
                                                                                                 "max_step_fraction" => 0.75)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip7,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false,
                                                                                                 "max_step_fraction" => 0.7)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip8,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                              MOI.Silent() => true),
                                                     "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                 "verbose" => false,
                                                                                                 "max_step_fraction" => 0.6,
                                                                                                 "max_iter" => 1500,
                                                                                                 "tol_gap_abs" => 1e-4,
                                                                                                 "tol_gap_rel" => 1e-4,
                                                                                                 "tol_ktratio" => 1e-3,
                                                                                                 "tol_feas" => 1e-4,
                                                                                                 "tol_infeas_abs" => 1e-4,
                                                                                                 "tol_infeas_rel" => 1e-4,
                                                                                                 "reduced_tol_gap_abs" => 1e-4,
                                                                                                 "reduced_tol_gap_rel" => 1e-4,
                                                                                                 "reduced_tol_ktratio" => 1e-3,
                                                                                                 "reduced_tol_feas" => 1e-4,
                                                                                                 "reduced_tol_infeas_abs" => 1e-4,
                                                                                                 "reduced_tol_infeas_rel" => 1e-4)),
                  check_sol = (; allow_local = true, allow_almost = true))]
all_industries = ["Energy", "Financials", "Health_Care", "Health_Care",
                  "Consumer_Discretionary", "Utilities", "Consumer_Staples", "Industrials",
                  "Information_Technology", "Consumer_Discretionary", "Financials",
                  "Industrials", "Financials", "Consumer_Staples", "Information_Technology",
                  "Utilities", "Industrials", "Financials", "Materials",
                  "Telecommunications_Services", "Consumer_Discretionary", "Health_Care",
                  "Industrials", "Telecommunications_Services", "Financials"]
sets = AssetSets(; dict = Dict("nx" => rd.nx, "nx_industries" => all_industries))
unique_industries = unique(all_industries)
idx = [all_industries .== i for i in unique_industries]
for (i, ui) in zip(idx, unique_industries)
    sets.dict[ui] = sets.dict["nx"][i]
end
m_idx = hcat(idx...)

res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rd)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)

res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = mip_slv, card = 8)), rd)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)

res = optimise(MeanRisk(;
                        opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                            lt = BuyInThreshold(0.02),
                                            gcard = LinearConstraintEstimator(;
                                                                              val = :(nx ==
                                                                                      8)))),
               rd)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)

res = optimise(MeanRisk(;
                        opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                            lt = BuyInThreshold(0.02),
                                            gcard = LinearConstraintEstimator(;
                                                                              val = :(nx >=
                                                                                      20)))),
               rd; str_names = true)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)

res = optimise(MeanRisk(;
                        opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                            lt = BuyInThreshold(0.02),
                                            gcard = LinearConstraintEstimator(;
                                                                              val = :(PSA +
                                                                                      MMC <=
                                                                                      1)))),
               rd; str_names = true)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)

res = optimise(MeanRisk(;
                        opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                            lt = BuyInThreshold(0.05),
                                            gcard = LinearConstraintEstimator(;
                                                                              val = :(PSA <=
                                                                                      MSFT)))),
               rd; str_names = true)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)

res = optimise(MeanRisk(;
                        opt = JuMPOptimiser(; slv = mip_slv, lt = BuyInThreshold(0.02),
                                            scard = 6,
                                            smtx = AssetSetsMatrixEstimator(;
                                                                            val = "nx_industries"),
                                            sets = sets)), rd; str_names = true)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)

#! fix this, probably adding an gsets parameter to jumpoptimiser
res = optimise(MeanRisk(;
                        opt = JuMPOptimiser(; slv = mip_slv, lt = BuyInThreshold(0.02),
                                            sgcard = LinearConstraintEstimator(;
                                                                               val = :(nx_industries <=
                                                                                       5)),
                                            sgmtx = AssetSetsMatrixEstimator(;
                                                                             val = "nx_industries"),
                                            sets = sets)), rd; str_names = true)
n_assets = count(round.(res.w, digits = 4) .> 0.0)
n_industries = count(round.(m_idx' * res.w, digits = 4) .> 0)