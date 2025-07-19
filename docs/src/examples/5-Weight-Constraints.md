The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
```@meta
EditURL = "../../../examples/5-Weight-Constraints.jl"
```

# Example 5: Weight constraints

This example deals with the use of basic weight and budget constraints.

````@example 5-Weight-Constraints
using PortfolioOptimisers, PrettyTables
# Format for pretty tables.
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
nothing #hide
````

## 1. Returns data

We will use the same data as the previous example.

````@example 5-Weight-Constraints
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = tsfmt)

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Preparatory steps

We'll provide a vector of continuous solvers beacause the optimisation type we'll be using is more complex, and will contain various constraints. We will also use a more exotic risk measure.

For the mixed interger solvers, we can use a single one.

````@example 5-Weight-Constraints
using Clarabel, HiGHS
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.8),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.70),
              check_sol = (; allow_local = true, allow_almost = true))];
mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true));
nothing #hide
````

This time we will use the `EntropicValueatRisk` measure and we will once again precompute prior.

````@example 5-Weight-Constraints
r = EntropicValueatRisk()
pr = prior(EmpiricalPriorEstimator(), rd)
````

## 3. Budget constraints

The `budget` is the value of the sum of a portfolio's weights.

Here we will showcase various budget constraints. We will start simple, with a strict budget constraint. We will also show the impact this has on the finite allocation.

### 3.1 Strict budget constraints

#### 3.1.1 Fully invested long only

First the default case, where the budget is equal to 1, `bgt = 1`. This means we are fully invested.

````@example 5-Weight-Constraints
opt1 = JuMPOptimiser(; pe = pr, slv = slv)
mr1 = MeanRisk(; r = r, opt = opt1)
````

You can see that `wb` is of type `WeightBoundsResult`, `lb = 0.0` (asset weights lower bound), and `ub = 1.0` (asset weights upper bound), and the `bgt = 1.0` (budget).

We can check that the constraints were satisfied.

````@example 5-Weight-Constraints
res1 = optimise!(mr1)
println("budget: $(sum(res1.w))")
println("long budget: $(sum(res1.w[res1.w .>= zero(eltype(res1.w))]))")
println("short budget: $(sum(res1.w[res1.w .< zero(eltype(res1.w))]))")
println("weight bounds: $(all(x -> zero(x) <= x <= one(x), res1.w))")
````

Now lets allocate a finite amount of capital, `4206.9`, to this portfolio.

````@example 5-Weight-Constraints
da = DiscreteAllocation(; slv = mip_slv)
mip_res1 = optimise!(da, res1.w, vec(values(X[end])), 4206.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res1.shares, :cost => mip_res1.cost,
                       :opt_weights => res1.w, :mip_weights => mip_res1.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res1.cost))")
println("long cost: $(sum(mip_res1.cost[mip_res1.cost .>= zero(eltype(mip_res1.cost))]))")
println("short cost: $(sum(mip_res1.cost[mip_res1.cost .< zero(eltype(mip_res1.cost))]))")
println("remaining cash: $(mip_res1.cash)")
println("used cash ≈ available cash: $(isapprox(sum(mip_res1.cost) + mip_res1.cash, 4206.9 * sum(res1.w)))")
````

#### 3.1.2 Maximum risk-return ratio market neutral portfolio

Now lets create a maximum risk-return ratio market neutral portfolio. In order to do this we need to set the short budget and lower the weight bounds constraints. If we simply set the budget to `0`, the solver will return all zeros.

For a market neutral portfolio, the portfolio weights must sum to `0`. This means the long and short budgets must be equal but opposite in sign. We also need to remember to allow the weights lower bounds to be negative, otherwise we will get all zeros as well.

Lets do the simple case where the long and short budgets are `1`, and the weights bounds are `±1`.

Both The short budget `sbgt` has to be given as an absolute value (it simplifies implementation details), but the weight lower bounds can be negative.

Minimising the risk under these constraints without additional constraints often yields all zeros. So we will maximise the risk-return ratio.

````@example 5-Weight-Constraints
rf = 4.2 / 100 / 252
opt2 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget and short budget absolute values.
                     bgt = 0, sbgt = 1,
                     # Weight bounds.
                     wb = WeightBoundsResult(; lb = -1.0, ub = 1.0))
mr2 = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt2)
res2 = optimise!(mr2)
println("budget: $(sum(res2.w))")
println("long budget: $(sum(res2.w[res2.w .>= zero(eltype(res2.w))]))")
println("short budget: $(sum(res2.w[res2.w .< zero(eltype(res2.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= one(x), res2.w))")
````

Lets see what happens when we allocate a finite amount of capital. Because we made the long and short budgets equal to 1, the entire available cash will be invested in both long and short positions.

There is a small variation due to the discrete nature of the allocation, but we can verify the statement above.

````@example 5-Weight-Constraints
mip_res2 = optimise!(da, res2.w, vec(values(X[end])), 4206.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res2.shares, :cost => mip_res2.cost,
                       :opt_weights => res2.w, :mip_weights => mip_res2.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res2.cost))")
println("long cost: $(sum(mip_res2.cost[mip_res2.cost .>= zero(eltype(mip_res2.cost))]))")
println("short cost: $(sum(mip_res2.cost[mip_res2.cost .< zero(eltype(mip_res2.cost))]))")
println("remaining cash: $(mip_res2.cash)")
println("used cash ≈ available cash: $(isapprox(sum(abs.(mip_res2.cost)) + mip_res2.cash, 4206.9 * sum(abs.(res2.w))))")
````

#### 3.1.3 Short-only portfolio

We will now create and discretely allocate a short-only portfolio. This is in general an anti-pattern but oen can use various combinations of budget, weight bounds and short budget constraints to create hedging portfolios.

````@example 5-Weight-Constraints
opt3 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget and short budget absolute values.
                     bgt = -1, sbgt = 1,
                     # Weight bounds.
                     wb = WeightBoundsResult(; lb = -1.0, ub = 0.0))
mr3 = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt3)
res3 = optimise!(mr3)
println("budget: $(sum(res3.w))")
println("long budget: $(sum(res3.w[res3.w .>= zero(eltype(res3.w))]))")
println("short budget: $(sum(res3.w[res3.w .< zero(eltype(res3.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= zero(x), res3.w))")

mip_res3 = optimise!(da, res3.w, vec(values(X[end])), 4206.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res3.shares, :cost => mip_res3.cost,
                       :opt_weights => res3.w, :mip_weights => mip_res3.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res3.cost))")
println("long cost: $(sum(mip_res3.cost[mip_res3.cost .>= zero(eltype(mip_res3.cost))]))")
println("short cost: $(sum(mip_res3.cost[mip_res3.cost .< zero(eltype(mip_res3.cost))]))")
println("remaining cash: $(mip_res3.cash)")
println("used cash ≈ available cash: $(isapprox(sum(mip_res3.cost) - mip_res3.cash, 4206.9 * sum(res3.w)))")
````

#### 3.1.4 Over and under leveraged portfolios

The discrete allocation procedure automatically adjusts the cash amount depending on the optimal long and short weights, so there is no need to split the cash amount into long and short allocations. This means we can fearlessly under and overleverage the portfolio, and the discrete allocation will follow suit.

We can do this regardless of what combination of budget, short budget and weight bounds constraints we use. Lets try an overleveraged long-only portfolio and an underleveraged long-short portfolio.

Lets try the overleveraged long-only portfolio first.

````@example 5-Weight-Constraints
opt4 = JuMPOptimiser(; pe = pr, slv = slv, bgt = 1.3)
mr4 = MeanRisk(; r = r, opt = opt4)
res4 = optimise!(mr4)
println("budget: $(sum(res4.w))")
println("long budget: $(sum(res4.w[res4.w .>= zero(eltype(res4.w))]))")
println("short budget: $(sum(res4.w[res4.w .< zero(eltype(res4.w))]))")
println("weight bounds: $(all(x -> zero(x) <= x <= one(x), res4.w))")

mip_res4 = optimise!(da, res4.w, vec(values(X[end])), 4206.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res4.shares, :cost => mip_res4.cost,
                       :opt_weights => res4.w, :mip_weights => mip_res4.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res4.cost))")
println("long cost: $(sum(mip_res4.cost[mip_res4.cost .>= zero(eltype(mip_res4.cost))]))")
println("short cost: $(sum(mip_res4.cost[mip_res4.cost .< zero(eltype(mip_res4.cost))]))")
println("remaining cash: $(mip_res4.cash)")
println("used cash ≈ available cash: $(isapprox(sum(mip_res4.cost) + mip_res4.cash, 4206.9 * sum(res4.w)))")
````

Lets try the underleveraged long-short portfolio next. Note that the short budget is not satisfied, that's because it is a relaxation constraint of the short weights. The the portfolio budget constraint satisfied, however. That is because it is an exact constraint. Later we will show how to set budget bounds constraint for the short budget as well as the portfolio budget. We will show these later.

We will be able to invest around half of our available cash (due to our finite resources), the rest will come from the short positions.

````@example 5-Weight-Constraints
opt5 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget and short budget absolute values.
                     bgt = 0.5, sbgt = 1,
                     # Weight bounds.
                     wb = WeightBoundsResult(; lb = -1.0, ub = 1.0))
mr5 = MeanRisk(; r = r, opt = opt5)
res5 = optimise!(mr5)
println("budget: $(sum(res5.w))")
println("long budget: $(sum(res5.w[res5.w .>= zero(eltype(res5.w))]))")
println("short budget: $(sum(res5.w[res5.w .< zero(eltype(res5.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= one(x), res5.w))")

mip_res5 = optimise!(da, res5.w, vec(values(X[end])), 4506.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res5.shares, :cost => mip_res5.cost,
                       :opt_weights => res5.w, :mip_weights => mip_res5.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res5.cost))")
println("long cost: $(sum(mip_res5.cost[mip_res5.cost .>= zero(eltype(mip_res5.cost))]))")
println("short cost: $(sum(mip_res5.cost[mip_res5.cost .< zero(eltype(mip_res5.cost))]))")
println("remaining cash: $(mip_res5.cash)")
println("used cash ≈ available cash: $(isapprox(sum(abs.(mip_res5.cost)) + mip_res5.cash, 4506.9 * sum(abs.(res5.w))))")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

