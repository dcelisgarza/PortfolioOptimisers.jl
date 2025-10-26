The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
```@meta
EditURL = "../../../examples/5_Budget_Constraints.jl"
```

# Example 5: Budget constraints

This example shows how to use basic budget constraints.

Before starting it is worth mentioning that portfolio budget constraints are implemented on the actual weights, while the short budget constraints are implemented on a relaxation variable stand-in for the short weights. This means that in some cases, it may appear the short budget constraints are not satisfied when they actually are. This is because the relaxation variables that stand in for the short weights can take on a range of values as long as they are greater than or equal to the absolute value of the actual negative weights, and still satisfy the budget constraint placed on them.

````@example 5_Budget_Constraints
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

## 1. ReturnsResult data

We will use the same data as the previous example.

````@example 5_Budget_Constraints
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = tsfmt)

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Preparatory steps

We'll provide a vector of continuous solvers because the optimisation type we'll be using is more complex, and will contain various constraints. We will also use a more exotic risk measure.

For the mixed integer solvers, we can use a single one.

````@example 5_Budget_Constraints
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

````@example 5_Budget_Constraints
r = EntropicValueatRisk()
pr = prior(EmpiricalPrior(), rd)
````

## 3. Exact budget constraints

The `budget` is the value of the sum of a portfolio's weights.

Here we will showcase various budget constraints. We will start simple, with a strict budget constraint. We will also show the impact this has on the finite allocation.

### 3.1 Strict budget constraints

#### 3.1.1 Fully invested long-only portfolio

First the default case, where the budget is equal to 1, `bgt = 1`. This means the portfolio will be fully invested.

````@example 5_Budget_Constraints
opt1 = JuMPOptimiser(; pe = pr, slv = slv)
mr1 = MeanRisk(; r = r, opt = opt1)
````

You can see that `wb` is of type `WeightBounds`, `lb = 0.0` (asset weights lower bound), and `ub = 1.0` (asset weights upper bound), and `bgt = 1.0` (budget).

We can check that the constraints were satisfied.

````@example 5_Budget_Constraints
res1 = optimise(mr1)
println("budget: $(sum(res1.w))")
println("long budget: $(sum(res1.w[res1.w .>= zero(eltype(res1.w))]))")
println("short budget: $(sum(res1.w[res1.w .< zero(eltype(res1.w))]))")
println("weight bounds: $(all(x -> zero(x) <= x <= one(x), res1.w))")
````

Now let's allocate a finite amount of capital, `4206.9`, to this portfolio.

````@example 5_Budget_Constraints
da = DiscreteAllocation(; slv = mip_slv)
mip_res1 = optimise(da, res1.w, vec(values(X[end])), 4206.9)
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

We will now create a maximum risk-return ratio market neutral portfolio. For a market neutral portfolio, the weights must sum to zero, which means the budget is zero. This means the long and short budgets must be equal in magnitude but opposite sign. In order to avoid all zero weights, we need to set a non-zero short budget, and negative lower weight bounds.

The short budget is given as an absolute value (simplifies implementation details). The weight bounds can be negative. We will set the maximum weight bounds to `±1`, the short budget to `1` (-1 in practice), and the portfolio budget to `0`, therefore the long budget is `1`.

Minimising the risk under without additional constraints often yields all zeros. So we will maximise the risk-return ratio.

````@example 5_Budget_Constraints
rf = 4.2 / 100 / 252
opt2 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget and short budget absolute values.
                     bgt = 0, sbgt = 1,
                     # Weight bounds.
                     wb = WeightBounds(; lb = -1.0, ub = 1.0))
mr2 = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt2)
res2 = optimise(mr2)
println("budget: $(sum(res2.w))")
println("long budget: $(sum(res2.w[res2.w .>= zero(eltype(res2.w))]))")
println("short budget: $(sum(res2.w[res2.w .< zero(eltype(res2.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= one(x), res2.w))")
````

Let's allocate a finite amount of capital. Since we set the long and short budgets equal to 1, the cost of the long and short positions will be approximately equal to the allocated value of `4206.9`, and the sum of the costs will be close to zero. The discrepancies are due to the fact that we are allocating a finite amount of capital.

The discrete allocation procedure automatically adjusts the cash amount depending on the optimal long and short weights, so there is no need to split the cash amount into long and short allocations.

````@example 5_Budget_Constraints
mip_res2 = optimise(da, res2.w, vec(values(X[end])), 4206.9)
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

We will now create and discretely allocate a short-only portfolio. This is in general an anti-pattern but one can use various combinations of budget, weight bounds and short budget constraints to create hedging portfolios.

````@example 5_Budget_Constraints
opt3 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget and short budget absolute values.
                     bgt = -1, sbgt = 1,
                     # Weight bounds.
                     wb = WeightBounds(; lb = -1.0, ub = 0.0))
mr3 = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt3)
res3 = optimise(mr3)
println("budget: $(sum(res3.w))")
println("long budget: $(sum(res3.w[res3.w .>= zero(eltype(res3.w))]))")
println("short budget: $(sum(res3.w[res3.w .< zero(eltype(res3.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= zero(x), res3.w))")
````

We can confirm that the finite allocation behaves as expected.

````@example 5_Budget_Constraints
mip_res3 = optimise(da, res3.w, vec(values(X[end])), 4206.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res3.shares, :cost => mip_res3.cost,
                       :opt_weights => res3.w, :mip_weights => mip_res3.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res3.cost))")
println("long cost: $(sum(mip_res3.cost[mip_res3.cost .>= zero(eltype(mip_res3.cost))]))")
println("short cost: $(sum(mip_res3.cost[mip_res3.cost .< zero(eltype(mip_res3.cost))]))")
println("remaining cash: $(mip_res3.cash)")
println("used cash ≈ available cash: $(isapprox(sum(mip_res3.cost) - mip_res3.cash, 4206.9 * sum(res3.w)))")
````

#### 3.1.4 Leveraged portfolios

Let's try a leveraged long-only portfolio.

````@example 5_Budget_Constraints
opt4 = JuMPOptimiser(; pe = pr, slv = slv, bgt = 1.3)
mr4 = MeanRisk(; r = r, opt = opt4)
res4 = optimise(mr4)
println("budget: $(sum(res4.w))")
println("long budget: $(sum(res4.w[res4.w .>= zero(eltype(res4.w))]))")
println("short budget: $(sum(res4.w[res4.w .< zero(eltype(res4.w))]))")
println("weight bounds: $(all(x -> zero(x) <= x <= one(x), res4.w))")
````

Again, the finite allocation respects the budget constraints.

````@example 5_Budget_Constraints
mip_res4 = optimise(da, res4.w, vec(values(X[end])), 4206.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res4.shares, :cost => mip_res4.cost,
                       :opt_weights => res4.w, :mip_weights => mip_res4.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res4.cost))")
println("long cost: $(sum(mip_res4.cost[mip_res4.cost .>= zero(eltype(mip_res4.cost))]))")
println("short cost: $(sum(mip_res4.cost[mip_res4.cost .< zero(eltype(mip_res4.cost))]))")
println("remaining cash: $(mip_res4.cash)")
println("used cash ≈ available cash: $(isapprox(sum(mip_res4.cost) + mip_res4.cash, 4206.9 * sum(res4.w)))")
````

We will now optimise an underleveraged long-short portfolio.

Note that the short budget is not satisfied, this is because it is implemented as an equality constraint on a relaxation variable stand-in for the short weights. However, the portfolio budget constraint is satisfied because it is an equality constraint on the actual weights.

It is also possible to set budget bounds for the short and portfolio budgets. They are implemented in the same way as the equality constraints. We will explore them in the next section.

````@example 5_Budget_Constraints
opt5 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget and short budget absolute values.
                     bgt = 0.5, sbgt = 1,
                     # Weight bounds.
                     wb = WeightBounds(; lb = -1.0, ub = 1.0))
mr5 = MeanRisk(; r = r, opt = opt5)
res5 = optimise(mr5)
println("budget: $(sum(res5.w))")
println("long budget: $(sum(res5.w[res5.w .>= zero(eltype(res5.w))]))")
println("short budget: $(sum(res5.w[res5.w .< zero(eltype(res5.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= one(x), res5.w))")
````

For this portfolio, the sum of the long and short cost will be approximately equal to half the allocated value of `4206.9`. Any discrepancies are due to the fact we are allocating a finite amount.

````@example 5_Budget_Constraints
mip_res5 = optimise(da, res5.w, vec(values(X[end])), 4506.9)
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res5.shares, :cost => mip_res5.cost,
                       :opt_weights => res5.w, :mip_weights => mip_res5.w);
             formatters = mipresfmt)
println("long cost + short cost = cost: $(sum(mip_res5.cost))")
println("long cost: $(sum(mip_res5.cost[mip_res5.cost .>= zero(eltype(mip_res5.cost))]))")
println("short cost: $(sum(mip_res5.cost[mip_res5.cost .< zero(eltype(mip_res5.cost))]))")
println("remaining cash: $(mip_res5.cash)")
println("used cash ≈ available cash: $(isapprox(sum(abs.(mip_res5.cost)) + mip_res5.cash, 4506.9 * sum(abs.(res5.w))))")
````

# 4. Budget range

The other type of budget constraint we will explore in this example is the budget range constraint, `BudgetRange`. It allows the user to define upper and lower bounds on the budget and short budget. When using a `BudgetRange`, it is necessary to provide at least one of the upper or lower bounds. If only one is provided, the other is assumed to be unbounded. If no budget bounds are desired, simply set `bgt` or `sbgt` to `nothing`.

We mentioned at the start of this example that the interaction between budget and short budget constraints might be unintuitive due to how the constraints are implemented. The following example will illustrate this.

````@example 5_Budget_Constraints
opt6 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget range.
                     bgt = BudgetRange(; lb = 0.3, ub = 0.8),
                     # Exact short budget
                     sbgt = 0.5,
                     # Weight bounds.
                     wb = WeightBounds(; lb = -1.0, ub = 1.0))
mr6 = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt6)
res6 = optimise(mr6)
println("budget: $(sum(res6.w))")
println("long budget: $(sum(res6.w[res6.w .>= zero(eltype(res6.w))]))")
println("short budget: $(sum(res6.w[res6.w .< zero(eltype(res6.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= one(x), res6.w))")
````

As you can see, the budget and weight constraints are satisfied, but not the short budget constraint. This happens even if we do not provide a short budget. This is a reflection of the fact that the weight and budget constraints are constraints on the actual weights. While the short budget constraints are constraints on relaxation variables, whose value must be greater than or equal to the absolute value of the negative weights. This gives them a unbounded wiggle room without violating the constraints, and without directly constraining the short weights.

In general, the short budget constraint will only constrain the portfolio weights when the unbounded optimal portfolio has a short budget whose absolute value is greater than or equal to the short budget constraint.

````@example 5_Budget_Constraints
opt7 = JuMPOptimiser(; pe = pr, slv = slv,
                     # Budget range.
                     bgt = BudgetRange(; lb = 0.3, ub = 0.8),
                     # Remove the slack from the short budget.
                     sbgt = 0.3,
                     # Weight bounds.
                     wb = WeightBounds(; lb = -1.0, ub = 1.0))
mr7 = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt7)
res7 = optimise(mr7)
println("budget: $(sum(res7.w))")
println("long budget: $(sum(res7.w[res7.w .>= zero(eltype(res7.w))]))")
println("short budget: $(sum(res7.w[res7.w .< zero(eltype(res7.w))]))")
println("weight bounds: $(all(x -> -one(x) <= x <= one(x), res7.w))")
````

The previous example has an essentially unbounded short budget. If we constrain the absolute value of the short budget to be less than the unconstrained value, then the constraint has an effect on the portfolio weights.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

