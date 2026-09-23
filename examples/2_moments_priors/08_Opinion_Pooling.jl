#=
```@meta
Description = "Opinion pooling in PortfolioOptimisers.jl: combine several entropy-pooling posteriors into one consensus prior weighted by trust."
```

# Opinion pooling

[Entropy pooling](07_Entropy_Pooling.md) turns one set of views into a reweighted prior. But
you often hold several sets of views from different sources, such as a fundamental analyst, a
quantitative signal and a macro desk, and they can conflict. Opinion pooling combines several
entropy pooling posteriors into one consensus prior, and weights each opinion by how much you
trust it. This page is the last of three on priors built from views. Each opinion is an
[`EntropyPoolingPrior`](@ref), and the pool blends them.

[`OpinionPoolingPrior`](@ref) takes a vector of entropy pooling priors in `pes`, optional
credibility weights in `w`, a pooling algorithm in `alg`, which is
[`LinearOpinionPooling`](@ref) or [`LogarithmicOpinionPooling`](@ref), and an optional penalty
`p` for robust pooling. We build three opinions and pool them. Sections 5 to 7 then go through
the parameters of the pool. Section 5 compares the two pooling algorithms. Section 6 shows the
credibility weights and the uniform prior that takes the weight they leave. Section 7 shows
robust pooling, which lowers the weight of an opinion far from the others.

!!! tip "When to reach for this"
    Reach for opinion pooling when you hold several sets of views that can conflict, and want
    one consensus rather than a choice of one set or an average of forecasts by hand. Give
    each opinion a weight for its credibility. If you hold one consistent set of views, entropy
    pooling is enough, and if your views are on the mean alone and Gaussian, Black-Litterman is
    simpler still.
=#

using PortfolioOptimisers, PrettyTables

mmtfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=4)) %" : v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. The data

We load one year of daily returns of 20 assets, and name two groups of them for the views.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                "energy" => ["CVX"]))

#=
## 2. Three opinions

Each opinion is an [`EntropyPoolingPrior`](@ref) with its own views.

  - Opinion A is bullish on Apple, `AAPL == 0.0008`.
  - Opinion B is a view on sectors. The three tech means added together are at least the energy
    mean, and Microsoft returns 6 bps a day.
  - Opinion C is defensive. It sets the expected returns of two low-volatility stocks,
    Johnson & Johnson at 4 bps a day and Coca-Cola at 3 bps a day.
=#

opinion_a = EntropyPoolingPrior(; sets = sets,
                                mu_views = LinearConstraintEstimator(;
                                                                     val = ["AAPL == 0.0008"]))
opinion_b = EntropyPoolingPrior(; sets = sets,
                                mu_views = LinearConstraintEstimator(;
                                                                     val = ["tech >= energy",
                                                                            "MSFT == 0.0006"]))
opinion_c = EntropyPoolingPrior(; sets = sets,
                                mu_views = LinearConstraintEstimator(;
                                                                     val = ["JNJ == 0.0004",
                                                                            "KO == 0.0003"]))

#=
## 3. Pooling the opinions

We pool the three with credibility weights. We trust the Apple view most, the sector view next
and the defensive view least. The weights sum to one here, and section 6 shows what happens
when they sum to less.
=#

op = OpinionPoolingPrior(; pes = [opinion_a, opinion_b, opinion_c], w = [0.5, 0.3, 0.2])

#=
## 4. The consensus and the individual opinions

We compute the posterior of each opinion and the pooled consensus, and compare their expected
returns with those of the plain empirical prior. With weights that sum to one, each pooled mean
is the average of the opinions' means, weighted by `w`.
=#

pr_emp = prior(EmpiricalPrior(), rd)
pr_a = prior(opinion_a, rd)
pr_b = prior(opinion_b, rd)
pr_c = prior(opinion_c, rd)
pr_op = prior(op, rd)

pretty_table(DataFrame(["Assets" => rd.nx, "Empirical" => pr_emp.mu, "Opinion A" => pr_a.mu,
                        "Opinion B" => pr_b.mu, "Opinion C" => pr_c.mu,
                        "Pooled" => pr_op.mu]); formatters = [mmtfmt],
             title = "Expected returns: individual opinions vs pooled consensus")

# The expected returns of the pooled consensus.
using StatsPlots, GraphRecipes
plot_mu(pr_op, rd.nx)

#=
## 5. Linear and logarithmic pooling

The pooling `alg` sets how the pool combines the distributions of the opinions.

  - [`LinearOpinionPooling`](@ref), the default, takes a weighted arithmetic average of the
    probabilities of the opinions. It is the mixture of experts rule. The consensus is a
    blend, so one confident opinion can move the mean a long way.
  - [`LogarithmicOpinionPooling`](@ref) takes a weighted geometric mean, the consensus that is
    optimal in Kullback-Leibler divergence. An asset moves far only when the opinions move it
    the same way, so a conflict between them has less effect.

The two pools give similar means when the opinions are compatible, and different means as the
opinions conflict. The table prints the pooled mean of every asset under each.
=#

pr_lin = prior(OpinionPoolingPrior(; pes = [opinion_a, opinion_b, opinion_c],
                                   w = [0.5, 0.3, 0.2], alg = LinearOpinionPooling()), rd)
pr_log = prior(OpinionPoolingPrior(; pes = [opinion_a, opinion_b, opinion_c],
                                   w = [0.5, 0.3, 0.2], alg = LogarithmicOpinionPooling()),
               rd)

pretty_table(DataFrame(["Assets" => rd.nx, "Linear pool" => pr_lin.mu,
                        "Logarithmic pool" => pr_log.mu]); formatters = [mmtfmt],
             title = "Linear vs logarithmic pooling")

#=
## 6. Credibility weights and the uniform-prior fallback

The weights `w` state how credible each opinion is. If they sum to less than one, the pool gives
the remaining weight to the uniform prior, which is the empirical distribution before any
reweighting. The total of `w` therefore acts as one confidence for all the views together, and a
smaller total pulls the consensus back toward the data and away from the views. We hold the
relative trust at 5:3:2 and scale the total from 1.0 down to 0.4. The table shows the pooled
expected returns of Apple and Microsoft as they move back toward the empirical prior.
=#

scales = [1.0, 0.7, 0.4]
pr_scaled = [prior(OpinionPoolingPrior(; pes = [opinion_a, opinion_b, opinion_c],
                                       w = s .* [0.5, 0.3, 0.2]), rd) for s in scales]

i_aapl = findfirst(==("AAPL"), rd.nx)
i_msft = findfirst(==("MSFT"), rd.nx)
pretty_table(DataFrame("weight total" => ["empirical (0.0)"; string.(scales)],
                       "AAPL posterior" => [pr_emp.mu[i_aapl];
                                            [p.mu[i_aapl] for p in pr_scaled]],
                       "MSFT posterior" => [pr_emp.mu[i_msft];
                                            [p.mu[i_msft] for p in pr_scaled]]);
             formatters = [mmtfmt],
             title = "Lower total weight shrinks the consensus toward the empirical prior")

#=
## 7. Robust pooling with `p`

By default the pool uses the weight of each opinion as given. Set `p`, a penalty above zero, to
turn on robust opinion pooling. The pool then multiplies the weight of opinion ``k`` by
``\exp(-p D_k)``, where ``D_k`` is the Kullback-Leibler divergence between the distribution of
that opinion and the linear consensus, and scales the weights to sum to one. A larger `p` moves
more weight to the opinions nearest the consensus. The adjustment runs before either pooling
algorithm.

```julia
pr_robust = prior(OpinionPoolingPrior(; pes = [opinion_a, opinion_b, opinion_c],
                                      w = [0.5, 0.3, 0.2], p = 0.1), rd)
```

The page shows this call without running it. On this data every opinion is close to the
consensus in Kullback-Leibler divergence, so a small `p` barely moves the weights. To move the
consensus mean on one short window, change the total of the credibility weights, as section 6
does.

## 8. A consensus portfolio

We maximise the risk-adjusted ratio under the empirical prior and under the pooled consensus,
and compare the weights.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

res_emp = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr_emp, slv = slv)))
res_op = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                           opt = JuMPOptimiser(; pe = pr_op, slv = slv)))

pretty_table(DataFrame(["Assets" => rd.nx, "Empirical" => res_emp.w,
                        "Pooled consensus" => res_op.w]); formatters = [resfmt],
             title = "Maximum-ratio weights: empirical vs pooled consensus")

#=
The composition plot stacks the weights of each portfolio into one bar, the empirical
portfolio first.
=#

plot_stacked_bar_composition([res_emp, res_op], rd; xticks = (1:2, ["Empirical", "Pooled"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Deep-dive pass (per "examples are deep dives" feedback): added linear-vs-logarithmic
#src   pooling and the credibility-weight total as a global confidence dial (sum(w)<1 →
#src   remainder to the uniform prior). Both verified to move mu on the real SP500 slice via
#src   kaimon (session bc3429ca): linear≠log; lowering the weight total monotonically pulls
#src   AAPL from +3.75 bps (total 1.0) back to −5.26 bps (total 0.4) toward empirical −11.26.
#src - BUG FIXED this session (→ #126): prior(OpinionPoolingPrior) used to MUTATE the stored
#src   weights `pe.w` in place — `ow = pe.w` aliased it and `push!(ow, rw)` grew it when
#src   sum(w)<1, so a second prior() call threw `length(w) == length(pes)`. Fixed in
#src   src/10_Prior/07_OpinionPoolingPrior.jl by `ow = vcat(ow, rw)` (also fixes the uniform
#src   `range` branch, which was immutable). Regression test added in test_12b_prior_core.jl.
#src - CORRECTED by #1227 (2026-09-23): the ~1e-14 effect of `p` that an earlier note recorded
#src   was wrong. Measured on this slice: p = 0.1 moves the opinion weights by ~3e-5 and mu by
#src   ~1e-7, p = 1 by ~3e-4 and ~1e-6, and p = 100 moves ow to [0.533, 0.304, 0.163]. Each
#src   opinion sits within KL 0.006 of the consensus, so exp(-p D) stays near 1 until p nears
#src   1/D. Robust pooling has no defect here.
