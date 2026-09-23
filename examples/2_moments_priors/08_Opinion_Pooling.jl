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
[`LinearOpinionPooling`](@ref) or [`LogarithmicOpinionPooling`](@ref), and an optional robust
confidence in `p`. We build three opinions and pool them. Sections 5 to 7 then go through the
parameters of the pool. Section 5 compares the two pooling algorithms. Section 6 shows the
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
## 1. ReturnsResult data

We use the same S&P 500 slice as the other examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                "energy" => ["CVX"]))

#=
## 2. Three opinions

Each opinion is an [`EntropyPoolingPrior`](@ref) with its own views. Think of them as three
analysts who studied the same market and reached different conclusions.

  - Opinion A is bullish on Apple, `AAPL == 0.0008`.
  - Opinion B is a view on sectors. Tech returns at least as much as energy, and Microsoft
    returns 6 bps a day.
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
returns with those of the plain empirical prior. The consensus holds each opinion in proportion
to its weight, and no single opinion decides it.
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

The weights `w` state how credible each opinion is. If they sum to less than one, the pool
gives the remaining weight to the uniform prior, which is the empirical distribution before any
reweighting. So the total of `w` acts as one confidence for all the views together, and a
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

Linear pooling uses the weight of each opinion as given. Set `p` to a confidence in ``(0, 1]``
to turn on robust opinion pooling. The pool then adjusts the weight of each opinion by the
Kullback-Leibler divergence between its distribution and the consensus, to discount an opinion
that is far from the rest.

```julia
pr_robust = prior(OpinionPoolingPrior(; pes = [opinion_a, opinion_b, opinion_c],
                                      w = [0.5, 0.3, 0.2], p = 0.1), rd)
```

The page shows this call without running it, because on this S&P 500 data it changes almost
nothing. When we wrote the page, the robust adjustment moved the posterior mean and covariance
by about ``10^{-14}``, which is numerical noise, even with a fourth opinion made extreme on
purpose. The divergence discount changes the pooled probabilities of the scenarios. All the
opinions share the same return scenarios, so that change barely reaches the first two moments
here. Use `p` when the distributions of the opinions are far apart. To move the consensus mean
on one short window, change the total of the credibility weights, as in section 6.

## 8. A consensus portfolio

We maximise the risk-adjusted ratio under the empirical prior and under the pooled consensus,
and compare the weights. The consensus portfolio is one allocation that takes all three
opinions into account.
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
#src - FINDING (→ #126): robust pooling `p` had NEGLIGIBLE effect here — even an extreme outlier
#src   (AMD == 0.006) at p∈{1.0,0.5,0.1,0.01} moved pooled mu by ~1e-14 (machine eps) and sigma
#src   likewise. The KL discount reshapes scenario probabilities but barely propagates into the
#src   first two moments on a single short window. Rewrote section 7 honestly (prose + caveat,
#src   no fake contrast table) rather than overclaim. Worth a docstring note on when `p` bites.
#src   Closes the BL→EP→OP view arc. → #126.
