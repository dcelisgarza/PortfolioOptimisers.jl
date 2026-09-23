#=
```@meta
Description = "The risk measures of PortfolioOptimisers.jl: variance, semi-moments, MAD, VaR, CVaR, EVaR, RLVaR, drawdowns and OWA, and the aliases that configure them."
```

# Risk measures

The previous page showed which optimiser to call. This page lists what you can ask it to
minimise, the risk measure in its `r` field. The library has many risk measures, and a name does
not always tell you what the measure is. Several well-known measures, such as the mean absolute
deviation, the semi-variance and the Gini mean difference, are not types. Each is a setting of a
generic type, and a short alias builds it.

This page is a reference. It answers three questions:

 1. What measures exist, and what does each penalise?
 2. Which optimisers accept it?
 3. What is its short alias, and what does that alias expand to?

We compute every table below from the types of the library and from
[`supports_risk_measure`](@ref), so the tables list what the optimisers accept. The one-line
meanings are written by hand, and the docs build fails if a measure has none.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, InteractiveUtils,
      StatsPlots, GraphRecipes

#=
## 1. How a measure is used

A risk measure has two uses.

Inside an optimiser, a measure is a configuration. You pass it to the `r` field, and the optimiser
turns it into constraints and an objective term.

Outside an optimiser, a measure is an object that you call on a portfolio to get a number.
[`expected_risk`](@ref) is one call for every measure. A measure takes one of three inputs: the
net returns of the portfolio, the weights with the returns and the fees, or the weights alone.
`expected_risk` gives each measure the input that it takes. We compute six measures on the
equal-weighted portfolio.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

w = optimise(EqualWeighted(), rd).w

measures = [SD(), MAD(), CVaR(), CDaR(), MDD(), UCI()]
pretty_table(DataFrame("Measure" => ["StandardDeviation", "MAD (LowOrderMoment)",
                                     "ConditionalValueatRisk", "ConditionalDrawdownatRisk",
                                     "MaximumDrawdown", "UlcerIndex"],
                       "Risk of the equal-weighted book" =>
                           [expected_risk(factory(r, pr), w, pr.X) for r in measures]))

#=
!!! note

    [`factory`](@ref) binds a measure to a prior. It fills `sigma` for [`Variance`](@ref) and
    [`StandardDeviation`](@ref), and `mu` for the moment measures. It also gives the observation
    weights of the prior to the measures that take them, such as
    [`ConditionalValueatRisk`](@ref). Without `factory`, such a measure gives every observation the
    same weight. The optimisers call `factory` on every measure, and you can call it on any
    measure. [`EntropicValueatRisk`](@ref), [`RelativisticValueatRisk`](@ref) and their drawdown
    forms solve a subproblem to compute the risk. Before you call one of them outside an
    optimiser, put a [`Solver`](@ref) in its `slv` field.

!!! note

    If you give `expected_risk` the prior, as in `expected_risk(r, w, pr)`, it calls `factory` for
    you. So you call `factory` yourself only when you want the bound measure. If you give it a
    returns matrix, and a field of the measure holds a [`DeferredQuantity`](@ref),
    `expected_risk` throws an error, because it has no prior to compute the quantity from.

    The fields `sigma`, `mu`, `kt` and `sk` each take a value or an estimator that computes it. The
    optimiser computes the estimator on the prior of each run, including each cross-validation
    fold and each resampled subset. The
    [subset resampling example](../examples/3_optimisers/14_Subset_Resampling_and_Cross_Validation.md)
    shows it.

## 2. The three usage classes

Each measure belongs to one of three classes, and the class sets where you can use it. The
`Optimisers` column of every table below comes from the class:

| Class | `Optimisers` column | Meaning |
| :-- | :-- | :-- |
| [`RiskMeasure`](@ref) | `JuMP + clustering` | Has a JuMP formulation, which is convex for most measures and mixed-integer for a few. You can use it in [`MeanRisk`](@ref), [`RiskBudgeting`](@ref), [`NearOptimalCentering`](@ref), [`FactorRiskContribution`](@ref) and the clustering optimisers. |
| [`HierarchicalRiskMeasure`](@ref) | `clustering only` | Has no JuMP formulation. Only a clustering optimiser can compute it, from the returns. `MeanRisk` throws a `TypeError` when you pass one. |
| `NonOptimisationRiskMeasure` | `diagnostic only` | Is not a target of an optimisation. It is a quantity to report, such as [`ExpectedReturn`](@ref) or [`Skewness`](@ref), or a ratio that scores a portfolio in cross-validation. |

The meta-optimisers, [`NestedClustered`](@ref), [`Stacking`](@ref) and [`SubsetResampling`](@ref),
have no `r` of their own. Each accepts a measure only when every optimiser it holds accepts it, so
the tables do not list them. Ask for the case you have:

```julia
supports_risk_measure(MeanRisk, ConditionalValueatRisk)   # true
supports_risk_measure(MeanRisk, EqualRisk)                # false, hierarchical only
supported_risk_measures(HierarchicalRiskParity)           # OptimisationRiskMeasure
```

## 3. The catalogue

Each table gives the meaning of each measure in one line, its alias and its class. The tables
group the measures by the part of the returns that they measure. Every name is exported and has a
docstring. The
[public API page on risk measures](../public_api/16_RiskMeasures/01_Base_RiskMeasures.md) gives the
signature, the fields and the references of each, and `?ConditionalValueatRisk` in the REPL shows
the same docstring.
=#

## Reverse-map the exported alias layer onto the measure types it names, so the alias column
## is read off the package rather than transcribed.
rm_alias = Dict{Symbol, String}()
for n in names(PortfolioOptimisers)
    v = getfield(PortfolioOptimisers, n)
    if isa(v, Type) && v <: PortfolioOptimisers.AbstractBaseRiskMeasure && nameof(v) != n
        rm_alias[nameof(v)] = String(n)
    end
end

## Class → the `Optimisers` column, derived from the compatibility trait.
function usage_class(T)
    return if supports_risk_measure(MeanRisk, T)
        "JuMP + clustering"
    elseif supports_risk_measure(HierarchicalRiskParity, T)
        "clustering only"
    else
        "diagnostic only"
    end
end

## The curated catalogue: family => [measure => what it penalises].
catalogue = ["Dispersion and moments" =>
                 [:Variance => "Portfolio variance from a covariance matrix, the default `r`.",
                  :StandardDeviation => "Square root of the variance; same ordering, different scale.",
                  :UncertaintySetVariance => "Worst-case variance over a covariance uncertainty set (robust).",
                  :LowOrderMoment => "Generic first/second moment measure; see §4 for its aliases.",
                  :HighOrderMoment => "Generic third/fourth moment measure; see §4 for its aliases.",
                  :MedianAbsoluteDeviation => "Median absolute deviation, a robust counterpart of `MAD()`.",
                  :Kurtosis => "Square-root kurtosis from the cokurtosis tensor (fat tails).",
                  :NegativeSkewness => "Downside asymmetry from the coskewness tensor.",
                  :VarianceSkewKurtosis => "Variance, skewness, and kurtosis combined in one expression.",
                  :BrownianDistanceVariance => "Distance variance: penalises *any* dependence, not only linear dependence."],
             "Tail: X-at-Risk" => [:WorstRealisation => "The single worst observed loss.",
                                   :ValueatRisk => "The `alpha` quantile of the loss distribution (MIP).",
                                   :ConditionalValueatRisk => "Mean loss beyond the `alpha` quantile; expected shortfall.",
                                   :DistributionallyRobustConditionalValueatRisk => "CVaR under a Wasserstein ball around the empirical distribution.",
                                   :EntropicValueatRisk => "Exponential-cone upper bound on VaR; tighter tail control than CVaR.",
                                   :RelativisticValueatRisk => "Power-cone family between EVaR (`kappa → 0`) and the worst realisation (`kappa → 1`).",
                                   :PowerNormValueatRisk => "Power-norm tail measure parameterised by the norm order."],
             "Tail ranges (both sides)" =>
                 [:Range => "Best realisation minus worst realisation.",
                  :ValueatRiskRange => "Loss-side VaR plus gain-side VaR.",
                  :ConditionalValueatRiskRange => "Loss-side CVaR plus gain-side CVaR.",
                  :DistributionallyRobustConditionalValueatRiskRange => "Two-sided distributionally robust CVaR.",
                  :EntropicValueatRiskRange => "Two-sided entropic VaR.",
                  :RelativisticValueatRiskRange => "Two-sided relativistic VaR.",
                  :PowerNormValueatRiskRange => "Two-sided power-norm VaR.",
                  :GenericValueatRiskRange => "Any pair of tail measures, one per side of the distribution."],
             "Drawdown: uncompounded" =>
                 [:AverageDrawdown => "Mean depth of the drawdown path.",
                  :UlcerIndex => "Root-mean-square drawdown depth; penalises long deep spells.",
                  :MaximumDrawdown => "Deepest peak-to-trough loss.",
                  :DrawdownatRisk => "The `alpha` quantile of the drawdown path (MIP).",
                  :ConditionalDrawdownatRisk => "Mean drawdown beyond the `alpha` quantile.",
                  :DistributionallyRobustConditionalDrawdownatRisk => "CDaR under a Wasserstein ball.",
                  :EntropicDrawdownatRisk => "Exponential-cone bound on the drawdown quantile.",
                  :RelativisticDrawdownatRisk => "Power-cone drawdown family between EDaR (`kappa → 0`) and the maximum drawdown (`kappa → 1`).",
                  :PowerNormDrawdownatRisk => "Power-norm drawdown measure."],
             "Drawdown: compounded (relative)" =>
                 [:RelativeAverageDrawdown => "Average drawdown of the compounded wealth path.",
                  :RelativeUlcerIndex => "Ulcer index of the compounded wealth path.",
                  :RelativeMaximumDrawdown => "Maximum drawdown of the compounded wealth path.",
                  :RelativeDrawdownatRisk => "Drawdown-at-risk of the compounded wealth path.",
                  :RelativeConditionalDrawdownatRisk => "Conditional drawdown-at-risk, compounded.",
                  :RelativeEntropicDrawdownatRisk => "Entropic drawdown-at-risk, compounded.",
                  :RelativeRelativisticDrawdownatRisk => "Relativistic drawdown-at-risk, compounded.",
                  :RelativePowerNormDrawdownatRisk => "Power-norm drawdown-at-risk, compounded."],
             "Ordered weights arrays" =>
                 [:OrderedWeightsArray => "Any weighting of the *sorted* losses; the most general family (§4).",
                  :OrderedWeightsArrayRange => "An OWA applied to both sides of the distribution."],
             "Path and mandate" =>
                 [:TrackingRiskMeasure => "Deviation of the book's returns from a benchmark.",
                  :RiskTrackingRiskMeasure => "Deviation of the book's *risk* from a benchmark's risk.",
                  :TurnoverRiskMeasure => "Distance from the previous weights; penalises trading."],
             "Composite and structural" =>
                 [:EqualRisk => "Drives every cluster to carry the same risk (hierarchical).",
                  :RiskRatio => "Ratio of two measures, used as a hierarchical objective.",
                  :NoRisk => "Contributes nothing: a null `r` for return-only problems."],
             "Non-optimisation (diagnostics and scoring)" =>
                 [:ExpectedReturn => "Prior expected return of the book.",
                  :MeanReturn => "Realised mean return of the book.",
                  :Skewness => "Standardised skewness of the return distribution.",
                  :ThirdCentralMoment => "Unstandardised third central moment.",
                  :NonOptimisationRiskRatio => "Ratio of any two non-optimisation measures.",
                  :ExpectedReturnRiskRatio => "Prior expected return over risk, a Sharpe-style score.",
                  :MeanReturnRiskRatio => "Realised mean return over risk."]]

## Every concrete measure must appear exactly once. This stops the page from drifting.
function leaf_measures(T, acc = Type[])
    subs = subtypes(T)
    isempty(subs) ? push!(acc, T) : foreach(S -> leaf_measures(S, acc), subs)
    return acc
end
all_measures = Set(nameof.(leaf_measures(PortfolioOptimisers.AbstractBaseRiskMeasure)))
listed = [first(p) for (_, fam) in catalogue for p in fam]
@assert allunique(listed)
@assert Set(listed) == all_measures

function family_table(name)
    entries = catalogue[findfirst(p -> first(p) == name, catalogue)][2]
    return DataFrame("Measure" => [String(first(e)) for e in entries],
                     "Alias" => [get(rm_alias, first(e), "") for e in entries],
                     "Penalises" => [last(e) for e in entries],
                     "Optimisers" => [usage_class(getfield(PortfolioOptimisers, first(e)))
                                      for e in entries])
end;

# ### Dispersion and moments
pretty_table(family_table("Dispersion and moments"))

# ### Tail measures
pretty_table(family_table("Tail: X-at-Risk"))

# ### Tail ranges (both sides)
pretty_table(family_table("Tail ranges (both sides)"))

# ### Drawdowns of uncompounded returns
pretty_table(family_table("Drawdown: uncompounded"))

#=
### Drawdowns of compounded returns

The `Relative*` measures compute the drawdown on the compounded wealth, not on the cumulative sum
of the returns. They have no convex JuMP formulation, so they are all `clustering only`.
=#
pretty_table(family_table("Drawdown: compounded (relative)"))

# ### Ordered weights arrays
pretty_table(family_table("Ordered weights arrays"))

# ### Path and mandate
pretty_table(family_table("Path and mandate"))

# ### Composite and structural
pretty_table(family_table("Composite and structural"))

#=
### Non-optimisation (diagnostics and scoring)

You cannot put these measures in an `r` field. They score a portfolio after the optimisation,
and they score the folds of a cross-validation. See
[validation and tuning](05_Validation_and_Tuning.md).
=#
pretty_table(family_table("Non-optimisation (diagnostics and scoring)"))

#=
## 4. Generic measures and their aliases

Three of the types above are generic. [`LowOrderMoment`](@ref), [`HighOrderMoment`](@ref) and
[`OrderedWeightsArray`](@ref) each compute a family of measures, and the `alg` or `w` field
selects the member. You can write `MAD()` for `LowOrderMoment(; alg = MeanAbsoluteDeviation())`.
Each row below is an exported alias that builds a measure in one call. An alias that builds a
`HighOrderMoment` gives a measure that only a clustering optimiser accepts.

We read the `Expands to` column off the object that each alias builds, so the column shows what the
alias builds in this version.
=#

alias_ctors = [("FLM", FLM, "First lower partial moment."),
               ("MAD", MAD, "Mean absolute deviation."),
               ("SCM", SCM,
                "Second central moment: scenario variance or standard deviation."),
               ("SLM", SLM, "Second lower moment: scenario semi-variance."),
               ("ECM", ECM, "Central even moment of order `2p`."),
               ("ELM", ELM, "Lower even moment of order `2p`."),
               ("TLM", TLM, "Third lower moment."),
               ("SSK", SSK, "Standardised third lower moment: semi-skewness."),
               ("FTCM", FTCM, "Fourth central moment."),
               ("FTLM", FTLM, "Fourth lower moment."),
               ("KT", KT, "Standardised fourth central moment: kurtosis."),
               ("SKT", SKT, "Standardised fourth lower moment: semi-kurtosis."),
               ("OWA_GMD", OWA_GMD, "Gini mean difference."),
               ("OWA_CVaR", OWA_CVaR, "CVaR as an OWA."), ("OWA_TG", OWA_TG, "Tail Gini."),
               ("OWA_WR", OWA_WR, "Worst realisation as an OWA."),
               ("OWA_RG", OWA_RG, "Range as an OWA."),
               ("OWA_CVaR_RG", OWA_CVaR_RG, "Two-sided CVaR as an OWA."),
               ("OWA_TG_RG", OWA_TG_RG, "Two-sided tail Gini."),
               ("OWA_LMoment", OWA_LMoment, "L-moment weights of order `k`.")]

## Walk the algorithm chain of a constructed measure so the expansion is observed, not asserted.
function expands_to(m)
    if isa(m, OrderedWeightsArray)
        return "w = $(isa(m.w, Function) ? nameof(m.w) : nameof(typeof(m.w)))"
    end
    parts, a = String[], m.alg
    while true
        push!(parts, String(nameof(typeof(a))))
        if hasproperty(a, :alg1)
            push!(parts, String(nameof(typeof(a.alg1))))
        end
        if hasproperty(a, :alg) && isa(a.alg, PortfolioOptimisers.AbstractAlgorithm)
            a = a.alg
        else
            break
        end
    end
    return join(parts, " → ")
end

pretty_table(DataFrame("Alias" => [a[1] * "()" for a in alias_ctors],
                       "Builds" => [String(nameof(typeof(a[2]()))) for a in alias_ctors],
                       "Expands to" => [expands_to(a[2]()) for a in alias_ctors],
                       "Meaning" => [a[3] for a in alias_ctors]))

#=
!!! warning "`MAD()` is not `MedianAbsoluteDeviation()`"

    [`MAD`](@ref) builds a [`LowOrderMoment`](@ref) with [`MeanAbsoluteDeviation`](@ref), the
    deviation around the mean. It is a [`RiskMeasure`](@ref), so every optimiser with an `r` field
    accepts it. [`MedianAbsoluteDeviation`](@ref) is a different type. By default it measures the
    deviation around the median, and only a clustering optimiser accepts it.

## 5. Choosing a measure

The choice depends on what you believe about the distribution of the returns:

  - If the returns are about symmetric and you care about their spread, use
    [`Variance`](@ref), the default, or [`StandardDeviation`](@ref).
  - If a loss in the left tail matters more than the spread, start with
    [`ConditionalValueatRisk`](@ref). [`EntropicValueatRisk`](@ref) is an upper bound on it, so it
    penalises the tail more. [`RelativisticValueatRisk`](@ref) generalises the entropic value at
    risk, and it tends to the entropic value at risk as its parameter `kappa` goes to zero.
  - If the path of the wealth matters as well as the distribution of the returns, use a
    drawdown measure, such as [`MaximumDrawdown`](@ref) or
    [`ConditionalDrawdownatRisk`](@ref).
  - If you do not trust the estimate of the covariance, use [`UncertaintySetVariance`](@ref).
  - If you want to give a weight to each sorted return, use [`OrderedWeightsArray`](@ref).
  - If trading costs matter, add [`TurnoverRiskMeasure`](@ref) to your main measure. It penalises
    the distance from the previous weights. To penalise the distance of the portfolio's returns
    from a benchmark, add [`TrackingRiskMeasure`](@ref).

[Multiple Risk Measures](../examples/3_optimisers/04_Multiple_Risk_Measures.md) shows how an
optimiser combines several measures in one objective. These pages cover the measures in depth:
[OWA Risk Measures](../examples/3_optimisers/05_OWA_Risk_Measures.md),
[Brownian Distance, Skew and Kurtosis](../examples/3_optimisers/06_Brownian_Distance_Variance_and_VarianceSkewKurtosis.md),
[Drawdown Risk Measures](../examples/3_optimisers/07_Drawdown_Risk_Measures.md), and
[Exotic Tail Risk Measures](../examples/3_optimisers/08_Exotic_Tail_Risk_Measures.md).

## 6. The measures on the distribution of the returns

[`plot_histogram`](@ref) draws the histogram of the returns of the equal-weighted portfolio, with
a vertical line for each of several measures. One line marks the mean. Three lines mark the mean
less the standard deviation, less the mean absolute deviation and less the Gini mean difference.
The other lines mark the value at risk, the conditional value at risk, the tail Gini and the worst
return. The value at risk is a quantile of the returns. The conditional value at risk is the mean
of the returns below that quantile, so its line is further to the left. The curve is the density
of the Normal distribution fitted to the returns. Where the histogram rises above the curve in the
left tail, the returns have a fatter tail than the Normal. `reference = false` removes the curve.
=#

plot_histogram(w, rd)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Closes card 9 of the 2026-07-19 ergonomics review: the ~57-measure family was reachable
#src   only from src/16_RiskMeasures + src/23_Aliases.jl, and the `LowOrderMoment(alg = …) ≡ MAD()`
#src   decoding had to be done by hand.
#src - Every table is generated: alias column reverse-maps the exported alias layer, `Optimisers`
#src   column comes from the ADR 0018 trait, `Expands to` is read off constructed objects. The
#src   only hand-written data is the one-line meaning, and the two `@assert`s make the docs build
#src   fail if a measure is added or removed without updating the catalogue.
#src - Absorbed the generated compatibility tick-table that previously lived in
#src   02_Optimisers.jl §2 (it carried the same trait information in a flat alphabetical list with
#src   no meanings); 02 now points here.
#src - Trap worth keeping: `MAD()` (mean, LowOrderMoment, JuMP-legal) vs `MedianAbsoluteDeviation`
#src   (median, hierarchical only).
