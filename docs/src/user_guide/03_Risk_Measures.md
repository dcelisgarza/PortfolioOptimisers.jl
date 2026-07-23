The source files can be found in [user_guide/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/user_guide/).

```@meta
EditURL = "../../../user_guide/03_Risk_Measures.jl"
```

# Risk measures

The previous page showed *which optimiser* to call. This one is the catalogue of *what you ask it
to minimise*: the `r` slot. `PortfolioOptimisers.jl` ships a large family of measures, and the
names alone do not tell you which is which — several of the best-known ones (mean absolute
deviation, semi-variance, Gini mean difference) are not types at all but *configurations* of a
generic type, reachable through a short alias.

This page is a reference, not a tutorial. It answers three questions in one place:

 1. What measures exist, and what does each penalise?
 2. Which optimisers accept it?
 3. What is its short alias, and what does that alias expand to?

Every table below is **generated from the type system and the compatibility trait**
([`supports_risk_measure`](@ref), ADR 0018), so it cannot drift from what the optimisers actually
dispatch on. The one-line meanings are curated, and the page fails the docs build if a measure is
added without one.

````@example 03_Risk_Measures
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, InteractiveUtils,
      StatsPlots, GraphRecipes
````

## 1. How a measure is used

A risk measure plays two roles, and they use two different call shapes.

**Inside an optimiser** it is a *configuration*: you hand it to the `r` field and the optimiser
turns it into constraints and an objective term.

**Outside an optimiser** it is a *callable functor*: you evaluate it on a book to get a number.
[`expected_risk`](@ref) is the uniform way in — it knows which of the three input shapes
(net returns, weights + returns + fees, weights alone) each measure wants, so you do not have to.

````@example 03_Risk_Measures
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
````

!!! note

    [`factory`](@ref) binds a measure to a prior — it is what fills in `sigma` for
    [`Variance`](@ref)/[`StandardDeviation`](@ref) and `mu` for the moment measures. Measures that
    read only the return path ([`ConditionalValueatRisk`](@ref), the drawdown family) do not need
    it, but calling `factory` unconditionally is always safe and is exactly what the optimisers do
    internally. The measures that solve a sub-problem to evaluate —
    [`EntropicValueatRisk`](@ref), [`RelativisticValueatRisk`](@ref), and their drawdown twins —
    additionally need a [`Solver`](@ref) in their `slv` field before you can call them outside an
    optimiser.

## 2. The three usage classes

Every measure sits in one of three classes, and the class is what determines where it is legal.
The `Optimisers` column in every table below is derived from it:

| Class | `Optimisers` column | Meaning |
| :-- | :-- | :-- |
| [`RiskMeasure`](@ref) | `JuMP + clustering` | Has a convex JuMP formulation. Usable everywhere: [`MeanRisk`](@ref), [`RiskBudgeting`](@ref), [`NearOptimalCentering`](@ref), [`FactorRiskContribution`](@ref), *and* the clustering optimisers. |
| [`HierarchicalRiskMeasure`](@ref) | `clustering only` | No JuMP formulation — it is only ever evaluated numerically, which clustering optimisers can do but a solver cannot. Passing one to `MeanRisk` is a type error, not a silent fallback. |
| `NonOptimisationRiskMeasure` | `diagnostic only` | Not an optimisation target at all: a reporting quantity ([`ExpectedReturn`](@ref), [`Skewness`](@ref)) or a ratio used for scoring in cross-validation. |

Meta-optimisers ([`NestedClustered`](@ref), [`Stacking`](@ref), [`SubsetResampling`](@ref)) have no
`r` of their own; they accept a measure only when *every* constituent optimiser does, so their
acceptance is instance-specific and is not tabulated. Ask directly:

```julia
supports_risk_measure(MeanRisk, ConditionalValueatRisk)   # true
supports_risk_measure(MeanRisk, EqualRisk)                # false — hierarchical only
supported_risk_measures(HierarchicalRiskParity)           # OptimisationRiskMeasure
```

## 3. The catalogue

The curated one-liner for each measure, its alias, and its class — grouped by what the measure
*looks at* rather than by source file. Every name below is exported and documented: the full
signature, fields, and references live under
[API → Risk Measures](../api/19_RiskMeasures/01_Base_RiskMeasures.md), and `?ConditionalValueatRisk`
in the REPL gets you there without leaving the terminal.

````@example 03_Risk_Measures
# Reverse-map the exported alias layer onto the measure types it names, so the alias column
# is read off the package rather than transcribed.
rm_alias = Dict{Symbol, String}()
for n in names(PortfolioOptimisers)
    v = getfield(PortfolioOptimisers, n)
    if isa(v, Type) && v <: PortfolioOptimisers.AbstractBaseRiskMeasure && nameof(v) != n
        rm_alias[nameof(v)] = String(n)
    end
end

# Class → the `Optimisers` column, derived from the ADR 0018 trait.
function usage_class(T)
    return if supports_risk_measure(MeanRisk, T)
        "JuMP + clustering"
    elseif supports_risk_measure(HierarchicalRiskParity, T)
        "clustering only"
    else
        "diagnostic only"
    end
end

# The curated catalogue: family => [measure => what it penalises].
catalogue = ["Dispersion and moments" =>
                 [:Variance => "Portfolio variance from a covariance matrix — the default `r`.",
                  :StandardDeviation => "Square root of the variance; same ordering, different scale.",
                  :UncertaintySetVariance => "Worst-case variance over a covariance uncertainty set (robust).",
                  :LowOrderMoment => "Generic first/second moment measure; see §4 for its aliases.",
                  :HighOrderMoment => "Generic third/fourth moment measure; see §4 for its aliases.",
                  :MedianAbsoluteDeviation => "Median absolute deviation — the robust cousin of `MAD()`.",
                  :Kurtosis => "Square-root kurtosis from the cokurtosis tensor (fat tails).",
                  :NegativeSkewness => "Downside asymmetry from the coskewness tensor.",
                  :VarianceSkewKurtosis => "Variance, skewness, and kurtosis combined in one expression.",
                  :BrownianDistanceVariance => "Distance variance — penalises *any* dependence, not just linear."],
             "Tail — X-at-Risk" => [:WorstRealisation => "The single worst observed loss.",
                                    :ValueatRisk => "The `alpha` quantile of the loss distribution (MIP).",
                                    :ConditionalValueatRisk => "Mean loss beyond the `alpha` quantile; expected shortfall.",
                                    :DistributionallyRobustConditionalValueatRisk => "CVaR under a Wasserstein ball around the empirical distribution.",
                                    :EntropicValueatRisk => "Exponential-cone upper bound on VaR; tighter tail control than CVaR.",
                                    :RelativisticValueatRisk => "Power-cone family interpolating between CVaR and worst realisation.",
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
             "Drawdown — uncompounded" =>
                 [:AverageDrawdown => "Mean depth of the drawdown path.",
                  :UlcerIndex => "Root-mean-square drawdown depth; penalises long deep spells.",
                  :MaximumDrawdown => "Deepest peak-to-trough loss.",
                  :DrawdownatRisk => "The `alpha` quantile of the drawdown path (MIP).",
                  :ConditionalDrawdownatRisk => "Mean drawdown beyond the `alpha` quantile.",
                  :DistributionallyRobustConditionalDrawdownatRisk => "CDaR under a Wasserstein ball.",
                  :EntropicDrawdownatRisk => "Exponential-cone bound on the drawdown quantile.",
                  :RelativisticDrawdownatRisk => "Power-cone drawdown family, CDaR → max drawdown.",
                  :PowerNormDrawdownatRisk => "Power-norm drawdown measure."],
             "Drawdown — compounded (relative)" =>
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
                  :NoRisk => "Contributes nothing — a null `r` for return-only problems."],
             "Non-optimisation (diagnostics and scoring)" =>
                 [:ExpectedReturn => "Prior expected return of the book.",
                  :MeanReturn => "Realised mean return of the book.",
                  :Skewness => "Standardised skewness of the return distribution.",
                  :ThirdCentralMoment => "Unstandardised third central moment.",
                  :NonOptimisationRiskRatio => "Ratio of any two non-optimisation measures.",
                  :ExpectedReturnRiskRatio => "Prior expected return over risk — a Sharpe-style score.",
                  :MeanReturnRiskRatio => "Realised mean return over risk."]]

# Every concrete measure must appear exactly once — this is what stops the page drifting.
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
                     "Alias" => [get(rm_alias, first(e), "—") for e in entries],
                     "Penalises" => [last(e) for e in entries],
                     "Optimisers" => [usage_class(getfield(PortfolioOptimisers, first(e)))
                                      for e in entries])
end;
nothing #hide
````

### Dispersion and moments

````@example 03_Risk_Measures
pretty_table(family_table("Dispersion and moments"))
````

### Tail — X-at-Risk

````@example 03_Risk_Measures
pretty_table(family_table("Tail — X-at-Risk"))
````

### Tail ranges (both sides)

````@example 03_Risk_Measures
pretty_table(family_table("Tail ranges (both sides)"))
````

### Drawdown — uncompounded

````@example 03_Risk_Measures
pretty_table(family_table("Drawdown — uncompounded"))
````

### Drawdown — compounded (relative)

The `Relative*` twins measure drawdown on the **compounded** wealth path rather than the
cumulative sum of returns. They have no convex JuMP formulation, which is why the whole family is
`clustering only`.

````@example 03_Risk_Measures
pretty_table(family_table("Drawdown — compounded (relative)"))
````

### Ordered weights arrays

````@example 03_Risk_Measures
pretty_table(family_table("Ordered weights arrays"))
````

### Path and mandate

````@example 03_Risk_Measures
pretty_table(family_table("Path and mandate"))
````

### Composite and structural

````@example 03_Risk_Measures
pretty_table(family_table("Composite and structural"))
````

### Non-optimisation (diagnostics and scoring)

These are *not* legal in an `r` slot. They exist to score a book after the fact and to serve as
cross-validation scorers — see [validation and tuning](05_Validation_and_Tuning.md).

````@example 03_Risk_Measures
pretty_table(family_table("Non-optimisation (diagnostics and scoring)"))
````

## 4. Measures that hide behind an `alg`

Three of the types above are *generic*: [`LowOrderMoment`](@ref), [`HighOrderMoment`](@ref), and
[`OrderedWeightsArray`](@ref) each host a whole family behind their `alg`/`w` field. Writing
`LowOrderMoment(; alg = MeanAbsoluteDeviation())` by hand is exactly the friction the alias layer
removes — every row below is a one-call constructor exported by the package.

The `Expands to` column is read off the constructed object, so it states what the alias actually
builds today.

````@example 03_Risk_Measures
alias_ctors = [("FLM", FLM, "First lower partial moment."),
               ("MAD", MAD, "Mean absolute deviation."),
               ("SCM", SCM,
                "Second central moment — scenario variance / standard deviation."),
               ("SLM", SLM, "Second lower moment — scenario semi-variance."),
               ("ECM", ECM, "Central even moment of order `2p`."),
               ("ELM", ELM, "Lower even moment of order `2p`."),
               ("TLM", TLM, "Third lower moment."),
               ("SSK", SSK, "Standardised third lower moment — semi-skewness."),
               ("FTCM", FTCM, "Fourth central moment."),
               ("FTLM", FTLM, "Fourth lower moment."),
               ("KT", KT, "Standardised fourth central moment — kurtosis."),
               ("SKT", SKT, "Standardised fourth lower moment — semi-kurtosis."),
               ("OWA_GMD", OWA_GMD, "Gini mean difference."),
               ("OWA_CVaR", OWA_CVaR, "CVaR as an OWA."), ("OWA_TG", OWA_TG, "Tail Gini."),
               ("OWA_WR", OWA_WR, "Worst realisation as an OWA."),
               ("OWA_RG", OWA_RG, "Range as an OWA."),
               ("OWA_CVaR_RG", OWA_CVaR_RG, "Two-sided CVaR as an OWA."),
               ("OWA_TG_RG", OWA_TG_RG, "Two-sided tail Gini."),
               ("OWA_LMoment", OWA_LMoment, "L-moment weights of order `k`.")]

# Walk the algorithm chain of a constructed measure so the expansion is observed, not asserted.
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
````

!!! warning "`MAD()` is not `MedianAbsoluteDeviation()`"

    [`MAD`](@ref) builds a [`LowOrderMoment`](@ref) with
    [`MeanAbsoluteDeviation`](@ref) — deviation around the **mean**, and a full
    [`RiskMeasure`](@ref) usable in any optimiser. [`MedianAbsoluteDeviation`](@ref) is a distinct
    type measuring deviation around the **median**, and it is `clustering only`. The names
    collide; the measures do not.

## 5. Picking one

The catalogue is long, but the choice collapses to what you believe about the return distribution:

- **Roughly symmetric, care about spread** — [`Variance`](@ref) (the default) or
    [`StandardDeviation`](@ref).
- **Left tail matters more than spread** — [`ConditionalValueatRisk`](@ref) first; reach for
    [`EntropicValueatRisk`](@ref) / [`RelativisticValueatRisk`](@ref) when you want to control the
    tail more tightly than CVaR does.
- **Path matters, not just the distribution** — the drawdown family, headed by
    [`MaximumDrawdown`](@ref) and [`ConditionalDrawdownatRisk`](@ref).
- **You distrust the covariance estimate** — [`UncertaintySetVariance`](@ref).
- **You want to shape the whole ordered loss curve** — [`OrderedWeightsArray`](@ref).
- **Trading costs bite** — add [`TurnoverRiskMeasure`](@ref) or
    [`TrackingRiskMeasure`](@ref) alongside your main measure.

Several measures can be combined in one objective — see
[Multiple Risk Measures](../examples/3_optimisers/04_Multiple_Risk_Measures.md) for how they are
scalarised. The deep dives are
[OWA Risk Measures](../examples/3_optimisers/05_OWA_Risk_Measures.md),
[Brownian Distance, Skew and Kurtosis](../examples/3_optimisers/06_Brownian_Distance_Variance_and_VarianceSkewKurtosis.md),
[Drawdown Risk Measures](../examples/3_optimisers/07_Drawdown_Risk_Measures.md), and
[Exotic Tail Risk Measures](../examples/3_optimisers/08_Exotic_Tail_Risk_Measures.md).

## 6. What the measures actually see

The clearest way to read the catalogue is to put the measures back on the return distribution they
summarise. [`plot_histogram`](@ref) draws the equal-weighted book's returns with each tail
measure marked where it falls: VaR cuts at a quantile, CVaR sits further left as the *mean* of
that tail, and the worst realisation anchors the end. Distance between the lines is exactly the
difference in what you are asking the optimiser to control.

````@example 03_Risk_Measures
plot_histogram(w, rd)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
