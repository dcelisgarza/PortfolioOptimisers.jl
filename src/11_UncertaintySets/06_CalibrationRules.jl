"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level, the tail probability that an `alpha` or a `beta` slot holds.

A rule of this family is a callable subtype of `AbstractSignificanceCalibrationAlgorithm`. [`resolve_calibration_slot`](@ref) runs a rule when it calls it, so a plain `Function` of the same five arguments is also a rule. [`Num_SigCal`](@ref) is the bound of every significance slot, and it names this family. So a constructor refuses a deformation rule in a significance slot. The slot names the end of the distribution, so one rule serves the lower tail and the upper tail.

# Interfaces

To add a significance rule, subtype `AbstractSignificanceCalibrationAlgorithm`, keep the parameters of the rule in the fields of the struct, and add the method below.

## The functor

  - `(alg::AbstractSignificanceCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv, ctx::CalibrationContext) -> Number`: Returns the significance level.

### Arguments

  - `key`: Name of the slot that the call resolves.
  - `pr`: Prior result from which the rule reads its sample size and its moments.
  - `w`: Effective observation weights, or `nothing` when neither the measure nor the prior gives any.
  - `slv`: Effective solver, or `nothing` when the measure has none.
  - `ctx`: The facts that the site knows and `key` does not give, as a [`CalibrationContext`](@ref). A rule reads the fields that it needs. A rule that reads no field writes `::CalibrationContext`.

### Returns

  - `alpha::Number`: The significance level.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`Num_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`ScenarioCount`](@ref): the rule of the family that keeps a stated count of tail scenarios.
  - [`RateSignificance`](@ref): the rule of the family that decreases the level at a square-root rate.
"""
abstract type AbstractSignificanceCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes a deformation parameter, the Kaniadakis ``\\kappa`` that a `kappa` slot holds.

A rule of this family is a callable subtype of `AbstractDeformationCalibrationAlgorithm`, as a rule of the significance family is. [`Num_DefCal`](@ref) is the bound of every deformation slot, and it names this family.

# Interfaces

To add a deformation rule, subtype `AbstractDeformationCalibrationAlgorithm`, keep the parameters of the rule in the fields of the struct, and add the method below.

## The functor

  - `(alg::AbstractDeformationCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv, ctx::CalibrationContext) -> Number`: Returns the deformation parameter.

### Arguments

  - `key`: Name of the slot that the call resolves.
  - `pr`: Prior result from which the rule reads its sample size and its moments.
  - `w`: Effective observation weights, or `nothing` when neither the measure nor the prior gives any.
  - `slv`: Effective solver, or `nothing` when the measure has none.
  - `ctx`: The facts that the site knows and `key` does not give, as a [`CalibrationContext`](@ref). A rule reads the fields that it needs. A rule that reads no field writes `::CalibrationContext`.

### Returns

  - `kappa::Number`: The deformation parameter.

## The series

A measure of the return distribution and a drawdown measure both have a deformation slot, and both resolve it under the key `:kappa`. So `key` does not tell which quantity the owner prices. A rule whose answer depends on that quantity reads `ctx.series`, which each slot owner sets in the [`CalibrationContext`](@ref) that it builds. A rule whose answer does not depend on it reads no field of the context. [`EntropyBudget`](@ref) is such a rule.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`CalibrationContext`](@ref)
  - [`Num_DefCal`](@ref)
  - [`kappa_log`](@ref)
  - [`EntropyBudget`](@ref): the rule of the family that spends a stated entropy budget.
  - [`HillTailDecay`](@ref): the rule of the family that reads the tail index of the series.
  - [`RadialTailDecay`](@ref): the rule of the family that reads the tail index of the radial series.
"""
abstract type AbstractDeformationCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
    const Num_SigCal = Union{AbstractSignificanceCalibrationAlgorithm, Function,
                             Number}

Field bound for a significance slot: the tail probability, a rule of the family, or a plain function of the same five arguments.

The slot names the end of the distribution, so the occupant does not have to. `alpha` names the lower tail and `beta` names the upper tail, and one rule serves both ends. So one bound serves every significance slot, and a Range measure sets its head slot to the occupant of its tail slot by default.

[`resolve_calibration_slot`](@ref) calls a rule, so a function and a callable struct are the same to it. A struct holds parameters and has a name that a docstring can describe. A function is the shortest way to state a rule for one use, and a closure over the data of a caller has no type. A function has no family, so this bound cannot refuse a deformation rule that is a closure. Only the name of the slot states the quantity there.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`Num_DefCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_SigCal = Union{AbstractSignificanceCalibrationAlgorithm, Function, Number}
"""
    const Num_DefCal = Union{AbstractDeformationCalibrationAlgorithm, Function,
                             Number}

Field bound for a deformation slot: the deformation parameter, a rule of the family, or a plain function of the same five arguments. It is the counterpart of [`Num_SigCal`](@ref) for the deformation family, and the description of that bound applies to it.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`Num_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_DefCal = Union{AbstractDeformationCalibrationAlgorithm, Function, Number}
"""
    is_calibration_rule(x) -> Bool

Returns `true` when the occupant `x` of a calibration slot is a **Calibration Rule**.

A slot can hold and call two kinds of rule: an [`AbstractCalibrationAlgorithm`](@ref), and a plain function of the same five arguments. Both return `true`. All other occupants return `false`: a stated number, `nothing`, and a child measure that a container declares.

A callable is not always a rule. [`calibration_slots`](@ref) also declares the children of a container. One such child is an ordered-weights builder in a reversal, which is a `Function` and computes no quantity. So this trait marks a rule, and the `Function` type does not. A type that is callable for a different purpose adds one method here that returns `false`. [`OWA_CalOccupant`](@ref) names the type that does this.

Three functions read the trait:

  - [`resolve_calibration_slot`](@ref) calls a rule and returns every other occupant unchanged.
  - [`assert_calibrated_slots`](@ref) refuses a rule that reached a value-level entry point.
  - [`assert_declared_calibration_resolver`](@ref) refuses a rule that the library did not resolve.

# Arguments

  - `x`: The occupant of the slot.

# Returns

  - `y::Bool`: Whether `x` is a rule.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`calibration_slots`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`assert_calibrated_slots`](@ref)
"""
is_calibration_rule(::Any)::Bool = false
is_calibration_rule(::AbstractCalibrationAlgorithm)::Bool = true
is_calibration_rule(::Function)::Bool = true
"""
    calibration_slots(x)

Declares the slots of `x` that can hold a **Calibration Rule**, as a `NamedTuple` from the name of each slot to its current value. The default is empty, so a type with no calibration slot needs no method.

It is the counterpart of [`deferred_slots`](@ref). Three functions read it:

  - [`assert_calibrated_slots`](@ref) refuses a rule that reached a value-level entry point.
  - [`assert_declared_calibration_resolver`](@ref) refuses a rule that the library did not resolve.
  - [`resolve_calibration_slots`](@ref) resolves the slots.

For most types, the library derives the resolution from this declaration. A type whose slots must resolve in an **order** writes its own [`resolve_deferred_quantities`](@ref) method instead. In such an order, one slot reads the resolved number of a sibling slot from a [`CalibrationContext`](@ref). A derivation cannot know which sibling to read. A slot that names its quantity under a key of its own also writes its own resolution. The three regularisation keys do this.

A slot that holds a child measure is declared here too. So a container names its children, and each child names its own slots. Such a slot is also declared in [`deferred_slots`](@ref), and the deferred recursion resolves the child. So only the two `assert_` walks read the declaration of a container. It names a child **only when that child can hold a slot**. A plain function in a calibration slot is a rule, so an occupant that has no slot is not named here. [`OWA_CalOccupant`](@ref) is the one bound in the library that states which occupants of a weight slot are children.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`assert_calibrated_slots`](@ref)
  - [`assert_declared_calibration_resolver`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`resolve_calibration_slots`](@ref)
  - [`deferred_slots`](@ref)
"""
calibration_slots(::Any) = (;)
"""
    assert_calibrated_slots(x)

Refuses a **Calibration Rule** that reached a value-level entry point, which has no prior result to resolve it against.

[`expected_risk`](@ref) takes a prior result or a plain returns matrix. With a prior result, it first resolves the measure through [`factory`](@ref). With a matrix, it cannot. A rule reads the sample size, the moments and the effective observation weights of the prior, and a returns matrix gives none of them. So the function refuses the rule, and the message names the slot, the rule in it and the remedy.

[`assert_resolved_slots`](@ref) does the same check for a Deferred Quantity. Both messages name a type with `nameof`, because a printed type has a module prefix when its name is not visible from `Main`, and the message must be the same in every process. A rule that is a plain function has no name of its own, so the message then names the slot alone.

The slots come from [`calibration_slots`](@ref), and the check recurses into each occupant. So the declarations of its children cover a container.

# Algorithm

 1. Walk the pairs that [`calibration_slots`](@ref) declares for `x`. Each pair gives the name `key` of a slot and its occupant `slot`.
 2. Refuse an occupant for which [`is_calibration_rule`](@ref) returns `true`.
 3. Recurse into the occupant, so the check also reads the slots of a child measure. Walk a slot that holds a vector of children element by element.
 4. Return `nothing` at the end of the walk.

# Validation

  - Throws an `ArgumentError` when a slot of `x`, or of a child that the walk reaches, holds a **Calibration Rule**. The message names the slot, the rule in it and the two remedies.

# Returns

  - `nothing`.

# Related

  - [`calibration_slots`](@ref)
  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`assert_declared_calibration_resolver`](@ref)
  - [`assert_resolved_slots`](@ref)
  - [`expected_risk`](@ref)
"""
function assert_calibrated_slots(x)
    for (key, slot) in pairs(calibration_slots(x))
        @argcheck(!is_calibration_rule(slot),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Calibration Rule, a `$(nameof(typeof(slot)))`, and this entry point has no prior result to resolve it against. A rule reads the sample size, the moments and the effective observation weights, which a bare returns matrix does not carry. Pass the prior result itself — `expected_risk(r, w, pr, fees)` — or resolve the measure first with `factory(r, pr)`."))
        assert_calibrated_slots(slot)
    end
    return nothing
end
function assert_calibrated_slots(xs::AbstractArray{<:Union{AbstractEstimator,
                                                           AbstractAlgorithm}})
    for x in xs
        assert_calibrated_slots(x)
    end
    return nothing
end
"""
    assert_declared_calibration_resolver(x, slots::NamedTuple)
    assert_declared_calibration_resolver(x)

Refuses a type that declares a calibration slot and does not resolve it.

`slots` is the output of the resolution. If a **Calibration Rule** remains in it, the type declared the slot in [`calibration_slots`](@ref) and wrote no resolution for it. With no check, the rule goes to the model builders, and they multiply it as a number. This function checks that each declaration has a resolution.

[`assert_declared_slot_resolver`](@ref) does the same check for a Deferred Quantity, and the two-argument method here has its form: the resolver gives the slots that it made. The one-argument method reads the slots from `x`. A caller that holds a resolved value, and not the slots that made it, calls this method.

The walk is the walk of [`assert_calibrated_slots`](@ref), and the two messages are different. That message names a caller who reached a value-level entry point. This message names a slot that the library did not resolve. So each message serves one failure.

The library derives no recursion for the calibration slots, so the pairing is a check and not a derivation. A rule that reads a sibling slot must resolve after that sibling, and a derivation cannot know the order. So each type writes its resolution next to its declaration, and this function refuses a declaration that has no resolution.

# Algorithm

 1. Walk the pairs of `slots`. Each pair gives the name `key` of a slot and its occupant `slot`.
 2. Refuse an occupant for which [`is_calibration_rule`](@ref) returns `true`.
 3. Return `nothing` at the end of the walk.

The one-argument method reads `slots` from [`calibration_slots`](@ref), and then recurses into each occupant, so it also checks the declarations of a child measure. It walks a slot that holds a vector of children element by element.

# Arguments

  - `x`: The slot owner, whose type the message names.
  - `slots`: The slots that the resolution made, or the declaration for the one-argument method.

# Validation

  - Throws an `ArgumentError` when an entry of `slots`, or of a child that the walk reaches, still holds a **Calibration Rule**. The message names the type, the slot and the resolution to write next to the declaration.

# Returns

  - `nothing`.

# Related

  - [`calibration_slots`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`assert_calibrated_slots`](@ref)
  - [`assert_declared_slot_resolver`](@ref)
"""
function assert_declared_calibration_resolver(x, slots::NamedTuple)
    for (key, slot) in pairs(slots)
        @argcheck(!is_calibration_rule(slot),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Calibration Rule, a `$(nameof(typeof(slot)))`, after the resolution ran, so `$(nameof(typeof(x)))` declares the slot in `calibration_slots` and resolves it nowhere. The declaration and the resolution travel together, because a rule that reads a sibling slot must be resolved after that sibling. Resolve the slot beside the declaration with `resolve_calibration_slot(x.$key, :$key, pr, pr.w, slv)`, or drop it from `calibration_slots`."))
    end
    return nothing
end
function assert_declared_calibration_resolver(x)
    slots = calibration_slots(x)
    assert_declared_calibration_resolver(x, slots)
    for slot in slots
        assert_declared_calibration_resolver(slot)
    end
    return nothing
end
function assert_declared_calibration_resolver(xs::AbstractArray{<:Union{AbstractEstimator,
                                                                        AbstractAlgorithm}})
    for x in xs
        assert_declared_calibration_resolver(x)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Names the series that a calibration rule reads, so that the rule reads the quantity that its slot owner prices.

A rule gets no portfolio, as [`resolve_calibration_slot`](@ref) states. So it cannot make the loss series of a portfolio, which exists only after the solver returns. A rule can make the series of each **column** of `pr.X`. This family names that series: the column, or the drawdown series of the column. A slot owner gives the series through the trait [`calibration_series`](@ref), and the [`CalibrationContext`](@ref) takes it to the rule.

The marker names a **quantity** and not an estimator, so each rule reads it in its own way:

  - [`HillTailDecay`](@ref) makes the series of each column and estimates its tail index.
  - [`RadialTailDecay`](@ref) cannot make a radial series of a path, because a Mahalanobis distance has no path. It whitens the rows of the drawdown sample instead.
  - The three radius rules read the per-asset dispersion of the series, which [`calibration_series_dispersion`](@ref) gives.
  - [`TailTermParity`](@ref) reads both terms of its ratio over the series.

# Related

  - [`ReturnsSeries`](@ref)
  - [`AbstractDrawdownSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_series_vec`](@ref)
  - [`CalibrationContext`](@ref)
"""
abstract type AbstractCalibrationSeries <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Names a drawdown series, the quantity that a drawdown measure prices.

A drawdown is a function of the path of one series, so a rule makes it per column and never per row. The two members use different compounding of the path. They are the series that [`absolute_drawdown_vec`](@ref) and [`relative_drawdown_vec`](@ref) make.

A drawdown series is not positive, so it has **one end**. So a rule that answers per end refuses a head key under this family. The refusal is a fact about the quantity and not about the sample.

**The tail of a drawdown series depends on the record.** A drawdown is a running function of the path, so its law over a finite record is not the law of one step. So a rule that reads the shape of a series reads a different shape here. A sample with a strong drift lets the drawdown process become stationary, and its drawdown series has a **heavier** tail than its returns. A sample with a weak drift gives the range of the path over the record, and that range has a **thinner** tail than the law of one step. Both are properties of the series that the measure prices over its record. So a rule describes the sample, and not drawdowns in general.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`RelativeDrawdownSeries`](@ref)
  - [`series_end_sign`](@ref)
"""
abstract type AbstractDrawdownSeries <: AbstractCalibrationSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the returns, the columns of `pr.X` with no change.

Every measure of the return distribution prices this series, and [`calibration_series`](@ref) returns it for every owner that names no other series.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct ReturnsSeries <: AbstractCalibrationSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the absolute drawdown series of a column, which [`absolute_drawdown_vec`](@ref) builds.

The path is the cumulative sum of the column, and the series is the distance of that path below its running peak. [`RelativisticDrawdownatRisk`](@ref) prices it. [`DrawdownatRisk`](@ref) uses the same series.

# Related

  - [`AbstractDrawdownSeries`](@ref)
  - [`RelativeDrawdownSeries`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct AbsoluteDrawdownSeries <: AbstractDrawdownSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the relative drawdown series of a column, which [`relative_drawdown_vec`](@ref) builds.

The path is the compounded value of the column, and the series is the fractional distance of that path below its running peak. [`RelativeRelativisticDrawdownatRisk`](@ref) prices it. [`RelativeDrawdownatRisk`](@ref) uses the same series.

# Related

  - [`AbstractDrawdownSeries`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct RelativeDrawdownSeries <: AbstractDrawdownSeries end
"""
    calibration_series(x)

Declares the series that `x` prices, so that a rule in one of its calibration slots reads that quantity.

The default is [`ReturnsSeries`](@ref), so a type that prices the return distribution needs no method. The two relativistic drawdown measures add a method, because they price a series that is not the sample that they hold.

This is a trait of the **slot owner** and not a field of the rule. The quantity belongs to the measure, and a rule cannot know which measure it reached. The [`CalibrationContext`](@ref) holds the norm order for the same reason. No rule holds a marker, so the answer of the owner overwrites nothing. A caller who runs a rule directly states the marker in the context.

# Arguments

  - `x`: The slot owner, usually a risk measure.

# Returns

  - `series::AbstractCalibrationSeries`: The series that the owner prices.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_slots`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
calibration_series(::Any) = ReturnsSeries()
"""
    calibration_series_vec(series::AbstractCalibrationSeries, x::VecNum)

Returns the univariate series that `series` names, made from one column of the sample.

A drawdown marker makes a new vector. [`ReturnsSeries`](@ref) returns the column, so the returns series allocates nothing.

The function takes one column and not the whole matrix, because a drawdown is a function of a path. It accumulates down a column and does not read across a row. A caller that needs the pool walks the columns and calls this function once for each column.

# Arguments

  - `series`: The series marker.
  - `x`: One column of the sample.

# Returns

  - `s::VecNum`: The series that the marker names.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`hill_tail_index`](@ref)
"""
function calibration_series_vec(::ReturnsSeries, x::VecNum)
    return x
end
function calibration_series_vec(::AbsoluteDrawdownSeries, x::VecNum)
    return absolute_drawdown_vec(x)
end
function calibration_series_vec(::RelativeDrawdownSeries, x::VecNum)
    return relative_drawdown_vec(x)
end
"""
    calibration_series_matrix(series::AbstractCalibrationSeries, X::AbstractMatrix)

Returns the sample that `series` names, one series for each column of `X`.

A drawdown marker calls the array function of its convention along `dims = 1`, the dimension of the observations. [`AbsoluteDrawdownSeries`](@ref) calls [`absolute_drawdown_arr`](@ref), and [`RelativeDrawdownSeries`](@ref) calls [`relative_drawdown_arr`](@ref). Those functions hold the convention of the running peak, so the matrix form and the vector form use one definition of a drawdown. The result has the shape of `X`. A drawdown series has one entry for each observation, so no observation is lost, and the count of a tail that the caller made on `X` stays correct.

[`ReturnsSeries`](@ref) returns `X`, so the returns sample allocates nothing. A rule that walks the columns one at a time calls [`calibration_series_vec`](@ref) instead, and allocates one column and not a matrix.

# Arguments

  - `series`: The series marker.
  - `X`: Returns matrix, `T × N`.

# Returns

  - `Y::AbstractMatrix`: The sample that the marker names, `T × N`.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_vec`](@ref)
  - [`absolute_drawdown_arr`](@ref)
  - [`relative_drawdown_arr`](@ref)
  - [`radial_series_inputs`](@ref)
"""
function calibration_series_matrix(::ReturnsSeries, X::AbstractMatrix)
    return X
end
function calibration_series_matrix(::AbsoluteDrawdownSeries, X::AbstractMatrix{<:Number})
    return absolute_drawdown_arr(X; dims = 1)
end
function calibration_series_matrix(::RelativeDrawdownSeries, X::AbstractMatrix{<:Number})
    return relative_drawdown_arr(X; dims = 1)
end
"""
    calibration_series_dispersion(series::AbstractCalibrationSeries, pr::AbstractPriorResult)

Returns the per-asset dispersion of the series that `series` names, one entry for each column of the sample.

A [`ReturnsSeries`](@ref) reads it from `pr.sigma`, as the square root of the diagonal of that matrix. So the dispersion of a prior that states a shrunk or a robust covariance matrix comes from that matrix.

A drawdown series reads it from the **drawdown sample**, because a prior result states no moment of a drawdown. [`calibration_series_matrix`](@ref) makes the sample, and the dispersion is the sample standard deviation of each of its columns. [`radial_series_inputs`](@ref) reads its centre and its whitening factor from the drawdown sample for the same reason. `pr.sigma` is a moment of the returns, and no scaling of it gives a moment of a drawdown.

A radius is a distance in the space of the scenarios that the model prices, so this vector gives the radius its units. [`ConcentrationRadius`](@ref) and [`DimensionalRateRadius`](@ref) average it into one scale. [`DualNormRadius`](@ref) divides it by the square root of the count that the site prices and takes a norm of the result.

# Mathematical definition

```math
\\begin{align}
\\hat{s}_{i} &= \\begin{cases}
\\sqrt{\\hat{\\mathbf{\\Sigma}}_{ii}} & \\textrm{for the returns series}\\\\
\\sqrt{\\dfrac{1}{T - 1} \\sum\\limits_{t=1}^{T} \\left(y_{ti} - \\bar{y}_{i}\\right)^{2}} & \\textrm{for a drawdown series}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:Sigma_hat_ii])
  - ``y_{ti}``: Entry ``t`` of the drawdown series of column ``i``.
  - ``\\bar{y}_{i}``: Sample mean of the drawdown series of column ``i``.
  - $(math_dict[:T])

# Arguments

  - `series`: The series that the slot owner prices.
  - `pr`: Prior result. A returns series reads its covariance matrix, and a drawdown series reads its sample.

# Returns

  - `s::AbstractVector`: The per-asset dispersion of that series, `N × 1`.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_matrix`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref)
"""
function calibration_series_dispersion(::ReturnsSeries, pr::AbstractPriorResult)
    return sqrt.(LinearAlgebra.diag(pr.sigma))
end
function calibration_series_dispersion(series::AbstractDrawdownSeries,
                                       pr::AbstractPriorResult)
    return vec(Statistics.std(calibration_series_matrix(series, pr.X); dims = 1))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the facts that a calibration site knows and the key of the slot does not give, from [`resolve_calibration_slot`](@ref) into the rule that it runs.

The site runs a rule as `alg(key, pr, w, slv, ctx)`. `key` names the **slot** and not the quantity. For example, [`RelativisticValueatRisk`](@ref) and [`RelativisticDrawdownatRisk`](@ref) both use `:kappa`, and they price different series. So the quantities below are properties of the site and not of the key, and the site states them in this type.

| Field    | The site that states it                                                       | The rules that read it                                                                                                                                                 |
|:-------- |:----------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`  | the slot owner, off a **sibling slot** it resolved first                      | [`EntropyBudget`](@ref), [`HillTailDecay`](@ref), [`RadialTailDecay`](@ref), [`TailTermParity`](@ref)                                                                  |
| `series` | the slot owner, through [`calibration_series`](@ref)                          | [`HillTailDecay`](@ref), [`RadialTailDecay`](@ref), [`ConcentrationRadius`](@ref), [`DimensionalRateRadius`](@ref), [`DualNormRadius`](@ref), [`TailTermParity`](@ref) |
| `p`      | the constraint or the penalty the quantity stands in                          | [`EffectiveAssetFloor`](@ref), [`DualNormRadius`](@ref)                                                                                                                |
| `cs`     | a [`CrossSectionalFactorPrior`](@ref), after the split of its Return Forecast | [`PrecisionBlend`](@ref), [`SteinShrinkage`](@ref), [`ForecastCalibrationSlope`](@ref)                                                                                 |
| `n`      | a site that prices the empirical measure of its own rows                      | [`ConcentrationRadius`](@ref), [`RateRadius`](@ref), [`DimensionalRateRadius`](@ref), [`DualNormRadius`](@ref)                                                         |

**No rule has a field for these quantities.** Each one belongs to the site, and a rule cannot know which site it reached. So a site overwrites no value of the rule, and no precedence between the two is necessary. A caller who runs a rule outside a measure builds the context that the site builds. That context is the only place where these quantities are written.

**Only the Cross-Sectional Factor Prior states `cs`.** Its Spanned Shrinkage `lambda` and its Orthogonal Forecast Scale `c` resolve inside [`prior`](@ref), after the prior splits the Return Forecast and before it makes the mean. The prior states `cs = (; g, ap, csfm, hist, cre)`. These are the spanned coefficients `g`, the unscaled orthogonal part `ap`, the factor-model block `csfm`, the Return Forecast history `hist` and the Cross-Sectional Regression Estimator `cre` of the prior. The block holds the idiosyncratic variance history `vs` and the regression weights `rw`, so a rule reads them from `csfm`. The block holds a zero `b` and no Return Forecast, because the scale is not resolved yet.

**The site states `n` because two sites price two different counts.** A distributionally robust measure centres its Wasserstein ball on the empirical measure of the rows of `pr.X` that its programme holds, weighted by its observation weights. The concentration bound of a radius is a statement about that measure, so its count is the count of those rows, which [`scenario_sample_size`](@ref) gives. A regularisation penalty prices the error of `pr.mu`, and that error decreases with the observations behind the moments, which [`effective_sample_size`](@ref) gives. The two counts differ under a Scenario Cap, which fits the moments over every observation and holds fewer rows. `key` cannot tell the rule which count it prices, so the site states it. A site that states no `n` prices the moments, and a rule then reads [`effective_sample_size`](@ref).

The default is the context of a site that states nothing. It has no sibling significance level, no norm order, no `cs` and no count. Its series is the [`ReturnsSeries`](@ref) that [`calibration_series`](@ref) returns for every owner that names no other series. A rule that needs a field that the default leaves at `nothing` refuses, and its message names that field.

**`alpha` is available only because the slots of one owner resolve in an order.** A deformation rule reads the significance level of a sibling slot. So the [`resolve_deferred_quantities`](@ref) method of the owner resolves `alpha` first, and puts the number in the context of the slot that reads it. No derivation can find that order, so each type writes its own resolution.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CalibrationContext(;
        alpha::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries(),
        p::Option{<:Number} = nothing,
        cs::Option{<:NamedTuple} = nothing,
        n::Option{<:Number} = nothing
    ) -> CalibrationContext

Keywords correspond to the fields of the struct. The default of each field is the value that a site that states nothing gives.

# Related

  - [`resolve_calibration_slot`](@ref)
  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_slots`](@ref)
  - [`ReturnsSeries`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
@concrete struct CalibrationContext <: AbstractResult
    """
    $(field_dict[:cal_ctx_alpha])
    """
    alpha
    """
    $(field_dict[:cal_ctx_series])
    """
    series
    """
    $(field_dict[:cal_ctx_p])
    """
    p
    """
    $(field_dict[:cal_ctx_cs])
    """
    cs
    """
    $(field_dict[:cal_ctx_n])
    """
    n
    function CalibrationContext(alpha::Option{<:Number}, series::AbstractCalibrationSeries,
                                p::Option{<:Number}, cs::Option{<:NamedTuple},
                                n::Option{<:Number})
        return new{typeof(alpha), typeof(series), typeof(p), typeof(cs), typeof(n)}(alpha,
                                                                                    series,
                                                                                    p, cs,
                                                                                    n)
    end
end
function CalibrationContext(; alpha::Option{<:Number} = nothing,
                            series::AbstractCalibrationSeries = ReturnsSeries(),
                            p::Option{<:Number} = nothing,
                            cs::Option{<:NamedTuple} = nothing,
                            n::Option{<:Number} = nothing)
    return CalibrationContext(alpha, series, p, cs, n)
end
"""
    resolve_calibration_slot(slot, key::Symbol, pr::AbstractPriorResult, w, slv = nothing,
                             ctx::CalibrationContext = CalibrationContext())

Resolves one calibration slot against the prior result `pr`, the effective observation weights `w`, the effective solver `slv` and the context `ctx` of the site, and returns a plain number.

The function calls a rule in a slot as `alg(key, pr, w, slv, ctx)`. So a callable rule and a plain function are the same here. It returns every other occupant, such as a stated number, unchanged.

A rule gets no portfolio. A prior result holds no portfolio weight vector, so no rule can measure the loss series of a portfolio. A rule can measure the series of each **column** of the sample, and `ctx.series` tells it which series the slot owner prices. A rule gets the solver on both routes that resolve a measure, so it can call [`ERM`](@ref) or [`RRM`](@ref):

  - On the [`factory`](@ref) route, [`@propagatable`](@ref) runs the `@cprop` selection before the resolution, so the struct holds the solver.
  - On the `JuMP` route, no selection runs. [`set_risk_constraints!`](@ref) passes the solver to [`resolve_deferred_quantities`](@ref), and the owner selects it as `sel(x.slv, slv)`.

**The context holds all the facts of the site, and this function rebuilds nothing.** `key` names the slot and not the quantity. So a rule reads from `ctx` the shape of a series, the significance level of a sibling slot, the norm order of its constraint, or the count of the scenarios that the site prices. No rule has a field for these facts, so the function rebuilds no occupant and overwrites nothing that a caller stated. [`CalibrationContext`](@ref) states which rule reads which field.

**This function checks the value of a ceiling rule and of a radius rule, and of no other family.** [`AbstractNormCeilingCalibrationAlgorithm`](@ref) and [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) share one method. It runs [`assert_nonempty_gt0_finite_val`](@ref) on the number that the rule returned, under the `key` of the slot. Both quantities reach a JuMP model from a field of [`JuMPOptimiser`](@ref): the ceiling from `l2c` and `linfc`, and the radius from `l1` and `linf`. No term is rebuilt on that route, so nothing later checks the range. The three other families are only in slots that their owner rebuilds. The rebuild runs the constructor of the owner on the calibrated number, and that constructor checks the range. Only the calibrated path pays for the check, because a stated number takes the first method.

**The site checks a rule that is a plain function.** A function has no family, so no method here can know the quantity that it computes. [`assemble_jump_model!`](@ref) is the one site that reads a calibrated number and rebuilds no term, and it checks the range of each of its four norm slots.

This function is the counterpart of [`resolve_slot`](@ref). It is a second function, and not a wider method of the first, for two reasons. The body of `resolve_slot` is `deferred_quantity(fit_deferred_quantity(dq, pr), key)`, a fit and then an extraction, and a rule fits nothing. Also, `resolve_slot` takes neither `w` nor `slv`, and a rule needs both. So the rule families are **not** members of the [`DeferredQuantity`](@ref) union.

The caller computes `w` as `sel(r.w, pr.w)`, and passes it with the `slv` of the measure. A parent that has no observation weights passes `pr.w`. A parent that has no solver leaves `slv` at its default.

**A [`TimeDependent`](@ref) reaches the estimator that holds the slot, and does not go further.** A schedule changes a *field of an estimator*, and [`update_time_dependent_fields`](@ref) applies it before the fit of a prior. A rule is always in a slot of an estimator, so a schedule replaces that estimator. If the estimator is a [`JuMPOptimiser`](@ref), its four norm fields can each have a schedule, and the schedule selects a rule for each fold. If the estimator is a risk measure, the bound of the slot admits no schedule. The caller then changes the whole measure through the risk-measure field of the optimiser, which can have a schedule. On both routes the selection runs first, and the rule then resolves against the prior of the selected period. So a rule needs no schedule of its own. A schedule in a rule names a fold that the rule cannot see, and it repeats the schedule of the estimator.

# Algorithm

 1. Return `slot` unchanged when [`is_calibration_rule`](@ref) returns `false` for it. A stated number takes this branch.
 2. Call the rule as `slot(key, pr, w, slv, ctx)`, and return the number that it gives.
 3. A norm-ceiling rule or an ambiguity-radius rule takes a more specific method. That method checks the number before it returns it.

# Arguments

  - `slot`: The occupant of the slot: a number or a rule.
  - `key`: Name of the slot that the call resolves.
  - `pr`: Prior result that the rule reads.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver, or `nothing` when the measure has none.
  - `ctx`: The facts that the site knows and `key` does not give, as a [`CalibrationContext`](@ref). The default has no sibling significance level, no norm order, and the returns series.

# Validation

  - A norm-ceiling rule or an ambiguity-radius rule must return a number that is `> 0` and finite. [`assert_nonempty_gt0_finite_val`](@ref) checks it under `key`. The function returns every other occupant with no check, and the owner of its slot checks the range when it rebuilds.

# Returns

  - `val::Number`: The calibrated quantity, or the stated value unchanged.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_slots`](@ref)
  - [`is_calibration_rule`](@ref)
  - [`Num_SigCal`](@ref)
  - [`resolve_slot`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
"""
function resolve_calibration_slot(slot, key::Symbol, pr::AbstractPriorResult, w,
                                  slv = nothing,
                                  ctx::CalibrationContext = CalibrationContext())
    return is_calibration_rule(slot) ? slot(key, pr, w, slv, ctx) : slot
end
"""
    effective_sample_size(pr::AbstractPriorResult, w::Option{<:ObsWeights})

Returns the number of observations behind the moments of `pr`, weighted by `w`.

A calibration rule that prices the sample size of the moments divides by this count, so those rules use one definition of it. The function reads three sources in order, and the first that gives a count wins:

 1. With weights, it is the effective sample size of [kish1965](@cite). That is the number of equally weighted observations that gives the precision of the weighted sample.
 2. With no weights, it is the count that the result **states** in `eoc`, when the result has no weights of its own. It is the smaller of the counts of the mean and of the covariance, because a rule does not know which moment it prices, and the sample is worth no more than its less informed moment. A Scenario Cap states there the number of observations over which it fitted the moments, and a weighted estimator of an [`EmpiricalPrior`](@ref) states the count of its weights. An entry of `nothing` is the row count of `pr.X`.
 3. Otherwise, it is the row count of `pr.X`, which is then the sample.

This function does not read an `eoc` that is next to a `w`. The weights `w` bind that `eoc`, because a pooling prior refits its moments under them. So a rule that reads no weights reads no count of them either. The function never reads `ens`, which is a diagnostic of `w`.

A rule that prices a property of the scenarios of the measure does not read this count. [`ScenarioCount`](@ref), [`RateSignificance`](@ref) and [`EntropyBudget`](@ref) read the rows that the measure holds when no weights are stated. A distributionally robust measure states the count of its rows, from [`scenario_sample_size`](@ref), in the `n` of its [`CalibrationContext`](@ref), and a radius rule reads that count in place of this one.

The function reads the weights through [`get_observation_weights`](@ref), so it accepts all of the [`ObsWeights`](@ref) union that the `w` field of a rule admits. A [`DynamicAbstractWeights`](@ref) that has no method for a column of the sample raises [`ObservationWeightsError`](@ref), which names the two signatures to write.

# Mathematical definition

```math
T_{e} = \\begin{cases}
\\dfrac{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}{\\sum\\limits_{t=1}^{T} w_{t}^{2}} & \\textrm{if } w \\textrm{ is given}\\\\
\\min\\left(T_{\\mu},\\, T_{\\Sigma}\\right) & \\textrm{if } w \\textrm{ is } \\texttt{nothing} \\textrm{ and the result states no weighting}\\\\
T & \\textrm{otherwise}
\\end{cases}\\,.
```

Where:

  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])
  - ``T_{\\mu}``, ``T_{\\Sigma}``: The effective numbers of observations of the mean and of the covariance that the result states in `eoc`. An entry of `nothing` is ``T``.

# Algorithm

 1. Resolve `w` against the first column of `pr.X` through [`get_observation_weights`](@ref), which gives `ws`.
 2. When `ws` is not `nothing`, return the count of Kish for it, through [`weight_observation_count`](@ref).
 3. Otherwise, return [`stated_sample_size`](@ref) of the `w`, the `eoc` and the `X` of the result. That is the smaller of the counts of the mean and of the covariance when the result has no `pr.w`, and `size(pr.X, 1)` otherwise.

# Arguments

  - `pr`: Prior result from which the function reads the sample length.
  - `w`: Effective observation weights, or `nothing`.

# Validation

  - `w` resolves through [`get_observation_weights`](@ref). That function raises [`ObservationWeightsError`](@ref) for a [`DynamicAbstractWeights`](@ref) that has no method for a column of `pr.X`.

# Returns

  - `T_e::Number`: The effective sample size.

# Related

  - [`get_observation_weights`](@ref)
  - [`stated_sample_size`](@ref)
  - [`ObsWeights`](@ref)
  - [`scenario_sample_size`](@ref): the count of the rows that a measure prices.
  - [`ScenarioCount`](@ref): reads this count only when observation weights are stated.
  - [`ConcentrationRadius`](@ref): reads this count when the site states no `n`.
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref)

# References

  - $(ref_dict[:kish1965]) Whole book. $(ref_dict[:no_original_source])
"""
function effective_sample_size(pr::AbstractPriorResult, w::Option{<:ObsWeights})
    ws = get_observation_weights(w, view(pr.X, :, 1))
    return if isnothing(ws)
        stated_sample_size(pr.w, pr.eoc, pr.X)
    else
        weight_observation_count(KishObservationCount(), ws)
    end
end
"""
    stated_sample_size(::Nothing, eoc::Tuple, X::MatNum)
    stated_sample_size(w, eoc, X::MatNum)

Returns the count that a Prior Result states in `eoc` next to no weights, and the row count of `X` otherwise.

It is the branch of [`effective_sample_size`](@ref) for no weights, and it dispatches on the `w` of the result. With no `w`, the count is the smaller of the counts of the mean and of the covariance, `eoc[1]` and `eoc[2]`, and an entry of `nothing` is `size(X, 1)`. A Scenario Cap writes there the number of observations behind its moments, and a weighted estimator the count of its weights. The weights `w` bind an `eoc` next to them, and a read with no weights does not take it. A result that states no count holds its sample in `X`.

# Arguments

  - `w`: The observation weights of the result, or `nothing`.
  - `eoc`: The counts that the result states. A [`HighOrderPrior`](@ref) states four, and the function reads the first two.
  - `X`: The returns matrix of the result, `observations × assets`.

# Returns

  - `T::Number`: The smaller count of the mean and the covariance when the result has no `w`, and `size(X, 1)` otherwise.

# Related

  - [`effective_sample_size`](@ref)
  - [`moment_observation_count`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function stated_sample_size(::Nothing, eoc::Tuple, X::MatNum)
    T = size(X, 1)
    return min(something(eoc[1], T), something(eoc[2], T))
end
function stated_sample_size(::Any, ::Any, X::MatNum)
    return size(X, 1)
end
"""
    scenario_sample_size(pr::AbstractPriorResult, w::Option{<:ObsWeights})

Returns the count of the empirical measure of the rows that `pr` holds, weighted by `w`.

A risk measure prices the rows of `pr.X`, and not the observations behind the moments. With weights, the measure is the weighted empirical measure of those rows, so the count is the effective sample size of [kish1965](@cite), as in [`effective_sample_size`](@ref). With no weights, the measure gives each row the same mass, so the count is the row count of `pr.X`.

The two counts differ under a Scenario Cap, and under a weighted estimator of the moments. A cap fits the moments over every observation, states that number in `eoc`, and holds fewer rows. [`effective_sample_size`](@ref) reads `eoc` there, and this function reads the rows. The `eoc` of an entropy-pooling prior is next to a `w`, so neither count reads it.

The function resolves `w` through [`get_observation_weights`](@ref), as the programme of a measure does. So a [`DynamicAbstractWeights`](@ref) that resolves to `nothing` gives the row count, which is the count of the measure that the programme then prices.

# Mathematical definition

```math
T_{n} = \\begin{cases}
\\dfrac{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}{\\sum\\limits_{t=1}^{T} w_{t}^{2}} & \\textrm{if } w \\textrm{ is given}\\\\
T & \\textrm{otherwise}
\\end{cases}\\,.
```

Where:

  - $(math_dict[:T]) It is the row count of the sample that the result holds.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:cal_T_n])

# Algorithm

 1. Resolve `w` against the first column of `pr.X` through [`get_observation_weights`](@ref), which gives `ws`.
 2. When `ws` is `nothing`, return `size(pr.X, 1)`.
 3. Otherwise, return the count of Kish for `ws`, through [`effective_sample_size`](@ref).

# Arguments

  - `pr`: Prior result whose rows the measure prices.
  - `w`: Effective observation weights, or `nothing`.

# Returns

  - `T_n::Number`: The count of the scenarios.

# Related

  - [`effective_sample_size`](@ref)
  - [`ScenarioCount`](@ref)
  - [`CalibrationContext`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)

# References

  - $(ref_dict[:kish1965]) Whole book. $(ref_dict[:no_original_source])
"""
function scenario_sample_size(pr::AbstractPriorResult, w::Option{<:ObsWeights})
    ws = get_observation_weights(w, view(pr.X, :, 1))
    return isnothing(ws) ? size(pr.X, 1) : effective_sample_size(pr, ws)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level from a count of scenarios, so that the tail keeps the same number of scenarios at every sample length.

A stated `alpha` fixes the probability of the tail. A fold of half the length then has half the scenarios in its tail, and the measure becomes less accurate as the sample becomes shorter. This rule fixes the count and returns `n / T`. So `alpha * T` is `n` at every sample length, and the tail of the measure holds `n` scenarios.

`T` is the row count of `pr.X` when no observation weights are stated, because the measure prices those rows. Under a Scenario Cap, the rows are fewer than the observations behind the moments, and the rule reads the rows. When weights are stated, `T` is the effective sample size of the weights. That is the number of equally weighted observations that gives the precision of the weighted sample. So a weighted tail holds `n` observations in that sense, and not `n` rows.

The rule does not check the range of its value. It returns the quantity of its slot, so the constructor of the slot owner does all the validation. That constructor refuses a value outside the range of the slot when the fold runs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ScenarioCount(;
        n::Option{<:Number} = nothing
    ) -> ScenarioCount

Keywords correspond to the fields of the struct. No default of `n` suits every sample, so the default is `nothing`, which is not a count. So the constructor refuses `ScenarioCount()`, and the message names the count and tells why the rule needs it. The keyword admits `nothing` only for this refusal, and the field never holds it.

## Validation

  - `n` is stated, else an `ArgumentError`.
  - `n > 0` and finite.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`RateSignificance`](@ref)
  - [`Num_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct ScenarioCount <: AbstractSignificanceCalibrationAlgorithm
    """
    $(field_dict[:cal_n])
    """
    n
    function ScenarioCount(n::Number)
        assert_nonempty_gt0_finite_val(n, :n)
        return new{typeof(n)}(n)
    end
end
function ScenarioCount(; n::Option{<:Number} = nothing)
    @argcheck(!isnothing(n),
              ArgumentError("`ScenarioCount` states no default for `n`, the number of observations the tail is to hold. It is a count and not a probability, and no count suits every sample and every measure, so the rule asks for one: `ScenarioCount(; n = 15)`. `RateSignificance` is the rule of this family that computes a significance level from a rate instead, and it constructs bare."))
    return ScenarioCount(n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the significance level that leaves `alg.n` scenarios in the tail of the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
\\alpha &= \\frac{n}{T_{n}}\\,,\\\\
T_{n} &= \\begin{cases}
\\dfrac{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}{\\sum\\limits_{t=1}^{T} w_{t}^{2}} & \\textrm{if } w \\textrm{ is given}\\\\
T & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_rm])
  - ``n``: Number of scenarios that the tail must hold.
  - $(math_dict[:cal_T_n])
  - $(math_dict[:T]) It is the row count of the sample that the result holds.
  - $(math_dict[:w_t_obs])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The count is the same for every key, so a tail slot and a head slot that hold one rule resolve to one number.
  - `pr`: Prior result from which the rule reads the sample length.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. This rule reads no field of it.

# Returns

  - `alpha::Number`: The significance level.

# Related

  - [`ScenarioCount`](@ref)
  - [`scenario_sample_size`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::ScenarioCount)(::Symbol, pr::AbstractPriorResult, w, ::Any,
                              ::CalibrationContext)
    return alg.n / scenario_sample_size(pr, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level that decreases with the square root of the sample length.

The tail probability is `c / sqrt(T)`, so the expected count of the tail is `c * sqrt(T)`. The count increases with the sample, but more slowly than the sample. The error of a sample mean decreases at the same rate. So a longer sample moves the tail further out and also puts more scenarios in it. [`ScenarioCount`](@ref) keeps the count fixed.

**`T` is the count of the scenarios that the measure prices, as for [`ScenarioCount`](@ref).** The tail is a set of the rows of `pr.X`, so the expected count of the tail is `c * sqrt(T)` only when `T` counts those rows. That is [`scenario_sample_size`](@ref): the row count of `pr.X` when no observation weights are stated, and the effective sample size of the weights otherwise. Under a Scenario Cap, the rows are fewer than the observations behind the moments, and the rule reads the rows. A read of the count in `eoc` there gave a tail of `sqrt(5)` times fewer scenarios than `c * sqrt(T)`, for `max_scenarios = 250` over 1250 observations.

The rule does not check the range of its value, for the reason that [`ScenarioCount`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RateSignificance(;
        c::Number = 1
    ) -> RateSignificance

Keywords correspond to the fields of the struct. The default `c = 1` gives the rate ``1/\\sqrt{T}``.

## Validation

  - `c > 0` and finite.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`ScenarioCount`](@ref)
  - [`Num_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct RateSignificance <: AbstractSignificanceCalibrationAlgorithm
    """
    $(field_dict[:cal_c])
    """
    c
    function RateSignificance(c::Number)
        assert_nonempty_gt0_finite_val(c, :c)
        return new{typeof(c)}(c)
    end
end
function RateSignificance(; c::Number = 1)
    return RateSignificance(c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the significance level at the square-root rate of the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
\\alpha &= \\frac{c}{\\sqrt{T_{n}}}\\,,\\\\
T_{n} &= \\begin{cases}
\\dfrac{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}{\\sum\\limits_{t=1}^{T} w_{t}^{2}} & \\textrm{if } w \\textrm{ is given}\\\\
T & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_rm])
  - ``c``: Rate coefficient.
  - $(math_dict[:cal_T_n])
  - $(math_dict[:T]) It is the row count of the sample that the result holds.
  - $(math_dict[:w_t_obs])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The rate is the same for every key.
  - `pr`: Prior result from which the rule reads the sample length.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. This rule reads no field of it.

# Returns

  - `alpha::Number`: The significance level.

# Related

  - [`RateSignificance`](@ref)
  - [`ScenarioCount`](@ref)
  - [`scenario_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::RateSignificance)(::Symbol, pr::AbstractPriorResult, w, ::Any,
                                 ::CalibrationContext)
    return alg.c / sqrt(scenario_sample_size(pr, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Kaniadakis deformation parameter that makes a relativistic measure spend a stated entropy budget.

[`RRM`](@ref) multiplies its dual variable by `kappa_log(inv(alpha * T), kappa)`, where `T` is the row count of the sample of the measure. So that coefficient is the price that the model pays for the deformation, and `target` states it. The rule returns the ``\\kappa`` that gives it. A stated `kappa` fixes the shape of the deformation and lets the price change with the sample. This rule fixes the price and lets the shape change. The Kaniadakis logarithm is Equation 63 of [kaniadakis2001](@cite).

The rule reads its sibling `alpha` from `ctx.alpha`. So the resolution method of the owner resolves `alpha` first and puts the number in the context of the slot that reads it. A caller who runs the rule outside a measure states `alpha` in the context.

The coefficient is monotone in ``\\kappa``, and the rule inverts it by bisection over ``(0, 1)``. Let ``l = \\ln(u)``. The coefficient is ``l \\sinh(\\kappa l) / (\\kappa l)``, and ``\\sinh(x)/x`` increases with ``|x|``. So the coefficient moves once from ``l`` at ``\\kappa \\to 0`` to ``\\sinh(l)`` at ``\\kappa = 1``. The function ``\\sinh(x)/x`` has no elementary inverse, so the rule runs a fixed sequence of 64 halvings.

The band that the coefficient reaches changes with the sample, and the rule refuses a target outside the band when the fold runs. This check does not limit the range of the value that the rule returns. The constructor of the slot owner does that, as it does for the two significance rules. The check states that the equation has a root. With no check, a target outside the band stops the bisection at an end of the interval. That ``\\kappa`` is inside the range that the slot owner accepts, but it does not give the target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropyBudget(;
        target::Option{<:Number} = nothing
    ) -> EntropyBudget

Keywords correspond to the fields of the struct. No default of `target` is useful, because the band of the target changes with the sample. So the default is `nothing`, which is not a budget, and the constructor refuses `EntropyBudget()`. The message names the budget and tells why the rule needs it. The keyword admits `nothing` only for this refusal, and the field never holds it.

## Validation

  - `target` is stated, else an `ArgumentError`.
  - `target` is finite. The band of the target changes with the sample, so the rule checks the band when it runs, and not here.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`CalibrationContext`](@ref)
  - [`Num_DefCal`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`RRM`](@ref)

# References

  - $(ref_dict[:kaniadakis2001]) Equation 63.
"""
@concrete struct EntropyBudget <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_target])
    """
    target
    function EntropyBudget(target::Number)
        assert_nonempty_finite_val(target, :target)
        return new{typeof(target)}(target)
    end
end
function EntropyBudget(; target::Option{<:Number} = nothing)
    @argcheck(!isnothing(target),
              ArgumentError("`EntropyBudget` states no default for `target`, the value of the Kaniadakis logarithm the measure is to spend. The band that value must lie in moves with the sample, so no default states anything and the rule asks for one: `EntropyBudget(; target = -1.3)`. `HillTailDecay` and `RadialTailDecay` are the rules of this family that read the tail of the sample instead, and both construct bare."))
    return EntropyBudget(target)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the deformation parameter whose Kaniadakis logarithm gives `alg.target` on the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
u &= \\frac{1}{\\alpha T}\\,,\\\\
\\ln_{\\kappa}(u) &= \\frac{u^{\\kappa} - u^{-\\kappa}}{2 \\kappa}\\,,\\\\
\\kappa &: \\ln_{\\kappa}(u) = \\tau\\,, \\quad \\kappa \\in (0,\\, 1)\\,.
\\end{align}
```

Where:

  - $(math_dict[:kappa_rm])
  - ``\\tau``: Target value of the Kaniadakis logarithm.
  - ``u``: Argument of the Kaniadakis logarithm, the reciprocal of the expected count of the tail.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T]) It is the row count of the sample of the measure, and under a Scenario Cap it is the count of the rows that the result holds.

# Algorithm

 1. Read the sample length `T` as the row count of `pr.X`, the scenarios over which the model of the measure forms its coefficient.
 2. Make the argument `u = inv(ctx.alpha * T)` and its natural logarithm `l = log(u)`.
 3. Make the band `(lo_b, hi_b)` as the ordered pair of `l` and `(u - inv(u)) / 2`. These are the values of the Kaniadakis logarithm at the two ends of ``\\kappa \\in (0,\\, 1)``. `# Validation` states the refusal of this band.
 4. Normalise the target as `target = alg.target / l`. The normalised coefficient increases once from `1`, so one comparison serves both signs of `l`, and the bisection needs no branch for the sign.
 5. Set the bracket of the answer to `lo = 0` and `hi = 1`.
 6. Halve the bracket 64 times. Each time, take the midpoint `kappa`. Set `lo` to it when `kappa_log(u, kappa) / l` is below `target`, and set `hi` to it otherwise. After 64 halvings, the bracket is smaller than the resolution of a `Float64`.
 7. Return the midpoint of the last bracket.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The budget is the same for every key.
  - `pr`: Prior result from which the rule reads the sample length.
  - `w`: Effective observation weights. The coefficient of the model counts rows, so the rule does not read the weights.
  - `slv`: Effective solver. This rule needs none, because the inversion is scalar.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.alpha` is the sibling significance level, and it must hold a number.

# Validation

  - `ctx.alpha` must not be `nothing`.
  - `alg.target` must be strictly between ``\\ln(u)`` and ``\\sinh(\\ln(u))``, which is the band of the coefficient over ``\\kappa \\in (0,\\, 1)``.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`EntropyBudget`](@ref)
  - [`CalibrationContext`](@ref)
  - [`kappa_log`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:kaniadakis2001]) Equation 63.
"""
function (alg::EntropyBudget)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any,
                              ctx::CalibrationContext)
    @argcheck(!isnothing(ctx.alpha),
              IsNothingError("`CalibrationContext.alpha` is `nothing`, so `EntropyBudget` cannot form `inv(alpha * T)`. The sibling `alpha` reaches the rule in the context the slot owner builds, after it resolves `alpha`. State `alpha` in the context to run the rule outside a measure."))
    T = size(pr.X, 1)
    u = inv(ctx.alpha * T)
    l = log(u)
    lo_b, hi_b = minmax(l, (u - inv(u)) / 2)
    @argcheck(lo_b < alg.target < hi_b,
              DomainError(alg.target,
                          "`EntropyBudget.target` must lie in ($lo_b, $hi_b), the band that `kappa_log(inv(alpha * T), kappa)` reaches over `kappa` in (0, 1) at `alpha = $(ctx.alpha)` and `T = $T`. No deformation parameter meets a target outside it, so the rule has nothing to return. The band moves with the sample, so a target that suits one fold need not suit another."))
    target = alg.target / l
    lo = zero(target)
    hi = one(target)
    # The normalised coefficient rises once from `1` to `sinh(l) / l` over `(0, 1)`, so one
    # comparison carries both signs of `l` and the sweep needs no sign branch. Sixty-four
    # halvings take the bracket below the resolution of a `Float64`.
    for _ in 1:64
        kappa = (lo + hi) / 2
        if kappa_log(u, kappa) / l < target
            lo = kappa
        else
            hi = kappa
        end
    end
    return (lo + hi) / 2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Estimates the tail index of the pool of standardised values of `X`, over the `k` most extreme order statistics of the pool.

`series` names the series of each column, and [`calibration_series_vec`](@ref) makes it. The function centres each series and divides it by its sample standard deviation, and `s` names the end. The estimator of [hill1975](@cite) reads the `k` of the `T N` standardised values that are furthest into that end. [`HillTailDecay`](@ref) states the assumptions of the pool. This function makes the estimate alone.

The pool holds `T` values for each column under every marker, because a drawdown series has one entry for each observation. So the function reads the count `k` that the caller made, for every series.

The signature binds the element type, so the pool and its sum have concrete types. A rule reads `pr.X` from an [`AbstractPriorResult`](@ref), whose field types no signature states, and this function is the boundary where the type becomes concrete.

# Algorithm

 1. Read the shape of `X` into `T` and `N`, and allocate the pool `pool` of `T * N` values.
 2. Walk the columns of `X`. Make the series of each column with [`calibration_series_vec`](@ref), take its sample mean `mu` and its sample standard deviation `sd`, and write `-s * (col[t] - mu) / sd` into `pool`.
 3. Partially sort `pool`, so that its `k + 1` smallest values are in order at the start, and read the last of them into `vkp1`. `partialsort!` with one index puts only that index in place, so the range is necessary to put the `k` entries of the estimate before `vkp1`. The sign of step 2 puts the end that the caller prices in the **lower** tail of the pool.
 4. Return `k` divided by the sum of `log(pool[i] / vkp1)` over those `k` entries. Both terms of each ratio are negative, so the ratio is a ratio of magnitudes, and the sum needs no change of sign.

# Arguments

  - `series`: The series of each column.
  - `X`: Returns matrix, `T × N`.
  - `s`: Sign of the end that the estimate reads. `1` reads the gain tail, and `-1` reads the loss tail.
  - `k`: Number of order statistics that the estimate reads. The caller states it and checks its floor.

# Validation

  - The `k + 1`-th largest pooled value must be positive.

# Returns

  - `a::Number`: The Hill estimate of the tail index of the pool.

# Related

  - [`HillTailDecay`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_vec`](@ref)

# References

  - $(ref_dict[:hill1975]) Pages 1163–1174. $(ref_dict[:no_original_source])
"""
function hill_tail_index(series::AbstractCalibrationSeries, X::AbstractMatrix{E},
                         s::Integer, k::Integer) where {E <: Number}
    T, N = size(X)
    pool = Vector{E}(undef, T * N)
    for j in axes(X, 2)
        col = calibration_series_vec(series, view(X, :, j))
        mu = Statistics.mean(col)
        sd = Statistics.std(col; mean = mu)
        o = (j - 1) * T
        for t in 1:T
            pool[o + t] = -s * (col[t] - mu) / sd
        end
    end
    # The pool is signed so that the end being priced is the LOWER tail, and the estimate
    # reads the `k + 1` smallest. The ratio of two negatives is the ratio of their
    # magnitudes, so the sum below is Hill's with no further sign, and the message reads the
    # magnitude the caller thinks in.
    partialsort!(pool, 1:(k + 1))
    vkp1 = pool[k + 1]
    ukp1 = -vkp1
    @argcheck(ukp1 > 0,
              DomainError(ukp1,
                          "The $(k + 1)-th largest of the pooled standardised values is $ukp1, which is not positive, so there is no Hill estimate: the estimator reads `log(u_i / u_(k+1))`, which needs a positive ratio. The pool holds fewer than $(k + 1) values on the side of the mean this end prices. Lower the count `k`."))
    return k / sum(i -> log(pool[i] / vkp1), 1:k)
end
"""
    series_end_sign(series::AbstractCalibrationSeries, key::Symbol)

Returns the sign of the end that `key` prices on `series`, and refuses an end that the series does not have.

`:kappa_b` is the only head key, so on a returns series every other key prices the loss end. A drawdown series is not positive and has only one end. So the head key names no end on it, and the function refuses it. The library has no drawdown Range measure, so only a caller who runs a rule directly gets this refusal.

# Arguments

  - `series`: The series of the estimate.
  - `key`: Name of the slot that the call resolves.

# Validation

  - `key` must not be `:kappa_b` when `series` is an [`AbstractDrawdownSeries`](@ref).

# Returns

  - `s::Int`: `1` for the gain end and `-1` for the loss end.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`AbstractDrawdownSeries`](@ref)
  - [`HillTailDecay`](@ref)
  - [`hill_tail_index`](@ref)
"""
function series_end_sign(::ReturnsSeries, key::Symbol)
    # `:kappa_b` is the only head key, so every other key prices the loss end. One estimator
    # then serves both ends.
    return key === :kappa_b ? 1 : -1
end
function series_end_sign(series::AbstractDrawdownSeries, key::Symbol)
    @argcheck(key !== :kappa_b,
              ArgumentError("The `:kappa_b` slot prices the gain end, and a $(nameof(typeof(series))) has no gain end: a drawdown series is non-positive, so its one end is the loss end that `:kappa` and `:kappa_a` price. No drawdown Range measure ships, so a rule reached this key outside a measure. Resolve the rule under `:kappa`, or state a `ReturnsSeries`."))
    return -1
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Kaniadakis deformation parameter whose tail decays at the rate of the tail of the sample.

The inverse of [`kappa_log`](@ref) is the ``\\kappa``-exponential ``\\exp_{\\kappa}(x) = \\left(\\kappa x + \\sqrt{1 + \\kappa^{2} x^{2}}\\right)^{1/\\kappa}`` of Equation 53 of [kaniadakis2001](@cite). It approaches ``(2 \\kappa x)^{1/\\kappa}`` for large ``x``, as page 20 of [kaniadakis2001](@cite) states. So a ``\\kappa``-deformed exponential tail is a power law of index ``1/\\kappa``, and ``\\kappa`` is the reciprocal of a tail index. The rule estimates the tail index ``\\hat{a}`` of the sample and returns ``1/\\hat{a}``, so the deformation decays at the rate of the sample. The band of the slot, ``(0,\\, 1)``, is ``\\hat{a} > 1``, which is the condition for a finite mean. So the rule refuses an estimate outside the band, as [`EntropyBudget`](@ref) refuses a target outside its band.

The estimator is that of [hill1975](@cite), over the `k` most extreme order statistics of a **pool**. The rule centres each column of `pr.X` and divides it by its sample standard deviation. It gives the standardised values the sign of the end that the slot prices, and it pools all `T * N` of them. **The rule reads only `pr.X`, and never `pr.sigma`.** The dispersion of a column comes from that column. This is the difference from [`RadialTailDecay`](@ref). This rule measures how far one end of one column moves, and its sibling measures how far the whole cross-section moves. So this rule keeps the sign of the end and gives one answer for each end. Its sibling reads a distance and gives one answer for both ends.

The pool has two assumptions:

  - The columns have one tail index after standardisation, so one estimate serves the whole sample.
  - The columns are cross-correlated. So the pool holds far fewer than `T * N` independent points, and the spread of the estimator is wider than the spread for `k` independent points.

The median of `N` estimates, one for each asset, needs neither assumption, but it is less accurate. At `alpha = 0.05` and `T = 250`, a column has 13 tail points, and a Hill estimate on 13 points changes from fold to fold with no cause in the data.

`alpha` fixes the depth of the estimate as well as the count. A Hill estimate reads the tail at the depth that `alpha` names. If the tail of a sample approaches its power law slowly, the estimate has a bias that decreases as the depth increases. For example, a Student-t sample at a depth of five per cent gives a ``\\kappa`` above the reciprocal of its degrees of freedom. The same sample at a depth of one per cent gives a ``\\kappa`` that is nearer to it. The spread of the estimate increases as the count decreases, and `kmin` is the floor of that count.

`key` names the end that the slot prices, so **the answer can be different for each key**. `:kappa` and `:kappa_a` read the loss tail, and `:kappa_b` reads the gain tail. [`EntropyBudget`](@ref) is different, because its budget is a price that the model pays, so it gives one number for both ends. A tail index is a property of one tail, and a skewed sample has two different tail indices. On a Range measure, the rule gives the two ends two different numbers.

**`ctx.series` names the quantity in the pool, and the slot owner states it.** A measure of the return distribution pools the columns of `pr.X`. A drawdown measure pools the drawdown series of each column of `pr.X`, because it prices that quantity. The resolution site names the marker in `ctx.series`. The standardisation, the count and the estimator do not change. The pool then gives the tail index of the drawdown series and not of the returns, and the two numbers are different. [`AbstractDrawdownSeries`](@ref) states how the length of the record decides the difference.

**The drawdown series of the columns stand for the drawdown of the portfolio**, under the assumption of the pool. The rule makes no portfolio, so it pools the drawdown series of the columns, as it pools the columns on a returns series. The columns have one tail index after standardisation, and one map takes a column to its drawdown series. So the pool gives the same estimate as the drawdown series of the portfolio. A portfolio drawdown is smaller than the average asset drawdown, but that is a difference of **scale**, and a tail index does not read the scale.

A drawdown series has one entry for each observation. So the pool holds `T N` entries under every marker, and the count `k = ceil(alpha * T * N)` does not change. The change of series drops no observation.

A drawdown measure resolves the significance level of its own drawdown series. So under a drawdown marker, the rule reads its tail at the level of that series.

The pool of a drawdown series has **more dependence** than the pool of a returns series. A drawdown series is a running function of the path, so consecutive entries move together, and one deep drawdown fills many of the `k` order statistics of the estimate. So the `T N` entries hold far fewer independent tail points than the same count of returns, and the spread of the estimate is wider than the count suggests. `kmin` counts entries and not independent points, so the same floor gives less accuracy here. Use a larger `kmin` for a drawdown owner.

The rule does not check the range of the parameter that it returns. The constructor of the slot owner does that, as for every calibration rule. The checks of the rule state that the estimate exists.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HillTailDecay(;
        kmin::Integer = 30
    ) -> HillTailDecay

Keywords correspond to the fields of the struct. The default `kmin = 30` is the floor of the count of order statistics that the estimate reads. The significance level and the series of the estimate are not fields of the rule. Both reach the rule in the [`CalibrationContext`](@ref) of the site.

## Validation

  - `kmin > 0`.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`CalibrationContext`](@ref)
  - [`Num_DefCal`](@ref)
  - [`EntropyBudget`](@ref)
  - [`hill_tail_index`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`series_end_sign`](@ref)

# References

  - $(ref_dict[:kaniadakis2001]) Equation 53 and page 20.
  - $(ref_dict[:hill1975]) Pages 1163–1174. $(ref_dict[:no_original_source])
"""
@concrete struct HillTailDecay <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_kmin])
    """
    kmin
    function HillTailDecay(kmin::Integer)
        assert_gt0(kmin, :kmin)
        return new{typeof(kmin)}(kmin)
    end
end
function HillTailDecay(; kmin::Integer = 30)
    return HillTailDecay(kmin)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the deformation parameter whose reciprocal is the Hill tail index of the series that `ctx.series` names, over the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
s &= \\begin{cases}
+1 & \\textrm{if the slot prices the gain end}\\\\
-1 & \\textrm{if the slot prices the loss end}
\\end{cases}\\,,\\\\
y_{tj} &= \\left(\\mathcal{S}\\left(\\boldsymbol{r}_{j}\\right)\\right)_{t}\\,,\\\\
u_{tj} &= s \\dfrac{y_{tj} - \\hat{\\mu}_{j}}{\\hat{\\sigma}_{j}}\\,,\\\\
k &= \\left\\lceil \\alpha T N \\right\\rceil\\,,\\\\
\\hat{a} &= \\dfrac{k}{\\sum\\limits_{i=1}^{k} \\ln\\left(\\dfrac{u_{(i)}}{u_{(k+1)}}\\right)}\\,,\\\\
\\kappa &= \\dfrac{1}{\\hat{a}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:kappa_rm])
  - $(math_dict[:alpha_rm]) It is the probability of the end that this slot prices, so a head slot reads its own ``\\beta``.
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:r_tj])
  - ``\\boldsymbol{r}_{j}``: Column ``j`` of the sample.
  - ``\\mathcal{S}``: The series that `ctx.series` names, made from one column. It is the identity for a [`ReturnsSeries`](@ref), and a drawdown series for the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``y_{tj}``: Entry ``t`` of the series of column ``j``.
  - ``\\hat{\\mu}_{j}``: Sample mean of the series of column ``j``. Under a drawdown marker, it is not the mean of the column.
  - ``\\hat{\\sigma}_{j}``: Sample standard deviation of the series of column ``j``. The rule reads it from that series, and never from ``\\hat{\\mathbf{\\Sigma}}``.
  - ``s``: Sign of the end that the slot prices.
  - ``u_{tj}``: Standardised value of asset ``j`` at time ``t``. Its sign makes the end that the slot prices the upper tail of the pool.
  - ``u_{(i)}``: ``i``-th largest of the ``T N`` pooled values, so that ``u_{(1)} \\geq \\ldots \\geq u_{(k+1)}``.
  - ``k``: Number of order statistics that the estimate reads.
  - ``\\hat{a}``: Hill estimate of the tail index of the pool.

# Algorithm

 1. Read the returns matrix of `pr` into `X`, and its element count into `np`.
 2. Take the sign `s` from `key` and `ctx.series` with [`series_end_sign`](@ref). A drawdown series has one end, so it refuses the head key.
 3. Make the count `k = ceil(Int, ctx.alpha * np)`, the number of order statistics that the estimate reads.
 4. Estimate the tail index `a` of the pool with [`hill_tail_index`](@ref). Under every marker, the pool holds the series that `ctx.series` names, with one entry for each observation of each column.
 5. Return `inv(a)`, the deformation parameter.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. It names the end. `:kappa` and `:kappa_a` read the loss tail, and `:kappa_b` reads the gain tail. So a skewed sample gives the two ends of a Range measure two different numbers.
  - `pr`: Prior result from which the rule reads the returns matrix.
  - `w`: Effective observation weights. A tail index is a property of the shape of a series and not of the count of observations behind it, so this rule does not read them.
  - `slv`: Effective solver. This rule needs none, because the estimate is a closed form.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.alpha` is the significance level of the end, and it must hold a number. `ctx.series` names the quantity in the pool.

# Validation

  - `ctx.alpha` must not be `nothing`.
  - `key` must name an end of the series. [`series_end_sign`](@ref) checks it.
  - `k` must be at least `alg.kmin`.
  - The pool must hold at least `k + 1` values.
  - ``u_{(k+1)}`` must be positive. [`hill_tail_index`](@ref) checks it.
  - ``\\hat{a}`` must be greater than one. That is the band ``\\kappa \\in (0,\\, 1)`` as a tail index.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`HillTailDecay`](@ref)
  - [`CalibrationContext`](@ref)
  - [`EntropyBudget`](@ref)
  - [`hill_tail_index`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`series_end_sign`](@ref)

# References

  - $(ref_dict[:hill1975]) Pages 1163–1174. $(ref_dict[:no_original_source])
"""
function (alg::HillTailDecay)(key::Symbol, pr::AbstractPriorResult, ::Any, ::Any,
                              ctx::CalibrationContext)
    @argcheck(!isnothing(ctx.alpha),
              IsNothingError("`CalibrationContext.alpha` is `nothing`, so `HillTailDecay` cannot form the count `k = ceil(alpha * T * N)`. The probability of the end reaches the rule in the context the slot owner builds, after it resolves that end's own probability. State `alpha` in the context to run the rule outside a measure."))
    X = pr.X
    # The sign puts the end the slot prices in the LOWER tail of the pool, and one estimator
    # then serves both ends. The series decides which ends there are to price.
    s = series_end_sign(ctx.series, key)
    np = prod(size(X))
    k = ceil(Int, ctx.alpha * np)
    @argcheck(k >= alg.kmin,
              DomainError(k,
                          "`HillTailDecay` reads the worst `k = ceil(alpha * T * N) = $k` of the $np pooled standardised values, and `HillTailDecay.kmin` puts the floor at $(alg.kmin). A Hill estimate over fewer order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. Lengthen the sample, widen `alpha`, or lower `kmin` and take the noise."))
    @argcheck(k + 1 <= np,
              DomainError(k,
                          "`HillTailDecay` needs $(k + 1) pooled values to form the estimate, and the pool of `T * N` holds $np. The count is `k = ceil(alpha * T * N)` at `alpha = $(ctx.alpha)`, so only a probability that takes the whole sample reaches this. Lower `alpha`."))
    a = hill_tail_index(ctx.series, X, s, k)
    @argcheck(a > 1,
              DomainError(a,
                          "`HillTailDecay` estimated a tail index of $a on the pool of the $(nameof(typeof(ctx.series))), so `kappa = 1 / a` is $(inv(a)) and lies outside the (0, 1) the slot admits. An index of one or less is a tail with no finite mean, so no admissible deformation parameter reads it. The series is heavier-tailed than the measure can price."))
    return inv(a)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns an upper triangular square factor of the covariance matrix of `pr`, so that the transpose of the factor whitens a centred observation.

The factor ``\\mathbf{U}`` satisfies ``\\mathbf{U}^{\\top} \\mathbf{U} = \\hat{\\mathbf{\\Sigma}}``. That is the convention of the `chol` field of a prior result, and the variance constraint layer uses it too. **`chol` has precedence over `sigma`**, as the `chol` field states. So the function whitens a prior that has a factor with that factor, and never with a factorisation of `sigma`. `chol` is optional, so usually the function factorises `sigma`.

The library checks `chol` only against the length of `mu`, so it can be ``K \\times N`` and not square, and a square `chol` need not be triangular. The function needs a triangular factor, because only the diagonal of a triangular factor gives its rank:

  - A `chol` of type `LinearAlgebra.UpperTriangular` is the factor.
  - Every other `chol` with ``K \\geq N`` gives the ``\\mathbf{R}`` factor of its QR factorisation. ``\\mathbf{R}^{\\top} \\mathbf{R} = \\mathbf{F}^{\\top} \\mathbf{F}``, so ``\\mathbf{R}`` is a triangular factor of the same covariance matrix. This includes a tall factor, and a square factor that is not triangular, such as a symmetric square root.
  - A wide `chol`, with ``K < N``, gives a singular covariance matrix, and no whitening exists for it.

[`RadialTailDecay`](@ref) calls this function. The function is separate because the precedence rule and its refusals are facts about a prior result and not about a tail index.

# Algorithm

 1. Read `pr.chol` into `F`.
 2. When `F` is `nothing`, factorise `pr.sigma` with a Cholesky factorisation that reports a failure and does not raise it, and return its upper factor.
 3. Otherwise, refuse a wide `F`. Take `F` when it is an `UpperTriangular`, and the `R` factor of its QR factorisation otherwise.
 4. Refuse a zero on the diagonal of that triangular factor, because it gives a singular covariance matrix, and return the factor.

# Arguments

  - `pr`: Prior result from which the function reads the covariance matrix.

# Validation

  - `pr.sigma` must be positive definite, when `pr.chol` is `nothing`.
  - `pr.chol` must have at least as many rows as columns.
  - The diagonal of the triangular factor must have no zero.

# Returns

  - `U::AbstractMatrix`: Upper triangular square factor of the covariance matrix.

# Related

  - [`RadialTailDecay`](@ref)
  - [`radial_tail_index`](@ref)
"""
function whitening_factor(pr::AbstractPriorResult)
    F = pr.chol
    if isnothing(F)
        f = LinearAlgebra.cholesky(pr.sigma; check = false)
        @argcheck(LinearAlgebra.issuccess(f),
                  DomainError(f.info,
                              "The Cholesky factorisation of `pr.sigma` failed at pivot $(f.info), so the covariance matrix the prior states is not positive definite and no whitening of the sample exists. A shrunk or a factor-model covariance matrix reaches this. Fit a prior whose covariance matrix is positive definite, or state the factor in `pr.chol`."))
        return f.U
    end
    @argcheck(size(F, 1) >= size(F, 2),
              DimensionMismatch("`pr.chol` is $(size(F, 1)) × $(size(F, 2)), so the covariance matrix `pr.chol' * pr.chol` it states has rank at most $(size(F, 1)) over $(size(F, 2)) assets and is singular. No whitening of the sample exists. `pr.chol` takes precedence over `pr.sigma` at every consumer, so dropping to `pr.sigma` here would state something the prior does not."))
    UT = LinearAlgebra.UpperTriangular
    U = F isa UT ? F : UT(LinearAlgebra.qr(F).R)
    j = findfirst(iszero, LinearAlgebra.diag(U))
    @argcheck(isnothing(j),
              DomainError(j,
                          "Entry $j of the diagonal of the triangular factor of `pr.chol` is zero, so the covariance matrix it states is singular and no whitening of the sample exists. The diagonal of a triangular factor gives its rank, so the check reads `pr.chol` itself when it is an `UpperTriangular`, and the R factor of its QR factorisation otherwise."))
    return U
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Estimates the tail index of the radial series of `X`, over the `k` largest order statistics of that series.

The function centres each row on `mu` and whitens it with `U`. The norm of the whitened row is one entry of the radial series. The estimator of [hill1975](@cite) then reads the `k` entries that are furthest out. [`RadialTailDecay`](@ref) states what the series measures. This function makes the estimate alone.

The signature binds the element type, so the series and its sum have concrete types. A rule reads `pr.X` from an [`AbstractPriorResult`](@ref), whose field types no signature states, and this function is the boundary where the type becomes concrete.

# Algorithm

 1. Solve `transpose(U) \\ transpose(X .- transpose(mu))`. The columns of the result `Z` are the whitened observations.
 2. Walk the columns of `Z`, and write the negative Euclidean norm of each column into the series `d`.
 3. Partially sort `d`, so that its `k + 1` smallest values are in order at the start, and read the last of them into `vkp1`. `partialsort!` with one index puts only that index in place, so the range is necessary to put the `k` entries of the estimate before `vkp1`. The negative sign of step 2 puts the largest distances at the start.
 4. Return `k` divided by the sum of `log(d[i] / vkp1)` over those `k` entries. Both terms of each ratio are negative, so the ratio is a ratio of magnitudes, and the sum needs no change of sign.

# Arguments

  - `X`: Returns matrix, `T × N`.
  - `mu`: Expected returns vector, `N × 1`. The function centres the rows on it.
  - `U`: Upper triangular square factor of the covariance matrix, which [`whitening_factor`](@ref) returns.
  - `k`: Number of order statistics that the estimate reads. The caller states it and checks its floor.

# Validation

  - The `k + 1`-th largest radial distance must be positive.

# Returns

  - `a::Number`: The Hill estimate of the tail index of the radial series.

# Related

  - [`RadialTailDecay`](@ref)
  - [`whitening_factor`](@ref)

# References

  - $(ref_dict[:hill1975]) Pages 1163–1174. $(ref_dict[:no_original_source])
"""
function radial_tail_index(X::AbstractMatrix{E}, mu::AbstractVector, U::AbstractMatrix,
                           k::Integer) where {E <: Number}
    Z = transpose(U) \ transpose(X .- transpose(mu))
    d = Vector{E}(undef, size(X, 1))
    for t in axes(X, 1)
        # The series is negated so that the end of it the estimate reads is the LOWER tail,
        # which lets the selection below run without `rev = true`. `hill_tail_index` signs
        # its pool for the same reason, so the two verbs read their series the same way.
        d[t] = -LinearAlgebra.norm(view(Z, :, t))
    end
    partialsort!(d, 1:(k + 1))
    vkp1 = d[k + 1]
    dkp1 = -vkp1
    @argcheck(dkp1 > 0,
              DomainError(dkp1,
                          "The $(k + 1)-th largest radial distance is $dkp1, which is not positive, so there is no Hill estimate: the estimator reads `log(d_i / d_(k+1))`, which needs a positive ratio. The sample holds fewer than $(k + 1) observations away from `pr.mu`. Lower the count `k`."))
    return k / sum(i -> log(d[i] / vkp1), 1:k)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Kaniadakis deformation parameter whose tail decays at the rate of the radial series of the sample.

The inverse of [`kappa_log`](@ref) is the ``\\kappa``-exponential ``\\exp_{\\kappa}(x) = \\left(\\kappa x + \\sqrt{1 + \\kappa^{2} x^{2}}\\right)^{1/\\kappa}`` of Equation 53 of [kaniadakis2001](@cite). It approaches ``(2 \\kappa x)^{1/\\kappa}`` for large ``x``, as page 20 of [kaniadakis2001](@cite) states. So a ``\\kappa``-deformed exponential tail is a power law of index ``1/\\kappa``, and ``\\kappa`` is the reciprocal of a tail index. The rule estimates a tail index ``\\hat{a}`` and returns ``1/\\hat{a}``, so the deformation decays at the rate of the sample. The band of the slot, ``(0,\\, 1)``, is ``\\hat{a} > 1``, which is the condition for a finite mean. So the rule refuses an estimate outside the band, as [`EntropyBudget`](@ref) refuses a target outside its band.

**The rule reads a radial series.** A covariance matrix is a scale and ``\\kappa`` is a shape. So a rule that gets a shape from a scale needs a second quantity to fix the units, and the sample is the only such quantity in a prior result. The covariance matrix turns the ``T \\times N`` sample into **one** univariate series. [`whitening_factor`](@ref) gives the factor of ``\\hat{\\mathbf{\\Sigma}}``, the rule centres and whitens each row, and the Euclidean norm of the whitened row is the Mahalanobis distance of that observation. Under an elliptical scale mixture, that series has the tail index of the mixture, and each entry reads the whole cross-section. A standardisation per column removes every off-diagonal term. So [`HillTailDecay`](@ref) cannot give this estimate, and this rule cannot give the estimate per column.

**The rule reads no direction.** A distance has no sign, so one ``\\kappa`` serves both ends of a Range measure, and the rule does not read `key`. [`HillTailDecay`](@ref) gives one answer for each end. A radial series measures how far the cross-section moves, and a series per column measures how far one end of one column moves. A head slot that defaults to its tail slot holds the same rule and gets the same number, and that is correct for a distance.

The series holds `T` entries, and the pool of [`HillTailDecay`](@ref) holds `T N`. So this rule reads **fewer** tail points from the same sample. At `alpha = 0.05` and `T = 250`, the series has 13 of them. `kmin` is the floor of that count, in the same units as the floor of the sibling, so this floor refuses more samples. The rule refuses a count below the floor. The series has one point for each observation and not `N`, so its points have none of the cross-correlation of the pool.

**`series` names the sample of the rows, and the slot owner states it.** A measure of the return distribution whitens the rows of `pr.X`. A drawdown measure whitens the rows of the **drawdown sample**. [`calibration_series_matrix`](@ref) turns each column into its drawdown series, and the rule whitens the rows of that sample and takes their norms in the same way. So the radial series is the same function, and only its sample changes. On a drawdown sample, the distance is the depth of the whole cross-section in drawdown at one date. On a returns series, it is the distance that the whole cross-section moved.

**A drawdown sample has its own moments, and the prior states none of them.** `pr.mu` and `pr.sigma` are the moments of the returns, and no scaling of them gives the moments of a drawdown. So under a drawdown marker, the centre is the vector of column means of the drawdown sample. The factor is the Cholesky factor of the covariance matrix of that sample. [`radial_series_inputs`](@ref) makes this choice, so the precedence of `pr.chol` over `pr.sigma` applies to the returns series alone.

The rule does not use a third estimate of ``\\kappa``. The excess kurtosis ``g`` of a Student-t distribution gives ``\\nu = 4 + 6/g``, and then ``\\kappa = 1/\\nu``. That estimate reads the whole sample and not the tail. So it changes less than a Hill estimate, but it assumes the shape of the whole distribution and not only of its tail.

The rule does not check the range of the parameter that it returns. The constructor of the slot owner does that, as for every calibration rule. The checks of the rule state that the estimate exists. They do not refuse a covariance matrix that is almost singular. The whitening then reads the direction of the smallest eigenvalue of the sample, the radial series follows it, and that is a property of the covariance matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RadialTailDecay(;
        kmin::Integer = 30
    ) -> RadialTailDecay

Keywords correspond to the fields of the struct. The default `kmin = 30` is the floor of the count of order statistics that the estimate reads. The significance level and the series of the estimate are not fields of the rule. Both reach the rule in the [`CalibrationContext`](@ref) of the site.

## Validation

  - `kmin > 0`.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`CalibrationContext`](@ref)
  - [`Num_DefCal`](@ref)
  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`radial_tail_index`](@ref)
  - [`whitening_factor`](@ref)

# References

  - $(ref_dict[:kaniadakis2001]) Equation 53 and page 20.
  - $(ref_dict[:hill1975]) Pages 1163–1174. $(ref_dict[:no_original_source])
"""
@concrete struct RadialTailDecay <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_kmin_rad])
    """
    kmin
    function RadialTailDecay(kmin::Integer)
        assert_gt0(kmin, :kmin)
        return new{typeof(kmin)}(kmin)
    end
end
function RadialTailDecay(; kmin::Integer = 30)
    return RadialTailDecay(kmin)
end
"""
    radial_series_inputs(series::AbstractCalibrationSeries, pr::AbstractPriorResult)

Returns the sample, the centre and the whitening factor from which [`RadialTailDecay`](@ref) makes the radial series of `series`.

A [`ReturnsSeries`](@ref) reads the three from the prior result: `pr.X`, `pr.mu` and the factor that [`whitening_factor`](@ref) takes from `pr.chol` or `pr.sigma`.

A drawdown series reads the three from the **drawdown sample**, because a prior result states no moment of a drawdown. [`calibration_series_matrix`](@ref) makes the sample. The centre is the vector of column means of that sample, and the factor is the Cholesky factor of its covariance matrix. So all three come from the quantity that the measure prices. `pr.mu` and `pr.sigma` are not read, because they are the moments of the returns, and no scaling of them gives the moments of a drawdown.

# Arguments

  - `series`: The series that the slot owner prices.
  - `pr`: Prior result. A returns series reads its sample and its moments, and a drawdown series reads its sample.

# Validation

  - The covariance matrix of the drawdown sample must be positive definite. A column that does not move fails this check. Two columns with the same path also fail it.

# Returns

  - `Y::AbstractMatrix`: The sample that the marker names, `T × N`.
  - `mu::AbstractVector`: The centre of that sample, `N × 1`.
  - `U::AbstractMatrix`: Upper triangular square factor of its covariance matrix.

# Related

  - [`RadialTailDecay`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_matrix`](@ref)
  - [`radial_tail_index`](@ref)
  - [`whitening_factor`](@ref)
"""
function radial_series_inputs(::ReturnsSeries, pr::AbstractPriorResult)
    return pr.X, pr.mu, whitening_factor(pr)
end
function radial_series_inputs(series::AbstractDrawdownSeries, pr::AbstractPriorResult)
    Y = calibration_series_matrix(series, pr.X)
    mu = vec(Statistics.mean(Y; dims = 1))
    # A prior result states the moments of the RETURNS, so the moments of the drawdown
    # sample come off that sample. `whitening_factor` is not the verb here: its precedence
    # rule and its refusals are statements about `pr.chol` and `pr.sigma`.
    f = LinearAlgebra.cholesky(Statistics.cov(Y); check = false)
    @argcheck(LinearAlgebra.issuccess(f),
              DomainError(f.info,
                          "The Cholesky factorisation of the covariance matrix of the $(nameof(typeof(series))) sample failed at pivot $(f.info), so that matrix is not positive definite and no whitening of the drawdown sample exists. A column that never moves has a drawdown series of zeros, and two columns with one path between them state one drawdown series twice. Drop the column, or state `kappa` on the slot."))
    return Y, mu, f.U
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the deformation parameter whose reciprocal is the Hill tail index of the radial series of the sample that `ctx.series` names.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{y}_{t} &= \\mathcal{S}\\left(\\boldsymbol{x}_{t}\\right)\\,,\\\\
\\hat{\\mathbf{\\Sigma}}_{y} &= \\mathbf{U}^{\\top} \\mathbf{U}\\,,\\\\
\\boldsymbol{z}_{t} &= \\mathbf{U}^{-\\top} \\left(\\boldsymbol{y}_{t} - \\hat{\\boldsymbol{\\mu}}_{y}\\right)\\,,\\\\
d_{t} &= \\left\\lVert \\boldsymbol{z}_{t} \\right\\rVert_{2}\\,,\\\\
k &= \\left\\lceil \\alpha T \\right\\rceil\\,,\\\\
\\hat{a} &= \\dfrac{k}{\\sum\\limits_{i=1}^{k} \\ln\\left(\\dfrac{d_{(i)}}{d_{(k+1)}}\\right)}\\,,\\\\
\\kappa &= \\dfrac{1}{\\hat{a}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:kappa_rm])
  - $(math_dict[:alpha_rm]) It is the probability of the end that the slot prices. The series has no sign, so the count that it fixes is the same for both ends.
  - $(math_dict[:T])
  - $(math_dict[:x_t_obs])
  - ``\\mathcal{S}``: The sample that `ctx.series` names, made one column at a time. It is the identity for a [`ReturnsSeries`](@ref), and a drawdown series for the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``\\boldsymbol{y}_{t}``: Row ``t`` of that sample.
  - ``\\hat{\\boldsymbol{\\mu}}_{y}``: Centre of that sample. It is `pr.mu` for a returns series, and the vector of column means of the drawdown sample for a drawdown series.
  - ``\\hat{\\mathbf{\\Sigma}}_{y}``: Covariance matrix of that sample. It is `pr.sigma` for a returns series, and the covariance matrix of the drawdown sample for a drawdown series.
  - ``\\mathbf{U}``: Upper triangular square factor of ``\\hat{\\mathbf{\\Sigma}}_{y}``. It comes from `pr.chol` when the prior has one and the slot prices the returns.
  - ``\\boldsymbol{z}_{t}``: Whitened observation ``t``.
  - ``d_{t}``: Mahalanobis distance of observation ``t``, the ``t``-th entry of the radial series.
  - ``d_{(i)}``: ``i``-th largest entry of the radial series, so that ``d_{(1)} \\geq \\ldots \\geq d_{(k+1)}``.
  - ``k``: Number of order statistics that the estimate reads.
  - ``\\hat{a}``: Hill estimate of the tail index of the radial series.

# Algorithm

 1. Read the returns matrix of `pr` into `X`, and its row count into `T`.
 2. Make the count `k = ceil(Int, ctx.alpha * T)`, the number of order statistics that the estimate reads. A drawdown series has one entry for each observation, so the count is the same under every marker.
 3. Take the sample, its centre and its whitening factor with [`radial_series_inputs`](@ref).
 4. Estimate the tail index `a` of the radial series with [`radial_tail_index`](@ref).
 5. Return `inv(a)`, the deformation parameter.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The radial series has no sign, so this rule reads no end from the key and returns one number for every key.
  - `pr`: Prior result from which the rule reads the returns matrix, the expected returns vector and the covariance matrix.
  - `w`: Effective observation weights. A tail index is a property of the shape of a series and not of the count of observations behind it, so this rule does not read them.
  - `slv`: Effective solver. This rule needs none, because the estimate is a closed form.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.alpha` is the significance level of the end, and it must hold a number. `ctx.series` names the sample of the radial series.

# Validation

  - `ctx.alpha` must not be `nothing`.
  - `k` must be at least `alg.kmin`.
  - The series must hold at least `k + 1` entries.
  - The sample must have a whitening. [`radial_series_inputs`](@ref) checks it for both series.
  - ``d_{(k+1)}`` must be positive. [`radial_tail_index`](@ref) checks it.
  - ``\\hat{a}`` must be greater than one. That is the band ``\\kappa \\in (0,\\, 1)`` as a tail index.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`RadialTailDecay`](@ref)
  - [`CalibrationContext`](@ref)
  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`radial_tail_index`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`whitening_factor`](@ref)

# References

  - $(ref_dict[:hill1975]) Pages 1163–1174. $(ref_dict[:no_original_source])
"""
function (alg::RadialTailDecay)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any,
                                ctx::CalibrationContext)
    @argcheck(!isnothing(ctx.alpha),
              IsNothingError("`CalibrationContext.alpha` is `nothing`, so `RadialTailDecay` cannot form the count `k = ceil(alpha * T)`. The probability of the end reaches the rule in the context the slot owner builds, after it resolves that end's own probability. State `alpha` in the context to run the rule outside a measure."))
    X = pr.X
    T = size(X, 1)
    k = ceil(Int, ctx.alpha * T)
    @argcheck(k >= alg.kmin,
              DomainError(k,
                          "`RadialTailDecay` reads the largest `k = ceil(alpha * T) = $k` of the $T radial distances, and `RadialTailDecay.kmin` puts the floor at $(alg.kmin). A Hill estimate over fewer order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. The radial series holds one entry per observation where the pool of `HillTailDecay` holds `N`, so the same floor binds harder here. Lengthen the sample, widen `alpha`, or lower `kmin` and take the noise."))
    @argcheck(k + 1 <= T,
              DomainError(k,
                          "`RadialTailDecay` needs $(k + 1) radial distances to form the estimate, and the series holds one per observation, which is $T. The count is `k = ceil(alpha * T)` at `alpha = $(ctx.alpha)`, so only a probability that takes the whole sample reaches this. Lower `alpha`."))
    Y, mu, U = radial_series_inputs(ctx.series, pr)
    a = radial_tail_index(Y, mu, U, k)
    @argcheck(a > 1,
              DomainError(a,
                          "`RadialTailDecay` estimated a tail index of $a on the radial series of the $(nameof(typeof(ctx.series))), so `kappa = 1 / a` is $(inv(a)) and lies outside the (0, 1) the slot admits. An index of one or less is a tail with no finite mean, so no admissible deformation parameter reads it. The series is heavier-tailed than the measure can price."))
    return inv(a)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from the data a prior result carries, so that the radius refits whenever the sample moves.

An **Ambiguity Radius** is the radius of the ball of probability measures that the model prices. It is the coefficient of a norm penalty on the weight vector, with no square. It is not a tail probability and not a deformation parameter. So it has its own family next to [`AbstractSignificanceCalibrationAlgorithm`](@ref) and [`AbstractDeformationCalibrationAlgorithm`](@ref), under the same root.

A rule of this family is a callable subtype of `AbstractAmbiguityRadiusCalibrationAlgorithm`, as a rule of the two other families is. [`Num_AmbRadCal`](@ref) is the bound of every radius slot, and it names this family.

# Interfaces

To add a radius rule, subtype `AbstractAmbiguityRadiusCalibrationAlgorithm`, keep the parameters of the rule in the fields of the struct, and add the method below.

## The functor

  - `(alg::AbstractAmbiguityRadiusCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv, ctx::CalibrationContext) -> Number`: Returns the ambiguity radius.

### Arguments

  - `key`: Name of the slot that the call resolves.
  - `pr`: Prior result from which the rule reads its sample size and its moments.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior gives any.
  - `slv`: Effective solver, or `nothing` when the owner has none.
  - `ctx`: The facts that the site knows and `key` does not give, as a [`CalibrationContext`](@ref). A rule reads the fields that it needs. A rule that reads no field writes `::CalibrationContext`.

### Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`ConcentrationRadius`](@ref): the rule of the family that makes the ball smaller as the sample becomes longer.
  - [`RateRadius`](@ref): the rule of the family that decreases the radius at a square-root rate.
  - [`DimensionalRateRadius`](@ref): the rule of the family that decreases the radius at the dimensional rate of a Wasserstein ball.
  - [`DualNormRadius`](@ref): the rule of the family that reads the ground metric that the slot names.
"""
abstract type AbstractAmbiguityRadiusCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes an Esfahani-Kuhn tail weight, the weight in the `l` slot of a distributionally robust measure.

The tail weight multiplies the tail term of the Esfahani-Kuhn loss, and it does not multiply the mean term. It is a different quantity from the radius, so it has its own family. [`Num_AmbTwtCal`](@ref) makes a constructor refuse a radius rule in a tail-weight slot.

[`TailTermParity`](@ref) is the rule of the family in the library. A stated tail weight has no units, but it is not independent of the scale of the sample. So one number is a different trade-off at each sampling frequency. The rule reads the units of the sample and leaves the preference in the `ratio` of the caller. [`Num_AmbTwtCal`](@ref) also admits a plain function, so a rule of a caller needs no type.

# Interfaces

To add a tail-weight rule, subtype `AbstractAmbiguityTailWeightCalibrationAlgorithm`, keep the parameters of the rule in the fields of the struct, and add the method below.

## The functor

  - `(alg::AbstractAmbiguityTailWeightCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv, ctx::CalibrationContext) -> Number`: Returns the tail weight.

### Arguments

  - `key`: Name of the slot that the call resolves.
  - `pr`: Prior result from which the rule reads its sample size and its moments.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior gives any.
  - `slv`: Effective solver, or `nothing` when the owner has none.
  - `ctx`: The facts that the site knows and `key` does not give, as a [`CalibrationContext`](@ref). A rule reads the fields that it needs. A rule that reads no field writes `::CalibrationContext`.

### Returns

  - `l::Number`: The tail weight.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`TailTermParity`](@ref): the rule of the family that prices the tail term at a stated multiple of the mean term.
"""
abstract type AbstractAmbiguityTailWeightCalibrationAlgorithm <:
              AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes a norm ceiling from the data a prior result carries, so that the ceiling refits whenever the universe moves.

A **Norm Ceiling** is an upper bound on a norm of the weight vector. The `l2c`, `lpc` and `linfc` slots of [`JuMPOptimiser`](@ref) hold it. It is not an **Ambiguity Radius**. A radius is the coefficient of a norm penalty in the objective, and a ceiling bounds that norm in a constraint. A ceiling on the norm is a floor on the effective number of assets. That is a statement about diversification, and not about the set of probability measures that the model prices. So the quantity has its own family next to the others, under the same root.

A rule of this family is a callable subtype of `AbstractNormCeilingCalibrationAlgorithm`, as a rule of the other families is. [`Num_NormCeilCal`](@ref) is the bound of every ceiling slot, and it names this family.

# Interfaces

To add a ceiling rule, subtype `AbstractNormCeilingCalibrationAlgorithm`, keep the parameters of the rule in the fields of the struct, and add the method below.

## The functor

  - `(alg::AbstractNormCeilingCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv, ctx::CalibrationContext) -> Number`: Returns the norm ceiling.

### Arguments

  - `key`: Name of the slot that the call resolves: `:l2c`, `:lpc` or `:linfc`.
  - `pr`: Prior result from which the rule reads its asset count and its sample size.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior gives any.
  - `slv`: Effective solver, or `nothing` when the owner has none.
  - `ctx`: The facts that the site knows and `key` does not give, as a [`CalibrationContext`](@ref). A rule reads the fields that it needs. A rule that reads no field writes `::CalibrationContext`.

### Returns

  - `val::Number`: The norm ceiling.

## The norm order

A ceiling applies to one norm order, and that order belongs to the constraint and not to the rule. A rule that needs the order reads `ctx.p`. Each constraint site sets it in the [`CalibrationContext`](@ref) that it builds. A rule that needs no order reads no field of the context.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`CalibrationContext`](@ref)
  - [`EffectiveAssetFloor`](@ref): the rule of the family that keeps a stated fraction of the universe effective.
"""
abstract type AbstractNormCeilingCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
    const Num_AmbRadCal = Union{AbstractAmbiguityRadiusCalibrationAlgorithm, Function,
                                Number}

Field bound for an ambiguity-radius slot: the radius, a rule of the family, or a plain function of the same five arguments.

The union names one rule family. So a tail-weight rule in a radius slot does not match the signature of the constructor, and the constructor refuses it. This is all the validation of the family, and no separate check exists for it. A function is a member of the union for the reason that [`Num_SigCal`](@ref) states. It has no family, so only the name of the slot states the quantity.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`Num_AmbRadNormCeilCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_AmbRadCal = Union{AbstractAmbiguityRadiusCalibrationAlgorithm, Function, Number}
"""
    const Num_AmbTwtCal = Union{AbstractAmbiguityTailWeightCalibrationAlgorithm,
                                Function, Number}

Field bound for an ambiguity-tail-weight slot: the tail weight, a rule of the family, or a plain function of the same five arguments. It is the counterpart of [`Num_AmbRadCal`](@ref) for the tail-weight family, and the description of that bound applies to it.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_AmbTwtCal = Union{AbstractAmbiguityTailWeightCalibrationAlgorithm, Function,
                            Number}
"""
    const Num_NormCeilCal = Union{AbstractNormCeilingCalibrationAlgorithm, Function,
                                  Number}

Field bound for a norm-ceiling slot: the ceiling, a rule of the family, or a plain function of the same five arguments.

The union names one rule family. So a constructor refuses a radius rule in a ceiling slot, and it refuses a ceiling rule in a radius slot in the same way. [`AbstractNormCeilingCalibrationAlgorithm`](@ref) states the difference between the two quantities.

# Related

  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Num_AmbRadNormCeilCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_NormCeilCal = Union{AbstractNormCeilingCalibrationAlgorithm, Function, Number}
"""
    const Num_AmbRadNormCeilCal = Union{AbstractAmbiguityRadiusCalibrationAlgorithm,
                                        AbstractNormCeilingCalibrationAlgorithm, Number}

Field bound for the one slot that the library reads as two quantities, the `val` field of [`LpRegularisation`](@ref).

That estimator is a penalty in the `lp` field of [`JuMPOptimiser`](@ref) and a norm constraint in its `lpc` field. So `val` is an ambiguity radius on one route and a norm ceiling on the other. One field cannot have two bounds, so this bound admits both rule families, and each route refuses the family that has no meaning on it. In this slot alone, the route and not the bound settles the quantity, after the construction.

This bound does **not** admit a plain function, and it is the only slot that refuses one. Every other slot names one quantity, so a function in it needs no family. This slot names two quantities, and the two checks read the family to tell them apart. A function has no family. So a function in this slot can reach a route that has no meaning for it, and no check can find the error.

# Related

  - [`LpRegularisation`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`assert_penalty_coefficient_role`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
"""
const Num_AmbRadNormCeilCal = Union{AbstractAmbiguityRadiusCalibrationAlgorithm,
                                    AbstractNormCeilingCalibrationAlgorithm, Number}
# The two families below name a quantity that reaches a JuMP model from a bare field of
# `JuMPOptimiser`: the ceiling from `l2c` and `linfc`, the radius from `l1` and `linf`.
# There is no term to rebuild on that route, and the constructor's guard read a rule rather
# than a number, so it selected the permissive fallback of the validator and stated nothing.
# The range of the quantity is therefore checked here, on the number the rule returned, and
# only on the calibrated path. The three other families stand only in slots their owner
# rebuilds, and that rebuild states their range. A rule written as a plain function names no
# family, so it reaches neither method here and `assemble_jump_model!` states its range.
function resolve_calibration_slot(alg::Union{AbstractNormCeilingCalibrationAlgorithm,
                                             AbstractAmbiguityRadiusCalibrationAlgorithm},
                                  key::Symbol, pr::AbstractPriorResult, w, slv = nothing,
                                  ctx::CalibrationContext = CalibrationContext())
    val = alg(key, pr, w, slv, ctx)
    assert_nonempty_gt0_finite_val(val, key)
    return val
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from a chi-squared quantile, so that the ball becomes smaller as the sample becomes longer.

The radius is a scale, in the units of the series that the slot owner prices, times the square root of a chi-squared quantile divided by the sample size. It simplifies Theorem 3 and Equation 22 of [blanchet2019](@cite). For the estimating equation of the mean, ``h(\\boldsymbol{x}, \\boldsymbol{\\theta}) = \\boldsymbol{x} - \\boldsymbol{\\theta}``, under the squared Euclidean cost, ``T`` times the robust Wasserstein profile function converges to ``\\boldsymbol{H}^{\\intercal} \\boldsymbol{H}``, with ``\\boldsymbol{H} \\sim \\mathcal{N}(\\boldsymbol{0}, \\mathbf{\\Sigma})``. The rule replaces ``\\mathbf{\\Sigma}`` with ``s^{2} \\mathbf{I}``, so the law is ``s^{2} \\chi^{2}_{N}``. The radius of that work is the quantile of this law divided by ``T``, for the squared cost. The rule takes its square root, because its slot multiplies a norm with no square. The isotropic covariance is an assumption of the rule and not of [blanchet2019](@cite).

The chi-squared factor has no units and increases with the number of assets. So at a fixed confidence level, a wider universe gives a wider ball, and a longer sample gives a smaller ball.

`scale` gives the units. A radius multiplies a norm of the weight vector, so it is in the units of the loss around which the ball is drawn. A caller cannot get that number from the confidence level alone. `scale = nothing` reads the average per-asset dispersion of the series that the slot owner prices. That scale changes with the sample, as the rest of the rule does.

**A drawdown owner gets a drawdown scale.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) measures the transport cost of its programme against the per-asset drawdown sample. So its ball is a ball over drawdown scenarios, and the radius has the units of a drawdown. `series` names that quantity, and [`calibration_series_dispersion`](@ref) takes the dispersion from the drawdown sample. `pr.sigma` is not read there, because it is a moment of the returns.

**`T` is the count that the site prices.** It is the `n` that the site states in its [`CalibrationContext`](@ref), and [`effective_sample_size`](@ref) when the site states none. A distributionally robust measure states the count of the rows on which its ball is centred, because the bound is a statement about the empirical measure of those rows. A regularisation slot states no count, because its radius prices the error of `pr.mu`, and that error decreases with the observations behind the moments. With observation weights, both counts are the effective sample size of the weights, because the error of a weighted estimate decreases with that count and not with the number of rows. The two counts differ only under a Scenario Cap. [`RateRadius`](@ref) and [`DimensionalRateRadius`](@ref) read the same count.

The rule does not check the range of its value. It returns the quantity of its slot, so the site that receives the number checks the range. [`resolve_calibration_slot`](@ref) refuses a radius that is not `> 0` and finite. The owner of a slot that rebuilds a term refuses a radius outside its range when the fold runs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConcentrationRadius(;
        confidence::Number = 0.95,
        scale::Option{<:Number} = nothing
    ) -> ConcentrationRadius

Keywords correspond to the fields of the struct. The default `confidence` is `0.95`. The default `scale` is `nothing`, which reads the average per-asset dispersion from the sample. The series of that dispersion is not a field of the rule. It reaches the rule in the [`CalibrationContext`](@ref) of the site.

## Validation

  - `0 < confidence < 1`.
  - If `scale` is not `nothing`: `scale > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`DualNormRadius`](@ref): gives the sampling error in the ground metric that the slot names, so its number changes with the key, and the number of this rule does not.
  - [`Num_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:blanchet2019]) Section 3.3, Theorem 3 and Equation 22.
"""
@concrete struct ConcentrationRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    $(field_dict[:cal_confidence])
    """
    confidence
    """
    $(field_dict[:cal_scale])
    """
    scale
    function ConcentrationRadius(confidence::Number, scale::Option{<:Number})
        assert_unit_interval(confidence, :confidence)
        assert_nonempty_gt0_finite_val(scale, :scale)
        return new{typeof(confidence), typeof(scale)}(confidence, scale)
    end
end
function ConcentrationRadius(; confidence::Number = 0.95, scale::Option{<:Number} = nothing)
    return ConcentrationRadius(confidence, scale)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the chi-squared ambiguity radius on the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
r &= s \\sqrt{\\dfrac{\\chi^{2}_{N,\\, q}}{T_{c}}}\\,,\\\\
s &= \\begin{cases}
\\dfrac{1}{N} \\sum\\limits_{i=1}^{N} \\hat{s}_{i} & \\textrm{if } \\texttt{scale} \\textrm{ is } \\texttt{nothing}\\\\
\\texttt{scale} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - $(math_dict[:cal_s_radius])
  - ``\\chi^{2}_{N,\\, q}``: Quantile of the chi-squared distribution with ``N`` degrees of freedom at confidence level ``q``.
  - $(math_dict[:N])
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_T_c])
  - $(math_dict[:w_t_obs])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The radius is the same for every key, so the two tails of a Range measure that hold one rule resolve to one number.
  - `pr`: Prior result from which the rule reads the sample size, the asset count and, for a returns series, the covariance matrix. A drawdown series reads the sample.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.series` names the quantity around which the ball is drawn. The rule reads the per-asset dispersion of that quantity when `scale` is `nothing`. `ctx.n` is the count that the site prices, and the rule reads [`effective_sample_size`](@ref) when it is `nothing`.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:blanchet2019]) Section 3.3, Theorem 3 and Equation 22.
"""
function (alg::ConcentrationRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any,
                                    ctx::CalibrationContext)
    N = size(pr.X, 2)
    T = isnothing(ctx.n) ? effective_sample_size(pr, w) : ctx.n
    scale = if isnothing(alg.scale)
        Statistics.mean(calibration_series_dispersion(ctx.series, pr))
    else
        alg.scale
    end
    q = Distributions.cquantile(Distributions.Chisq(N),
                                one(alg.confidence) - alg.confidence)
    return scale * sqrt(q / T)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius that decreases with the square root of the sample length.

The radius is `c / sqrt(T)`. The rate has a theoretical basis, and the coefficient does not. So calibrate `c` with a cross-validation, and use this rule as the form of the radius over a grid of `c`.

**`T` is the count that the site prices, as for [`ConcentrationRadius`](@ref).** The rate ``1/\\sqrt{T}`` is the rate at which the empirical measure of `T` draws concentrates around the true law, and it is the basis of the rate of the three other radius rules. So `T` is the `n` that the site states in its [`CalibrationContext`](@ref), and [`effective_sample_size`](@ref) when the site states none. A distributionally robust measure states the count of the rows on which its ball is centred. Under a Scenario Cap, a read of the count in `eoc` there made the ball `sqrt(5)` times too small for `max_scenarios = 250` over 1250 observations. With observation weights, both counts are the effective sample size of the weights, because a weighted empirical measure concentrates at the rate of that count.

`c` has the units of the returns, because the rate has no units.

The rule does not check the range of its value, for the reason that [`ConcentrationRadius`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RateRadius(;
        c::Number = 1
    ) -> RateRadius

Keywords correspond to the fields of the struct. The default `c = 1` gives the rate ``1/\\sqrt{T}``.

## Validation

  - `c > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref): gives the sampling error in the ground metric that the slot names, so its number changes with the key, and the number of this rule does not.
  - [`Num_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct RateRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    $(field_dict[:cal_rate_c])
    """
    c
    function RateRadius(c::Number)
        assert_nonempty_gt0_finite_val(c, :c)
        return new{typeof(c)}(c)
    end
end
function RateRadius(; c::Number = 1)
    return RateRadius(c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the ambiguity radius at the square-root rate of the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
r &= \\frac{c}{\\sqrt{T_{c}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - ``c``: Rate coefficient, in the units of the returns.
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_T_c])
  - $(math_dict[:w_t_obs])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The rate is the same for every key.
  - `pr`: Prior result from which the rule reads the sample size.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.n` is the count that the site prices, and the rule reads [`effective_sample_size`](@ref) when it is `nothing`.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`RateRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`CalibrationContext`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::RateRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any,
                           ctx::CalibrationContext)
    return alg.c / sqrt(isnothing(ctx.n) ? effective_sample_size(pr, w) : ctx.n)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius that decreases at the dimensional rate of a Wasserstein ball, and not at the square-root rate.

The radius is the radius of Equation 8 of [drcvar](@cite) with the constants ``c_{1} = c_{2} = 1``, times a scale in the units of the series that the slot owner prices. It is the sample factor `log(1 / (1 - confidence)) / T` to the power `1 / max(N, 2)`. Equation 8 inverts the concentration bound of Theorem 3.4 of [drcvar](@cite), which is Theorem 2 of [fournier2015](@cite) for the Wasserstein distance of order one. A ball of probability measures must contain the true **measure**, and not only the true mean, and the concentration bound needs that exponent for a measure.

The constants ``c_{1}`` and ``c_{2}`` depend on the dimension and on the light-tail condition of the bound. Section 3 of [drcvar](@cite) states that the proof of Theorem 2 of [fournier2015](@cite) gives them, and that the ball that they give is larger than necessary. Neither work states them as numbers. The rule sets both to one. So the radius has the rate of the bound, but it is not a calibrated confidence bound. Theorem 3.4 of [drcvar](@cite) excludes ``N = 2``. At that dimension, Theorem 2 of [fournier2015](@cite) has an additional logarithmic factor, and the rule does not apply it. At `N = 1` and `N = 2`, the rule gives the square-root rate of [`ConcentrationRadius`](@ref) and [`RateRadius`](@ref).

**The ball becomes only a little smaller over a wide universe.** At `N = 20` and `T = 250`, the factor `T^(-1/20)` is `0.76`, and a record ten times longer gives `0.68`. This is the curse of dimensionality of the Wasserstein ball. A square-root rate is a rate for the error of a *mean*, and this rule states a rate for a *measure*.

**In practice the radius is conservative.** Section 3 of [drcvar](@cite) recommends that the radius come from a bootstrap or a cross-validation, and not from this bound, and Section 7.2 compares a holdout method with a k-fold cross-validation. A portfolio that a ball this wide prices can hold only cash. [`RateRadius`](@ref) is the form for such a grid. So calibrate the radius with that rule, and use this rule to know the rate.

Equation 8 of [drcvar](@cite) has a second branch for a short record. Its exponent is `1 / a` and not `1 / max(N, 2)`. The branch applies when ``T < \\ln(c_{1} / \\beta) / c_{2}``, with ``\\beta = 1 - q``. With the constants of the rule, that is fewer than 3 rows at `confidence = 0.95`. The rule does not use this branch.

[`HillTailDecay`](@ref) does not give the `a` of that branch. The `a` of Equation 8 is the exponent of the light-tail condition ``\\mathbb{E}\\left[\\exp\\left(\\lVert \\boldsymbol{\\xi} \\rVert^{a}\\right)\\right] < \\infty`` of Assumption 3.3 of [drcvar](@cite). [`HillTailDecay`](@ref) estimates the index of a **power-law** tail, which does not meet that condition. So the two quantities have the same letter and different meanings.

`N` is the asset count, `size(pr.X, 2)`, because the ball is over the assets. A factor prior has a smaller effective dimension, but the ball is still over the assets, so a factor prior does not change `N`.

`scale` gives the units, as for [`ConcentrationRadius`](@ref). `scale = nothing` reads the average per-asset dispersion of the series that the slot owner prices.

[`DistributionallyRobustConditionalDrawdownatRisk`](@ref) prices a ball around the drawdown scenarios. The dimension of a scenario is still `N`, so the rate does not change. The site states the marker of the owner in `ctx.series`, and [`calibration_series_dispersion`](@ref) then reads the per-asset dispersion from the drawdown sample and not from `pr.sigma`. A drawdown column is a running function of its returns, so its dispersion is the larger of the two. So with `scale = nothing`, that owner gets a wider ball than a returns owner of the same sample. A stated `scale` has precedence, and it is the way to price a ball whose units are neither of the two.

`T` is the count that the site prices, as for [`ConcentrationRadius`](@ref): the `n` of its [`CalibrationContext`](@ref), and [`effective_sample_size`](@ref) when the site states none. Theorem 2 of [fournier2015](@cite) bounds the distance between the true law and the empirical measure of `T` draws. A distributionally robust measure centres its ball on the empirical measure of its rows, so it states the count of those rows. The rate is a statement of concentration, so it prices the effective sample size of the weights when weights are stated. [`RateRadius`](@ref) reads the same count, at the square-root rate.

The rule does not check the range of its value, for the reason that [`ConcentrationRadius`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionalRateRadius(;
        confidence::Number = 0.95,
        scale::Option{<:Number} = nothing
    ) -> DimensionalRateRadius

Keywords correspond to the fields of the struct. The default `confidence` is `0.95`. The default `scale` is `nothing`, which reads the average per-asset dispersion from the sample. The series of that dispersion is not a field of the rule. It reaches the rule in the [`CalibrationContext`](@ref) of the site.

## Validation

  - `0 < confidence < 1`.
  - If `scale` is not `nothing`: `scale > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar]) Section 3, Assumption 3.3, Theorem 3.4 and Equation 8, and Section 7.2.
  - $(ref_dict[:fournier2015]) Theorem 2.
"""
@concrete struct DimensionalRateRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    $(field_dict[:cal_dim_confidence])
    """
    confidence
    """
    $(field_dict[:cal_dim_scale])
    """
    scale
    function DimensionalRateRadius(confidence::Number, scale::Option{<:Number})
        assert_unit_interval(confidence, :confidence)
        assert_nonempty_gt0_finite_val(scale, :scale)
        return new{typeof(confidence), typeof(scale)}(confidence, scale)
    end
end
function DimensionalRateRadius(; confidence::Number = 0.95,
                               scale::Option{<:Number} = nothing)
    return DimensionalRateRadius(confidence, scale)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the ambiguity radius of the measure-concentration bound on the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
r &= s \\left(\\dfrac{\\ln\\left(\\dfrac{1}{1 - q}\\right)}{T_{c}}\\right)^{\\frac{1}{\\max(N,\\, 2)}}\\,,\\\\
s &= \\begin{cases}
\\dfrac{1}{N} \\sum\\limits_{i=1}^{N} \\hat{s}_{i} & \\textrm{if } \\texttt{scale} \\textrm{ is } \\texttt{nothing}\\\\
\\texttt{scale} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - $(math_dict[:cal_s_radius])
  - ``q``: Confidence level of the bound.
  - $(math_dict[:N])
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_T_c])
  - $(math_dict[:w_t_obs])

The first equation is Equation 8 of [drcvar](@cite) with ``c_{1} = c_{2} = 1`` and ``\\beta = 1 - q``, times the scale. The exponent is at most one half, so a universe of one or two assets gives the square-root rate and not a faster rate.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The radius is the same for every key, so the two tails of a Range measure that hold one rule resolve to one number.
  - `pr`: Prior result from which the rule reads the sample size, the asset count and, for a returns series, the covariance matrix. A drawdown series reads the sample.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.series` names the quantity around which the ball is drawn. The rule reads the per-asset dispersion of that quantity when `scale` is `nothing`. `ctx.n` is the count that the site prices, and the rule reads [`effective_sample_size`](@ref) when it is `nothing`.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`DimensionalRateRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar]) Section 3, Equation 8.
  - $(ref_dict[:fournier2015]) Theorem 2.
"""
function (alg::DimensionalRateRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any,
                                      ctx::CalibrationContext)
    N = size(pr.X, 2)
    T = isnothing(ctx.n) ? effective_sample_size(pr, w) : ctx.n
    scale = if isnothing(alg.scale)
        Statistics.mean(calibration_series_dispersion(ctx.series, pr))
    else
        alg.scale
    end
    return scale * (log(inv(one(alg.confidence) - alg.confidence)) / T)^inv(max(N, 2))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius in the ground metric that its slot names, so that two slots of two different norms get two different numbers.

The eight radius slots of the library do not measure distance in one norm. A radius multiplies a norm of the weight vector, and the ground metric of the ball is the dual of that norm. For a loss that is linear in the sample point, Theorem 6.3 and Proposition 6.5 of [drcvar](@cite) give the worst case over the ball as the sample mean of the loss plus the radius times the dual norm of the weights. So the `l1` coefficient of [`JuMPOptimiser`](@ref) is a distance in the ∞-norm, and its `linf` coefficient is a distance in the 1-norm. This rule reads `key`, selects the ground metric of that slot, and returns the sampling error of the empirical measure in that metric. [`ConcentrationRadius`](@ref) and [`RateRadius`](@ref) return one number for every key. That is correct inside one measure, and not correct across the eight slots.

A linear loss reads the sampling error of the mean vector. The per-asset scale of that error is the dispersion of the series that the slot owner prices, divided by the square root of the count that the site prices. The radius is a norm of that error vector, at the confidence level that `confidence` states.

**The vector has no term for the error of the covariance matrix.** A radius multiplies a norm of the weight vector, so it prices the part of the loss that is **linear** in the weights. The loss of a distributionally robust measure is linear in the sample point. The error of the second moment reaches the objective through the tail term and not through the ball. The tail term has its own coefficient, which [`TailTermParity`](@ref) sets. A radius with both errors prices one error two times.

`confidence` is a level **per coordinate**, and the rule does not correct it for the number of assets. The ∞-norm case is a maximum over `N` coordinates, so a level per coordinate is too low for it. A caller who wants a level over the whole vector states the corrected level, for example `1 - (1 - c) / N`.

**The rule leaves the correction to the caller, because the correction is large and it is a choice.** A Bonferroni correction from `0.95` to `1 - 0.05 / N` increases the quantile by about a third at four assets, and by more at twenty. Other corrections exist. A level over the vector for `N` correlated coordinates and one for `N` independent coordinates differ by an amount that only the correlation gives. So the rule uses the level per coordinate, and this docstring names the corrected level. A rule that corrects the level with no notice gives a caller a level that the caller did not ask for and cannot see.

The 1-norm case adds the per-asset errors, so it prices them as if they moved together. That is the worst case over the correlations, so it is the conservative choice for a radius. A form that reads the correlations gives a smaller ball, and this rule does not compute it.

**A radius must not be too small, so the rule keeps the worst case.** A radius that is too small gives a model that is robust to less than the caller asked for. That failure is silent: the optimisation solves, and the weights are not robust. A radius that is too large is visible, because the portfolio moves towards cash. The two errors are not symmetric, so the rule takes the error that a caller can see.

`ctx.p` serves the `:lpreg_val` slot alone. The ground metric of [`LpRegularisation`](@ref) is the metric of order ``q``, with ``1/p + 1/q = 1``, and `key` names the slot and not the norm order. The order belongs to the penalty, so that site states it in the [`CalibrationContext`](@ref) before it resolves the slot. The rule holds no order, so a caller who runs it outside that site states the order in the context. No other key reads it.

**`series` gives the drawdown owner a drawdown scale.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) measures the transport cost of its programme against the per-asset drawdown sample. So its ball is a ball over drawdown scenarios, and the ground metric is a distance between two such vectors. Under a drawdown marker, [`calibration_series_dispersion`](@ref) reads the error scale from that sample. `pr.sigma` is not read, because it is a moment of the returns, and no scaling of it gives a moment of a drawdown. The ground metric does not change with the series. Only the vector of the norm changes.

**The error scale of a drawdown series is too small, because of the record.** A drawdown is a running function of the path, so its entries are strongly dependent down a column. ``\\hat{s}_{i} / \\sqrt{T_{c}}`` prices a record of ``T_{c}`` independent draws, and a drawdown sample does not have them. The rule does not correct for this, as it does not correct for the number of assets. A correction needs a model of the dependence, and the sample gives none. So the rule uses the scale of the quantity as if the record were independent.

`T_c` is the count that the site prices, as for [`ConcentrationRadius`](@ref): the `n` of its [`CalibrationContext`](@ref), and [`effective_sample_size`](@ref) when the site states none. On a distributionally robust measure the error vector is the sampling error of the mean of the rows on which the ball is centred, so the count is the count of those rows. On a regularisation slot it is the sampling error of `pr.mu`, so the count is the count of the observations behind the moments.

The rule does not check the range of its value, for the reason that [`ConcentrationRadius`](@ref) states.

**A sample with no dispersion gives a radius of zero, and the slot accepts it.** Every entry of the dispersion vector is zero or more, so the radius is zero or more, and finite, for every admissible input. A sample whose columns never move gives an error vector of zeros, and a ball of no width is correct for it. The empirical measure is the only measure that such a sample supports, and the robust measure becomes the plain measure at the centre of the ball. In floating point, such a sample gives a very small number and not an exact zero. The `> 0` check of the slot owner then passes, and the model prices a ball of almost no width. The two results have the same meaning.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DualNormRadius(;
        confidence::Number = 0.95
    ) -> DualNormRadius

Keywords correspond to the fields of the struct. The default `confidence` is `0.95`. The norm order and the series are not fields of the rule. Both reach the rule in the [`CalibrationContext`](@ref) of the site.

## Validation

  - `0 < confidence < 1`.
  - The norm order is not a field, so the constructor cannot check it. [`dual_norm_radius_scale`](@ref) checks `ctx.p` when it resolves the `:lpreg_val` slot. `ctx.p` must be stated, finite and more than one, as for [`LpRegularisation`](@ref).

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref): gives the width of the ball at a confidence level, with one factor that has no units and reads no norm.
  - [`RateRadius`](@ref): gives the rate at which the ball becomes smaller with the record, and leaves the coefficient to a cross-validation.
  - [`Num_AmbRadCal`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar]) Section 6, Theorem 6.3 and Proposition 6.5.
"""
@concrete struct DualNormRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    `confidence`: Confidence level per coordinate of the normal quantile that multiplies the error scale. The rule does not correct it for the number of assets, so the ∞-norm case reads a level over one coordinate and not over the vector.
    """
    confidence
    function DualNormRadius(confidence::Number)
        assert_unit_interval(confidence, :confidence)
        return new{typeof(confidence)}(confidence)
    end
end
function DualNormRadius(; confidence::Number = 0.95)
    return DualNormRadius(confidence)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the ambiguity radius of the slot that `key` names, in the ground metric of that slot, on the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
r &= z_{c} \\lVert \\boldsymbol{e} \\rVert_{g}\\,,\\\\
e_{i} &= \\dfrac{\\hat{s}_{i}}{\\sqrt{T_{c}}}\\,,\\\\
q &= \\dfrac{p}{p - 1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - ``z_{c}``: Quantile of the standard normal distribution at the per-coordinate confidence level ``c``.
  - ``\\boldsymbol{e}``: Per-asset scale of the sampling error of the mean vector, in the units of the series that the slot owner prices.
  - ``g``: Order of the ground metric, which `key` names.
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_T_c])
  - $(math_dict[:w_t_obs])
  - ``p``: Norm order of the penalty, the `p` field.
  - ``q``: Order of the ground metric of that penalty.

# The ground metric of each key

| `key`          | Slot                                                                                                              | Penalised norm | Ground metric ``g`` |
|:-------------- |:----------------------------------------------------------------------------------------------------------------- |:-------------- |:------------------- |
| `:l1`          | `l1` of [`JuMPOptimiser`](@ref)                                                                                   | 1              | ``\\infty``         |
| `:linf`        | `linf` of [`JuMPOptimiser`](@ref)                                                                                 | ``\\infty``    | 1                   |
| `:r`           | [`DistributionallyRobustConditionalValueatRisk`](@ref), [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) | ``\\infty``    | 1                   |
| `:r_a`, `:r_b` | [`DistributionallyRobustConditionalValueatRiskRange`](@ref)                                                       | ``\\infty``    | 1                   |
| `:l2reg_val`   | `val` of [`L2Regularisation`](@ref)                                                                               | 2              | 2                   |
| `:lpreg_val`   | `val` of [`LpRegularisation`](@ref)                                                                               | ``p``          | ``q``               |

The two ends of a Range measure have one ground metric, so a rule on both ends resolves to one number. A radius names no end of the distribution.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. It selects the ground metric, so this is the one rule of its family for which the key has a meaning.
  - `pr`: Prior result from which the rule reads the sample size and, for a returns series, the covariance matrix. A drawdown series reads the sample.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.series` names the quantity in which the rule measures the error vector. `ctx.p` is the norm order of the penalty, and only the `:lpreg_val` key reads it. `ctx.n` is the count that the site prices, and the rule reads [`effective_sample_size`](@ref) when it is `nothing`.

# Validation

  - `key` must be one of the seven keys of the table, else an `ArgumentError` that names the key that the rule got and the keys that it serves.
  - The `:lpreg_val` key needs a stated `p`, else an `ArgumentError` that names the field.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`DualNormRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar]) Section 6, Theorem 6.3 and Proposition 6.5.
"""
function (alg::DualNormRadius)(key::Symbol, pr::AbstractPriorResult, w, ::Any,
                               ctx::CalibrationContext)
    T = isnothing(ctx.n) ? effective_sample_size(pr, w) : ctx.n
    e = calibration_series_dispersion(ctx.series, pr) ./ sqrt(T)
    z = Distributions.quantile(Distributions.Normal(), alg.confidence)
    return z * dual_norm_radius_scale(key, ctx.p, e)
end
"""
    dual_norm_radius_scale(key::Symbol, p::Option{<:Number}, e::AbstractVector)

Reduces the per-asset error vector `e` to one number, in the ground metric that `key` names.

This function holds all the meaning of the key, apart from the functor. Six of the seven keys name a fixed norm order, so they are a literal table and not a chain of branches. The seventh key, `:lpreg_val`, gets its order from the `p` of the context, because no key can name it. The function refuses any other key. A caller who writes a measure gets this refusal first, so the message names the keys that the rule serves.

# Arguments

  - `key`: Name of the slot that the call resolves.
  - `p`: Norm order that the site states in its [`CalibrationContext`](@ref). Only the `:lpreg_val` branch reads it.
  - `e`: Per-asset scale of the sampling error of the mean vector.

# Validation

  - `key` must be one of the seven keys that [`DualNormRadius`](@ref) serves, else an `ArgumentError`.
  - `p` must be stated when `key` is `:lpreg_val`, else an `ArgumentError`. It must then be finite and more than one, so that the conjugate order `q = p / (p - 1)` exists.

# Returns

  - `s::Number`: The norm of `e` in the ground metric of the slot.

# Related

  - [`DualNormRadius`](@ref)
  - [`LpRegularisation`](@ref)
"""
function dual_norm_radius_scale(key::Symbol, p::Option{<:Number}, e::AbstractVector)
    metrics = (; l1 = Inf, linf = 1, r = 1, r_a = 1, r_b = 1, l2reg_val = 2)
    g = if key === :lpreg_val
        @argcheck(!isnothing(p),
                  ArgumentError("`CalibrationContext.p` is `nothing` while the `:lpreg_val` slot is being resolved. That slot's ground metric is the type-`q` metric of an `LpRegularisation` penalty, with `1/p + 1/q = 1`, and the order belongs to the penalty. The penalty site states it in the context, so a `nothing` here means the rule was resolved somewhere that states no order. Place the rule in the `val` field of an `LpRegularisation`, or state `p` in the context."))
        @argcheck(isfinite(p) && p > one(p),
                  DomainError(p,
                              "`CalibrationContext.p` is $p, and the type-`q` ground metric of an `LpRegularisation` penalty needs a finite order above one, because `q = p / (p - 1)` is what `DualNormRadius` takes the norm in. An order of one, or an infinite one, names no conjugate the rule can read."))
        p / (p - one(p))
    else
        get(() -> throw(ArgumentError("`DualNormRadius` reads `key` to pick the ground metric of the slot it stands in, and it received `:$key`, which names no slot it serves. The keys it serves are `:l1`, `:linf`, `:r`, `:r_a`, `:r_b`, `:l2reg_val` and `:lpreg_val`. A measure of your own that holds a radius resolves its slot under one of those keys, or carries a rule of its own.")),
            metrics, key)
    end
    return LinearAlgebra.norm(e, g)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Esfahani-Kuhn tail weight that prices the tail term of the loss at a stated multiple of its mean term.

At ``r \\to 0``, the loss of [`DistributionallyRobustConditionalValueatRisk`](@ref) becomes ``-\\mathbb{E}[\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}] + l \\, \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi})``, the mean-risk objective of Equation 26 of [drcvar](@cite), where the tail weight is ``\\rho``. So `l` is the exchange rate between one unit of expected return and one unit of tail loss. Both terms have the units of the returns, so `l` has no units. **But `l` is not independent of the scale of the sample.** For returns that are independent over time, the mean of a return column increases with the horizon, and its CVaR increases about as the square root of the horizon. So the ratio of the CVaR to the mean decreases about as the square root of the horizon. A stated `l = 1.0` thus prices the tail term at a much larger multiple of the mean term on a daily sample than on a monthly sample. One stated number is a different trade-off at each sampling frequency, and at each fit that changes the record.

The rule reads the units of the sample and nothing else, so the preference stays with the caller. `ratio` states how many mean terms one tail term is worth. The rule returns the `l` that gives this ratio on the sample of the prior result. `ratio = 1` is parity, and `ratio = 2` prices the tail term at two times the mean term on every sample.

A rule reads no portfolio, so it cannot make ``\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}``. It reads the cross-section of the per-asset losses of the quantity that the slot owner prices. The mean-term scale ``m`` is the mean of that pool, and the tail-term scale ``c`` is the mean of the ``\\mathrm{CVaR}_{\\alpha}`` of each column. **The mean of the values per column is not the pooled ``\\mathrm{CVaR}_{\\alpha}``, and the difference is intentional.** A pooled tail comes from the worst columns, so the most volatile assets control it, and it gives a smaller weight. The measure prices a portfolio and not a single asset, so the rule uses the mean over the columns.

**A caller cannot state a reference portfolio.** The pooled cross-section gives the rate for equal weights, and a reference portfolio gives a more exact rate. But a reference portfolio makes the rule a second place where a portfolio enters the problem. [`resolve_calibration_slot`](@ref) gives a rule the Prior and nothing else, so a reference vector needs a separate route. The rule must also check the vector against the universe of each fold, and the caller must state it again when a subset view or a cluster changes that universe. The exchange rate is a property of the **sample**, and the cross-section with equal weights needs no second input. A caller who wants the rate at a portfolio computes `l` and states the number.

``m`` is negative for a sample with a positive expected return, and the rule takes ``\\lvert m \\rvert``. No field states the sign. The slot does not accept a negative weight, and a sample with a negative expected return does not reverse the trade-off.

The rule reads its sibling `alpha`, because ``c`` is a ``\\mathrm{CVaR}`` at the significance level of the measure. So the resolution method of the owner resolves `alpha` first and puts the number in `ctx.alpha`, as it does for `alpha` and `kappa`. A caller who runs the rule outside a measure states `alpha` in the context.

**Both scales read the observation weights.** They are sample statistics and not counts, so the rule reads a weighted sample with its weights. A rule that prices a count, such as [`RateRadius`](@ref) or [`RateSignificance`](@ref), reads `w` only through the effective sample size of the weights.

**`ctx.series` names the quantity of both terms, and the slot owner states it.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) also has the key `:l`, and its tail term is a ``\\mathrm{CDaR}`` of the drawdown series of the portfolio. A rule gets no portfolio, but it can make the drawdown series of each **column**, and the resolution site names the marker in `ctx.series`. Both terms then change together. The mean term is the mean drawdown of the pool, and the tail term is the mean of the ``\\mathrm{CDaR}_{\\alpha}`` of each column. The rule computes it with [`ConditionalValueatRisk`](@ref), because the tail mean of a drawdown column that is not positive **is** the ``\\mathrm{CDaR}`` of that column. So the rule and the measure that it calibrates use one definition.

The rule does not use a second meaning of `l`. `l` can be read as a coefficient of risk aversion and mapped from a mean-variance coefficient. But a variance penalty is quadratic in the weight vector, and a ``\\mathrm{CVaR}`` term is positively homogeneous, so the two objectives cannot be compared term by term. The map is correct at one reference portfolio only, and a rule gets no portfolio.

The rule does not check the range of the weight that it returns, for the reason that [`RateRadius`](@ref) states. The constructor of the slot owner checks it. The two refusals of the rule state that the ratio exists, as the check of [`EntropyBudget`](@ref) states that its root exists. A sample whose mean loss is near zero gives a very large weight, and that is correct for such a sample. No threshold separates a small mean loss from a smaller one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TailTermParity(;
        ratio::Number = 1
    ) -> TailTermParity

Keywords correspond to the fields of the struct. The default `ratio = 1` is parity between the two terms. The significance level and the series of both terms are not fields of the rule. Both reach the rule in the [`CalibrationContext`](@ref) of the site.

## Validation

  - `ratio > 0` and finite.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`CalibrationContext`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`RateRadius`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar]) Section 7.1, Equation 26.
"""
@concrete struct TailTermParity <: AbstractAmbiguityTailWeightCalibrationAlgorithm
    """
    $(field_dict[:cal_ratio])
    """
    ratio
    function TailTermParity(ratio::Number)
        assert_nonempty_gt0_finite_val(ratio, :ratio)
        return new{typeof(ratio)}(ratio)
    end
end
function TailTermParity(; ratio::Number = 1)
    return TailTermParity(ratio)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the tail weight that prices the tail term at `ratio` mean terms on the sample of `pr`.

# Mathematical definition

```math
\\begin{align}
l &= \\rho \\dfrac{\\lvert m \\rvert}{c}\\,,\\\\
y_{tj} &= \\left(\\mathcal{S}\\left(\\boldsymbol{r}_{j}\\right)\\right)_{t}\\,,\\\\
m &= -\\dfrac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{j=1}^{N} w_{t} y_{tj}}{N \\sum\\limits_{t=1}^{T} w_{t}}\\,,\\\\
c &= \\dfrac{1}{N} \\sum\\limits_{j=1}^{N} \\mathrm{CVaR}_{\\alpha}\\left(\\boldsymbol{y}_{j}\\right)\\,.
\\end{align}
```

Where:

  - ``l``: Esfahani-Kuhn tail weight, the weight of the tail term of the loss.
  - ``\\rho``: Number of mean terms that one tail term is worth.
  - ``m``: Mean-term scale, the mean of the pooled cross-section of the per-asset losses.
  - ``c``: Tail-term scale, the mean over the columns of the ``\\mathrm{CVaR}_{\\alpha}`` of the loss.
  - ``\\boldsymbol{r}_{j}``: Column ``j`` of the returns matrix.
  - ``\\mathcal{S}``: The series that `ctx.series` names, made from one column. It is the identity for a [`ReturnsSeries`](@ref), and a drawdown series for the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``y_{tj}``: Entry ``t`` of the series of column ``j``.
  - ``\\boldsymbol{y}_{j}``: The series of column ``j``. The ``\\mathrm{CVaR}_{\\alpha}`` of a drawdown series that is not positive is the ``\\mathrm{CDaR}_{\\alpha}`` of that column.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:r_tj])

Every observation weight is one when no weight is stated. Every column holds ``T`` entries, so the pooled mean and the mean of the column means are equal, and the definition uses the pooled form.

# Algorithm

 1. Make the sample that `ctx.series` names with [`calibration_series_matrix`](@ref) into `X`, and read the effective observation weights from `w`. A [`ReturnsSeries`](@ref) returns `pr.X`.
 2. Make the mean-term scale `m`, the negative weighted mean of the pooled cross-section of `X`.
 3. Make a [`ConditionalValueatRisk`](@ref) at `ctx.alpha` with the same weights. Make the tail-term scale `c`, the mean over the columns of `X` of the value of that measure on each column. The rule uses the measure itself, so the rule and the measure that it calibrates use one definition.
 4. Return `alg.ratio * abs(m) / c`.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The rule reads the scales from the asset columns, so the key does not select the value. The two ends of a Range measure differ only through their two probabilities.
  - `pr`: Prior result from which the rule reads the returns matrix. The rule makes the series from its columns.
  - `w`: Effective observation weights, or `nothing`. Both scales are sample statistics, so this rule reads them.
  - `slv`: Effective solver. This rule needs none, because both scales are closed forms.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.alpha` is the significance level of the measure, and it must hold a number. `ctx.series` names the quantity of both terms.

# Validation

  - `ctx.alpha` must not be `nothing`.
  - ``\\lvert m \\rvert`` must be positive.
  - ``c`` must be positive.

# Returns

  - `l::Number`: The Esfahani-Kuhn tail weight.

# Related

  - [`TailTermParity`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series_matrix`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar]) Section 7.1, Equation 26.
"""
function (alg::TailTermParity)(::Symbol, pr::AbstractPriorResult, w, ::Any,
                               ctx::CalibrationContext)
    @argcheck(!isnothing(ctx.alpha),
              IsNothingError("`CalibrationContext.alpha` is `nothing`, so `TailTermParity` cannot form the tail-term scale `c`, which is a CVaR at the measure's own significance level. The probability reaches the rule in the context the slot owner builds, after it resolves `alpha`. State `alpha` in the context to run the rule outside a measure."))
    # Both scales are read over the series the OWNER prices. Under a returns marker this is
    # `pr.X` itself, and under a drawdown marker the tail term is a CDaR: the CVaR kernel
    # over a non-positive drawdown column is the mean of its worst `alpha`.
    X = calibration_series_matrix(ctx.series, pr.X)
    N = size(X, 2)
    ws = get_observation_weights(w, view(X, :, 1))
    m = if isnothing(ws)
        -Statistics.mean(X)
    else
        -sum(j -> LinearAlgebra.dot(view(X, :, j), ws), axes(X, 2)) / (N * sum(ws))
    end
    @argcheck(abs(m) > 0,
              DomainError(m,
                          "`TailTermParity` read a mean-term scale of $m on the sample, so the ratio `ratio * abs(m) / c` is zero and the slot admits no such weight. The rule prices one term of the loss against the other, and a sample whose pooled mean loss is zero states no exchange rate between them: every tail weight prices a mean term of zero alike. State `l` on the slot instead."))
    # The tail term the weight scales is a CVaR, so the reading is the measure's own rather
    # than a second encoding of it here. A rule that carried its own copy would drift from
    # the measure it calibrates the moment either moved.
    rm = ConditionalValueatRisk(; alpha = ctx.alpha, w = w)
    c = Statistics.mean(j -> rm(view(X, :, j)), axes(X, 2))
    @argcheck(c > 0,
              DomainError(c,
                          "`TailTermParity` read a tail-term scale of $c at `alpha = $(ctx.alpha)`, which is not positive, so the ratio `ratio * abs(m) / c` has no admissible value. The scale is the mean of the per-column CVaR of the loss, and a non-positive one is a sample whose worst `alpha` of every column holds no loss at all. Widen `alpha`, or state `l` on the slot."))
    return alg.ratio * abs(m) / c
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a norm ceiling that keeps a stated fraction of the universe effective, so that the floor changes when the universe changes.

A bound on the norm of the weights is a floor on the effective number of assets. This rule states the floor as a **fraction of the universe** and not as a count. The asset count comes from the prior result, so a subset view, a cluster and a cross-validation fold each get the floor of their own universe. No count stays fixed to the universe for which a caller wrote it.

`ctx.p` is the norm order of the ceiling, and it belongs to the constraint. Each of the three constraint sites states it in the [`CalibrationContext`](@ref) before it resolves the slot. The rule holds no order, so a caller who runs it outside those sites states the order in the context.

The rule does not check the range of its value, for the reason that [`ConcentrationRadius`](@ref) states. [`resolve_calibration_slot`](@ref) checks that the ceiling is `> 0` and finite.

# Mathematical definition

```math
\\begin{align}
m &= f N\\,,\\\\
\\mathrm{val} &= \\begin{cases}
m^{-1} & \\textrm{if } p \\textrm{ is infinite}\\\\
m^{1/p - 1} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{val}``: Norm ceiling.
  - ``m``: Effective number of assets that the ceiling keeps as a floor.
  - ``f``: Fraction of the universe, the `fraction` field.
  - $(math_dict[:N])
  - ``p``: Norm order of the constraint of the ceiling.

The effective number of assets of order ``p`` is ``N_{\\mathrm{eff},\\, p}(\\boldsymbol{w}) = \\left(\\sum_{i} \\lvert w_{i} \\rvert^{p}\\right)^{1/(1 - p)}``. So for ``p > 1``, ``N_{\\mathrm{eff},\\, p} \\geq m`` is ``\\lVert \\boldsymbol{w} \\rVert_{p} \\leq m^{1/p - 1}``. At ``p = 2``, this count is the number that [`number_effective_assets`](@ref) gives. At every order, a portfolio with equal weights over ``m`` assets gives exactly ``m``. The exponent is also ``-1/q`` for the conjugate order ``q``, because ``1/p - 1 = -1/q``. At ``p = 1`` the ceiling is one for every ``m``.

The infinite branch is the limit of the finite branch. ``m^{1/p - 1} \\to m^{-1}`` as ``p`` increases. A cap of ``1/m`` on the largest weight of a fully invested portfolio spreads it over at least ``m`` assets. So the two branches agree, and as a caller increases ``p``, the ceiling moves towards ``1/m``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EffectiveAssetFloor(;
        fraction::Number = 0.5
    ) -> EffectiveAssetFloor

Keywords correspond to the fields of the struct. The default `fraction = 0.5` keeps half of the universe effective. The norm order of the ceiling is not a field of the rule. It reaches the rule as `ctx.p` in the [`CalibrationContext`](@ref) of the constraint site.

## Validation

  - `0 < fraction <= 1`.
  - The norm order is not a field, so the constructor cannot check it. The rule checks `ctx.p` when it runs. It must be stated, and it must be one or more.

# Related

  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`CalibrationContext`](@ref)
  - [`number_effective_assets`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct EffectiveAssetFloor <: AbstractNormCeilingCalibrationAlgorithm
    """
    $(field_dict[:cal_fraction])
    """
    fraction
    function EffectiveAssetFloor(fraction::Number)
        assert_nonempty_gt0_finite_val(fraction, :fraction)
        @argcheck(fraction <= one(fraction),
                  DomainError(fraction,
                              "`EffectiveAssetFloor.fraction` is $fraction, and a fraction of the universe is at most one. The rule multiplies it by the asset count of the prior result to get the effective number of assets it holds as a floor, so a fraction above one names a floor above the universe, which no portfolio meets. State a value in `(0, 1]`."))
        return new{typeof(fraction)}(fraction)
    end
end
function EffectiveAssetFloor(; fraction::Number = 0.5)
    return EffectiveAssetFloor(fraction)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the norm ceiling that keeps the fraction `alg.fraction` of the universe of `pr` effective.

The universe is the column count of the returns matrix of the prior result, so the ceiling changes with the universe of the model.

# Mathematical definition

```math
\\begin{align}
\\mathrm{val} &= \\begin{cases}
(f N)^{-1} & \\textrm{if } p \\textrm{ is infinite}\\\\
(f N)^{1/p - 1} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{val}``: Norm ceiling.
  - ``f``: Fraction of the universe, the `fraction` field.
  - $(math_dict[:N])
  - ``p``: Norm order of the constraint of the ceiling, `ctx.p`.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that the call resolves. The refusal below names the site with it, and the ceiling does not depend on it. The ceiling depends on the norm order.
  - `pr`: Prior result from which the rule reads the asset count.
  - `w`: Effective observation weights. A count of the universe is not a count of the sample, so this rule does not read them.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The [`CalibrationContext`](@ref) of the site. `ctx.p` is the norm order of the ceiling, and it must hold a number.

# Validation

  - `ctx.p` must not be `nothing`, and it must be one or more. Every constraint site states it, so `nothing` means that a context with no order reached the rule.

# Returns

  - `val::Number`: The norm ceiling.

# Related

  - [`EffectiveAssetFloor`](@ref)
  - [`CalibrationContext`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::EffectiveAssetFloor)(key::Symbol, pr::AbstractPriorResult, ::Any, ::Any,
                                    ctx::CalibrationContext)
    p = ctx.p
    @argcheck(isnothing(p) || p >= one(p),
              DomainError(p,
                          "`CalibrationContext.p` is $p, and a norm ceiling is read against an order of one or more. The three constraint sites state `2`, `Inf` and the order of their own term, so an order below one reaches this rule only from a context a caller built by hand."))
    @argcheck(!isnothing(p),
              ArgumentError("`$(nameof(CalibrationContext)).p` is `nothing` while the rule in `$key` is being resolved. A ceiling is read against one norm order, the order belongs to the constraint, and each constraint site states it in the context. Place the rule in `l2c`, `lpc` or `linfc`, or state `p` in the context."))
    m = alg.fraction * size(pr.X, 2)
    return isinf(p) ? inv(m) : m^(inv(p) - one(p))
end

export CalibrationContext, ReturnsSeries, AbsoluteDrawdownSeries, RelativeDrawdownSeries,
       ScenarioCount, RateSignificance, EntropyBudget, HillTailDecay, RadialTailDecay,
       ConcentrationRadius, RateRadius, DimensionalRateRadius, DualNormRadius,
       TailTermParity, EffectiveAssetFloor
public AbstractSignificanceCalibrationAlgorithm, AbstractDeformationCalibrationAlgorithm,
       AbstractAmbiguityRadiusCalibrationAlgorithm,
       AbstractAmbiguityTailWeightCalibrationAlgorithm,
       AbstractNormCeilingCalibrationAlgorithm
