---
status: accepted
---

# An estimator value algorithm is a value, and the branch a slot admits is in its bound

## Context

`AbstractEstimatorValueAlgorithm` names an algorithm that computes a value over a universe.
`estimator_to_val` runs it, and `UniformValues` — the one concrete member the library ships —
returns the equal-weight range over the universe a `UniverseSets` names.

Two aliases in [`src/01_Base/08_TypeAliases.jl`](../../src/01_Base/08_TypeAliases.jl) carried
the family root.

- `EstValType` bounds a slot that holds a **value**: `WeightBoundsEstimator`'s `lb` and `ub`,
  `RiskBudgetEstimator`'s `val`, `ThresholdEstimator`'s `val`, `TurnoverEstimator`'s `val`,
  `FeesEstimator`'s `l`, `s`, `fl` and `fs`, and `PortfolioTarget`'s `w`.
- `EqnType` bounds a slot that holds **equation text**: `LinearConstraintEstimator`'s `val`, and the
  `eqn` argument of `linear_constraints`, `parse_equation` and `black_litterman_views`.

The second was wrong, and
[#633](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/633) is what it cost.
`parse_equation` has no method for an algorithm, so an algorithm reaching an `EqnType` consumer
raised a `MethodError` — everywhere but one site. That site was a method of `linear_constraints`
written for the case:

```julia
function linear_constraints(lcs::LinearConstraintEstimator{<:AbstractEstimatorValueAlgorithm},
                            sets::UniverseSets, key::Option{<:AbstractString} = nothing;
                            datatype::DataType = Float64, strict::Bool = false,
                            args...)::Option{<:LinearConstraint}
    return estimator_to_val(lcs.val, sets, ..., key; datatype = datatype, strict = strict)
end
```

It declared `::Option{<:LinearConstraint}` and returned the `Num_VecNum` that `estimator_to_val`
computes, so the annotation refused every value the body could compute. The method raised on every
call it could receive, and it was the more specific of the two methods that answer the two-argument
call, so it won the dispatch. It was uncovered on CI, which is how it survived the sweep of
[#513](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/513).

Three optimiser slots take a `LinearConstraintEstimator` — `lcse`, `gcarde` and `sgcarde` in
[`src/20_Optimisation/10_JuMPOptimiser.jl`](../../src/20_Optimisation/10_JuMPOptimiser.jl) — and
each is bounded downstream by a constraint type rather than by a value. So
`JuMPOptimiser(; gcarde = LinearConstraintEstimator(; val = UniformValues()))` raised inside
`processed_jump_optimiser_attributes`, three frames from the slot the caller wrote, with a message
that named neither the slot nor the value.

## Decision

### The family splits by the shape of the value, and the branch carries a name

```julia
abstract type AbstractEstimatorValueAlgorithm <: AbstractAlgorithm end
abstract type VectorAbstractEstimatorValueAlgorithm <: AbstractEstimatorValueAlgorithm end
```

The root stays the family. `VectorAbstractEstimatorValueAlgorithm` is the branch whose
`estimator_to_val` returns a `Num_VecNum`, and `UniformValues` subtypes it. A shape a slot cannot
resolve is now a shape a slot can *name*, which is the whole point: Julia writes a bound as a
positive statement of what is admitted, never as a statement of what is refused.

### `EstValType` takes the branch as a type parameter

```julia
const EstValType{T <: AbstractEstimatorValueAlgorithm} = Union{<:Num_VecNum, <:MatNum,
                                                               <:PairStrNum,
                                                               <:MultiEstValType, T}
```

Every value slot listed above writes `EstValType{<:VectorAbstractEstimatorValueAlgorithm}`, so the
branch a slot admits is part of the slot's own type. The alias written without a parameter is the
widest bound the family gives, and a branch added later reaches every slot that names it without a
second alias.

### `EqnType` holds equation text and nothing else

```julia
const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr}
```

No branch of the family is an `EqnType`. An algorithm computes a value over the universe, and a
value carries neither a comparison operator nor a side, so no constraint row can be assembled from
it. The `linear_constraints` method quoted above is deleted, and `LinearConstraintEstimator(; val =
UniformValues())` raises at the constructor, naming the `val` keyword and the value it refused.

## Consequences

- The raise moves from `processed_jump_optimiser_attributes` to the estimator the caller built, and
  the message names the slot. A caller who wanted the equal-weight vector wanted a value slot, and
  the three optimiser slots that take a `LinearConstraintEstimator` never accepted one.
- No documented route moves. Every documented user of `UniformValues` is a `WeightBoundsEstimator`
  slot, and every such slot keeps it.
- The residue of #513 for
  [`02_LinearConstraintGeneration.jl`](../../src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl)
  falls to zero: the two uncovered lines were the deleted method's signature and its body.
- `test/test_18k_constraints.jl` gates the split, the refusal and every value slot that keeps the
  algorithm.

## Alternatives refused

1. **Correct the annotation to `Num_VecNum`.** One line, and the body then returns what it computes.
   But no consumer of `linear_constraints` accepts a numeric vector, so the raise moves one frame
   later to `ProcessedJuMPOptimiserAttributes`'s field bound. The route stays unusable and the method
   stays uncovered.
2. **Delete the method and leave `EqnType` carrying the root.** The estimator then falls to the
   method that reads `val` off it, which lands on the equation method and reaches `parse_equation`
   with an algorithm. That is a `MethodError` from inside the parser rather than a bound at the slot.
3. **Name an equation-producing branch as well, and let `EqnType` carry that.** It would be an
   abstract type with no subtype and no method. The design of such an algorithm is not settled, so it
   is deferred: when one is written it subtypes the root, takes its own branch, and `EqnType` names
   that branch then.
