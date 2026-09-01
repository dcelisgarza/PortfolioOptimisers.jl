"""
$(DocStringExtensions.TYPEDEF)

Computes a calibrated quantity — a tail probability, a deformation parameter, an ambiguity radius, an Esfahani-Kuhn tail weight or a norm ceiling — from the data a prior result carries, so that the quantity refits whenever the sample moves.

All concrete subtypes should subtype one of the families under this root rather than the root itself, and `# Related` names them. A plain number in place of a rule is the quantity itself, exactly as it is today.

This root carries the **rules** alone. A **role**, the type that places a rule in the slot of one quantity, is an Estimator under [`AbstractCalibrationEstimator`](@ref) and subtypes nothing here, so a role placed inside another role's `alg` field is refused by the field's bound.

A **Calibration Rule** is not a [`DeferredQuantity`](@ref), and the two mechanisms stay parallel end to end. A Deferred Quantity is *fitted* and the quantity is read off the fit; a rule fits nothing, and reads the sample size and the moments the prior result already carries. A rule also sees the effective observation weights, which [`resolve_slot`](@ref) does not carry. So a rule resolves through [`resolve_calibration_slot`](@ref), is declared through [`calibration_slots`](@ref), and is refused at a value-level entry point by [`assert_calibrated_slots`](@ref).

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`DeferredQuantity`](@ref)
"""
abstract type AbstractCalibrationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Places a **Calibration Rule** in the slot of one quantity, and names the quantity the slot holds.

A role is configuration that carries an algorithm, so a role is an Estimator and the rule it carries is the Algorithm. [`AbstractOrderedWeightsArrayEstimator`](@ref) and [`AbstractOrderedWeightsArrayAlgorithm`](@ref) are the same pair, and the two calibration roots are separate for the same reason.

The separation is the whole of the role validation. Every `alg` field is bounded by a `Func_` union that names one rule family under [`AbstractCalibrationAlgorithm`](@ref), and no role subtypes that root. So a role placed inside another role's `alg` field is refused **at construction, by the bound**, and no guard method is written for it.

The root is flat, and no role carries a per-family supertype, because nothing dispatches on a family of roles. Each slot's `Num_` bound names one concrete role, and the three sites that read a role — [`resolve_calibration_slot`](@ref), [`assert_calibrated_slots`](@ref) and [`sel`](@ref) — read this root itself.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`assert_calibrated_slots`](@ref)
"""
abstract type AbstractCalibrationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level, the tail probability that an `alpha` or a `beta` slot holds.

All concrete subtypes should subtype `AbstractSignificanceCalibrationAlgorithm`, and should be **callable**, because [`resolve_calibration_slot`](@ref) runs a rule by calling it. A plain `Function` of the same four arguments is therefore a rule as well, and needs no type at all. The family's two role types, [`SignificanceTailCalibration`](@ref) and [`SignificanceHeadCalibration`](@ref), subtype [`AbstractCalibrationEstimator`](@ref) instead: a role names the end of the distribution the slot addresses and holds the rule in its `alg` field, and the same rule serves both ends. Neither role subtypes this family, so neither is admitted by [`Func_SigCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractSignificanceCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractSignificanceCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the significance level.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the measure nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the measure carries none.

### Returns

  - `alpha::Number`: The significance level.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`Func_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
abstract type AbstractSignificanceCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes a deformation parameter, the Kaniadakis ``\\kappa`` that a `kappa` slot holds.

All concrete subtypes should subtype `AbstractDeformationCalibrationAlgorithm`, and should be **callable**, on the same terms as the significance family. The family's two role types, [`DeformationTailCalibration`](@ref) and [`DeformationHeadCalibration`](@ref), subtype [`AbstractCalibrationEstimator`](@ref) instead, and neither is admitted by [`Func_DefCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractDeformationCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractDeformationCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the deformation parameter.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the measure nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the measure carries none.

### Returns

  - `kappa::Number`: The deformation parameter.

## The series

A deformation slot sits on a measure of the return distribution and on a drawdown measure alike, and both resolve the key `:kappa`, so `key` does not say which quantity the owner prices. A rule whose answer moves with that quantity should carry a `series` field and implement a [`bind_series`](@ref) method, which each slot owner calls before it resolves the slot. A rule whose answer does not move with it needs no method, and [`EntropyBudget`](@ref) is that case.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_series`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`Func_DefCal`](@ref)
  - [`kappa_log`](@ref)
"""
abstract type AbstractDeformationCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
    const Func_SigCal = Union{<:Function, <:AbstractSignificanceCalibrationAlgorithm}

Field bound for the `alg` field of a significance role: a rule of the family, or a plain function of the same four arguments.

A rule is run by calling it, so a function and a callable struct are the same thing to [`resolve_calibration_slot`](@ref). The struct earns its keep by carrying parameters and a name that a docstring can describe; the function is the shortest way to state a one-off rule, and a closure over a caller's own data is the case that has no type.

A function carries no family, so it is admitted by both role families and the family split cannot refuse it. Only a rule that names its family is checked.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`Func_DefCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_SigCal = Union{<:Function, <:AbstractSignificanceCalibrationAlgorithm}
"""
    const Func_DefCal = Union{<:Function, <:AbstractDeformationCalibrationAlgorithm}

Field bound for the `alg` field of a deformation role: a rule of the family, or a plain function of the same four arguments. It is the counterpart of [`Func_SigCal`](@ref), and carries its reading unchanged.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`Func_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_DefCal = Union{<:Function, <:AbstractDeformationCalibrationAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Places a significance rule in a slot that addresses the lower tail of the return distribution.

The role is the whole of the type: the rule itself lives in `alg`, and both ends of the distribution take the same rule. A slot bounded by [`Num_SigTailCal`](@ref) admits this role and refuses [`SignificanceHeadCalibration`](@ref), so a head rule placed in a tail slot is refused at construction rather than at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignificanceTailCalibration(;
        alg::Func_SigCal
    ) -> SignificanceTailCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`Num_SigTailCal`](@ref)
  - [`Func_SigCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct SignificanceTailCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_sig])
    """
    alg
    function SignificanceTailCalibration(alg::Func_SigCal)
        return new{typeof(alg)}(alg)
    end
end
function SignificanceTailCalibration(; alg::Func_SigCal)
    return SignificanceTailCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a significance rule in a slot that addresses the upper tail of the return distribution.

It is the counterpart of [`SignificanceTailCalibration`](@ref), and takes the same `alg` members. Every head slot in the library sits on a Range measure, so the head role never appears on a scalar measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignificanceHeadCalibration(;
        alg::Func_SigCal
    ) -> SignificanceHeadCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`Num_SigHeadCal`](@ref)
  - [`Func_SigCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct SignificanceHeadCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_sig])
    """
    alg
    function SignificanceHeadCalibration(alg::Func_SigCal)
        return new{typeof(alg)}(alg)
    end
end
function SignificanceHeadCalibration(; alg::Func_SigCal)
    return SignificanceHeadCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a deformation rule in a slot that addresses the lower tail of the return distribution.

It is the deformation family's counterpart of [`SignificanceTailCalibration`](@ref), and carries the same shape.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DeformationTailCalibration(;
        alg::Func_DefCal
    ) -> DeformationTailCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`Num_DefTailCal`](@ref)
  - [`Func_DefCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct DeformationTailCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_def])
    """
    alg
    function DeformationTailCalibration(alg::Func_DefCal)
        return new{typeof(alg)}(alg)
    end
end
function DeformationTailCalibration(; alg::Func_DefCal)
    return DeformationTailCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a deformation rule in a slot that addresses the upper tail of the return distribution.

It is the counterpart of [`DeformationTailCalibration`](@ref), and takes the same `alg` members.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DeformationHeadCalibration(;
        alg::Func_DefCal
    ) -> DeformationHeadCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`Num_DefHeadCal`](@ref)
  - [`Func_DefCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct DeformationHeadCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_def])
    """
    alg
    function DeformationHeadCalibration(alg::Func_DefCal)
        return new{typeof(alg)}(alg)
    end
end
function DeformationHeadCalibration(; alg::Func_DefCal)
    return DeformationHeadCalibration(alg)
end
"""
    const Num_SigTailCal = Union{<:SignificanceTailCalibration, <:Number}

Field bound for a lower-tail significance slot: the tail probability itself, or the role that computes it.

The union names one role and no other, so a head role placed in a tail slot fails the constructor's signature and is refused at construction. That is the whole of the role validation, and no guard method is written for it.

# Related

  - [`SignificanceTailCalibration`](@ref)
  - [`Num_SigHeadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_SigTailCal = Union{<:SignificanceTailCalibration, <:Number}
"""
    const Num_SigHeadCal = Union{<:SignificanceHeadCalibration, <:Number}

Field bound for an upper-tail significance slot: the tail probability itself, or the role that computes it.

# Related

  - [`SignificanceHeadCalibration`](@ref)
  - [`Num_SigTailCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_SigHeadCal = Union{<:SignificanceHeadCalibration, <:Number}
"""
    const Num_DefTailCal = Union{<:DeformationTailCalibration, <:Number}

Field bound for a lower-tail deformation slot: the deformation parameter itself, or the role that computes it.

# Related

  - [`DeformationTailCalibration`](@ref)
  - [`Num_DefHeadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_DefTailCal = Union{<:DeformationTailCalibration, <:Number}
"""
    const Num_DefHeadCal = Union{<:DeformationHeadCalibration, <:Number}

Field bound for an upper-tail deformation slot: the deformation parameter itself, or the role that computes it.

# Related

  - [`DeformationHeadCalibration`](@ref)
  - [`Num_DefTailCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_DefHeadCal = Union{<:DeformationHeadCalibration, <:Number}
"""
    resolve_calibration_slot(slot, key::Symbol, pr::AbstractPriorResult, w, slv = nothing)

Resolve one calibration slot against prior result `pr`, the effective observation weights `w` and the effective solver `slv`, and return a plain number.

A slot that holds a role type is unwrapped, and the rule in its `alg` field is **called** as `alg(key, pr, w, slv)`. So a callable rule and a plain function are the same thing here, and a rule never sees the role it was placed in. Anything else, a stated number above all, is returned unchanged.

A rule gets no portfolio. A prior result carries no portfolio weight vector, so no rule can measure a portfolio's own loss series. What it can measure is the series of each **column** of the sample, and [`bind_series`](@ref) tells it which series the slot owner prices. It does get the solver, on both of the routes that resolve a measure, so a rule may call [`ERM`](@ref) or [`RRM`](@ref). On the [`factory`](@ref) route [`@propagatable`](@ref) runs the `@cprop` selection before the resolution, so the solver is on the struct. On the `JuMP` route no selection runs, so [`set_risk_constraints!`](@ref) threads it into [`resolve_deferred_quantities`](@ref) and the owner settles it as `sel(x.slv, slv)`.

This is the parallel of [`resolve_slot`](@ref), and it is a second verb rather than a widening of the first for two reasons. `resolve_slot`'s body is `deferred_quantity(fit_deferred_quantity(dq, pr), key)`, a fit followed by an extraction, and a rule fits nothing. `resolve_slot` also carries neither `w` nor `slv`, which a rule needs. So the role types stay **out** of the [`DeferredQuantity`](@ref) union.

The caller computes `w` itself, as `sel(r.w, pr.w)`, and threads it with the measure's own `slv`. A parent that carries no observation weights of its own passes `pr.w`, and one that carries no solver leaves `slv` at its default.

**A [`TimeDependent`](@ref) reaches the host that holds the slot, and no further.** A schedule varies a *field of an estimator*, and it is consumed by [`update_time_dependent_fields`](@ref) before any prior is fitted. A rule is never standalone: it stands in a slot of a host, so the host is what a schedule swaps. Where the host is a [`JuMPOptimiser`](@ref) the four norm fields are themselves schedulable, and a schedule over them selects a rule per fold. Where the host is a risk measure the slot's own bound admits no schedule, and the caller varies the whole measure instead, through the schedulable risk-measure field of the optimiser. Both routes land in the same place, because the selection runs first and the rule then resolves against the prior of the period that was selected. A schedule *inside* a rule is therefore not a gap: it would name a fold the rule cannot see, and it would duplicate the channel the host already carries.

# Algorithm

 1. Return `slot` unchanged when it is not an [`AbstractCalibrationEstimator`](@ref). A stated number takes that arm.
 2. Read the rule out of the role's `alg` field.
 3. Call the rule as `alg(key, pr, w, slv)`, and return the number it gives. A callable struct and a plain function are the same thing here, so a rule never sees the role it was placed in.

# Arguments

  - `slot`: The slot's occupant: a number, or a role under [`AbstractCalibrationEstimator`](@ref).
  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver, or `nothing` when the measure carries none.

# Returns

  - `val::Number`: The calibrated quantity, or the stated value unchanged.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`calibration_slots`](@ref)
  - [`Func_SigCal`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_calibration_slot(slot, ::Symbol, ::AbstractPriorResult, ::Any,
                                  ::Any = nothing)
    return slot
end
function resolve_calibration_slot(r::AbstractCalibrationEstimator, key::Symbol,
                                  pr::AbstractPriorResult, w, slv = nothing)
    return r.alg(key, pr, w, slv)
end
"""
    calibration_slots(x)

Declare the slots of `x` that may hold a **Calibration Rule**, as a `NamedTuple` mapping each slot's name to its current value. The default is empty: a type with no calibration slot needs no method.

This is the parallel of [`deferred_slots`](@ref), and [`assert_calibrated_slots`](@ref) reads it. A type that names its slots here writes the resolution beside them, because a rule that reads a sibling slot must be resolved after that sibling and no derivation can know the order.

A slot that holds a child measure is declared here too, so a container names its children and each child names its own slots.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`assert_calibrated_slots`](@ref)
  - [`assert_declared_calibration_resolver`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`deferred_slots`](@ref)
"""
calibration_slots(::Any) = (;)
"""
    assert_calibrated_slots(x)

Refuse a **Calibration Role** that reached a value-level entry point, which has no prior result to resolve its rule against.

[`expected_risk`](@ref) takes either a prior result or a plain returns matrix. Given the prior it resolves the measure through [`factory`](@ref) first. Given the matrix it cannot: a rule reads the sample size and the moments the prior carries, and it reads the effective observation weights, none of which a bare returns matrix supplies. So it refuses instead, and names the slot, the role standing in it and the way out.

This is the shape [`assert_resolved_slots`](@ref) already uses on the Deferred-Quantity side, and the message names both types with `nameof` for the same reason: a printed type carries a module prefix wherever the name is not visible from `Main`, and the message must read the same in every process.

The slots come from [`calibration_slots`](@ref) and the check recurses into whatever they hold, so a container is covered by its children's declarations.

# Algorithm

 1. Walk the pairs that [`calibration_slots`](@ref) declares for `x`, giving each slot's name `key` and its occupant `slot`.
 2. Refuse an occupant that holds an [`AbstractCalibrationEstimator`](@ref).
 3. Recurse into the occupant, so a child measure's own slots are checked as well. A slot that holds a vector of children is walked element by element.
 4. Return `nothing` once the walk is spent.

# Validation

  - Throws an `ArgumentError` when a slot of `x`, or of any child the walk reaches, holds an [`AbstractCalibrationEstimator`](@ref). The message names the slot, the role standing in it and the two ways out.

# Returns

  - `nothing`.

# Related

  - [`calibration_slots`](@ref)
  - [`AbstractCalibrationEstimator`](@ref)
  - [`assert_declared_calibration_resolver`](@ref)
  - [`assert_resolved_slots`](@ref)
  - [`expected_risk`](@ref)
"""
function assert_calibrated_slots(x)
    for (key, slot) in pairs(calibration_slots(x))
        @argcheck(!isa(slot, AbstractCalibrationEstimator),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Calibration Role, a `$(nameof(typeof(slot)))`, and this entry point has no prior result to resolve its rule against. A rule reads the sample size, the moments and the effective observation weights, which a bare returns matrix does not carry. Pass the prior result itself — `expected_risk(r, w, pr, fees)` — or resolve the measure first with `factory(r, pr)`."))
        assert_calibrated_slots(slot)
    end
    return nothing
end
function assert_calibrated_slots(xs::AbstractArray{<:Union{<:AbstractEstimator,
                                                           <:AbstractAlgorithm}})
    for x in xs
        assert_calibrated_slots(x)
    end
    return nothing
end
"""
    assert_declared_calibration_resolver(x, slots::NamedTuple)
    assert_declared_calibration_resolver(x)

Refuse a type that declares a calibration slot and no way to resolve it.

`slots` is what the resolution produced. A **Calibration Role** that survives it names a type that declared the slot in [`calibration_slots`](@ref) and then wrote no resolution for it, so the role would reach the model builders and be multiplied as though it were a number. This is where the declaration and the resolver are paired.

The two channels check alike. [`assert_declared_slot_resolver`](@ref) is the Deferred-Quantity half, and the two-argument method here carries its shape: the resolver holds the slots it produced and hands them over. The one-argument method reads them off `x` instead, and is what a funnel takes, which holds a resolved value and not the slots that made it.

The walk is the one [`assert_calibrated_slots`](@ref) makes, and the message is the other half of the pair. That one names a caller who reached a value-level entry point, and this one names a slot the library itself left unresolved. So one message never has to serve two failures.

The calibration channel derives no recursion of its own, which is why the pairing is a check and not a derivation: a rule that reads a sibling slot must be resolved after that sibling, and no derivation can know the order. So each type writes its resolution beside its declaration, and this refuses the pair that does not meet.

# Algorithm

 1. Walk the pairs of `slots`, giving each slot's name `key` and its occupant `slot`.
 2. Refuse an occupant that holds an [`AbstractCalibrationEstimator`](@ref).
 3. Return `nothing` once the walk is spent.

The one-argument method reads `slots` from [`calibration_slots`](@ref), then recurses into each occupant, so a child measure's own declarations are paired as well. A slot that holds a vector of children is walked element by element.

# Arguments

  - `x`: The slot owner, whose type the message names.
  - `slots`: The slots the resolution produced, or the declaration itself for the one-argument method.

# Validation

  - Throws an `ArgumentError` when an entry of `slots`, or of any child the walk reaches, still holds an [`AbstractCalibrationEstimator`](@ref). The message names the type, the slot and the resolution to write beside the declaration.

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
        @argcheck(!isa(slot, AbstractCalibrationEstimator),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Calibration Role, a `$(nameof(typeof(slot)))`, after the resolution ran, so `$(nameof(typeof(x)))` declares the slot in `calibration_slots` and resolves it nowhere. The declaration and the resolution travel together, because a rule that reads a sibling slot must be resolved after that sibling. Resolve the slot beside the declaration with `resolve_calibration_slot(x.$key, :$key, pr, pr.w, slv)`, or drop it from `calibration_slots`."))
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
function assert_declared_calibration_resolver(xs::AbstractArray{<:Union{<:AbstractEstimator,
                                                                        <:AbstractAlgorithm}})
    for x in xs
        assert_declared_calibration_resolver(x)
    end
    return nothing
end
"""
    mirror_role(x)

Carry the occupant of a lower-tail slot across to its upper-tail counterpart, and keep the rule.

Every Range type defaults its head slot to whatever its tail slot holds, and this verb is what carries the occupant across. A number crosses unchanged, and a tail role crosses as the head role of the same family holding the same `alg`, so a rule stated on one end reaches both ends and no stated number moves. A caller who states the head slot themselves gets their own occupant, and the two ends then resolve independently.

The two role families are the whole domain of the second and third methods, because a head slot's bound admits nothing else. An ambiguity role names no end of the distribution, so it takes no method here, and the ambiguity slots of a Range type keep numbers of their own.

# Arguments

  - `x`: The lower-tail slot's occupant.

# Returns

  - `y`: The upper-tail slot's occupant.

# Related

  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
"""
function mirror_role(x::Number)
    return x
end
function mirror_role(r::SignificanceTailCalibration)
    return SignificanceHeadCalibration(; alg = r.alg)
end
function mirror_role(r::DeformationTailCalibration)
    return DeformationHeadCalibration(; alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Names the series a calibration rule reads, so that a rule reads the quantity its slot owner prices.

A rule gets no portfolio, which [`resolve_calibration_slot`](@ref) states, so it cannot form the loss series of a portfolio that does not exist until the solver returns. What it can form is the series of each **column** of `pr.X`. This family names which one: the column itself, or the drawdown series that column carries. [`calibration_series`](@ref) is the trait a slot owner answers, and [`bind_series`](@ref) carries the answer into the rule.

The marker states the **quantity** and not the estimator, so each rule reads it on its own terms. [`HillTailDecay`](@ref) forms the series and estimates its tail index. [`RadialTailDecay`](@ref) cannot form it, because a Mahalanobis distance carries no path, so it whitens the drawdown sample instead. The three radius rules read the per-asset dispersion of the series, which [`calibration_series_dispersion`](@ref) takes, and [`TailTermParity`](@ref) reads both terms of its ratio over it.

# Related

  - [`ReturnsSeries`](@ref)
  - [`AbstractDrawdownSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_series_vec`](@ref)
  - [`bind_series`](@ref)
"""
abstract type AbstractCalibrationSeries <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Names a drawdown series, the quantity a drawdown measure prices.

A drawdown is a path functional of one series, so it is formed per column and never per row. The two members differ in how the path compounds, and they are the two series [`absolute_drawdown_vec`](@ref) and [`relative_drawdown_vec`](@ref) build.

A drawdown series is non-positive, so it carries **one end**. A rule that answers per end therefore refuses a head key under this family, and the refusal is a statement about the quantity rather than about the sample.

**What a drawdown reading says depends on the record.** A drawdown is a running functional, so its law over a finite record is not the law of one step, and a rule that reads the shape of a series reads a different shape here. A sample whose drift is strong enough for the drawdown process to settle reads **heavier** than its own returns, which is the reading a stationary drawdown carries. A sample whose drift is weak reads the range of the path over the record instead, and that range is **thinner** in the tail than the step law that made it. Both are readings of the series the measure prices over the record the measure prices it on, so a rule states the sample rather than a claim about drawdowns in general.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`RelativeDrawdownSeries`](@ref)
  - [`series_end_sign`](@ref)
"""
abstract type AbstractDrawdownSeries <: AbstractCalibrationSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the returns themselves, the columns of `pr.X` unchanged.

It is the series every measure of the return distribution prices, and it is the default of every rule that carries a `series` field. A rule that never left this marker reads what it read before the family existed.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct ReturnsSeries <: AbstractCalibrationSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the absolute drawdown series of a column, which [`absolute_drawdown_vec`](@ref) builds.

The path is the cumulative sum of the column, and the series is the distance of that path below its own running peak. [`RelativisticDrawdownatRisk`](@ref) prices it, and [`DrawdownatRisk`](@ref) is the measure it takes its name from.

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

The path is the compounded value of the column, and the series is the fractional distance of that path below its own running peak. [`RelativeRelativisticDrawdownatRisk`](@ref) prices it, and [`RelativeDrawdownatRisk`](@ref) is the measure it takes its name from.

# Related

  - [`AbstractDrawdownSeries`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct RelativeDrawdownSeries <: AbstractDrawdownSeries end
"""
    calibration_series(x)

Declare the series that `x` prices, so that a rule in one of its calibration slots reads that quantity.

The default is [`ReturnsSeries`](@ref), so a type that prices the return distribution needs no method. The two relativistic drawdown measures write one, because the series they price is not the sample they carry.

This is a trait on the **slot owner** and not a field on the rule, for the reason [`bind_norm_order`](@ref) carries a norm order rather than reading one off the rule: the quantity belongs to the measure, and a rule cannot know which measure it reached. So a marker a caller states on a rule is overwritten wherever a measure resolves it, and it serves a caller who runs the rule by hand.

# Arguments

  - `x`: The slot owner, a risk measure above all.

# Returns

  - `series::AbstractCalibrationSeries`: The series the owner prices.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_slots`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
calibration_series(::Any) = ReturnsSeries()
"""
    calibration_series_vec(series::AbstractCalibrationSeries, x::VecNum)

Return the univariate series that `series` names, built from one column of the sample.

A drawdown marker builds a fresh vector, and [`ReturnsSeries`](@ref) returns the column itself. So the returns reading allocates nothing and reads the sample it was always given.

The verb takes one column rather than the whole matrix because a drawdown is a path functional: it accumulates down a column and says nothing across a row. A caller that needs the pool walks the columns and calls this once per column.

# Arguments

  - `series`: The series marker.
  - `x`: One column of the sample.

# Returns

  - `s::VecNum`: The series the marker names.

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

Return the sample that `series` names, one series per column of `X`.

A drawdown series reaches the array builder of its own convention along `dims = 1`, which is the observations dimension: [`absolute_drawdown_arr`](@ref) for [`AbsoluteDrawdownSeries`](@ref), and [`relative_drawdown_arr`](@ref) for [`RelativeDrawdownSeries`](@ref). Those builders hold the running-peak convention, so the matrix reading and the vector reading state one definition of a drawdown and cannot drift apart. The result holds the same shape as `X`: a drawdown series carries one entry per observation, so no observation is lost and the count of a tail is the count the caller formed on `X`.

[`ReturnsSeries`](@ref) returns `X` itself, so the returns reading allocates nothing. A rule that walks the columns one at a time should call [`calibration_series_vec`](@ref) instead, and allocate one column rather than a matrix.

# Arguments

  - `series`: The series marker.
  - `X`: Returns matrix, `T × N`.

# Returns

  - `Y::AbstractMatrix`: The sample the marker names, `T × N`.

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

Return the per-asset dispersion of the series that `series` names, one entry per column of the sample.

A [`ReturnsSeries`](@ref) reads it off `pr.sigma`, as the square root of that matrix's diagonal. That is the reading the radius family has always carried, so a prior that states a shrunk or a robust covariance matrix is read through it, and this verb leaves that untouched.

A drawdown series reads it off the **drawdown sample** instead, because a prior result states no drawdown moment. [`calibration_series_matrix`](@ref) builds the sample, and the dispersion is the sample dispersion of each of its columns. This is the reading [`radial_series_inputs`](@ref) takes a centre and a whitening factor by, and it parts from the returns reading for the same reason: `pr.sigma` is a moment of the returns, and no scaling of it states a moment of a drawdown.

A radius is a distance in the space of the scenarios the model prices, so this vector carries its units. [`ConcentrationRadius`](@ref) and [`DimensionalRateRadius`](@ref) average it into one scale, and [`DualNormRadius`](@ref) divides it by the square root of the effective sample size and takes a norm of the result.

# Arguments

  - `series`: The series the slot owner prices.
  - `pr`: Prior result the covariance matrix is read off on a returns series, and the sample on a drawdown series.

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
    effective_sample_size(pr::AbstractPriorResult, w::Option{<:ObsWeights})

Return the number of observations the record of `pr` carries, weighted by `w`.

This is the count a calibration rule divides by, so the four rules that read a record read one definition of it. Without weights it is the row count of `pr.X`. With weights it is Kish's effective sample size, which is the number of equally weighted observations that carries the information the weighted sample carries.

The weights are read through [`get_observation_weights`](@ref), so the whole of the [`ObsWeights`](@ref) union that a rule's own `w` field admits is served here. A [`DynamicAbstractWeights`](@ref) that states no method for a column of the sample raises [`ObservationWeightsError`](@ref), which names the two signatures to write, rather than a bare `MethodError` off `sum`.

# Mathematical definition

```math
T_{e} = \\begin{cases}
T & \\textrm{if } w \\textrm{ is } \\texttt{nothing}\\\\
\\dfrac{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}{\\sum\\limits_{t=1}^{T} w_{t}^{2}} & \\textrm{otherwise}
\\end{cases}\\,.
```

Where:

  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])

# Arguments

  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights, or `nothing`.

# Validation

  - `w` resolves through [`get_observation_weights`](@ref), which raises [`ObservationWeightsError`](@ref) for a [`DynamicAbstractWeights`](@ref) that states no method for a column of `pr.X`.

# Returns

  - `T_e::Number`: The effective sample size.

# Related

  - [`get_observation_weights`](@ref)
  - [`ObsWeights`](@ref)
  - [`ScenarioCount`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref)
"""
function effective_sample_size(pr::AbstractPriorResult, w::Option{<:ObsWeights})
    ws = get_observation_weights(w, view(pr.X, :, 1))
    return isnothing(ws) ? size(pr.X, 1) : sum(ws)^2 / sum(abs2, ws)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level from a count of observations, so that the tail keeps the same number of scenarios whatever the sample length becomes.

A stated `alpha` fixes the tail's probability. A fold half as long then leaves half as many observations in the tail, and the measure it feeds grows noisier as the sample shrinks. This rule fixes the count instead and returns `n / T`, so `ceil(alpha * T) == n` at every sample length the resolution meets.

`T` is the effective sample size when observation weights are stated, and the raw row count when they are not. Kish's effective sample size is the number of equally weighted observations that carries the same information as the weighted sample, so a weighted tail holds `n` observations in that sense rather than `n` rows. This is the only one of the three rules that reads the weights.

The rule carries no range check of its own. It returns the quantity of the slot it stands in, so the slot owner's constructor is the whole validation, and a count that produces a value outside the slot's range is refused there, at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ScenarioCount(;
        n::Number
    ) -> ScenarioCount

Keywords correspond to the struct's fields. `n` has no default, because a scenario count that suits every sample does not exist.

## Validation

  - `n > 0` and finite.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`RateSignificance`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
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
function ScenarioCount(; n::Number)
    return ScenarioCount(n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the significance level that leaves `alg.n` observations in the tail of the sample that `pr` carries.

# Mathematical definition

```math
\\alpha = \\frac{n}{T_{e}}\\,.
```

Where:

  - $(math_dict[:alpha_rm])
  - ``n``: Number of observations the tail is to hold.
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The count is the same for every key, so a tail slot and a head slot that carry one rule resolve to one number.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `alpha::Number`: The significance level.

# Related

  - [`ScenarioCount`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::ScenarioCount)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    T = effective_sample_size(pr, w)
    return alg.n / T
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level that shrinks with the square root of the sample length.

The tail probability is `c / sqrt(T)`, so the tail's expected count is `c * sqrt(T)`. It grows with the sample, but more slowly than the sample does, which is the rate at which a sample mean's own error falls. A longer sample therefore buys a further tail rather than only a fuller one, and [`ScenarioCount`](@ref) is the rule that buys neither.

The rule reads the raw row count, and not the effective sample size that [`ScenarioCount`](@ref) reads. The rate is a statement about the length of the record, whereas a scenario count is a statement about the observations the tail holds.

The rule carries no range check of its own, on the same terms as [`ScenarioCount`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RateSignificance(;
        c::Number = 1
    ) -> RateSignificance

Keywords correspond to the struct's fields. `c` defaults to `1`, which is the plain ``1/\\sqrt{T}`` rate.

## Validation

  - `c > 0` and finite.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`ScenarioCount`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
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

Compute the significance level at the square-root rate of the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
\\alpha &= \\frac{c}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_rm])
  - ``c``: Rate coefficient.
  - $(math_dict[:T])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The rate is the same for every key.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights. This rule reads the raw row count, so it ignores them.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `alpha::Number`: The significance level.

# Related

  - [`RateSignificance`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::RateSignificance)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    return alg.c / sqrt(size(pr.X, 1))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Kaniadakis deformation parameter that makes a relativistic measure spend a stated entropy budget.

[`RRM`](@ref) multiplies its dual variable by `kappa_log(inv(alpha * T), kappa)`, so that coefficient is the price the model pays for the deformation, and `target` states it directly. The rule returns the ``\\kappa`` that meets it. A stated `kappa` fixes the shape of the deformation and lets the price move with the sample; this rule fixes the price and lets the shape move.

The two quantities travel together. The rule reads its sibling `alpha`, which is why the `alpha` field exists: [`bind_alpha`](@ref) fills it with the number the slot owner resolved, and the owner's own resolution method resolves `alpha` first for that reason. A caller who runs the rule outside a measure states `alpha` instead.

The inversion is monotone, and it is solved by bisection over ``(0, 1)``. Writing ``l = \\ln(u)``, the coefficient is ``l \\sinh(\\kappa l) / (\\kappa l)``, and ``\\sinh(x)/x`` rises with ``|x|``, so the coefficient walks once from ``l`` at ``\\kappa \\to 0`` to ``\\sinh(l)`` at ``\\kappa = 1``. There is no elementary inverse of ``\\sinh(x)/x``, so the solve is a fixed sweep of 64 halvings rather than a formula.

The band the coefficient reaches moves with the sample, and a target outside it is refused at fold time. This is the one check any of the three rules carries, and it is not a range check on the quantity the rule returns — the slot owner's constructor keeps that job, as it does for the two significance rules. It states that the equation has a root at all: a target the band does not reach leaves the sweep at an end of the interval, where ``\\kappa`` is far too small or too large to be the answer to the question the caller asked and yet still inside the range the slot owner admits.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropyBudget(;
        target::Number,
        alpha::Option{<:Number} = nothing
    ) -> EntropyBudget

Keywords correspond to the struct's fields. `target` has no default, because the budget is the whole content of the rule. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in.

## Validation

  - `target` is finite. The band the target must lie in moves with the sample, so the rule checks the band when it runs, not here.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`bind_alpha`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`RRM`](@ref)
"""
@concrete struct EntropyBudget <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_target])
    """
    target
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    function EntropyBudget(target::Number, alpha::Option{<:Number})
        assert_nonempty_finite_val(target, :target)
        return new{typeof(target), typeof(alpha)}(target, alpha)
    end
end
function EntropyBudget(; target::Number, alpha::Option{<:Number} = nothing)
    return EntropyBudget(target, alpha)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deformation parameter whose Kaniadakis logarithm meets `alg.target` on the sample that `pr` carries.

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
  - ``u``: Argument of the Kaniadakis logarithm, the reciprocal of the tail's expected count.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

# Algorithm

 1. Read the sample length `T` off `pr.X`.
 2. Form the argument `u = inv(alg.alpha * T)`, and its plain logarithm `l = log(u)`.
 3. Form the band `(lo_b, hi_b)` as the ordered pair of `l` and `(u - inv(u)) / 2`, which are the values the Kaniadakis logarithm reaches at the two ends of ``\\kappa \\in (0,\\, 1)``. `# Validation` states the refusal this band carries.
 4. Normalise the target as `target = alg.target / l`. The normalised coefficient rises once from `1`, so one comparison carries both signs of `l` and the sweep needs no sign branch.
 5. Bracket the answer with `lo = 0` and `hi = 1`.
 6. Halve the bracket 64 times. Take the midpoint `kappa` each time, and raise `lo` to it when `kappa_log(u, kappa) / l` is below `target`, or lower `hi` to it otherwise. Sixty-four halvings take the bracket below the resolution of a `Float64`.
 7. Return the midpoint of the last bracket.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The budget is the same for every key.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights. This rule reads the raw row count, so it ignores them.
  - `slv`: Effective solver. This rule needs none, because the inversion is a scalar one.

# Validation

  - `alg.alpha` must not be `nothing`.
  - `alg.target` must lie strictly between ``\\ln(u)`` and ``\\sinh(\\ln(u))``, which is the band the coefficient reaches over ``\\kappa \\in (0,\\, 1)``.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`EntropyBudget`](@ref)
  - [`bind_alpha`](@ref)
  - [`kappa_log`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::EntropyBudget)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`EntropyBudget.alpha` is `nothing`, so the rule cannot form `inv(alpha * T)`. The sibling `alpha` travels to the rule through `bind_alpha`, which the slot owner calls after it resolves `alpha`. State `alpha` on the rule itself to run it outside a measure."))
    T = size(pr.X, 1)
    u = inv(alg.alpha * T)
    l = log(u)
    lo_b, hi_b = minmax(l, (u - inv(u)) / 2)
    @argcheck(lo_b < alg.target < hi_b,
              DomainError(alg.target,
                          "`EntropyBudget.target` must lie in ($lo_b, $hi_b), the band that `kappa_log(inv(alpha * T), kappa)` reaches over `kappa` in (0, 1) at `alpha = $(alg.alpha)` and `T = $T`. No deformation parameter meets a target outside it, so the rule has nothing to return. The band moves with the sample, so a target that suits one fold need not suit another."))
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

Estimate the tail index of the pool of standardised values of `X`, over the worst `k` order statistics of the pool.

`series` names the series each column carries, and [`calibration_series_vec`](@ref) builds it. Every such series is centred and divided by its own sample dispersion, and `s` names the end. Hill's estimator reads the `k` of the `T N` standardised values that lie furthest into that end. [`HillTailDecay`](@ref) states the reading and the assumptions the pool carries. This verb is the estimate alone.

The pool holds `T` values per column under every marker, because a drawdown series holds one entry per observation. So the count `k` the caller formed is the count this verb reads, whatever the series is.

The element type is bound by the signature, so the pool and the sum it feeds are concrete. A rule reads `pr.X` off an [`AbstractPriorResult`](@ref), whose field types no signature states, and this is the boundary that type crosses at.

# Algorithm

 1. Read the shape of `X` into `T` and `N`, and allocate the pool `pool` of `T * N` values.
 2. Walk the columns of `X`. Build each column's series with [`calibration_series_vec`](@ref), take its sample mean `mu` and its sample dispersion `sd`, and write `-s * (col[t] - mu) / sd` into `pool`.
 3. Partially sort `pool` so that its `k + 1` smallest values sit in order at the front, and read the last of them into `vkp1`. `partialsort!` places one index alone, and it makes no promise about the rest of the vector, so the range is what puts the `k` entries the estimate reads before `vkp1`. The sign of step 2 puts the end the caller prices in the **lower** tail of the pool.
 4. Return `k` over the sum of `log(pool[i] / vkp1)` across those `k` entries. Both terms of each ratio are negative, so the ratio is one of magnitudes and the sum is Hill's with no further sign.

# Arguments

  - `series`: The series each column carries.
  - `X`: Returns matrix, `T × N`.
  - `s`: Sign of the end the estimate reads: `1` reads the gain tail and `-1` reads the loss tail.
  - `k`: Number of order statistics the estimate reads. The caller states it, and the caller keeps the floor under it.

# Validation

  - The `k + 1`-th largest pooled value must be positive.

# Returns

  - `a::Number`: The Hill estimate of the tail index of the pool.

# Related

  - [`HillTailDecay`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_vec`](@ref)
"""
function hill_tail_index(series::AbstractCalibrationSeries, X::AbstractMatrix{E},
                         s::Integer, k::Integer) where {E <: Number}
    T, N = size(X)
    pool = Vector{float(E)}(undef, T * N)
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

Return the sign of the end that `key` prices on `series`, and refuse an end the series does not have.

`:kappa_b` is the only head key, so on a returns series every other key prices the loss end. A drawdown series is non-positive and carries one end alone, so the head key names nothing on it and is refused. No drawdown Range measure ships, so the refusal is reached by a caller who runs a rule by hand.

# Arguments

  - `series`: The series the estimate is taken over.
  - `key`: Name of the slot that is being resolved.

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

Computes the Kaniadakis deformation parameter whose tail decays at the rate the sample's own tail decays at.

The inverse of [`kappa_log`](@ref) is the ``\\kappa``-exponential ``\\exp_{\\kappa}(x) = \\left(\\kappa x + \\sqrt{1 + \\kappa^{2} x^{2}}\\right)^{1/\\kappa}``, which goes to ``(2 \\kappa x)^{1/\\kappa}`` for large ``x``. A ``\\kappa``-deformed exponential tail is therefore a power law of index ``1/\\kappa``, and ``\\kappa`` is a reciprocal tail index. So the rule estimates the sample's tail index ``\\hat{a}`` and returns ``1/\\hat{a}``, which makes the deformation decay at the rate the sample decays at. The band the slot admits, ``(0,\\, 1)``, reads as ``\\hat{a} > 1``, and that is the condition for a finite mean. The refusal is built into the reading, on the same terms as the band of [`EntropyBudget`](@ref).

The estimator is Hill's, over the worst `k` order statistics of a **pool**. Every column of `pr.X` is centred and divided by its own sample dispersion, the standardised values are signed to the end the slot prices, and the `T * N` of them are pooled. **The rule reads `pr.X` alone, and never `pr.sigma`.** A column's dispersion comes from that column, which is the line that separates this rule from [`RadialTailDecay`](@ref). The two answer two questions: this rule asks how far one end of one column moves, and its sibling asks how far the whole cross-section moves. So this rule keeps the sign of the end and answers per end, and its sibling reads a distance and answers one number for both.

The pool carries two assumptions, and both are stated rather than hidden. The columns share one tail index after standardisation, which is what lets one estimate stand for the whole sample. The columns are also cross-correlated, so the pool holds far fewer than `T * N` independent points and the estimator's spread is wider than the nominal one of `k` points. The median of `N` per-asset estimates drops both assumptions, and pays for it in noise: at `alpha = 0.05` and `T = 250` a column leaves 12 tail points, and a Hill estimate on 12 points moves from fold to fold for no reason in the data.

`alpha` fixes the depth of the reading as well as the count. A Hill estimate reads the tail at the depth `alpha` names, and a sample whose tail approaches its power law slowly is read with a bias that falls as the depth rises: a Student-t sample read at a five per cent depth returns a ``\\kappa`` above the reciprocal of its degrees of freedom, and the same sample read at one per cent returns most of that distance. The spread of the estimate rises as the count falls, and `kmin` is the floor under that trade.

`key` says which end the slot prices, and **the answer is not the same for every key**. `:kappa` and `:kappa_a` read the loss tail, and `:kappa_b` reads the gain tail. This is the opposite of [`EntropyBudget`](@ref), whose budget is a price the model pays and is therefore one number for both ends. A tail index is a statement about a tail, and a skewed sample has two different ones, which is the whole point of the rule on a Range measure.

**`series` says which quantity the pool holds, and the slot owner states it.** A measure of the return distribution pools the columns of `pr.X`. A drawdown measure pools the per-column drawdown series of `pr.X` instead, because that is the quantity it prices, and [`bind_series`](@ref) puts the marker there at the resolution site. Nothing else in the reading moves: the same standardisation, the same count and the same estimator run over the drawdown sample. The estimate the pool then carries is the index of the drawdown series rather than of the returns, and the two are different numbers. [`AbstractDrawdownSeries`](@ref) states which way they part, and it is the record that decides.

**The per-column reading stands for the portfolio's own drawdown**, under the assumption the pool already states. The rule forms no portfolio, so it reads the drawdown series of each column and pools them, exactly as it pools the columns themselves on a returns series. The columns share one tail index after standardisation, and one map carries a column to its drawdown series, so the reading the pool gives is the reading the portfolio's own drawdown series would give. A portfolio drawdown is shallower than the average asset drawdown, and that is a statement about the **scale**, which a tail index does not read.

The count is the same count. A drawdown series holds one entry per observation, so the pool holds `T N` entries under every marker and `k = ceil(alpha * T * N)` is unchanged. No observation is dropped by the change of series.

`alpha` reaches the right series as well. A drawdown measure resolves the significance level of its own drawdown series, and the depth the rule reads is that same level under a drawdown marker.

The pool of a drawdown series carries **more dependence** than the pool of a returns series. A drawdown series is a running functional, so consecutive entries move together and one deep episode fills many of the `k` order statistics the estimate reads. The `T N` entries therefore hold far fewer independent tail points than the same count of returns, and the spread of the estimate is wider than the count states. `kmin` is stated in entries and not in independent points, so the same floor buys less here. Raise it for a drawdown owner.

The rule carries no range check on the parameter it returns. The slot owner's constructor keeps that job, as it does for every calibration rule. Its checks are statements that the estimate exists at all.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HillTailDecay(;
        kmin::Integer = 30,
        alpha::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> HillTailDecay

Keywords correspond to the struct's fields. `kmin` defaults to `30`, which is the floor under the count of order statistics the estimate reads. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in. `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `kmin > 0`.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`EntropyBudget`](@ref)
  - [`hill_tail_index`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`series_end_sign`](@ref)
"""
@concrete struct HillTailDecay <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_kmin])
    """
    kmin
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    """
    $(field_dict[:cal_series])
    """
    series
    function HillTailDecay(kmin::Integer, alpha::Option{<:Number},
                           series::AbstractCalibrationSeries)
        assert_gt0(kmin, :kmin)
        return new{typeof(kmin), typeof(alpha), typeof(series)}(kmin, alpha, series)
    end
end
function HillTailDecay(; kmin::Integer = 30, alpha::Option{<:Number} = nothing,
                       series::AbstractCalibrationSeries = ReturnsSeries())
    return HillTailDecay(kmin, alpha, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deformation parameter whose reciprocal is the Hill tail index of the series that `alg.series` names, over the sample that `pr` carries.

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
  - $(math_dict[:alpha_rm]) It is the probability of the end this slot prices, so a head slot reads its own ``\\beta``.
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:r_tj])
  - ``\\boldsymbol{r}_{j}``: Column ``j`` of the sample.
  - ``\\mathcal{S}``: The series `alg.series` names, built from one column. It is the identity on a [`ReturnsSeries`](@ref), and a drawdown series on the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``y_{tj}``: Entry ``t`` of the series of column ``j``.
  - ``\\hat{\\mu}_{j}``: Sample mean of the series of column ``j``, and not of the column under a drawdown marker.
  - ``\\hat{\\sigma}_{j}``: Sample dispersion of the series of column ``j``, read off that series alone and never off ``\\hat{\\mathbf{\\Sigma}}``.
  - ``s``: Sign of the end the slot prices.
  - ``u_{tj}``: Standardised value of asset ``j`` at time ``t``, signed so that the end the slot prices is the upper tail of the pool.
  - ``u_{(i)}``: ``i``-th largest of the ``T N`` pooled values, so that ``u_{(1)} \\geq \\ldots \\geq u_{(k+1)}``.
  - ``k``: Number of order statistics the estimate reads.
  - ``\\hat{a}``: Hill estimate of the tail index of the pool.

# Algorithm

 1. Read the returns matrix off `pr` into `X`, and its element count into `np`.
 2. Take the sign `s` from `key` and `alg.series` with [`series_end_sign`](@ref). A drawdown series carries one end, so it refuses the head key rather than signing it.
 3. Form the count `k = ceil(Int, alg.alpha * np)`, the number of order statistics the estimate reads.
 4. Estimate the tail index of the pool with [`hill_tail_index`](@ref), giving `a`. The pool holds the series `alg.series` names, one entry per observation per column under every marker.
 5. Return `inv(a)`, which is the deformation parameter.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there, and its `series` field names the quantity, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. It names the end, so `:kappa` and `:kappa_a` read the loss tail and `:kappa_b` reads the gain tail. A skewed sample therefore resolves the two ends of a Range measure to two different numbers.
  - `pr`: Prior result the returns matrix is read off.
  - `w`: Effective observation weights. A tail index is a statement about the shape of a series rather than about the count of observations behind it, so this rule ignores them.
  - `slv`: Effective solver. This rule needs none, because the estimate is a closed form.

# Validation

  - `alg.alpha` must not be `nothing`.
  - `key` must name an end the series has, which [`series_end_sign`](@ref) checks.
  - `k` must be at least `alg.kmin`.
  - The pool must hold at least `k + 1` values.
  - ``u_{(k+1)}`` must be positive, which [`hill_tail_index`](@ref) checks.
  - ``\\hat{a}`` must be greater than one, which is the band ``\\kappa \\in (0,\\, 1)`` read as a tail index.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`HillTailDecay`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`EntropyBudget`](@ref)
  - [`hill_tail_index`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`series_end_sign`](@ref)
"""
function (alg::HillTailDecay)(key::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`HillTailDecay.alpha` is `nothing`, so the rule cannot form the count `k = ceil(alpha * T * N)`. The probability of the end travels to the rule through `bind_alpha`, which the slot owner calls after it resolves that end's own probability. State `alpha` on the rule itself to run it outside a measure."))
    X = pr.X
    # The sign puts the end the slot prices in the LOWER tail of the pool, and one estimator
    # then serves both ends. The series decides which ends there are to price.
    s = series_end_sign(alg.series, key)
    np = prod(size(X))
    k = ceil(Int, alg.alpha * np)
    @argcheck(k >= alg.kmin,
              DomainError(k,
                          "`HillTailDecay` reads the worst `k = ceil(alpha * T * N) = $k` of the $np pooled standardised values, and `HillTailDecay.kmin` puts the floor at $(alg.kmin). A Hill estimate over fewer order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. Lengthen the sample, widen `alpha`, or lower `kmin` and take the noise."))
    @argcheck(k + 1 <= np,
              DomainError(k,
                          "`HillTailDecay` needs $(k + 1) pooled values to form the estimate, and the pool of `T * N` holds $np. The count is `k = ceil(alpha * T * N)` at `alpha = $(alg.alpha)`, so only a probability that takes the whole sample reaches this. Lower `alpha`."))
    a = hill_tail_index(alg.series, X, s, k)
    @argcheck(a > 1,
              DomainError(a,
                          "`HillTailDecay` estimated a tail index of $a on the pool of the $(nameof(typeof(alg.series))), so `kappa = 1 / a` is $(inv(a)) and lies outside the (0, 1) the slot admits. An index of one or less is a tail with no finite mean, so no admissible deformation parameter reads it. The series is heavier-tailed than the measure can price."))
    return inv(a)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the square factor of the covariance matrix that `pr` states, so that the factor's transpose whitens a centred observation.

The factor ``\\mathbf{U}`` satisfies ``\\mathbf{U}^{\\top} \\mathbf{U} = \\hat{\\mathbf{\\Sigma}}``, which is the convention a prior result's `chol` carries and the one the variance constraint layer reads. **`chol` takes precedence over `sigma`**, which is the rule the `chol` field states, so a prior that carries a factor is whitened by that factor and never by a factorisation of `sigma`. `chol` is optional, so the common case is a `sigma` this verb factorises itself.

`chol` is checked against the length of `mu` alone, so it can be ``K \\times N`` rather than square. A rectangular factor has no inverse. A tall factor states a covariance matrix all the same, because ``\\mathbf{U}^{\\top} \\mathbf{U}`` is ``N \\times N`` at any ``K``, and the ``\\mathbf{R}`` factor of its QR factorisation is the square factor of that same matrix. A wide factor states a singular covariance matrix, and no whitening exists for it.

[`RadialTailDecay`](@ref) is the caller. The verb is separate because the precedence rule and its two refusals are a statement about a prior result rather than about a tail index.

# Algorithm

 1. Read `pr.chol` into `F`.
 2. When `F` is `nothing`, factorise `pr.sigma` with a Cholesky factorisation that reports rather than raises, and return its upper factor.
 3. Otherwise refuse a wide `F`, and take `F` itself when it is square and the `R` factor of its QR factorisation when it is tall.
 4. Refuse a zero on the diagonal of that factor, which is the rank of a triangular factor, and return the factor.

# Arguments

  - `pr`: Prior result the covariance matrix is read off.

# Validation

  - `pr.sigma` must be positive definite, when `pr.chol` is `nothing`.
  - `pr.chol` must have at least as many rows as columns.
  - The diagonal of the square factor must carry no zero.

# Returns

  - `U::AbstractMatrix`: Square factor of the covariance matrix, upper triangular whenever the prior's own factor is.

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
    U = size(F, 1) == size(F, 2) ? F : LinearAlgebra.UpperTriangular(LinearAlgebra.qr(F).R)
    j = findfirst(iszero, LinearAlgebra.diag(U))
    @argcheck(isnothing(j),
              DomainError(j,
                          "Entry $j of the diagonal of the square factor of `pr.chol` is zero, so the covariance matrix it states is singular and no whitening of the sample exists. The diagonal is the rank of a triangular factor, which is what `pr.chol` carries."))
    return U
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Estimate the tail index of the radial series of `X`, over the largest `k` order statistics of that series.

Every row is centred on `mu` and whitened by `U`, and the norm of the whitened row is one entry of the radial series. Hill's estimator then reads the `k` entries that lie furthest out. [`RadialTailDecay`](@ref) states the reading the series carries. This verb is the estimate alone.

The element type is bound by the signature, so the series and the sum it feeds are concrete. A rule reads `pr.X` off an [`AbstractPriorResult`](@ref), whose field types no signature states, and this is the boundary that type crosses at.

# Algorithm

 1. Solve `transpose(U) \\ transpose(X .- transpose(mu))`, giving the whitened observations as the columns of `Z`.
 2. Walk the columns of `Z`, and write the negated Euclidean norm of each into the series `d`.
 3. Partially sort `d` so that its `k + 1` smallest values sit in order at the front, and read the last of them into `vkp1`. `partialsort!` places one index alone, and it makes no promise about the rest of the vector, so the range is what puts the `k` entries the estimate reads before `vkp1`. The negation of step 2 puts the largest distances at the front.
 4. Return `k` over the sum of `log(d[i] / vkp1)` across those `k` entries. Both terms of each ratio are negative, so the ratio is one of magnitudes and the sum is Hill's with no further sign.

# Arguments

  - `X`: Returns matrix, `T × N`.
  - `mu`: Expected returns vector, `N × 1`. The rows are centred on it.
  - `U`: Square factor of the covariance matrix, which [`whitening_factor`](@ref) returns.
  - `k`: Number of order statistics the estimate reads. The caller states it, and the caller keeps the floor under it.

# Validation

  - The `k + 1`-th largest radial distance must be positive.

# Returns

  - `a::Number`: The Hill estimate of the tail index of the radial series.

# Related

  - [`RadialTailDecay`](@ref)
  - [`whitening_factor`](@ref)
"""
function radial_tail_index(X::AbstractMatrix{E}, mu::AbstractVector, U::AbstractMatrix,
                           k::Integer) where {E <: Number}
    Z = transpose(U) \ transpose(X .- transpose(mu))
    d = Vector{float(E)}(undef, size(X, 1))
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

Computes the Kaniadakis deformation parameter whose tail decays at the rate the sample's radial series decays at.

The inverse of [`kappa_log`](@ref) is the ``\\kappa``-exponential ``\\exp_{\\kappa}(x) = \\left(\\kappa x + \\sqrt{1 + \\kappa^{2} x^{2}}\\right)^{1/\\kappa}``, which goes to ``(2 \\kappa x)^{1/\\kappa}`` for large ``x``. A ``\\kappa``-deformed exponential tail is therefore a power law of index ``1/\\kappa``, and ``\\kappa`` is a reciprocal tail index. So the rule estimates a tail index ``\\hat{a}`` and returns ``1/\\hat{a}``, which makes the deformation decay at the rate the sample decays at. The band the slot admits, ``(0,\\, 1)``, reads as ``\\hat{a} > 1``, and that is the condition for a finite mean. The refusal is built into the reading, on the same terms as the band of [`EntropyBudget`](@ref).

**The series the rule reads is radial.** A covariance matrix is a scale and ``\\kappa`` is a shape, so a rule that turns the one into the other needs a second quantity to fix the units, and the sample is the only one a prior result carries. The covariance matrix is what turns the ``T \\times N`` sample into **one** univariate series: [`whitening_factor`](@ref) gives the factor of ``\\hat{\\mathbf{\\Sigma}}``, each row is centred and whitened, and the Euclidean norm of the whitened row is that observation's Mahalanobis distance. Under an elliptical scale mixture that series carries the mixture's tail index, and the whole cross-section stands behind every entry of it. A per-column standardisation drops every off-diagonal term, so [`HillTailDecay`](@ref) cannot buy that reading and this rule cannot buy the per-column one.

**The rule is direction-blind, and that is a statement.** A distance has no sign, so one ``\\kappa`` answers both ends of a Range measure and `key` is ignored. [`HillTailDecay`](@ref) answers per end, and the difference is the whole of what the two rules say: a radial reading asks how far the cross-section moves, and a per-column reading asks how far one end of one column moves. [`mirror_role`](@ref) is therefore trivially correct for this rule, because a head role holding it returns the number a tail role holding it returns.

The series holds `T` entries where the pool of [`HillTailDecay`](@ref) holds `T N`, so this rule reads **fewer** tail points from the same sample: `alpha = 0.05` at `T = 250` leaves 12 of them. `kmin` is the floor under that count, and it is stated in the same units as its sibling's, so the floor binds harder here. A count below it is refused rather than estimated. The points the series does hold are one per observation rather than `N` per observation, so they carry none of the cross-correlation the pool carries.

**`series` says which sample the rows are read off, and the slot owner states it.** A measure of the return distribution whitens the rows of `pr.X`. A drawdown measure whitens the rows of the **drawdown sample**: [`calibration_series_matrix`](@ref) turns each column into the drawdown series it carries, and the rows of that sample are whitened and normed on the same terms. So the radial reading itself is unchanged, and only the sample it reads moves. The distance is then the depth of the whole cross-section in drawdown at one date, where on a returns series it is the distance the whole cross-section moved.

**A drawdown sample carries its own moments, and the prior states none of them.** `pr.mu` and `pr.sigma` are the moments of the returns, and no scaling of them states the moments of a drawdown. So under a drawdown marker the centre is the column means of the drawdown sample and the factor is the Cholesky factor of its covariance matrix, both taken off that sample. [`radial_series_inputs`](@ref) is where the two readings part, and the precedence of `pr.chol` over `pr.sigma` therefore governs the returns reading alone.

A third reading of ``\\kappa`` exists and this rule does not take it. The excess kurtosis ``g`` of a Student-t gives ``\\nu = 4 + 6/g``, and ``\\kappa = 1/\\nu`` follows. It reads the whole sample rather than the tail, so it is steady where a Hill estimate is noisy, and it pays for that with an assumption about the shape of the whole distribution rather than of its tail.

The rule carries no range check on the parameter it returns. The slot owner's constructor keeps that job, as it does for every calibration rule. Its checks are statements that the estimate exists at all. A near-singular covariance matrix is not among them: the whitening then reads the sample's smallest eigen-direction, the radial series follows it, and that is the covariance matrix speaking rather than a defect.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RadialTailDecay(;
        kmin::Integer = 30,
        alpha::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> RadialTailDecay

Keywords correspond to the struct's fields. `kmin` defaults to `30`, which is the floor under the count of order statistics the estimate reads. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in. `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `kmin > 0`.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`radial_tail_index`](@ref)
  - [`whitening_factor`](@ref)
"""
@concrete struct RadialTailDecay <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_kmin_rad])
    """
    kmin
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    """
    $(field_dict[:cal_series_rad])
    """
    series
    function RadialTailDecay(kmin::Integer, alpha::Option{<:Number},
                             series::AbstractCalibrationSeries)
        assert_gt0(kmin, :kmin)
        return new{typeof(kmin), typeof(alpha), typeof(series)}(kmin, alpha, series)
    end
end
function RadialTailDecay(; kmin::Integer = 30, alpha::Option{<:Number} = nothing,
                         series::AbstractCalibrationSeries = ReturnsSeries())
    return RadialTailDecay(kmin, alpha, series)
end
"""
    radial_series_inputs(series::AbstractCalibrationSeries, pr::AbstractPriorResult)

Return the sample, the centre and the whitening factor that the radial series of `series` is built from.

A [`ReturnsSeries`](@ref) reads the three off the prior result: `pr.X`, `pr.mu` and the factor [`whitening_factor`](@ref) takes off `pr.chol` or `pr.sigma`. That is the reading [`RadialTailDecay`](@ref) has always carried, and this verb leaves it untouched.

A drawdown series reads the three off the **drawdown sample** instead, because a prior result states no drawdown moment. [`calibration_series_matrix`](@ref) builds the sample, the centre is the column means of that sample, and the factor is the Cholesky factor of its covariance matrix. So the whole reading moves to the quantity the measure prices, and `pr.mu` and `pr.sigma` reach nothing: they are the moments of the returns, and no scaling of them states the moments of a drawdown.

# Arguments

  - `series`: The series the slot owner prices.
  - `pr`: Prior result the sample and, on a returns series, the moments are read off.

# Validation

  - The covariance matrix of the drawdown sample must be positive definite. A column with no movement, and two columns with one path between them, are what reach this.

# Returns

  - `Y::AbstractMatrix`: The sample the marker names, `T × N`.
  - `mu::AbstractVector`: The centre of that sample, `N × 1`.
  - `U::AbstractMatrix`: Square factor of its covariance matrix.

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

Compute the deformation parameter whose reciprocal is the Hill tail index of the radial series of the sample that `alg.series` names.

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
  - $(math_dict[:alpha_rm]) It is the probability of the end the slot prices, and the count it fixes is the same for both ends because the series has no sign.
  - $(math_dict[:T])
  - $(math_dict[:x_t_obs])
  - ``\\mathcal{S}``: The sample `alg.series` names, built one column at a time. It is the identity on a [`ReturnsSeries`](@ref), and a drawdown series on the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``\\boldsymbol{y}_{t}``: Row ``t`` of that sample.
  - ``\\hat{\\boldsymbol{\\mu}}_{y}``: Centre of that sample. It is `pr.mu` on a returns series, and the column means of the drawdown sample on a drawdown series.
  - ``\\hat{\\mathbf{\\Sigma}}_{y}``: Covariance matrix of that sample. It is `pr.sigma` on a returns series, and the covariance matrix of the drawdown sample on a drawdown series.
  - ``\\mathbf{U}``: Square factor of ``\\hat{\\mathbf{\\Sigma}}_{y}``, which is `pr.chol` when the prior carries one and the slot prices the returns.
  - ``\\boldsymbol{z}_{t}``: Whitened observation ``t``.
  - ``d_{t}``: Mahalanobis distance of observation ``t``, the ``t``-th entry of the radial series.
  - ``d_{(i)}``: ``i``-th largest entry of the radial series, so that ``d_{(1)} \\geq \\ldots \\geq d_{(k+1)}``.
  - ``k``: Number of order statistics the estimate reads.
  - ``\\hat{a}``: Hill estimate of the tail index of the radial series.

# Algorithm

 1. Read the returns matrix off `pr` into `X`, and its row count into `T`.
 2. Form the count `k = ceil(Int, alg.alpha * T)`, the number of order statistics the estimate reads. A drawdown series holds one entry per observation, so the count is the same count under every marker.
 3. Take the sample, its centre and its whitening factor with [`radial_series_inputs`](@ref).
 4. Estimate the tail index of the radial series with [`radial_tail_index`](@ref), giving `a`.
 5. Return `inv(a)`, which is the deformation parameter.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there, and its `series` field names the quantity, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The radial series has no sign, so this rule reads no end from it and returns one number for every key.
  - `pr`: Prior result the returns matrix, the expected returns vector and the covariance matrix are read off.
  - `w`: Effective observation weights. A tail index is a statement about the shape of a series rather than about the count of observations behind it, so this rule ignores them.
  - `slv`: Effective solver. This rule needs none, because the estimate is a closed form.

# Validation

  - `alg.alpha` must not be `nothing`.
  - `k` must be at least `alg.kmin`.
  - The series must hold at least `k + 1` entries.
  - The sample must state a whitening, which [`radial_series_inputs`](@ref) checks on both readings.
  - ``d_{(k+1)}`` must be positive, which [`radial_tail_index`](@ref) checks.
  - ``\\hat{a}`` must be greater than one, which is the band ``\\kappa \\in (0,\\, 1)`` read as a tail index.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`RadialTailDecay`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`radial_tail_index`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`whitening_factor`](@ref)
"""
function (alg::RadialTailDecay)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`RadialTailDecay.alpha` is `nothing`, so the rule cannot form the count `k = ceil(alpha * T)`. The probability of the end travels to the rule through `bind_alpha`, which the slot owner calls after it resolves that end's own probability. State `alpha` on the rule itself to run it outside a measure."))
    X = pr.X
    T = size(X, 1)
    k = ceil(Int, alg.alpha * T)
    @argcheck(k >= alg.kmin,
              DomainError(k,
                          "`RadialTailDecay` reads the largest `k = ceil(alpha * T) = $k` of the $T radial distances, and `RadialTailDecay.kmin` puts the floor at $(alg.kmin). A Hill estimate over fewer order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. The radial series holds one entry per observation where the pool of `HillTailDecay` holds `N`, so the same floor binds harder here. Lengthen the sample, widen `alpha`, or lower `kmin` and take the noise."))
    @argcheck(k + 1 <= T,
              DomainError(k,
                          "`RadialTailDecay` needs $(k + 1) radial distances to form the estimate, and the series holds one per observation, which is $T. The count is `k = ceil(alpha * T)` at `alpha = $(alg.alpha)`, so only a probability that takes the whole sample reaches this. Lower `alpha`."))
    Y, mu, U = radial_series_inputs(alg.series, pr)
    a = radial_tail_index(Y, mu, U, k)
    @argcheck(a > 1,
              DomainError(a,
                          "`RadialTailDecay` estimated a tail index of $a on the radial series of the $(nameof(typeof(alg.series))), so `kappa = 1 / a` is $(inv(a)) and lies outside the (0, 1) the slot admits. An index of one or less is a tail with no finite mean, so no admissible deformation parameter reads it. The series is heavier-tailed than the measure can price."))
    return inv(a)
end
"""
    bind_alpha(slot, alpha::Number)

Hand a resolved `alpha` to the rule that reads it, and return the slot's occupant with the number in place.

Two pairs **travel** through this verb. `alpha` and `kappa`: [`EntropyBudget`](@ref), [`HillTailDecay`](@ref) and [`RadialTailDecay`](@ref) each read the significance level of a sibling slot. `alpha` and `l`: [`TailTermParity`](@ref) prices a tail term at the measure's own significance level. [`resolve_calibration_slot`](@ref) carries a `Symbol` and no number, so the number travels through the rule itself. The slot owner's own resolution method resolves `alpha` first, calls this verb on the slot that reads it, and resolves the result:

```julia
alpha = resolve_calibration_slot(x.alpha, :alpha, pr, w, slv)
kappa = resolve_calibration_slot(bind_alpha(x.kappa, alpha), :kappa, pr, w, slv)
```

The default is the identity, so a stated number, a plain function and a rule that reads no sibling all pass through untouched. A travelling role is rebuilt around the bound rule, which is what lets the verb take the slot rather than the rule the caller has to unwrap first. The significance family and the radius family need no method, because no rule of either reads a sibling and the identity is already the right answer for both.

# Arguments

  - `slot`: The slot's occupant: a number, a travelling role, or a rule.
  - `alpha`: The sibling slot's resolved significance level.

# Returns

  - `bound`: The occupant, with `alpha` in place wherever a rule reads it.

# Related

  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`TailTermParity`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function bind_alpha(slot, ::Number)
    return slot
end
function bind_alpha(r::DeformationTailCalibration, alpha::Number)
    return DeformationTailCalibration(; alg = bind_alpha(r.alg, alpha))
end
function bind_alpha(r::DeformationHeadCalibration, alpha::Number)
    return DeformationHeadCalibration(; alg = bind_alpha(r.alg, alpha))
end
function bind_alpha(alg::EntropyBudget, alpha::Number)
    return EntropyBudget(; target = alg.target, alpha = alpha)
end
function bind_alpha(alg::HillTailDecay, alpha::Number)
    return HillTailDecay(; kmin = alg.kmin, alpha = alpha, series = alg.series)
end
function bind_alpha(alg::RadialTailDecay, alpha::Number)
    return RadialTailDecay(; kmin = alg.kmin, alpha = alpha, series = alg.series)
end
"""
    bind_series(slot, series::AbstractCalibrationSeries)

Hand the series a slot owner prices to the rule that reads it, and return the slot's occupant with the marker in place.

A rule gets a prior result and no portfolio, so the quantity it can read is the sample the prior carries. Which series of that sample is the right one is a property of the **owner**: [`RelativisticValueatRisk`](@ref) prices the returns and [`RelativisticDrawdownatRisk`](@ref) prices a drawdown series of them. [`resolve_calibration_slot`](@ref) carries a `Symbol` and no marker, and the key `:kappa` serves both owners, so the marker travels through the rule itself. This is the shape [`bind_alpha`](@ref) uses to carry a significance level, and the shape [`bind_norm_order`](@ref) uses to carry a norm order.

The owner's series **wins**, on the terms [`bind_norm_order`](@ref) states. A rule that already carries a marker has it replaced, because the quantity belongs to the measure and a rule cannot know which measure it reached. So a stated `series` serves a caller who runs the rule by hand, and nothing else.

The default is the identity, so a stated number, a travelling role, a plain function and a rule that reads no series all pass through untouched. [`EntropyBudget`](@ref) needs no method: it reads the sample length and the sibling `alpha`, and neither moves with the series. The significance and norm-ceiling families need none either. A significance level is a probability and carries no units, and a norm ceiling is a bound on the weight vector rather than on the sample.

**The radius and tail-weight families do need one, and it is the units that say so.** A radius is a distance in the space of the scenarios the model prices, and a tail weight is the exchange rate between two terms of a loss, so both are read off a scale of that quantity. Under [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) the quantity is a per-asset drawdown, which is what the transport cost of its own programme is measured against.

The slot owner's own resolution method calls this beside [`bind_alpha`](@ref):

```julia
alpha = resolve_calibration_slot(x.alpha, :alpha, pr, w, slv)
kappa = resolve_calibration_slot(bind_series(bind_alpha(x.kappa, alpha),
                                             calibration_series(x)), :kappa, pr, w, slv)
```

# Arguments

  - `slot`: The slot's occupant: a number, a travelling role, or a rule.
  - `series`: The series the slot owner prices, which [`calibration_series`](@ref) states.

# Returns

  - `bound`: The occupant, with `series` in place wherever a rule reads one.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_norm_order`](@ref)
  - [`HillTailDecay`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref)
  - [`TailTermParity`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function bind_series(slot, ::AbstractCalibrationSeries)
    return slot
end
function bind_series(r::DeformationTailCalibration, series::AbstractCalibrationSeries)
    return DeformationTailCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(r::DeformationHeadCalibration, series::AbstractCalibrationSeries)
    return DeformationHeadCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(alg::HillTailDecay, series::AbstractCalibrationSeries)
    return HillTailDecay(; kmin = alg.kmin, alpha = alg.alpha, series = series)
end
function bind_series(alg::RadialTailDecay, series::AbstractCalibrationSeries)
    return RadialTailDecay(; kmin = alg.kmin, alpha = alg.alpha, series = series)
end
# The two ambiguity roles and their rules stand below the deformation pair for the reason
# the two `bind_alpha` methods above them do: the types are declared further down the file,
# and the role is rebuilt around the bound rule on the same terms.
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from the data a prior result carries, so that the radius refits whenever the sample moves.

An **Ambiguity Radius** is the radius of the ball of probability measures the model prices, and it is the coefficient of an un-squared norm penalty on the weight vector. It is neither a tail probability nor a deformation parameter, so it takes its own family beside the two of [`AbstractSignificanceCalibrationAlgorithm`](@ref) and [`AbstractDeformationCalibrationAlgorithm`](@ref), under the same root.

All concrete subtypes should subtype `AbstractAmbiguityRadiusCalibrationAlgorithm`, and should be **callable**, on the same terms as the two other families. The family's role type, [`AmbiguityRadiusCalibration`](@ref), subtypes [`AbstractCalibrationEstimator`](@ref) instead, and is not admitted by [`Func_AmbRadCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractAmbiguityRadiusCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractAmbiguityRadiusCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the ambiguity radius.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the owner carries none.

### Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
"""
abstract type AbstractAmbiguityRadiusCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes an Esfahani-Kuhn tail weight, the weight that the `l` slot of a distributionally robust measure holds.

The tail weight scales the tail term of the Esfahani-Kuhn loss, and the mean term is not scaled by it. It is a different quantity from the radius, so it takes a different family and a different role: a radius rule placed in a tail-weight slot is refused at construction. The family's role type, [`AmbiguityTailWeightCalibration`](@ref), subtypes [`AbstractCalibrationEstimator`](@ref) rather than this family, and is not admitted by [`Func_AmbTwtCal`](@ref).

[`TailTermParity`](@ref) is the rule the family ships. A stated tail weight is dimensionless and is not scale-free in the sample, so one number is a different trade-off at every sampling frequency; the rule reads the sample's own units and leaves the preference in the caller's `ratio`. [`Func_AmbTwtCal`](@ref) admits a plain function beside it, so a caller's own rule needs no type.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractAmbiguityTailWeightCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractAmbiguityTailWeightCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the tail weight.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the owner carries none.

### Returns

  - `l::Number`: The tail weight.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`Func_AmbTwtCal`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`TailTermParity`](@ref)
"""
abstract type AbstractAmbiguityTailWeightCalibrationAlgorithm <:
              AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes a norm ceiling from the data a prior result carries, so that the ceiling refits whenever the universe moves.

A **Norm Ceiling** is an upper bound on a norm of the weight vector, the quantity the `l2c`, `lpc` and `linfc` slots of [`JuMPOptimiser`](@ref) hold. It is not an **Ambiguity Radius**. A radius is the coefficient of a norm penalty in the objective, and a ceiling bounds that norm in a constraint. The reciprocal of a ceiling is a floor on the effective number of assets, which is a diversification statement rather than a statement about the set of measures the model prices. So the quantity takes its own family beside the others, under the same root.

All concrete subtypes should subtype `AbstractNormCeilingCalibrationAlgorithm`, and should be **callable**, on the same terms as the other families. The family's role type, [`NormCeilingCalibration`](@ref), subtypes [`AbstractCalibrationEstimator`](@ref) instead, and is not admitted by [`Func_NormCeilCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractNormCeilingCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractNormCeilingCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the norm ceiling.

### Arguments

  - `key`: Name of the slot that is being resolved: `:l2c`, `:lpc` or `:linfc`.
  - `pr`: Prior result the rule reads its asset count and sample size off.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the owner carries none.

### Returns

  - `val::Number`: The norm ceiling.

## The norm order

A ceiling is read against one norm order, and that order belongs to the constraint rather than to the rule. A rule that needs the order should implement a [`bind_norm_order`](@ref) method, which each constraint site calls before it resolves the slot. A rule that needs no order needs no method.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`NormCeilingCalibration`](@ref)
  - [`Func_NormCeilCal`](@ref)
  - [`bind_norm_order`](@ref)
  - [`EffectiveAssetFloor`](@ref)
"""
abstract type AbstractNormCeilingCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
    const Func_AmbRadCal = Union{<:Function, <:AbstractAmbiguityRadiusCalibrationAlgorithm}

Field bound for the `alg` field of an ambiguity-radius role: a rule of the family, or a plain function of the same four arguments. It is the counterpart of [`Func_SigCal`](@ref), and carries its reading unchanged.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`Func_AmbTwtCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_AmbRadCal = Union{<:Function, <:AbstractAmbiguityRadiusCalibrationAlgorithm}
"""
    const Func_AmbTwtCal = Union{<:Function,
                                 <:AbstractAmbiguityTailWeightCalibrationAlgorithm}

Field bound for the `alg` field of an ambiguity-tail-weight role: a rule of the family, or a plain function of the same four arguments. [`TailTermParity`](@ref) is the rule the family ships, and the plain function carries a caller's own.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`TailTermParity`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_AmbTwtCal = Union{<:Function, <:AbstractAmbiguityTailWeightCalibrationAlgorithm}
"""
    const Func_NormCeilCal = Union{<:Function,
                                   <:AbstractNormCeilingCalibrationAlgorithm}

Field bound for the `alg` field of a norm-ceiling role: a rule of the family, or a plain function of the same four arguments. It is the counterpart of [`Func_AmbRadCal`](@ref), and carries its reading unchanged.

# Related

  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_NormCeilCal = Union{<:Function, <:AbstractNormCeilingCalibrationAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Places an ambiguity-radius rule in a slot that holds the radius of the ball the model prices.

The role is the whole of the type: the rule itself lives in `alg`. A radius has no lower and upper end to name, so the family carries one role rather than the two that a tail probability carries, and [`mirror_role`](@ref) needs no method for it.

A slot bounded by [`Num_AmbRadCal`](@ref) admits this role and refuses [`AmbiguityTailWeightCalibration`](@ref), so a tail-weight rule placed in a radius slot is refused at construction rather than at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AmbiguityRadiusCalibration(;
        alg::Func_AmbRadCal
    ) -> AmbiguityRadiusCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
"""
@concrete struct AmbiguityRadiusCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_amb_rad])
    """
    alg
    function AmbiguityRadiusCalibration(alg::Func_AmbRadCal)
        return new{typeof(alg)}(alg)
    end
end
function AmbiguityRadiusCalibration(; alg::Func_AmbRadCal)
    return AmbiguityRadiusCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a tail-weight rule in a slot that holds the weight of the tail term of an Esfahani-Kuhn loss.

It is the counterpart of [`AmbiguityRadiusCalibration`](@ref), and carries the same shape. Its `alg` holds [`TailTermParity`](@ref) or a caller's own function.

The role **travels**, and the radius role does not. A tail weight prices a tail at the measure's own significance level, so [`bind_alpha`](@ref) rebuilds the role around the bound rule before the slot is resolved, on the same terms as the two deformation roles.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AmbiguityTailWeightCalibration(;
        alg::Func_AmbTwtCal
    ) -> AmbiguityTailWeightCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`Func_AmbTwtCal`](@ref)
  - [`TailTermParity`](@ref)
  - [`bind_alpha`](@ref)
"""
@concrete struct AmbiguityTailWeightCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_amb_twt])
    """
    alg
    function AmbiguityTailWeightCalibration(alg::Func_AmbTwtCal)
        return new{typeof(alg)}(alg)
    end
end
function AmbiguityTailWeightCalibration(; alg::Func_AmbTwtCal)
    return AmbiguityTailWeightCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a norm-ceiling rule in a slot that bounds a norm of the weight vector from above.

A ceiling is a different quantity from a radius, so it takes a different role: a radius rule placed in a ceiling slot is refused at construction, and a ceiling rule placed in a radius slot is refused the same way. [`AbstractNormCeilingCalibrationAlgorithm`](@ref) states the difference between the two quantities.

The role is the whole of the type: the rule itself lives in `alg`. A ceiling has no lower and upper end to name, so the family carries one role rather than two, and [`mirror_role`](@ref) needs no method for it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormCeilingCalibration(;
        alg::Func_NormCeilCal
    ) -> NormCeilingCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`Func_NormCeilCal`](@ref)
  - [`EffectiveAssetFloor`](@ref)
"""
@concrete struct NormCeilingCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_norm_ceil])
    """
    alg
    function NormCeilingCalibration(alg::Func_NormCeilCal)
        return new{typeof(alg)}(alg)
    end
end
function NormCeilingCalibration(; alg::Func_NormCeilCal)
    return NormCeilingCalibration(alg)
end
"""
    const Num_AmbRadCal = Union{<:AmbiguityRadiusCalibration, <:Number}

Field bound for an ambiguity-radius slot: the radius itself, or the role that computes it.

The union names one role and no other, so a tail-weight role placed in a radius slot fails the constructor's signature and is refused at construction. That is the whole of the role validation, and no guard method is written for it.

# Related

  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_AmbRadCal = Union{<:AmbiguityRadiusCalibration, <:Number}
"""
    const Num_AmbTwtCal = Union{<:AmbiguityTailWeightCalibration, <:Number}

Field bound for an ambiguity-tail-weight slot: the tail weight itself, or the role that computes it.

# Related

  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_AmbTwtCal = Union{<:AmbiguityTailWeightCalibration, <:Number}
"""
    const Num_NormCeilCal = Union{<:NormCeilingCalibration, <:Number}

Field bound for a norm-ceiling slot: the ceiling itself, or the role that computes it.

The union names one role and no other, so a radius role placed in a ceiling slot fails the constructor's signature and is refused at construction. That is the whole of the role validation, and no guard method is written for it.

# Related

  - [`NormCeilingCalibration`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Num_AmbRadNormCeilCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_NormCeilCal = Union{<:NormCeilingCalibration, <:Number}
"""
    const Num_AmbRadNormCeilCal = Union{<:AmbiguityRadiusCalibration,
                                        <:NormCeilingCalibration, <:Number}

Field bound for the one slot the library reads as two quantities, the `val` field of [`LpRegularisation`](@ref).

That estimator is a penalty in the `lp` field of [`JuMPOptimiser`](@ref) and a norm constraint in its `lpc` field, so `val` is an ambiguity radius on one route and a norm ceiling on the other. One field cannot carry two bounds, so this bound admits both roles and each route refuses the role that has no reading on it. It is the only slot in the library whose role is settled after construction rather than by its bound.

# Related

  - [`LpRegularisation`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`assert_penalty_coefficient_role`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
"""
const Num_AmbRadNormCeilCal = Union{<:AmbiguityRadiusCalibration, <:NormCeilingCalibration,
                                    <:Number}
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from the concentration of measure, so that the ball shrinks as the sample grows.

The radius is the Blanchet-Kang-Murthy form, a scale in the units of the series the slot owner prices times the square root of a chi-squared quantile over the sample size. The chi-squared factor is dimensionless and grows with the number of assets, so a wider universe buys a wider ball at a fixed confidence level, and a longer sample shrinks it.

`scale` states the units. A radius multiplies a norm of the weight vector, so it is in the units of the loss the ball is drawn around, and no caller can intuit that number from the confidence level alone. `scale = nothing` reads the average per-asset dispersion of the series the slot owner prices instead, which is the first pass the form's source recommends, and it moves with the sample the way the rest of the rule does.

**A drawdown owner is read on a drawdown scale.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) measures the transport cost of its own programme against the per-asset drawdown sample, so the ball it prices is a ball over drawdown scenarios and the radius carries drawdown units. `series` names that quantity, [`calibration_series_dispersion`](@ref) takes the dispersion off the drawdown sample, and `pr.sigma` reaches nothing there: it is a moment of the returns.

`T` is the effective sample size when observation weights are stated, and the raw row count when they are not, on the same terms as [`ScenarioCount`](@ref). The radius prices estimation error, and the error of a weighted estimate falls with Kish's effective sample size rather than with the number of rows. [`RateRadius`](@ref) reads the raw row count instead, because a rate speaks of the length of the record.

The rule carries no range check of its own. It returns the quantity of the slot it stands in, so the slot owner's constructor is the whole validation, and a radius outside the slot's range is refused there, at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConcentrationRadius(;
        confidence::Number = 0.95,
        scale::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> ConcentrationRadius

Keywords correspond to the struct's fields. `confidence` defaults to `0.95`, `scale` defaults to `nothing`, which reads the average per-asset dispersion off the sample, and `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `0 < confidence < 1`.
  - If `scale` is not `nothing`: `scale > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`DualNormRadius`](@ref): answers what the sampling error is in the ground metric the slot names, so its number changes with the key while this one's does not.
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
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
    """
    $(field_dict[:cal_series_scale])
    """
    series
    function ConcentrationRadius(confidence::Number, scale::Option{<:Number},
                                 series::AbstractCalibrationSeries)
        assert_unit_interval(confidence, :confidence)
        assert_nonempty_gt0_finite_val(scale, :scale)
        return new{typeof(confidence), typeof(scale), typeof(series)}(confidence, scale,
                                                                      series)
    end
end
function ConcentrationRadius(; confidence::Number = 0.95, scale::Option{<:Number} = nothing,
                             series::AbstractCalibrationSeries = ReturnsSeries())
    return ConcentrationRadius(confidence, scale, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius that the concentration of measure gives on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= s \\sqrt{\\dfrac{\\chi^{2}_{N,\\, q}}{T_{e}}}\\,,\\\\
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
  - $(math_dict[:w_t_obs])

# Arguments

  - `alg`: The rule. Its `series` field names the quantity the ball is drawn around, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The radius is the same for every key, so the two tails of a Range measure that carry one rule resolve to one number.
  - `pr`: Prior result the sample size, the asset count and, on a returns series, the covariance matrix are read off. A drawdown series reads the sample instead.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::ConcentrationRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    N = size(pr.X, 2)
    T = effective_sample_size(pr, w)
    scale = if isnothing(alg.scale)
        Statistics.mean(calibration_series_dispersion(alg.series, pr))
    else
        alg.scale
    end
    q = Distributions.cquantile(Distributions.Chisq(N),
                                one(alg.confidence) - alg.confidence)
    return scale * sqrt(q / T)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius that shrinks with the square root of the sample length.

The radius is `c / sqrt(T)`. The rate is the part of the form to trust, and the coefficient is the part to calibrate: a cross-validation over `c` is the honest route to a radius, and this is the shape a grid moves over.

The rule reads the raw row count, and not the effective sample size that [`ConcentrationRadius`](@ref) reads. The rate is a statement about the length of the record, on the same terms as [`RateSignificance`](@ref).

`c` carries the units of the returns, because the rate itself is dimensionless.

The rule carries no range check of its own, on the same terms as [`ConcentrationRadius`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RateRadius(;
        c::Number = 1
    ) -> RateRadius

Keywords correspond to the struct's fields. `c` defaults to `1`, which is the plain ``1/\\sqrt{T}`` rate.

## Validation

  - `c > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref): answers what the sampling error is in the ground metric the slot names, so its number changes with the key while this one's does not.
  - [`AmbiguityRadiusCalibration`](@ref)
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

Compute the ambiguity radius at the square-root rate of the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= \\frac{c}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - ``c``: Rate coefficient, in the units of the returns.
  - $(math_dict[:T])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The rate is the same for every key.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights. This rule reads the raw row count, so it ignores them.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`RateRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::RateRadius)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    return alg.c / sqrt(size(pr.X, 1))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius that shrinks at the dimensional rate a Wasserstein ball earns, not at the square-root rate.

The radius is the Esfahani-Kuhn [drcvar](@cite) form, a scale in the units of the series the slot owner prices times the sample factor `log(1 / (1 - confidence)) / T` raised to the power `1 / max(N, 2)`. A ball of probability measures must hold the true **measure**, and not merely the true mean, and the measure-concentration result the form is read off charges that exponent for it. At `N = 2` the rule returns the square-root rate that [`ConcentrationRadius`](@ref) and [`RateRadius`](@ref) both carry, so the family's existing behaviour is the two-dimensional corner of this one.

**The ball hardly shrinks over a wide universe, and that is the model speaking rather than a defect.** At `N = 20` and `T = 250` the factor `T^(-1/20)` is `0.76`, so a record ten times longer buys almost nothing. This is the curse of dimensionality of the Wasserstein ball. A caller who reads a square-root rate and expects an estimation error reads a statement about a *mean*, and this rule makes one about a *measure*.

**The radius is conservative in practice, and its source recommends a cross-validation over the radius in its place.** A portfolio priced against a ball this wide can hold nothing but cash. [`RateRadius`](@ref) is the shape such a grid moves over, so calibrate there and read this rule as the statement of what the rate is.

The source result carries a second branch for a short record, whose exponent is `1 / a` for a tail-decay exponent `a` rather than `1 / max(N, 2)`. That branch binds only below a threshold that depends on constants the source leaves to the caller, so this rule drops it and ships the one branch. A reader who holds the paper beside this form finds the difference stated here, rather than by deriving it.

**The branch stays dropped, and [`HillTailDecay`](@ref) does not supply it.** That rule estimates the tail index of a **power law**, which is the heavy-tailed regime. The `a` of the second branch is the exponent of the **light-tail** moment condition the concentration result assumes, and the same condition carries the two constants that place the threshold. So the two exponents share a letter and nothing else, and a rule that read one for the other would state a rate the result does not give. The threshold is also unreachable in practice: it puts the short branch below a record of a few dozen rows, which no fold of a portfolio problem uses.

`N` is the asset count, `size(pr.X, 2)`, because the ball is over the assets. A factor prior carries a smaller effective dimension, and the ball is still over the assets, so a factor prior does not move `N`.

`scale` states the units. A radius multiplies a norm of the weight vector, so it is in the units of the loss the ball is drawn around, and no caller can intuit that number from the confidence level alone. `scale = nothing` reads the average per-asset dispersion of the series the slot owner prices instead, on the same terms as [`ConcentrationRadius`](@ref), and a drawdown owner is read on a drawdown scale there for the reason that rule states.

[`DistributionallyRobustConditionalDrawdownatRisk`](@ref) prices a ball around the drawdown scenarios. The scenario dimension there is still `N`, so the rate carries, and the scale moves with the series: [`bind_series`](@ref) puts the owner's marker on the rule, and [`calibration_series_dispersion`](@ref) then reads the per-asset dispersion off the drawdown sample rather than off `pr.sigma`. A drawdown column is a running functional of its returns, so its dispersion is the wider of the two, and a `scale` of `nothing` therefore gives a wider ball on that owner than on a returns owner of the same sample. A stated `scale` still wins, and it is the way to price a ball whose units are neither.

`T` is the effective sample size when observation weights are stated, and the raw row count when they are not, on the same terms as [`ConcentrationRadius`](@ref) and [`ScenarioCount`](@ref). The rate is a concentration statement, so the record it prices is the one Kish's count measures. [`RateRadius`](@ref) reads the raw row count instead, because its rate speaks of the length of the record.

The rule carries no range check of its own, on the same terms as [`ConcentrationRadius`](@ref). It returns the quantity of the slot it stands in, so the slot owner's constructor is the whole validation, and a radius outside the slot's range is refused there, at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionalRateRadius(;
        confidence::Number = 0.95,
        scale::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> DimensionalRateRadius

Keywords correspond to the struct's fields. `confidence` defaults to `0.95`, `scale` defaults to `nothing`, which reads the average per-asset dispersion off the sample, and `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `0 < confidence < 1`.
  - If `scale` is not `nothing`: `scale > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar])
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
    """
    $(field_dict[:cal_series_scale])
    """
    series
    function DimensionalRateRadius(confidence::Number, scale::Option{<:Number},
                                   series::AbstractCalibrationSeries)
        assert_unit_interval(confidence, :confidence)
        assert_nonempty_gt0_finite_val(scale, :scale)
        return new{typeof(confidence), typeof(scale), typeof(series)}(confidence, scale,
                                                                      series)
    end
end
function DimensionalRateRadius(; confidence::Number = 0.95,
                               scale::Option{<:Number} = nothing,
                               series::AbstractCalibrationSeries = ReturnsSeries())
    return DimensionalRateRadius(confidence, scale, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius that the measure-concentration bound gives on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= s \\left(\\dfrac{\\ln\\left(\\dfrac{1}{1 - q}\\right)}{T_{e}}\\right)^{\\frac{1}{\\max(N,\\, 2)}}\\,,\\\\
s &= \\begin{cases}
\\dfrac{1}{N} \\sum\\limits_{i=1}^{N} \\hat{s}_{i} & \\textrm{if } \\texttt{scale} \\textrm{ is } \\texttt{nothing}\\\\
\\texttt{scale} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - $(math_dict[:cal_s_radius])
  - ``q``: Confidence level the bound is read at.
  - $(math_dict[:N])
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])

The exponent is floored at one half, so a universe of one or two assets returns the square-root rate rather than a faster one. The bound states no rate above that floor.

# Arguments

  - `alg`: The rule. Its `series` field names the quantity the ball is drawn around, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The radius is the same for every key, so the two tails of a Range measure that carry one rule resolve to one number.
  - `pr`: Prior result the sample size, the asset count and, on a returns series, the covariance matrix are read off. A drawdown series reads the sample instead.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`DimensionalRateRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::DimensionalRateRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    N = size(pr.X, 2)
    T = effective_sample_size(pr, w)
    scale = if isnothing(alg.scale)
        Statistics.mean(calibration_series_dispersion(alg.series, pr))
    else
        alg.scale
    end
    return scale * (log(inv(one(alg.confidence) - alg.confidence)) / T)^inv(max(N, 2))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius in the ground metric that the slot it stands in names, so that two slots of two different norms get two different numbers.

The eight radius slots of the library do not measure distance in one norm. A radius multiplies a norm of the weight vector, and the ground metric of the ball is the dual of that norm, so the `l1` coefficient of [`JuMPOptimiser`](@ref) is a distance in the ∞-norm while its `linf` coefficient is a distance in the 1-norm. This rule reads `key`, picks the ground metric of that slot, and returns the sampling error of the empirical measure in it. [`ConcentrationRadius`](@ref) and [`RateRadius`](@ref) return one number for every key, which is right inside one measure and wrong across the eight slots.

The sampling error of the mean vector is the part a linear loss sees, and its per-asset scale is the dispersion of the series the slot owner prices over the square root of the effective sample size. The radius is a norm of that error vector, at the confidence level `confidence` states.

**No term of the covariance error belongs in that vector.** A radius multiplies a norm of the weight vector, so it prices the part of the loss that is **linear** in the weights, and the loss of a distributionally robust measure is linear in the sample point. The error of the second moment reaches the objective through the tail term rather than through the ball, and the tail term carries its own coefficient, which [`TailTermParity`](@ref) sets. A radius that carried both would price one error twice.

`confidence` is a **per-coordinate** level, and it is not corrected for the number of assets. The ∞-norm case is a maximum over `N` coordinates, so a per-coordinate level understates it, and a caller who wants a level over the whole vector states the corrected number themselves, as `1 - (1 - c) / N`.

**The correction stays with the caller, and it is not a rounding.** A Bonferroni step from `0.95` to `1 - 0.05 / N` moves the quantile by about a third at four assets and by more at twenty. It is also one choice of several: a level over the vector read off `N` correlated coordinates and one read off `N` independent ones disagree by an amount only the correlation states. The source result states a per-coordinate level and no correction, so the rule ships what the result states and names the corrected number in prose. A rule that corrected silently would give a caller a level they did not ask for and cannot see.

The 1-norm case sums the per-asset errors, which prices them as if they moved together. That is the worst case over the correlations, and it is therefore the conservative reading for a radius. A correlation-aware form would give a smaller ball, and this rule does not compute one.

**The worst case is the reading a radius wants, so the correlation-aware form stays out.** A radius that understates the ball gives a model that is robust to less than the caller asked for, and the failure is silent: the optimisation solves and the weights are not robust. A radius that overstates it is visible, because the portfolio moves towards cash. The two errors are not symmetric, so the rule takes the one a caller can see.

`p` serves the `:lpreg_val` slot alone. The ground metric of [`LpRegularisation`](@ref) is the type-``q`` metric with ``1/p + 1/q = 1``, and `key` names the slot rather than the norm order. The order belongs to the penalty, so that site fills this field through [`bind_norm_order`](@ref) before it resolves the slot, and the call **overwrites** whatever the field holds. A stated `p` therefore serves a caller who runs the rule outside that site, and nothing else. Every other key ignores the field.

**The drawdown owner is served on a drawdown scale, and `series` is what says so.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) measures the transport cost of its own programme against the per-asset drawdown sample, so the ball it prices is a ball over drawdown scenarios and the ground metric is a distance between two such vectors. [`calibration_series_dispersion`](@ref) reads the error scale off that sample under a drawdown marker, and `pr.sigma` reaches nothing there: it is a moment of the returns, and no scaling of it states a moment of a drawdown. The ground metric does not move with the series, only the vector it is taken of.

**The error scale of a drawdown series is a floor, and the record is the reason.** A drawdown is a running functional, so its entries are strongly dependent down a column, and ``\\hat{s}_{i} / \\sqrt{T_{e}}`` prices a record of ``T_{e}`` independent draws that a drawdown sample does not hold. The rule states no correction for it, on the same terms it states none for the number of assets: a correction needs a model of the dependence, and the sample states none. So the reading is the honest scale of the quantity, taken as if the record were independent.

`T_e` is Kish's effective sample size when observation weights are stated, and the raw row count when they are not, on the same terms as [`ConcentrationRadius`](@ref).

The rule carries no range check of its own, on the same terms as [`ConcentrationRadius`](@ref).

**A sample of no dispersion gives a radius of zero, and the slot admits it.** Every diagonal of the dispersion is non-negative, so the radius is non-negative and finite for every admissible input. A sample whose columns never move gives an error vector of zeros, and a ball of no width is the right answer for it: the empirical measure is the only measure such a sample supports, and the robust measure reduces to the plain one it is drawn around. In floating point that sample gives a number many orders below the smallest one a caller would state rather than an exact zero, so the slot owner's `> 0` check passes and the model prices a ball of no width. Both outcomes read the same way, and neither is a defect.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DualNormRadius(;
        confidence::Number = 0.95,
        p::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> DualNormRadius

Keywords correspond to the struct's fields. `confidence` defaults to `0.95`, `p` defaults to `nothing`, which serves every slot but `:lpreg_val`, and `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `0 < confidence < 1`.
  - If `p` is not `nothing`: `isfinite(p)` and `p > 1`, on the same terms as [`LpRegularisation`](@ref).

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref): answers how wide the ball is at a confidence level, in one dimensionless factor that no norm enters.
  - [`RateRadius`](@ref): answers how fast the ball shrinks with the record, and leaves the coefficient to a cross-validation.
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`bind_series`](@ref)
  - [`bind_norm_order`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct DualNormRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    `confidence`: Per-coordinate confidence level of the normal quantile the error scale is multiplied by. It is not corrected for the number of assets, so the ∞-norm case reads a level over one coordinate and not over the vector.
    """
    confidence
    """
    `p`: Norm order of the [`LpRegularisation`](@ref) penalty the radius stands in, or `nothing`. It is read for the `:lpreg_val` key alone, where the ground metric is the type-``q`` metric with ``1/p + 1/q = 1`` and no key can name ``q``. The penalty site overwrites it through [`bind_norm_order`](@ref), so state it only to run the rule outside that site.
    """
    p
    """
    $(field_dict[:cal_series_scale])
    """
    series
    function DualNormRadius(confidence::Number, p::Option{<:Number},
                            series::AbstractCalibrationSeries)
        assert_unit_interval(confidence, :confidence)
        if !isnothing(p)
            @argcheck(isfinite(p), IsNonFiniteError)
            @argcheck(p > one(p), DomainError)
        end
        return new{typeof(confidence), typeof(p), typeof(series)}(confidence, p, series)
    end
end
function DualNormRadius(; confidence::Number = 0.95, p::Option{<:Number} = nothing,
                        series::AbstractCalibrationSeries = ReturnsSeries())
    return DualNormRadius(confidence, p, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius of the slot `key` names, in the ground metric of that slot, on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= z_{c} \\lVert \\boldsymbol{e} \\rVert_{g}\\,,\\\\
e_{i} &= \\dfrac{\\hat{s}_{i}}{\\sqrt{T_{e}}}\\,,\\\\
q &= \\dfrac{p}{p - 1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - ``z_{c}``: Quantile of the standard normal distribution at the per-coordinate confidence level ``c``.
  - ``\\boldsymbol{e}``: Per-asset scale of the sampling error of the mean vector, in the units of the series the slot owner prices.
  - ``g``: Order of the ground metric, which `key` names.
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])
  - ``p``: Norm order of the penalty, the `p` field.
  - ``q``: Order of the type-``q`` ground metric of that penalty.

# The ground metric of each key

| `key`          | Slot                                                                                                              | Penalised norm | Ground metric ``g`` |
|:-------------- |:----------------------------------------------------------------------------------------------------------------- |:-------------- |:------------------- |
| `:l1`          | `l1` of [`JuMPOptimiser`](@ref)                                                                                   | 1              | ``\\infty``         |
| `:linf`        | `linf` of [`JuMPOptimiser`](@ref)                                                                                 | ``\\infty``    | 1                   |
| `:r`           | [`DistributionallyRobustConditionalValueatRisk`](@ref), [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) | ``\\infty``    | 1                   |
| `:r_a`, `:r_b` | [`DistributionallyRobustConditionalValueatRiskRange`](@ref)                                                       | ``\\infty``    | 1                   |
| `:l2reg_val`   | `val` of [`L2Regularisation`](@ref)                                                                               | 2              | 2                   |
| `:lpreg_val`   | `val` of [`LpRegularisation`](@ref)                                                                               | ``p``          | ``q``               |

The two ends of a Range measure carry one ground metric, so a rule stated on both resolves to one number. A radius names no end of the distribution.

# Arguments

  - `alg`: The rule. Its `series` field names the quantity the ball is drawn around, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. It selects the ground metric, so this is the one rule of its family for which the key carries meaning.
  - `pr`: Prior result the sample size and, on a returns series, the covariance matrix are read off. A drawdown series reads the sample instead.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Validation

  - `key` is one of the seven the table names, else an `ArgumentError` naming the key it received and the keys it serves.
  - The `:lpreg_val` key needs a stated `p`, else an `ArgumentError` naming the field.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`DualNormRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::DualNormRadius)(key::Symbol, pr::AbstractPriorResult, w, ::Any)
    T = effective_sample_size(pr, w)
    e = calibration_series_dispersion(alg.series, pr) ./ sqrt(T)
    z = Distributions.quantile(Distributions.Normal(), alg.confidence)
    return z * dual_norm_radius_scale(alg, key, e)
end
"""
    dual_norm_radius_scale(alg::DualNormRadius, key::Symbol, e::AbstractVector)

Reduce the per-asset error vector `e` to one number, in the ground metric that `key` names.

This is the whole of the key's meaning, held apart from the functor. Six of the seven keys name a fixed norm order, so they are a literal table and not a chain of branches. The seventh, `:lpreg_val`, derives its order from the `p` the rule carries, because no key can name it. An eighth key is a refusal, because a caller who writes their own measure reaches it first and the message must name the keys the rule serves.

# Arguments

  - `alg`: The rule, read for `p` on the `:lpreg_val` arm alone.
  - `key`: Name of the slot that is being resolved.
  - `e`: Per-asset scale of the sampling error of the mean vector.

# Validation

  - `key` is one of the seven [`DualNormRadius`](@ref) serves, else an `ArgumentError`.
  - `alg.p` is stated when `key` is `:lpreg_val`, else an `ArgumentError`.

# Returns

  - `s::Number`: The norm of `e` in the ground metric of the slot.

# Related

  - [`DualNormRadius`](@ref)
  - [`LpRegularisation`](@ref)
"""
function dual_norm_radius_scale(alg::DualNormRadius, key::Symbol, e::AbstractVector)
    metrics = (; l1 = Inf, linf = 1, r = 1, r_a = 1, r_b = 1, l2reg_val = 2)
    g = if key === :lpreg_val
        @argcheck(!isnothing(alg.p),
                  ArgumentError("`DualNormRadius.p` is `nothing` while the `:lpreg_val` slot is being resolved. That slot's ground metric is the type-`q` metric of an `LpRegularisation` penalty, with `1/p + 1/q = 1`, and the order belongs to the penalty. The penalty site fills the field through `bind_norm_order`, so a `nothing` here means the rule was resolved somewhere that binds no order. Place the rule in the `val` field of an `LpRegularisation`, or state `p` on the rule."))
        alg.p / (alg.p - one(alg.p))
    else
        get(() -> throw(ArgumentError("`DualNormRadius` reads `key` to pick the ground metric of the slot it stands in, and it received `:$key`, which names no slot it serves. The keys it serves are `:l1`, `:linf`, `:r`, `:r_a`, `:r_b`, `:l2reg_val` and `:lpreg_val`. A measure of your own that holds a radius resolves its slot under one of those keys, or carries a rule of its own.")),
            metrics, key)
    end
    return LinearAlgebra.norm(e, g)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Esfahani-Kuhn tail weight that prices the tail term of the loss at a stated multiple of its mean term.

At ``r \\to 0`` the loss of [`DistributionallyRobustConditionalValueatRisk`](@ref) reduces to ``-\\mathbb{E}[\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}] + l \\, \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi})``, so `l` is the exchange rate between one unit of expected return and one unit of tail loss. Both terms carry the units of the returns, so `l` is dimensionless. **A dimensionless number is not a scale-free one, and that is the trap this rule exists for.** The mean of a daily return column sits two orders of magnitude below its five per cent CVaR, and on a monthly column the gap is far smaller, so a stated `l = 1.0` prices the tail term about forty times the mean term on a daily sample and under ten times on a monthly one. One stated number is a different trade-off at every sampling frequency, and at every re-fit that changes the record.

The rule carries the sample's own units and nothing else, so the preference stays the caller's. `ratio` states how many mean terms one tail term is worth, and the rule returns the `l` that prices it so on the sample the prior result carries. `ratio = 1` is parity, and `ratio = 2` prices the tail term at twice the mean term on every sample.

A rule reads no portfolio, so it cannot form ``\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}``. The series it reads is the cross-section of the per-asset losses of the quantity the slot owner prices: the mean-term scale ``m`` is the mean of that pool, and the tail-term scale ``c`` is the mean of the per-column ``\\mathrm{CVaR}_{\\alpha}``. **The mean of the per-column values is not the pooled ``\\mathrm{CVaR}_{\\alpha}``, and the difference is deliberate.** A pooled tail is drawn from the worst columns, so it is dominated by the most volatile assets and gives a smaller weight. The measure prices a portfolio rather than a single asset, so the per-column mean is the reading that answers it.

**A caller who holds a reference portfolio still cannot state one, and that is the design.** The pooled cross-section is the equal-weight reading, and a stated reference portfolio would give a sharper one. It would also make the rule a second place where a portfolio enters the problem, and the calibration channel carries no portfolio by construction: [`resolve_calibration_slot`](@ref) hands a rule the Prior and nothing else, so a reference vector would have to travel a channel of its own, be checked against the universe of the fold, and be re-stated whenever a subset view or a cluster changed that universe. The exchange rate the rule prices is a property of the **sample**, and the equal-weight cross-section is the reading of it that needs no second input. A caller who wants the rate at their own portfolio computes `l` themselves and states the number.

``m`` is negative for a sample of positive expected return, and the rule takes ``\\lvert m \\rvert``. No field states the sign: a negative weight is not admissible in the slot, and a sample of negative expected return does not turn the trade-off around.

The rule reads its sibling `alpha`, because ``c`` is a ``\\mathrm{CVaR}`` at the measure's own significance level. `alpha` and `l` are a **travelling pair**, on the same terms as `alpha` and `kappa`: [`bind_alpha`](@ref) fills the `alpha` field with the number the slot owner resolved, and the owner's own resolution method resolves `alpha` first for that reason. A caller who runs the rule outside a measure states `alpha` instead.

**Both scales read the observation weights.** They are sample statistics rather than counts, so a weighted sample is read weighted. [`RateRadius`](@ref) and [`RateSignificance`](@ref) ignore `w` because a rate speaks of the length of the record, and that reading does not carry to a moment.

**`series` says which quantity both terms are read over, and the slot owner states it.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) carries the key `:l` as well, and its tail term is a ``\\mathrm{CDaR}`` of the portfolio drawdown series. A rule is given no portfolio, but it can form the drawdown series of each **column**, and [`bind_series`](@ref) puts the marker there at the resolution site. Both terms then move together: the mean term is the mean drawdown of the pool, and the tail term is the mean of the per-column ``\\mathrm{CDaR}_{\\alpha}``. The same [`ConditionalValueatRisk`](@ref) reading forms it, because the tail mean of a non-positive drawdown column **is** the ``\\mathrm{CDaR}`` of that column, so the rule and the measure it calibrates still cannot drift apart.

A second reading of `l` exists, and this rule does not take it. `l` can be read as a risk-aversion coefficient and mapped from a mean-variance one, but a variance penalty is quadratic in the weight vector and a ``\\mathrm{CVaR}`` term is positively homogeneous, so the two objectives are not comparable term by term. The map holds at one reference portfolio and nowhere else, and a rule gets no portfolio.

The rule carries no range check on the weight it returns, on the same terms as [`RateRadius`](@ref). The slot owner's constructor keeps that job. Its two refusals state that the ratio exists at all, which is the shape of the one check [`EntropyBudget`](@ref) carries. A sample whose mean loss is near zero returns a very large weight, and that is the sample speaking rather than a defect: no threshold separates a small mean loss from a smaller one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TailTermParity(;
        ratio::Number = 1,
        alpha::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> TailTermParity

Keywords correspond to the struct's fields. `ratio` defaults to `1`, which is parity between the two terms. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in. `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `ratio > 0` and finite.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`bind_alpha`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`RateRadius`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar])
"""
@concrete struct TailTermParity <: AbstractAmbiguityTailWeightCalibrationAlgorithm
    """
    $(field_dict[:cal_ratio])
    """
    ratio
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    """
    $(field_dict[:cal_series_twt])
    """
    series
    function TailTermParity(ratio::Number, alpha::Option{<:Number},
                            series::AbstractCalibrationSeries)
        assert_nonempty_gt0_finite_val(ratio, :ratio)
        return new{typeof(ratio), typeof(alpha), typeof(series)}(ratio, alpha, series)
    end
end
function TailTermParity(; ratio::Number = 1, alpha::Option{<:Number} = nothing,
                        series::AbstractCalibrationSeries = ReturnsSeries())
    return TailTermParity(ratio, alpha, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the tail weight that prices the tail term at `ratio` mean terms on the sample that `pr` carries.

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
  - ``c``: Tail-term scale, the mean of the per-column ``\\mathrm{CVaR}_{\\alpha}`` of the loss.
  - ``\\boldsymbol{r}_{j}``: Column ``j`` of the returns matrix.
  - ``\\mathcal{S}``: The series `alg.series` names, built from one column. It is the identity on a [`ReturnsSeries`](@ref), and a drawdown series on the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``y_{tj}``: Entry ``t`` of the series of column ``j``.
  - ``\\boldsymbol{y}_{j}``: The series of column ``j``. A ``\\mathrm{CVaR}_{\\alpha}`` of a non-positive drawdown series is the ``\\mathrm{CDaR}_{\\alpha}`` of that column.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:r_tj])

Every observation weight is one when none is stated. Every column holds ``T`` entries, so the pooled mean and the mean of the per-column means are one number, and the pooled form is the one written.

# Algorithm

 1. Build the sample `alg.series` names with [`calibration_series_matrix`](@ref) into `X`, and read the effective observation weights off `w`. A [`ReturnsSeries`](@ref) returns `pr.X` itself.
 2. Form the mean-term scale `m`, the negated weighted mean of the pooled cross-section of `X`.
 3. Build a [`ConditionalValueatRisk`](@ref) at `alg.alpha` carrying the same weights, and form the tail-term scale `c`, the mean over the columns of `X` of the value that measure takes on each. The measure's own reading is the one used, so the rule and the measure it calibrates cannot drift apart.
 4. Return `alg.ratio * abs(m) / c`.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there, and its `series` field names the quantity, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The scales are read off the asset columns, so the key never selects the value, and the two ends of a Range measure part company through their two probabilities alone.
  - `pr`: Prior result the returns matrix is read off. The series is built from its columns.
  - `w`: Effective observation weights, or `nothing`. Both scales are sample statistics, so this rule reads them.
  - `slv`: Effective solver. This rule needs none, because both scales are closed forms.

# Validation

  - `alg.alpha` must not be `nothing`.
  - ``\\lvert m \\rvert`` must be positive.
  - ``c`` must be positive.

# Returns

  - `l::Number`: The Esfahani-Kuhn tail weight.

# Related

  - [`TailTermParity`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_matrix`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::TailTermParity)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`TailTermParity.alpha` is `nothing`, so the rule cannot form the tail-term scale `c`, which is a CVaR at the measure's own significance level. The probability travels to the rule through `bind_alpha`, which the slot owner calls after it resolves `alpha`. State `alpha` on the rule itself to run it outside a measure."))
    # Both scales are read over the series the OWNER prices. Under a returns marker this is
    # `pr.X` itself, and under a drawdown marker the tail term is a CDaR: the CVaR kernel
    # over a non-positive drawdown column is the mean of its worst `alpha`.
    X = calibration_series_matrix(alg.series, pr.X)
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
    rm = ConditionalValueatRisk(; alpha = alg.alpha, w = w)
    c = Statistics.mean(j -> rm(view(X, :, j)), axes(X, 2))
    @argcheck(c > 0,
              DomainError(c,
                          "`TailTermParity` read a tail-term scale of $c at `alpha = $(alg.alpha)`, which is not positive, so the ratio `ratio * abs(m) / c` has no admissible value. The scale is the mean of the per-column CVaR of the loss, and a non-positive one is a sample whose worst `alpha` of every column holds no loss at all. Widen `alpha`, or state `l` on the slot."))
    return alg.ratio * abs(m) / c
end
# The other methods of `bind_alpha` stand beside the deformation rules, which are the pair
# the verb was built for. These two stand here because the types they name are declared
# above them, and the role is rebuilt around the bound rule on the same terms.
function bind_alpha(r::AmbiguityTailWeightCalibration, alpha::Number)
    return AmbiguityTailWeightCalibration(; alg = bind_alpha(r.alg, alpha))
end
function bind_alpha(alg::TailTermParity, alpha::Number)
    return TailTermParity(; ratio = alg.ratio, alpha = alpha, series = alg.series)
end
# The same holds for the six `bind_series` methods below. The two ambiguity families are
# declared between the verb and this point, so their methods cannot stand beside the
# deformation pair the verb was written for.
function bind_series(r::AmbiguityRadiusCalibration, series::AbstractCalibrationSeries)
    return AmbiguityRadiusCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(r::AmbiguityTailWeightCalibration, series::AbstractCalibrationSeries)
    return AmbiguityTailWeightCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(alg::ConcentrationRadius, series::AbstractCalibrationSeries)
    return ConcentrationRadius(; confidence = alg.confidence, scale = alg.scale,
                               series = series)
end
function bind_series(alg::DimensionalRateRadius, series::AbstractCalibrationSeries)
    return DimensionalRateRadius(; confidence = alg.confidence, scale = alg.scale,
                                 series = series)
end
function bind_series(alg::DualNormRadius, series::AbstractCalibrationSeries)
    return DualNormRadius(; confidence = alg.confidence, p = alg.p, series = series)
end
function bind_series(alg::TailTermParity, series::AbstractCalibrationSeries)
    return TailTermParity(; ratio = alg.ratio, alpha = alg.alpha, series = series)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a norm ceiling that holds a stated fraction of the universe effective, so that the floor refits whenever the universe changes.

A norm ceiling and the effective number of assets are reciprocally related, so a bound on the norm is a floor on that count. This rule states the floor as a **fraction of the universe** rather than as a count. The asset count comes off the prior result, so a subset view, a cluster and a cross-validation fold each get the floor their own universe earns, and no number is pinned to the universe it was written for.

`p` is the norm order the ceiling is read against, and it belongs to the constraint. Each of the three constraint sites calls [`bind_norm_order`](@ref) before it resolves the slot, and that call **overwrites** whatever this field holds. So a stated `p` serves a caller who runs the rule outside those sites, and nothing else.

The rule carries no range check on its answer, on the same terms as [`ConcentrationRadius`](@ref). It returns the quantity of the slot it stands in, so the slot owner is the whole validation.

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
  - ``m``: Effective number of assets the ceiling holds as a floor.
  - ``f``: Fraction of the universe, the `fraction` field.
  - $(math_dict[:N])
  - ``p``: Norm order of the constraint the ceiling stands in.

The order-``p`` effective number of assets is ``N_{\\mathrm{eff},\\, p}(\\boldsymbol{w}) = \\left(\\sum_{i} \\lvert w_{i} \\rvert^{p}\\right)^{1/(1 - p)}``, so ``N_{\\mathrm{eff},\\, p} \\geq m`` is ``\\lVert \\boldsymbol{w} \\rVert_{p} \\leq m^{1/p - 1}``. It is the reading [`number_effective_assets`](@ref) states, taken to an arbitrary order: at ``p = 2`` the two are the same number, and at every order an equal-weight portfolio over ``m`` assets reports exactly ``m``. The exponent is also ``-1/q`` for the conjugate order ``q``, because ``1/p - 1 = -1/q``.

The infinite arm is the limit of the finite one and not a second reading. ``m^{1/p - 1} \\to m^{-1}`` as ``p`` grows, and a cap of ``1/m`` on the largest weight of a fully invested portfolio spreads it over at least ``m`` assets. So the two arms meet, and a caller who raises ``p`` towards the infinite order sees the ceiling move towards ``1/m`` rather than away from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EffectiveAssetFloor(;
        fraction::Number = 0.5,
        p::Option{<:Number} = nothing
    ) -> EffectiveAssetFloor

Keywords correspond to the struct's fields. `fraction` defaults to `0.5`, which holds half of the universe effective, and `p` defaults to `nothing`, which every constraint site fills.

## Validation

  - `0 < fraction <= 1`.
  - If `p` is not `nothing`: `p >= 1`.

# Related

  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`NormCeilingCalibration`](@ref)
  - [`bind_norm_order`](@ref)
  - [`number_effective_assets`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct EffectiveAssetFloor <: AbstractNormCeilingCalibrationAlgorithm
    """
    $(field_dict[:cal_fraction])
    """
    fraction
    """
    $(field_dict[:cal_norm_order])
    """
    p
    function EffectiveAssetFloor(fraction::Number, p::Option{<:Number})
        assert_nonempty_gt0_finite_val(fraction, :fraction)
        @argcheck(fraction <= one(fraction), DomainError)
        if !isnothing(p)
            @argcheck(p >= one(p), DomainError)
        end
        return new{typeof(fraction), typeof(p)}(fraction, p)
    end
end
function EffectiveAssetFloor(; fraction::Number = 0.5, p::Option{<:Number} = nothing)
    return EffectiveAssetFloor(fraction, p)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the norm ceiling that holds `fraction` of the universe that `pr` carries effective.

The universe is the column count of the prior result's returns matrix, so the ceiling moves with the universe the model actually sees.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. It names the site in the refusal below, and the ceiling itself does not depend on it: the norm order carries that dependence instead.
  - `pr`: Prior result the asset count is read off.
  - `w`: Effective observation weights. A universe count is not a sample count, so this rule ignores them.
  - `slv`: Effective solver. This rule needs none.

# Validation

  - `alg.p` is not `nothing`. Every constraint site fills it through [`bind_norm_order`](@ref), so a `nothing` here means the rule was resolved somewhere that binds no order.

# Returns

  - `val::Number`: The norm ceiling.

# Related

  - [`EffectiveAssetFloor`](@ref)
  - [`bind_norm_order`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::EffectiveAssetFloor)(key::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    p = alg.p
    @argcheck(!isnothing(p),
              ArgumentError("`$(nameof(EffectiveAssetFloor)).p` is `nothing` while the rule in `$key` is being resolved. A ceiling is read against one norm order, the order belongs to the constraint, and each constraint site fills it through `bind_norm_order`. Place the rule in `l2c`, `lpc` or `linfc`, or state `p` on the rule."))
    m = alg.fraction * size(pr.X, 2)
    return isinf(p) ? inv(m) : m^(inv(p) - one(p))
end
"""
    bind_norm_order(slot, p::Number)

Hand the norm order of a weight-norm constraint to the rule that computes its ceiling.

A **Norm Ceiling** is read against one norm order, and that order is a property of the constraint rather than of the rule: one rule placed in `lpc` serves every term, and each term carries its own `p`. [`resolve_calibration_slot`](@ref) carries a `Symbol` and no number, so the order travels through the rule itself. This is the shape [`bind_alpha`](@ref) already uses to carry a significance level to a deformation rule.

An **Ambiguity Radius** on the `val` field of [`LpRegularisation`](@ref) reads the same order, and reads it for the same reason. The ground metric of that penalty is the type-``q`` metric with ``1/p + 1/q = 1``, so [`DualNormRadius`](@ref) needs the owner's `p`, and `key` names the slot rather than the norm order. So that site binds too, and the radius family carries the same pair of methods the ceiling family carries.

The default is the identity, so a stated number crosses unchanged, and so does a caller's own plain function. A plain function reads the slot's name from `key` instead.

The order the constraint site holds **wins**. A rule that already carries one has it replaced, because the constraint is the thing the quantity is read against and the rule cannot know which site it reached.

# Arguments

  - `slot`: The slot's occupant: a number, a [`NormCeilingCalibration`](@ref), or an [`AmbiguityRadiusCalibration`](@ref).
  - `p`: Norm order of the constraint or penalty the quantity stands in.

# Returns

  - `y`: The occupant, with the order filled wherever it holds a rule that reads one.

# Related

  - [`NormCeilingCalibration`](@ref)
  - [`EffectiveAssetFloor`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`DualNormRadius`](@ref)
  - [`bind_alpha`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function bind_norm_order(slot, ::Number)
    return slot
end
function bind_norm_order(r::NormCeilingCalibration, p::Number)
    return NormCeilingCalibration(; alg = bind_norm_order(r.alg, p))
end
function bind_norm_order(alg::EffectiveAssetFloor, p::Number)
    return EffectiveAssetFloor(; fraction = alg.fraction, p = p)
end
function bind_norm_order(r::AmbiguityRadiusCalibration, p::Number)
    return AmbiguityRadiusCalibration(; alg = bind_norm_order(r.alg, p))
end
function bind_norm_order(alg::DualNormRadius, p::Number)
    return DualNormRadius(; confidence = alg.confidence, p = p, series = alg.series)
end

export SignificanceTailCalibration, SignificanceHeadCalibration, DeformationTailCalibration,
       DeformationHeadCalibration, ReturnsSeries, AbsoluteDrawdownSeries,
       RelativeDrawdownSeries, ScenarioCount, RateSignificance, EntropyBudget,
       HillTailDecay, RadialTailDecay, AmbiguityRadiusCalibration,
       AmbiguityTailWeightCalibration, ConcentrationRadius, RateRadius,
       DimensionalRateRadius, DualNormRadius, TailTermParity, NormCeilingCalibration,
       EffectiveAssetFloor
