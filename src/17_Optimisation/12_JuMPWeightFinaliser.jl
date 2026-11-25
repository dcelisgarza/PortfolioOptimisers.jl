"""
$(DocStringExtensions.TYPEDEF)

Moves the weights into their bounds, and into the feasible set of a [`ConstraintBlock`](@ref), with the solution of a JuMP programme.

The programme keeps the sum of the input weights, holds every weight between its bounds, and holds every constraint of the block in `cbe`. The field `alg` selects the objective, the distance from the input that the programme minimises. So a clustering or naive optimiser, which has no solver of its own, takes every constraint kind of a JuMP optimiser.

Each slot of the block is a keyword and a property of the finaliser, as in [`JuMPOptimiser`](@ref), so `wf.card` reads `wf.cbe.card`. The optimiser that holds the finaliser is the one source of the weight bounds `wb` and of the `sets`, and the block holds neither. The split of the optimiser, the finaliser and the result then read one set of bounds.

A finaliser with no slots solves no programme for weights that break no bound, and returns them. With a slot it solves the programme for every input, because it does not test the constraints on the input.

# The block

The finaliser is the owner of its block. The head gives it its prior result, its returns data, its sets, its fees and its `strict` flag, and the block resolves in the two stages of a JuMP optimiser, [`resolve_constraint_block_names`](@ref) and [`resolve_constraint_block`](@ref). These rules hold for the slots:

  - The net budget is the sum of the input weights. A long-short book takes `sbgt`, `gbgt` and `xbgt`.
  - A slot that reads a prior, such as `tr`, `cte`, an estimator in `ple`, an exposure row in `lcse` or a Calibration Rule, needs a head that fits one. [`EqualWeighted`](@ref), [`RandomWeighted`](@ref) and [`BestConstantRebalancedPortfolio`](@ref) refuse it.
  - A turnover ceiling in `tn` takes the reference that the caller gives, and a fold loop replaces it with the weights of the previous fold, as in [`JuMPOptimiser`](@ref).
  - A tracking error in `tr` reads the fees of the head. No other slot reads them.
  - The penalties `l1`, `l2`, `lp` and `linf` and the custom objective terms `cobj` add to the objective. A penalty weighs against the distance of `alg`. A penalty added to a norm does not move the answer off a feasible input while the dual norm of its gradient is at most one, so a penalty goes well with [`SquaredL2Norm`](@ref), with [`EuclideanProjection`](@ref) or with another geometry.
  - No slot holds a [`TimeDependent`](@ref) schedule. The optimiser schedules the whole finaliser, `wf = TimeDependent([...])`.
  - The constraints that a risk measure states, such as a risk ceiling or a return floor, are not slots of the block.

# The objective

  - A [`JuMPWeightFinaliserFormulation`](@ref), such as [`ErrorWeightFinaliser`](@ref), minimises a norm of the deviation from the input. A [`RelativeWeightDeviation`](@ref), the default, divides by the input weights, so an input weight at or near zero gives a coefficient of up to `1 / eps`, and some solvers refuse it. Give a sparse book an [`AbsoluteWeightDeviation`](@ref) or a geometry.
  - An [`AbstractProjectionGeometry`](@ref) minimises the divergence of [`set_projection_objective!`](@ref). [`EntropicProjection`](@ref) keeps the ratios of the free weights of a long-only book, and it refuses a negative input weight and a negative lower bound. [`GramProjection`](@ref) with `A` the covariance matrix gives the book that is nearest to the input in tracking variance. It needs its `A`, and it solves on its own `slv` and not on the solver of the finaliser.

# A failed solve

The finaliser logs a warning and returns the weights of a default [`IterativeWeightFinaliser`](@ref). With no slots, that repair meets every constraint that the finaliser holds, so [`finalise_weight_bounds`](@ref) tests it against the bounds. With a slot, the repair can break the slot, so the return code is an [`OptimisationFailure`](@ref) with the solver trials, and the fallback chain of the head runs.

!!! note

    A mixed-integer slot makes each finalisation a mixed-integer programme, which needs a mixed-integer solver in `slv`. These slots are `card`, `scard`, `gcarde`, `sgcarde`, `smtx` and `sgmtx`, `lt`, `st` and the sub-group thresholds, `conc`, an integer phylogeny in `ple`, and `xbgt`. A fixed fee of the head does the same beside a tracking error. A clustering or naive head has no solve of its own, so the finaliser can be the whole cost of a fit. A cross-validation loop solves the programme once for each fold, and a meta head once for each inner optimisation that holds the finaliser.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPWeightFinaliser(;
        slv::Slv_VecSlv,
        cbe::Option{<:ConstraintBlock} = nothing,
        sc::Number = 1.0,
        so::Number = 1.0,
        alg::WfF_ProjG = ErrorWeightFinaliser(),
        slots...
    ) -> JuMPWeightFinaliser

Keywords correspond to the struct's fields and to the slots of [`ConstraintBlock`](@ref), except `wb` and `sets`. `slots...` takes each keyword of the block with its type and its default, and the constructor builds the block in `cbe` from them. When the caller gives `cbe`, a flat keyword replaces that slot of the given block, see [`constraint_block`](@ref). The default bounds of a [`ConstraintBlock`](@ref) are not `nothing`, so a block that the caller builds states `wb = nothing`. A name that is neither a field nor a slot raises a `MethodError`.

## Validation

The constructor of [`ConstraintBlock`](@ref) checks each slot on its own.

  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - `sc > 0`, `so > 0`.
  - `cbe.wb` and `cbe.sets` are `nothing`. An `ArgumentError` is thrown otherwise: state them on the optimiser.
  - No slot of `cbe` holds a [`TimeDependent`](@ref). An `ArgumentError` is thrown otherwise.
  - A [`GramProjection`](@ref) in `alg` holds its matrix `A`. An `ArgumentError` is thrown otherwise.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `cbe`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `cbe`: Recursively viewed through [`port_opt_view`](@ref).
  - `alg`: Recursively viewed through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `cbe`: Recursively indexed through [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> wf = JuMPWeightFinaliser(; slv = Solver(; solver = nothing), card = 10);

julia> wf.card
10

julia> wf.alg
ErrorWeightFinaliser
  alg ┼ L1Norm
      │   ddof ┴ Int64: 0
  dev ┴ RelativeWeightDeviation()
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`ConstraintBlock`](@ref)
  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`AbstractProjectionGeometry`](@ref)
  - [`finalise_weight_bounds`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct JuMPWeightFinaliser <: WeightFinaliser
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:cbe]) In a weight finaliser its `wb` and its `sets` are `nothing`, because the optimiser states both.
    """
    @fprop @vprop @oprop cbe
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:wfalg])
    """
    @vprop alg
    function JuMPWeightFinaliser(slv::Slv_VecSlv, cbe::ConstraintBlock, sc::Number,
                                 so::Number, alg::WfF_ProjG)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc > zero(sc), DomainError(sc, "sc must be positive"))
        @argcheck(so > zero(so), DomainError(so, "so must be positive"))
        @argcheck(isnothing(cbe.wb),
                  ArgumentError("the block of a `JuMPWeightFinaliser` holds no weight bounds: the `wb` of the optimiser is the one source of the bounds. State `wb` on the optimiser that holds the finaliser, and leave it `nothing` in the block."))
        @argcheck(isnothing(cbe.sets),
                  ArgumentError("the block of a `JuMPWeightFinaliser` holds no sets: the `sets` of the optimiser are the one source of the names. State `sets` on the optimiser that holds the finaliser."))
        tdfs = typed_time_dependent_fields(cbe)
        @argcheck(isempty(tdfs),
                  ArgumentError("the slot `$(first(tdfs))` of a `JuMPWeightFinaliser` holds a `TimeDependent` schedule, and no fold loop resolves a schedule inside a weight finaliser. Schedule the finaliser, `wf = TimeDependent([...])`, and give each entry a static slot."))
        @argcheck(!isa(alg, GramProjection{<:Any, Nothing}),
                  ArgumentError("a `GramProjection` in a `JuMPWeightFinaliser` has no rule that binds its matrix: state it, `GramProjection(; slv = slv, A = A)`."))
        return new{typeof(slv), typeof(cbe), typeof(sc), typeof(so), typeof(alg)}(slv, cbe,
                                                                                  sc, so,
                                                                                  alg)
    end
end
function JuMPWeightFinaliser(; slv::Slv_VecSlv, cbe::Option{<:ConstraintBlock} = nothing,
                             sc::Number = 1.0, so::Number = 1.0,
                             alg::WfF_ProjG = ErrorWeightFinaliser(),
                             slots...)::JuMPWeightFinaliser
    # The default bounds of a block are not `nothing`, and a finaliser holds none.
    cbe = if isnothing(cbe)
        constraint_block(cbe; wb = nothing, slots...)
    else
        constraint_block(cbe; slots...)
    end
    return JuMPWeightFinaliser(slv, cbe, sc, so, alg)
end
# Each slot of the block is a property of the finaliser, as it is of every owner (ADR 0237).
@forward_properties JuMPWeightFinaliser begin
    forward(cbe, wb, sbgt, gbgt, lt, st, lcse, cte, gcarde, sgcarde, smtx, sgmtx, slt, sst,
            sglt, sgst, conc, tn, sets, tr, ple, ccnt, cobj, ss, card, scard, l2c, lpc,
            linfc, l1, l2, lp, linf, xbgt; settable)
end
function needs_previous_weights(wf::JuMPWeightFinaliser)::Bool
    return needs_previous_weights(wf.cbe)
end
function name_keyed(wf::JuMPWeightFinaliser)
    return name_keyed(wf.cbe)
end
"""
    reader_frames(wf::JuMPWeightFinaliser, path::AbstractString, refit::Bool) -> ReturnsFrames

Walk the readers in the block of a [`JuMPWeightFinaliser`](@ref), for the frame walk of the head that holds it.

# Arguments

  - `wf`: The weight finaliser.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the slots of the block.

# Related

  - [`field_frames`](@ref)
  - [`reader_frames`](@ref)
"""
function reader_frames(wf::JuMPWeightFinaliser, path::AbstractString, refit::Bool)
    return field_frames(wf, path, refit)
end
function finaliser_readers(wf::JuMPWeightFinaliser)::Tuple
    return wf.ple, wf.cte
end
function investable_finaliser(wf::JuMPWeightFinaliser, sets, strict::Bool, imsk, X::MatNum)
    cbe = resolve_constraint_block_names(rebuild_struct(wf.cbe, (; sets = sets)), strict,
                                         eltype(X))
    wf = rebuild_struct(wf, (; cbe = rebuild_struct(cbe, (; sets = nothing))))
    return isnothing(imsk) ? wf : port_opt_view(wf, findall(imsk), X)
end
function fit_finaliser_readers(wf::JuMPWeightFinaliser, pr::AbstractPriorResult,
                               rd::ReturnsResult, strict::Bool)
    if !any(fitted_on_rows, finaliser_readers(wf))
        return wf
    end
    cte = centrality_constraints(wf.cte, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd,
                                 strict = strict)
    ple = phylogeny_constraints(wf.ple, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd)
    return rebuild_struct(wf, (; cbe = rebuild_struct(wf.cbe, (; cte = cte, ple = ple))))
end
function assert_prior_free_finaliser(wf::JuMPWeightFinaliser, head::Symbol)::Nothing
    @argcheck(!block_reads_prior(wf.cbe),
              IsNothingError("a slot of the `JuMPWeightFinaliser` reads a prior, and `$head` has none: a tracking error, a centrality or phylogeny estimator, an exposure row or a Calibration Rule. Use a head that fits a prior, such as `InverseVolatility`, or state the slot as values."))
    return nothing
end
"""
    projection_solver(alg::WfF_ProjG, wf::JuMPWeightFinaliser)
    projection_solver(proj::GramProjection, wf::JuMPWeightFinaliser)

Returns the solver that the programme of a [`JuMPWeightFinaliser`](@ref) runs on.

A [`GramProjection`](@ref) carries its own solver, as on every Allocation Set. Every other objective uses the solver of the finaliser.

# Returns

  - `slv`: The solver, or a vector of solvers.

# Related

  - [`JuMPWeightFinaliser`](@ref)
  - [`solve_projection_programme!`](@ref)
"""
function projection_solver(::WfF_ProjG, wf::JuMPWeightFinaliser)
    return wf.slv
end
function projection_solver(proj::GramProjection, ::JuMPWeightFinaliser)
    return proj.slv
end
function risk_constraint_solver(wf::JuMPWeightFinaliser)
    return wf.slv
end
"""
    risk_contribution_constraints(r::Variance, wf::JuMPWeightFinaliser, pr)

Gives the risk contribution rows of a [`Variance`](@ref) that a tracking error of a weight finaliser tracks.

The finaliser holds no sets, so it resolves no names. The rows are values.

# Arguments

  - `r`: The variance.
  - `wf`: The weight finaliser.
  - `pr`: The prior result of the head.

# Validation

  - `r.rc` holds no estimator that is keyed by name ([`name_keyed`](@ref)). An `ArgumentError` is thrown otherwise.

# Returns

  - `rc::Option{<:LinearConstraint}`: The rows, or `nothing` when `r.rc` is `nothing`.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
function risk_contribution_constraints(r::Variance, ::JuMPWeightFinaliser, ::Any)
    @argcheck(!name_keyed(r.rc),
              ArgumentError("the risk contribution rows of a variance that a `JuMPWeightFinaliser` tracks are keyed by name, and the finaliser holds no sets: state them as a `LinearConstraint`."))
    return r.rc
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Tells whether a block holds a slot other than the bounds, the names and the settings.

A [`JuMPWeightFinaliser`](@ref) with no such slot has the programme of its bounds alone.

# Arguments

  - `cbe`: The block.

# Returns

  - `flag::Bool`: `true` when a slot other than `wb`, `sets`, `ss` and `xbgt` is not `nothing`, or when `xbgt` is `true`.

# Related

  - [`ConstraintBlock`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
function holds_block_slot(cbe::ConstraintBlock)::Bool
    slots = Base.structdiff(constraint_block_slots(cbe),
                            NamedTuple{(:wb, :sets, :ss, :xbgt)})
    return cbe.xbgt || any(!isnothing, values(slots))
end
"""
    set_clustering_weight_finaliser_alg!(model::JuMP.Model, alg::ErrorWeightFinaliser,
                                         wi::VecNum)
    set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                         proj::AbstractProjectionGeometry, wi::VecNum)

Adds the objective of a [`JuMPWeightFinaliser`](@ref) to its programme.

[`opt_weight_bounds`](@ref) adds the weights `w` and the rows before it calls this function. A [`RelativeWeightDeviation`](@ref) divides by `wi`, so [`weight_deviation`](@ref) replaces each zero entry of a copy of `wi` with `eps(eltype(wi))`. It does not change `wi`. The method for a geometry is [`set_projection_objective!`](@ref), which states its rows.

# JuMP formulation

The rows of the method for an [`ErrorWeightFinaliser`](@ref).

## Variables

  - `w`: The weights, read from the model.
  - `t`: The bound on the deviation, created.

## Constraints

  - The rows of [`set_norm_error_epigraph!`](@ref) for the norm `alg.alg`, which bound the norm of the residual ``\\boldsymbol{d}`` by ``t``. They are unnamed.
  - An unnamed row, in place of these rows, for a [`SquaredL2Norm`](@ref) when the model holds an Objective Penalty: ``(s_c t,\\, s_c / 2,\\, s_c \\boldsymbol{d}) \\in \\mathcal{K}_{\\mathrm{RSOC}}``, that is ``t \\geq \\lVert \\boldsymbol{d} \\rVert_2^2``. Without a penalty the norm and its square have the same minimiser.

## Objective

  - `Min`: ``s_o (t + \\pi)``. The objective pulls ``t`` down, so ``t`` equals the norm, or its square, at the optimum.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - ``\\boldsymbol{d}``: The residual of [`weight_deviation`](@ref) for `alg.dev`, ``\\boldsymbol{w} \\oslash \\tilde{\\boldsymbol{w}}_{0} - \\boldsymbol{1}`` for a [`RelativeWeightDeviation`](@ref), and ``\\boldsymbol{w} - \\boldsymbol{w}_{0}`` for an [`AbsoluteWeightDeviation`](@ref).
  - ``\\tilde{\\boldsymbol{w}}_{0}``: ``\\boldsymbol{w}_{0}`` with each zero entry replaced by `eps(eltype(wi))`, so that the division is defined.
  - ``t``: Bound on the norm of the deviation, or on its square.
  - ``\\pi``: The Objective Penalty of the block, zero when the model holds none.
  - ``\\mathcal{K}_{\\mathrm{RSOC}}``: Rotated second-order cone.
  - $(math_dict[:sc_scale])
  - $(math_dict[:so_scale])
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

# Arguments

  - `model`: The JuMP model. It must already hold `w` and the two scales `sc` and `so`.
  - `alg`: The deviation and its norm.
  - `proj`: The Projection Geometry.
  - `wi`: The weights that the optimisation produced.

# Returns

  - `nothing`.

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`AbstractProjectionGeometry`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`set_divergence_objective!`](@ref)
  - [`opt_weight_bounds`](@ref)
"""
function set_clustering_weight_finaliser_alg!(model::JuMP.Model, alg::ErrorWeightFinaliser,
                                              wi::VecNum)
    w = get_w(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, t)
    d = weight_deviation(alg.dev, w, wi, one(eltype(wi)))
    if isa(alg.alg, SquaredL2Norm) && shared_has(model, :op)
        JuMP.@constraint(model, [sc * t; sc / 2; sc * d] in JuMP.RotatedSecondOrderCone())
    else
        set_norm_error_epigraph!(model, t, d, alg.alg)
    end
    set_divergence_objective!(model, t, get_objective_scale(model))
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              proj::AbstractProjectionGeometry, wi::VecNum)
    return set_projection_objective!(model, proj, wi)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gives the net budget of the programme of a [`JuMPWeightFinaliser`](@ref), the sum of the input weights.

The sum is a float, and it can pass the ceiling of a gross budget by a rounding error, such as `1 + eps` beside `gbgt = 1`. The budget is then that ceiling, so [`assert_gross_budget_admissible`](@ref) does not refuse it.

# Arguments

  - `wi`: The input weights.
  - `gbgt`: The gross budget of the block, or `nothing`.

# Returns

  - `bgt::Number`: The net budget.

# Related

  - [`weights_bound_tolerance`](@ref)
  - [`gross_budget_ceiling`](@ref)
"""
function finaliser_budget(wi::VecNum, gbgt::Option{<:Num_BgtRg})
    s = sum(wi)
    hi = gross_budget_ceiling(gbgt)
    past = !isnothing(hi) && zero(hi) < abs(s) - hi <= weights_bound_tolerance(eltype(wi))
    return past ? copysign(oftype(s, hi), s) : s
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the bounds and the budget of a [`JuMPWeightFinaliser`](@ref) with no slots to its programme.

It adds no long and short parts of the weights, because no row reads them.

# JuMP formulation

## Constraints

  - `bgt`: ``s_c (\\boldsymbol{1}^\\intercal \\boldsymbol{w} - \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}) = 0``.
  - `w_lb`, when `wb.lb` holds a finite bound: ``s_c (\\boldsymbol{w} - \\boldsymbol{l}) \\geq \\boldsymbol{0}``.
  - `w_ub`, when `wb.ub` holds a finite bound: ``s_c (\\boldsymbol{w} - \\boldsymbol{u}) \\leq \\boldsymbol{0}``.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - $(math_dict[:sc_scale])
  - ``\\boldsymbol{1}``: Vector of ones.

# Arguments

  - $(arg_dict[:model])
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `bgt`: The net budget, the sum of the input weights.

# Returns

  - `nothing`.

# Related

  - [`set_finaliser_block!`](@ref)
  - [`set_budget_constraints!`](@ref)
"""
function set_finaliser_bounds!(model::JuMP.Model, wb::WeightBounds, bgt::Number)::Nothing
    (; lb, ub) = wb
    w = get_w(model)
    sc = get_constraint_scale(model)
    set_budget_constraints!(model, bgt, w)
    if w_finite_flag(lb)
        state_set!(model, Symbol(""), :w_lb, JuMP.@constraint(model, sc * (w ⊖ lb) >= 0))
    end
    if w_finite_flag(ub)
        state_set!(model, Symbol(""), :w_ub, JuMP.@constraint(model, sc * (w ⊖ ub) <= 0))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves the block of a [`JuMPWeightFinaliser`](@ref), and adds its rows and its penalties to the programme.

# Algorithm

 1. Refuse a gross budget that the net budget and the bounds do not admit, with [`assert_gross_budget_admissible`](@ref).
 2. Expand a scalar or an absent bound to one bound for each asset. Put the bounds and the sets of the call into the block, and resolve both stages of it, giving `cbr`. A head already resolved the names on its full universe, with [`investable_finaliser`](@ref), and a resolved slot comes back unchanged.
 3. When the bounds admit a short weight and the block states no `sbgt`, write the row of the net budget here, and give the block builder no net budget. The builder would pin the long part at the net budget, which forbids a short.
 4. Add both phases of the block with the block builder [`set_block_constraints!`](@ref), with the finaliser as the owner. The builder gets the fees only beside a tracking error, so a fixed fee makes no other programme mixed-integer.
 5. Add the custom objective terms of the block with [`set_block_objective_terms!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `wf`: The weight finaliser.
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `bgt`: The net budget, the sum of the input weights.
  - $(arg_dict[:pr]) It is `nothing` for a head that fits no prior.
  - $(arg_dict[:rd])
  - $(arg_dict[:sets]) It is `nothing` when the head holds none.
  - $(arg_dict[:fees_opt])
  - $(arg_dict[:strict])

# Returns

  - `nothing`.

# Related

  - [`programme_finalised_weights`](@ref)
  - [`resolve_constraint_block_names`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`set_block_constraints!`](@ref)
"""
function set_finaliser_block!(model::JuMP.Model, wf::JuMPWeightFinaliser, wb::WeightBounds,
                              bgt::Number, pr::Option{<:AbstractPriorResult},
                              rd::Option{<:ReturnsResult}, sets::Option{<:UniverseSets},
                              fees::Option{<:Fees}, strict::Bool)::Nothing
    # A mixed-integer builder reads one bound for each asset.
    wb = weight_bounds_constraints(wb; N = length(get_w(model)), datatype = typeof(bgt))
    assert_gross_budget_admissible(bgt, wf.sbgt, wf.gbgt, wb)
    cbe = resolve_constraint_block_names(rebuild_struct(wf.cbe, (; wb = wb, sets = sets)),
                                         strict, typeof(bgt))
    cbr = resolve_constraint_block(cbe, pr, rd; slv = wf.slv, strict = strict)
    tr = bind_data_benchmark(cbe.tr, rd)
    # A net budget with no short budget pins the long part of a long-short book at that
    # budget, and so forbids a short. The bounds admit one, so the row of the net budget
    # stands alone, as on the bare projection of an Allocation Set.
    alone = isnothing(cbe.sbgt) && (w_neg_flag(wb.lb) || w_neg_flag(wb.ub))
    if alone
        set_budget_constraints!(model, bgt, get_w(model))
    end
    set_block_constraints!(model, cbr, wf, alone ? nothing : bgt,
                           isnothing(tr) ? nothing : fees; tr = tr, rd = rd)
    set_block_post_risk_constraints!(model, cbr, wf, nothing)
    set_block_objective_terms!(model, cbr, wf.alg, wf)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds and solves the programme of a [`JuMPWeightFinaliser`](@ref).

[`opt_weight_bounds`](@ref) and [`finalise_weight_bounds`](@ref) both call it. The return code of the second reads the result of the solve.

# Algorithm

 1. Return `wi` with no solve when an entry is not finite, or when the block holds no slot and `wi` breaks no bound.
 2. Refuse a slot that reads a prior when `pr` is `nothing`, with [`assert_prior_free_finaliser`](@ref), and a lower bound that the geometry does not admit, with [`assert_geometry_admits_bounds`](@ref).
 3. Build an empty model with the scales of `wf`, the row count of the prior, `k = 1` and the weights `w`.
 4. With a slot, add the resolved block with [`set_finaliser_block!`](@ref). With none, add the bounds and the budget with [`set_finaliser_bounds!`](@ref).
 5. Add the objective of `wf.alg` with [`set_clustering_weight_finaliser_alg!`](@ref).
 6. Solve with [`solve_projection_programme!`](@ref) on [`projection_solver`](@ref). Return the answer of [`programme_answer`](@ref) when the solve succeeds.
 7. Else log a warning, and return the weights of a default [`IterativeWeightFinaliser`](@ref) for `wi`.

# JuMP formulation

## Variables

  - `w`: The weights, created.

## Expressions

  - `k`: the constant one.
  - `sc`: ``s_c``.
  - `so`: ``s_o``.

## Constraints

  - The rows of [`set_finaliser_block!`](@ref) or of [`set_finaliser_bounds!`](@ref).

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:so_scale])

# Arguments

  - `wf`: The weight finaliser.
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `wi`: The weights that the optimisation produced.
  - $(arg_dict[:pr]) It is `nothing` for a head that fits no prior.
  - $(arg_dict[:rd])
  - $(arg_dict[:sets]) It is `nothing` when the head holds none.
  - $(arg_dict[:fees_opt])
  - $(arg_dict[:strict])

# Returns

  - `(w, res)::Tuple`: The finalised weights, and the result of the solve, or `nothing` when there was none.

# Related

  - [`JuMPWeightFinaliser`](@ref)
  - [`opt_weight_bounds`](@ref)
  - [`finalise_weight_bounds`](@ref)
  - [`set_block_constraints!`](@ref)
"""
function programme_finalised_weights(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum,
                                     pr::Option{<:AbstractPriorResult},
                                     rd::Option{<:ReturnsResult},
                                     sets::Option{<:UniverseSets}, fees::Option{<:Fees},
                                     strict::Bool)
    (; cbe, alg) = wf
    slotted = holds_block_slot(cbe)
    if !all(isfinite, wi) || !slotted && !weights_break_bounds(wb, wi)
        return wi, nothing
    end
    if isnothing(pr)
        assert_prior_free_finaliser(wf, :finalise_weight_bounds)
    end
    assert_geometry_admits_bounds(alg, wb)
    model = JuMP.Model()
    set_model_scales!(model, wf.sc, wf.so)
    set_model_observations!(model, isnothing(pr) ? 0 : size(pr.X, 1))
    JuMP.@expression(model, k, 1)
    state_set!(model, Symbol(""), :w,
               JuMP.@variable(model, [1:length(wi)], base_name = "w"))
    bgt = finaliser_budget(wi, cbe.gbgt)
    if slotted
        set_finaliser_block!(model, wf, wb, bgt, pr, rd, sets, fees, strict)
    else
        set_finaliser_bounds!(model, wb, bgt)
    end
    set_clustering_weight_finaliser_alg!(model, alg, wi)
    res, w = solve_projection_programme!(model, alg, wi, projection_solver(alg, wf))
    if !isnothing(w)
        return programme_answer(alg, w, wb), res
    end
    @warn("The weight finalisation model of $(alg) could not be solved, so the weights are finalised with IterativeWeightFinaliser() instead, which reads the bounds and no slot of the block.",
          maxlog = log_limit(:warn), _id = @log_id(typeof(alg)))
    return opt_weight_bounds(IterativeWeightFinaliser(), wb, wi), res
end
"""
    opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum, pr = nothing,
                      rd = nothing, sets = nothing, fees = nothing, strict::Bool = false) -> VecNum

Moves a weight vector into the bounds `wb` and into the feasible set of the block of `wf`, and keeps the sum of the vector.

[`programme_finalised_weights`](@ref) states the programme. When the solve fails, the method logs a warning and gives the weights of a default [`IterativeWeightFinaliser`](@ref).

# Arguments

  - `wf`: The weight finaliser.
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `wi`: The weights that the optimisation produced.
  - $(arg_dict[:pr]) It is `nothing` for a head that fits no prior.
  - $(arg_dict[:rd])
  - $(arg_dict[:sets]) It is `nothing` when the head holds none.
  - $(arg_dict[:fees_opt])
  - $(arg_dict[:strict])

# Returns

  - `w::VecNum`: The finalised weights. They are the input when the block holds no slot and the input breaks no bound.

# Related

  - [`JuMPWeightFinaliser`](@ref)
  - [`programme_finalised_weights`](@ref)
  - [`finalise_weight_bounds`](@ref)
"""
function opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum,
                           pr = nothing, rd = nothing, sets = nothing, fees = nothing,
                           strict::Bool = false)
    return first(programme_finalised_weights(wf, wb, wi, pr, rd, sets, fees, strict))
end
"""
    finalise_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum,
                           pr = nothing, rd = nothing, sets = nothing, fees = nothing,
                           strict::Bool = false)

Moves the weights into their bounds and into the feasible set of the block of `wf`, and gives the return code that states whether the move succeeded.

# Algorithm

 1. Solve the programme with [`programme_finalised_weights`](@ref), giving `w` and the result `res` of the solve.
 2. When the block holds a slot and there was a solve, the solve decides. A solve that succeeds gives an [`OptimisationSuccess`](@ref), because the solver met every row to its own tolerance. A solve that fails gives an [`OptimisationFailure`](@ref) that holds the solver trials.
 3. Otherwise give the return code of [`weight_bounds_retcode`](@ref).

# Arguments

  - `wf`: The weight finaliser.
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `wi`: The weights that the optimisation produced.
  - $(arg_dict[:pr]) It is `nothing` for a head that fits no prior.
  - $(arg_dict[:rd])
  - $(arg_dict[:sets]) It is `nothing` when the head holds none.
  - $(arg_dict[:fees_opt])
  - $(arg_dict[:strict])

# Returns

  - `(retcode, w)`: The return code and the finalised weights.

# Related

  - [`JuMPWeightFinaliser`](@ref)
  - [`programme_finalised_weights`](@ref)
  - [`weight_bounds_retcode`](@ref)
"""
function finalise_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum,
                                pr = nothing, rd = nothing, sets = nothing, fees = nothing,
                                strict::Bool = false)
    w, res = programme_finalised_weights(wf, wb, wi, pr, rd, sets, fees, strict)
    retcode = if isnothing(res) || !holds_block_slot(wf.cbe)
        weight_bounds_retcode(wf, wb, w, sum(wi))
    elseif res.success
        OptimisationSuccess()
    else
        OptimisationFailure(; res = res.trials)
    end
    return retcode, w
end

export JuMPWeightFinaliser
public set_clustering_weight_finaliser_alg!
