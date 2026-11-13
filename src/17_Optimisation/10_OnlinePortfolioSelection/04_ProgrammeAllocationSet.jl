"""
$(DocStringExtensions.TYPEDEF)

Constrains an online allocation with every constraint kind of a JuMP optimiser, and projects onto the set with a solver.

The set holds its constraint kinds and its direct objective penalties in a [`ConstraintBlock`](@ref), the block that [`JuMPOptimiser`](@ref) holds, in its field `cbe`. Each slot of the block is a keyword and a property of the set, so `set.wb` reads `set.cbe.wb`. The risk ceilings `r` and the return floor `ret` are fields of the set, and not slots of the block. A projection onto the set is a programme, so `slv` has no default. The budget is one, and the set owns it, so the set has no budget field. Cash is an asset whose price relative is one, and the rule allocates it as any other asset.

The projection is a bare JuMP model with the variables and rows of the Weight Finaliser, which are `w`, `k = 1`, `Σw = 1`, the constraint scale and the objective scale. [`set_allocation_set_constraints!`](@ref) adds the constraint kinds through the builders of the optimiser, in the order that [`assemble_jump_model!`](@ref) runs them. [`set_projection_objective!`](@ref) sets the objective, the divergence of the rule's Projection Geometry plus the set's penalties.

The block resolves in the two stages of a JuMP head. Once per fold, [`resolve_allocation_set`](@ref) runs stage 1, [`resolve_constraint_block_names`](@ref), and resolves `wb` when it reads no rows: the slots that hold a name over `sets` and whose value follows a view at the Investable Mask. These are `wb`, `lt`, `st`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst` and `tn`. At every step, [`resolve_allocation_set_step`](@ref) runs stage 2, [`resolve_constraint_block`](@ref), on the set viewed at the mask of the step and on the prior result of `pe` fitted on the head's rows. It gives the [`ResolvedConstraintBlock`](@ref) of the set, which the builders read. Stage 2 resolves these kinds:

  - `wb`, when it reads the returns ([`bounds_read_rows`](@ref)), as a [`ScoreWeightBounds`](@ref) does. Each step scores the assets on the head's rows at the Investable Mask of the step, as the batch fit of a JuMP head on those rows does, so the bound reads no row after the step. The bounds rows, the floor of a barrier geometry and the read-out of the head all read this bound of the step.

  - `lcse` and `gcarde`, the linear rows. A linear constraint does not follow a view at the mask, so they resolve after it, over the `sets` that carry the Non-Investable Axis. An exposure row in `lcse` takes its loadings by the precedence of [`factor_space_regression`](@ref): the result that its `FactorSpace` states, else the loadings `rr` of the set's prior, else a fit of the regression of the space on the head's rows.

  - `cte`, the centrality rows, and `ple`, the integer and semidefinite phylogeny kinds.

  - A Calibration Rule in `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp` or `linf`, against the prior and the solver `slv`.

The settings `sbgt`, `gbgt`, `xbgt`, `card`, `scard`, `ss`, `ccnt` and `cobj` need no resolution. Each kind goes to the builder that [`JuMPOptimiser`](@ref) gives it to. The fields of the set read the prior and the rows of its block Result at every step:

  - `r`, the risk ceilings. Each measure goes through its own JuMP builder, with the set as the [`RiskConstraintOwner`](@ref) and `settings.ub` as the ceiling. [`factory`](@ref) first materialises the measure against the prior, so a moment that the measure carries is the moment its row uses. A [`Variance`](@ref) or [`StandardDeviation`](@ref) that holds its matrix reads no rows and no prior. It goes through the same builder with no prior, so the source of the matrix does not change the formulation ([`set_allocation_risk_ceiling!`](@ref)).
  - `ret`, a floor on the prior's expected return of the allocation.

The tracking errors `tr` also read the prior and the rows of the block Result.

At every step [`factory`](@ref) replaces the reference of a turnover ceiling, and of a [`WeightsTracking`](@ref), with the Price-Adjusted Allocation of the step. That is the book the fund trades from. A `fixed` object keeps the caller's reference.

**Each reader of the set states its own source.** The set holds no source. It gives the head's rows to its prior `pe`, to each graph slot in `cte` and `ple`, and to the regression of each exposure slot in `lcse`. Each one reads the rows as its own `x_src` states, as the same slot of a [`JuMPOptimiser`](@ref) does. When `pe` states `x_src = ExcessSource()`, the prior fits the rows on `X - B`, which [`benchmark_relative_returns`](@ref) gives. A [`Variance`](@ref) ceiling then becomes a ceiling on the tracking variance against the benchmark `B` that the rows carry. The head's rows keep the raw returns and the benchmark, so each reader reads its own source, whatever the source of the head's step is.

The readers of the set join the group of the head, see [`group_frames`](@ref). A [`DataBenchmarkTracking`](@ref) in `tr` or `r` refuses a `pe` that states `ExcessSource()`, because it would subtract the benchmark twice. A reader of the Excess Frame refuses rows with no benchmark.

A set that reads the rows, for which [`rows_needed`](@ref) returns `nothing`, fits its prior once over the whole prefix at every step, and then solves its programme. A caller who wants a window states it on `pe`. A covariance of one observation does not exist. So while the head holds fewer than two rows, the set fits no prior, and the step is a Held Step with its reason recorded ([`allocation_set_ready`](@ref)). A caller who wants a covariance ceiling from the first row gives `r` its matrix. A MIP projection is not unique. So the identity between the Causal Pass and the Recursion Result is a property of the code path, the same solves in the same order on the same rows, and it holds exactly only with a deterministic solver.

The set has no field for a budget, fees, the return term as an objective, the scalariser or the execution settings of the optimiser, and no slot of its block holds a `TimeDependent` schedule. At construction [`assert_risk_ceiling`](@ref) refuses a frontier or a per-asset `ub` on a measure, because a ceiling is one number. The set does not read the `rke` or the `scale` of a measure. The ceiling is a constraint, and the objective is the divergence plus the penalties. The Euclidean, Gram and diagonal geometries admit a negative lower bound. The entropic, Tsallis and log-barrier geometries refuse it, at the head's construction when the bound is a value ([`assert_geometry_admits_set`](@ref)) and at the projection when an estimator resolves it. The wealth factor of a leveraged allocation can reach zero on an extreme day. At that point the log wealth and the next gradient are undefined, and the set does not guard against it.

# Mathematical definition

```math
\\begin{align}
\\mathcal{W} &= \\left\\{ \\boldsymbol{w} \\in \\mathbb{R}^N : \\boldsymbol{1}^\\intercal \\boldsymbol{w} = 1,\\; \\boldsymbol{w} \\in \\mathcal{C}_k \\text{ for every kind } k \\text{ of the set} \\right\\}\\,, \\\\
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q}) + \\pi(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:W_aset])
  - $(math_dict[:w_port])
  - $(math_dict[:N])
  - ``\\mathcal{C}_k``: Feasible set of the constraint kind ``k``, which the builder of [`JuMPOptimiser`](@ref) for that kind writes. A risk ceiling, a tracking error, a centrality row, a phylogeny kind, an exposure row and a return floor take ``\\mathcal{C}_k`` from the prior of the step, and a turnover ceiling takes it from the Price-Adjusted Allocation.
  - $(math_dict[:w_plus_proj])
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:q_raw])
  - $(math_dict[:pi_obj_pen])

[`set_projection_objective!`](@ref) states the divergence of each geometry.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProgrammeAllocationSet(;
        slv::Slv_VecSlv,
        pe::AbstractPriorEstimator = EmpiricalPrior(),
        cbe::Option{<:ConstraintBlock} = nothing,
        r::Option{<:RM_VecRM} = nothing,
        ret::Option{<:JRE_VecJRE} = nothing,
        sc::Number = 1,
        so::Number = 1,
        slots...,
    ) -> ProgrammeAllocationSet

Keywords correspond to the struct's fields and to the slots of [`ConstraintBlock`](@ref). `slv` has no default, because every projection onto the set is a programme. `slots...` takes each keyword of the block with its type and its default: `wb`, `sbgt`, `gbgt`, `lt`, `st`, `lcse`, `cte`, `gcarde`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn`, `sets`, `tr`, `ple`, `ccnt`, `cobj`, `ss`, `card`, `scard`, `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp`, `linf` and `xbgt`. The constructor builds the block in `cbe` from these flat keywords, as the keyword constructor of [`JuMPOptimiser`](@ref) does. When the caller gives `cbe`, a flat keyword replaces that slot of the given block, and a slot that the caller does not name keeps its value, see [`constraint_block`](@ref). A name that is neither a field nor a slot raises a `MethodError`.

`lcse`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn` and `ple` take a vector that mixes the members of their element type. The keyword constructor of the block narrows it through [`narrow_mixed_vector`](@ref).

The positional constructor takes the fields in the order of the struct: `pe`, `slv`, `cbe`, `r`, `ret`, `sc` and `so`. Before the block, it took one argument for each slot.

## Validation

The constructor of [`ConstraintBlock`](@ref) checks each slot on its own: the values of `sbgt`, `gbgt`, `card`, `l2c`, `linfc`, `l1` and `linf`, an empty vector in a slot, the roles of `lp` and `lpc`, the shapes of the sub-group MIP slots, and `sets` beside a slot keyed by name ([`name_keyed`](@ref)). This constructor checks the fields of the set, and the rules that read the set and its block together.

  - If `slv` is a vector: `!isempty(slv)`. An `IsEmptyError` is thrown otherwise.
  - No slot of `cbe` holds a [`TimeDependent`](@ref). No fold loop resolves a schedule inside an Allocation Set, so the constructor refuses it with an `ArgumentError`. Schedule the whole set instead, `OnlinePortfolioSelection(; set = TimeDependent([…]))`.
  - If `gbgt` is given: `sbgt` is not a number, because the budget of one and a number in `sbgt` already fix the gross exposure. If `wb` is a [`WeightBounds`](@ref), one of its bounds admits a short weight. If `gbgt` has an upper end, it is at least one, because the gross exposure is at least the budget of one ([`assert_gross_budget_admissible`](@ref)). An `ArgumentError` is thrown otherwise.
  - If `r` is given: the `settings.ub` of every measure is a finite non-negative number, the ceiling, and no measure holds a [`ReturnsTracking`](@ref), because the ceiling reads rows that change at every step ([`assert_risk_ceiling`](@ref)). An `ArgumentError` is thrown otherwise, and an `IsEmptyError` for an empty vector.
  - If `ret` is given: the `settings.lb` of every return term is `nothing` or a number, the floor, because the projection solves once and returns one allocation ([`assert_return_floor`](@ref)). An `ArgumentError` is thrown for a [`Frontier`](@ref) or a vector. An empty vector raises an `IsEmptyError`.
  - If `tr` or `r` holds a [`DataBenchmarkTracking`](@ref): `pe` must not state `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).

## View parameters

`ProgrammeAllocationSet` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method views the block through `port_opt_view(cbe::ConstraintBlock, i, args...)`, which passes the arguments after `i` to the tracking errors `tr` alone. It views the risk ceilings `r` with `nothing` in place of those arguments.
  - The prior estimator `pe`, the return floor `ret` and the risk ceilings recurse through [`port_opt_view`](@ref). The block views its per-asset slots, and keeps `cte`, `gcarde`, `sgcarde` and `ple`, as it does in [`JuMPOptimiser`](@ref). A precomputed [`LinearConstraint`](@ref) is the identity under a view.
  - When a sub-group slot and its sub-grouped slot hold one object, such as `smtx` and `sgmtx`, the block views it once, and the two slots stay one object.

# Examples

```jldoctest
julia> ProgrammeAllocationSet(; slv = Solver(; solver = nothing),
                              tn = Turnover(; w = fill(0.25, 4), val = 0.1))
ProgrammeAllocationSet
   pe ┼ EmpiricalPrior
      │           ce ┼ PortfolioOptimisersCovariance
      │              │   ce ┼ Covariance
      │              │      │    me ┼ SimpleExpectedReturns
      │              │      │       │   w ┴ nothing
      │              │      │    ce ┼ GeneralCovariance
      │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │              │      │       │    w ┴ nothing
      │              │      │   alg ┼ FullMoment()
      │              │      │     w ┴ nothing
      │              │   mp ┼ MatrixProcessing
      │              │      │     pdm ┼ Posdef
      │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │              │      │      dn ┼ nothing
      │              │      │      dt ┼ nothing
      │              │      │     alg ┼ nothing
      │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │           me ┼ SimpleExpectedReturns
      │              │   w ┴ nothing
      │      horizon ┼ nothing
      │   fill_limit ┼ nothing
      │        x_src ┴ nothing
  slv ┼ Solver
      │          name ┼ String: \"\"
      │        solver ┼ nothing
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
  cbe ┼ ConstraintBlock
      │        wb ┼ WeightBounds
      │           │   lb ┼ Float64: 0.0
      │           │   ub ┴ Float64: 1.0
      │      sbgt ┼ nothing
      │      gbgt ┼ nothing
      │        lt ┼ nothing
      │        st ┼ nothing
      │      lcse ┼ nothing
      │       cte ┼ nothing
      │    gcarde ┼ nothing
      │   sgcarde ┼ nothing
      │      smtx ┼ nothing
      │     sgmtx ┼ nothing
      │       slt ┼ nothing
      │       sst ┼ nothing
      │      sglt ┼ nothing
      │      sgst ┼ nothing
      │        tn ┼ Turnover
      │           │       w ┼ Vector{Float64}: [0.25, 0.25, 0.25, 0.25]
      │           │     val ┼ Float64: 0.1
      │           │   fixed ┴ Bool: false
      │      sets ┼ nothing
      │        tr ┼ nothing
      │       ple ┼ nothing
      │      ccnt ┼ nothing
      │      cobj ┼ nothing
      │        ss ┼ nothing
      │      card ┼ nothing
      │     scard ┼ nothing
      │       l2c ┼ nothing
      │       lpc ┼ nothing
      │     linfc ┼ nothing
      │        l1 ┼ nothing
      │        l2 ┼ nothing
      │        lp ┼ nothing
      │      linf ┼ nothing
      │      xbgt ┴ Bool: false
    r ┼ nothing
  ret ┼ nothing
   sc ┼ Int64: 1
   so ┴ Int64: 1
```

# Related

  - [`AbstractAllocationSet`](@ref)
  - [`BoundedAllocationSet`](@ref): the set of weight bounds alone, with a scalar root for every geometry but the Gram one.
  - [`JuMPOptimiser`](@ref)
  - [`project`](@ref)
  - [`projection_programme`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`set_projection_objective!`](@ref)
  - [`resolve_allocation_set`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`HeldStep`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct ProgrammeAllocationSet <: AbstractProgrammeAllocationSet
    """
    The prior estimator that the set fits on the head's rows at every step. The risk ceilings, the tracking errors, the centrality and phylogeny rows, an exposure row and the return floor use its result. The set does not read it when no slot reads the rows.
    """
    pe
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:cbe]) In a programme set, the turnover ceilings `tn` and a [`WeightsTracking`](@ref) in `tr` that is not `fixed` take the Price-Adjusted Allocation of the step as their reference. A [`ReturnsTracking`](@ref) series in `tr` that is stated over the fold is cut to the prefix that the head has folded, and the centrality rows `cte` are resolved on the head's rows at every step.
    """
    cbe
    """
    The risk ceilings, a [`RiskMeasure`](@ref) or a vector of them, each with `settings.ub` the ceiling, or `nothing`.
    """
    r
    """
    The return floor: a return term, or a vector of them, whose `settings.lb` bounds the prior's expected return of the projected allocation from below; the term's objective role is not read.
    """
    ret
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    function ProgrammeAllocationSet(pe::AbstractPriorEstimator, slv::Slv_VecSlv,
                                    cbe::ConstraintBlock, r::Option{<:RM_VecRM},
                                    ret::Option{<:JRE_VecJRE}, sc::Number, so::Number)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        # The constructor of the block checks each slot on its own. The checks below read a
        # field of the set, or a rule that the set adds to the slots of the block. A bound
        # that reads the returns resolves at stage 2, on the rows of each step (#2059).
        tdfs = typed_time_dependent_fields(cbe)
        @argcheck(isempty(tdfs),
                  ArgumentError("the slot `$(first(tdfs))` of a `ProgrammeAllocationSet` holds a `TimeDependent` schedule, and no fold loop resolves a schedule inside an Allocation Set: the online head resolves a schedule of the whole set. Schedule the set, `OnlinePortfolioSelection(; set = TimeDependent([...]))`, and give each entry a static slot."))
        assert_gross_budget_admissible(1, cbe.sbgt, cbe.gbgt, cbe.wb)
        assert_risk_ceiling(r)
        assert_return_floor(ret)
        if isa(ret, AbstractVector)
            @argcheck(!isempty(ret), IsEmptyError("ret cannot be empty"))
        end
        assert_data_benchmark_prior(cbe.tr, pe, :tr, :ProgrammeAllocationSet)
        assert_data_benchmark_prior(r, pe, :r, :ProgrammeAllocationSet)
        return new{typeof(pe), typeof(slv), typeof(cbe), typeof(r), typeof(ret), typeof(sc),
                   typeof(so)}(pe, slv, cbe, r, ret, sc, so)
    end
end
function ProgrammeAllocationSet(; slv::Slv_VecSlv,
                                pe::AbstractPriorEstimator = EmpiricalPrior(),
                                cbe::Option{<:ConstraintBlock} = nothing,
                                r::Option{<:RM_VecRM} = nothing,
                                ret::Option{<:JRE_VecJRE} = nothing, sc::Number = 1,
                                so::Number = 1, slots...)::ProgrammeAllocationSet
    return ProgrammeAllocationSet(pe, slv, constraint_block(cbe; slots...), r, ret, sc, so)
end
# Each slot of the block is a property of the set, and a patch of `Accessors.set` can set
# it, so `set.wb` keeps its meaning (ADR 0237). `r` and `ret` are fields of the set.
@forward_properties ProgrammeAllocationSet begin
    forward(cbe, wb, sbgt, gbgt, lt, st, lcse, cte, gcarde, sgcarde, smtx, sgmtx, slt, sst,
            sglt, sgst, tn, sets, tr, ple, ccnt, cobj, ss, card, scard, l2c, lpc, linfc, l1,
            l2, lp, linf, xbgt; settable)
end
function port_opt_view(set::ProgrammeAllocationSet, i, args...)
    return rebuild_estimator(set,
                             (; pe = port_opt_view(set.pe, i),
                              cbe = port_opt_view(set.cbe, i, args...),
                              r = port_opt_view(set.r, i, nothing),
                              ret = port_opt_view(set.ret, i)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `set` whose parts that read the previous weights hold the Price-Adjusted Allocation `w` of a step.

The block takes `w` through `factory(cbe::ConstraintBlock, w)`: its turnover ceilings, its tracking errors, its custom constraints and its custom objective terms. The risk ceilings `r` take it through `factory(r, w)`, so a [`WeightsTracking`](@ref) benchmark or a [`TurnoverRiskMeasure`](@ref) reference that is not fixed reads the book of the step. A `fixed` object keeps the caller's reference. The return floor reads no weights. [`resolve_allocation_set_step`](@ref) calls it at each step, before stage 2 of the block.

# Arguments

  - `set`: The programme set.
  - `w`: The Price-Adjusted Allocation of the step.

# Returns

  - `set'::ProgrammeAllocationSet`: The set with the parts rebuilt for `w`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`resolve_allocation_set_step`](@ref)
"""
function factory(set::ProgrammeAllocationSet, w::AbstractVector)::ProgrammeAllocationSet
    return rebuild_estimator(set, (; cbe = factory(set.cbe, w), r = factory(set.r, w)))
end
# The set declares the Non-Investable Axis on its `sets` after the view at the mask of a step,
# as a JuMP head does in `investable_reduction` (`programme_investable_reduction`).
function non_investable_universe(set::ProgrammeAllocationSet,
                                 ni::VecStr)::ProgrammeAllocationSet
    return rebuild_estimator(set, (; sets = non_investable_sets(set.sets, ni)))
end
"""
    reader_frames(set::ProgrammeAllocationSet, path::AbstractString, refit::Bool) -> ReturnsFrames

Walk the readers of a [`ProgrammeAllocationSet`](@ref) for the group of the head that holds it.

The set holds no source of its own. Its prior `pe`, each graph slot in `cte` and `ple`, the regression of each exposure slot in `lcse`, and the readers inside its measures are the outermost readers of the head's rows, so the walk counts each one. [`field_frames`](@ref) walks every field, and it enters the [`ConstraintBlock`](@ref) in `cbe` through `reader_frames(cbe::ConstraintBlock, path, refit)`. So the path of a slot passes through the block, as `"set.cbe.ple"`. The walk reads the refit rule of the set's own prior, because a regression reader of the set reads the loadings of that prior, see [`refits_regression`](@ref).

# Arguments

  - `set`: The programme set.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit]) The set ignores it and reads the rule of its own prior.

# Returns

  - `fr::ReturnsFrames`: The frames that the set adds to the group.

# Related

  - [`field_frames`](@ref)
  - [`group_frames`](@ref)
"""
function reader_frames(set::ProgrammeAllocationSet, path::AbstractString, ::Bool)
    return field_frames(set, path, refits_regression(set.pe))
end
"""
    name_keyed(x)
    name_keyed(x::AbstractVector)

Tells whether a constraint slot holds an estimator that resolves its names over `sets`.

The owner of such a slot must carry `sets`. The estimators are every [`AbstractWeightBoundsEstimator`](@ref), [`LinearConstraintEstimator`](@ref), [`ExposureConstraintEstimator`](@ref), [`ThresholdEstimator`](@ref), [`AssetSetsMatrixEstimator`](@ref) and [`TurnoverEstimator`](@ref). A [`ScoreWeightBounds`](@ref) is the exception: it is keyed by name only when its `val` or its static bounds name assets. A vector is keyed by name when one of its entries is.

# Returns

  - `::Bool`: `true` when the slot holds such an estimator.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function name_keyed(::Any)
    return false
end
function name_keyed(::Union{AbstractWeightBoundsEstimator, LinearConstraintEstimator,
                            ExposureConstraintEstimator, ThresholdEstimator,
                            AssetSetsMatrixEstimator, TurnoverEstimator})
    return true
end
function name_keyed(x::AbstractVector)
    return any(name_keyed, x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves the slots of a [`ProgrammeAllocationSet`](@ref) that hold a name or an estimator, once per fold.

This is stage 1 of the resolution of the set's block, [`resolve_constraint_block_names`](@ref), the stage that a [`JuMPOptimiser`](@ref) runs on its own block. The online head runs it once for each fold, before it knows the Investable Mask of a step. At every step, [`resolve_allocation_set_step`](@ref) runs stage 2, [`resolve_constraint_block`](@ref), on the set viewed at the mask of the step. So the set orders the names and the reduction by the rule of a JuMP head.

The weight bounds `wb` are a stage 2 slot of the block, because their resolver can read the rows. A bound that reads only names, a [`WeightBounds`](@ref) or a [`WeightBoundsEstimator`](@ref), resolves here, over the `N` assets, once per fold. Stage 2 then gives the resolved bounds back unchanged. A bound that reads the rows, such as a [`ScoreWeightBounds`](@ref), stays in the set, and stage 2 resolves it at each step on the head's rows ([`bounds_read_rows`](@ref)).

# Algorithm

 1. When `wb` reads no rows, resolve it over the `N` assets with [`weight_bounds_constraints`](@ref). Otherwise keep it.
 2. Resolve stage 1 of the block with [`resolve_constraint_block_names`](@ref): the thresholds, `sgcarde`, the sub-group matrices and thresholds, and the turnover ceilings `tn`.
 3. Return `set` rebuilt with [`rebuild_estimator`](@ref), with the resolved block in `cbe` and the resolved bounds in its slot `wb`. Every other field keeps its value.

# Arguments

  - `set`: The programme set.
  - `N`: The number of assets.
  - $(arg_dict[:strict])
  - `datatype`: The element type of the resolved values. The sub-grouped cardinality constraints take `Int`.

# Returns

  - `set'::ProgrammeAllocationSet`: The set with its stage 1 slots resolved.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`resolve_constraint_block_names`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function resolve_allocation_set(set::ProgrammeAllocationSet, N::Integer, strict::Bool,
                                datatype::DataType)
    wb = if bounds_read_rows(set.wb)
        set.wb
    else
        weight_bounds_constraints(set.wb, set.sets; N = N, strict = strict,
                                  datatype = datatype)
    end
    cbe = resolve_constraint_block_names(set.cbe, strict, datatype)
    # The block and its slot `wb` go in one call, so the flat keyword replaces that slot of
    # the resolved block.
    return rebuild_estimator(set, (; cbe = cbe, wb = wb))
end
"""
    exposure_keyed(x)
    exposure_keyed(x::AbstractVector)

Tells whether a linear-constraint slot holds an [`ExposureConstraintEstimator`](@ref).

An exposure estimator writes its rows in the basis of the factors, and the loadings of a factor prior take them to the assets. So such a slot resolves on the head's rows at every step, and not once per fold. A vector is keyed by exposure when one of its entries is.

# Returns

  - `::Bool`: `true` when the slot holds an exposure estimator.

# Related

  - [`resolve_allocation_set`](@ref)
  - [`resolve_constraint_block`](@ref)
"""
function exposure_keyed(::Any)
    return false
end
function exposure_keyed(::ExposureConstraintEstimator)
    return true
end
function exposure_keyed(x::AbstractVector)
    return any(exposure_keyed, x)
end
"""
    fitted_on_rows(x)
    fitted_on_rows(x::AbstractVector)

Tells whether a centrality or phylogeny slot holds an estimator that the set fits on the head's rows at every step.

A precomputed [`LinearConstraint`](@ref), [`IntegerPhylogeny`](@ref) or [`SemiDefinitePhylogeny`](@ref) reads no rows. A vector reads the rows when one of its entries does.

# Returns

  - `::Bool`: `true` when the slot holds a centrality or phylogeny estimator.

# Related

  - [`rows_needed`](@ref)
  - [`resolve_constraint_block`](@ref)
"""
function fitted_on_rows(::Any)
    return false
end
function fitted_on_rows(::Union{AbstractCentralityConstraint,
                                AbstractPhylogenyConstraintEstimator})
    return true
end
function fitted_on_rows(x::AbstractVector)
    return any(fitted_on_rows, x)
end
"""
    bounds_read_rows(wb)
    bounds_read_rows(wb::AbstractWeightBoundsEstimator)
    bounds_read_rows(wb::WeightBoundsEstimator)

Tells whether the weight bounds of a programme set read the head's rows.

A [`WeightBounds`](@ref), a [`WeightBoundsEstimator`](@ref) and `nothing` read no rows, so [`resolve_allocation_set`](@ref) resolves them once per fold. Every other [`AbstractWeightBoundsEstimator`](@ref), such as a [`ScoreWeightBounds`](@ref), can read the returns of the fit. Stage 2, [`resolve_constraint_block`](@ref), resolves it at each step on the head's rows, so it reads no row after the step.

# Returns

  - `::Bool`: `true` when the bounds read the rows.

# Related

  - [`rows_needed`](@ref)
  - [`resolve_allocation_set`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
"""
function bounds_read_rows(::Any)
    return false
end
function bounds_read_rows(::AbstractWeightBoundsEstimator)
    return true
end
function bounds_read_rows(::WeightBoundsEstimator)
    return false
end
"""
    rows_needed(set::ProgrammeAllocationSet)

Returns the number of rows that a programme set reads at a step.

The set reads every row that the head has folded when one of its slots reads the rows. These slots are weight bounds that read them ([`bounds_read_rows`](@ref)), a risk ceiling that reads them ([`risk_reads_rows`](@ref)), a tracking error, a centrality or phylogeny estimator ([`fitted_on_rows`](@ref)), an exposure row ([`exposure_keyed`](@ref)), a return floor, and a Calibration Rule in a norm ceiling or in a penalty ([`calibrated`](@ref)).

# Returns

  - `::Union{Nothing, Int}`: `nothing`, every row folded, when a slot reads the rows, and `0` otherwise.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`rows_needed`](@ref)
  - [`risk_reads_rows`](@ref)
"""
function rows_needed(set::ProgrammeAllocationSet)
    reads = bounds_read_rows(set.wb) ||
            risk_reads_rows(set.r) ||
            !isnothing(set.tr) ||
            any(fitted_on_rows, (set.cte, set.ple)) ||
            exposure_keyed(set.lcse) ||
            !isnothing(set.ret) ||
            any(calibrated, (set.l2c, set.lpc, set.linfc, set.l1, set.l2, set.lp, set.linf))
    return reads ? nothing : 0
end
"""
$(DocStringExtensions.TYPEDEF)

Projects the raw step onto the Allocation Set in the Bregman divergence of the Tsallis potential of power `alpha`.

This is the geometry of the Tsallis-entropy steps of Section 3 of [abernethy2015](@cite) and of the Tsallis-INF algorithm of Section 3.2 of [zimmertseldin2021](@cite). The potential of [abernethy2015](@citet) is ``\\alpha \\Psi_\\alpha`` plus a constant, and the potential of [zimmertseldin2021](@citet) is ``\\Psi_\\alpha`` plus an affine term, which is constant on the simplex. So each divergence is ``D_{\\Psi_\\alpha}`` times a constant factor, which a learning rate absorbs. [abernethy2015](@citet) and [zimmertseldin2021](@citet) both bound the regret of the multi-armed bandit with bounded losses. At the tuned rate, Corollary 3.2 of [abernethy2015](@cite) bounds the expected regret by ``\\sqrt{2 T N / (\\alpha (1 - \\alpha))}``. The log-wealth loss of a portfolio has no bounded gradient, so that bound does not apply to it as it stands. The geometry cannot set a positive entry to zero, and a zero entry stays zero. The geometry refuses a negative lower bound, because the potential is undefined below zero.

# Mathematical definition

```math
\\begin{align}
\\Psi_\\alpha(\\boldsymbol{w}) &= \\frac{1}{\\alpha (\\alpha - 1)} \\sum_i w_i^{\\alpha}\\,, \\\\
\\mathrm{Proj}^{\\alpha}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_{\\Psi_\\alpha}(\\boldsymbol{w}, \\boldsymbol{q})\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(\\left(q_i^{\\alpha - 1} + \\theta\\right)^{1 / (\\alpha - 1)},\\, l_i\\right),\\, u_i\\right)\\,, \\\\
w_{t+1, i} &= \\left( w_{t, i}^{\\alpha - 1} + (1 - \\alpha) \\eta_t g_{t, i} + \\theta \\right)^{1 / (\\alpha - 1)}\\,.
\\end{align}
```

Where:

  - ``\\Psi_\\alpha``: Tsallis potential, the potential of this geometry.
  - ``\\alpha``: Power of the potential, in ``(0, 1)``.
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:g_t_loss])

The mirror image is ``\\nabla \\Psi_\\alpha(\\boldsymbol{w})_i = w_i^{\\alpha - 1} / (\\alpha - 1)``, so the projection sets ``\\nabla \\Psi_\\alpha(\\boldsymbol{w}) - \\nabla \\Psi_\\alpha(\\boldsymbol{q})`` to one constant over the free entries. On a [`BoundedAllocationSet`](@ref) the projection is ``\\boldsymbol{w}(\\theta)`` at the root. A base ``q_i^{\\alpha - 1} + \\theta`` at or below zero puts its asset at ``u_i``. The budget ``\\sum_i w_i(\\theta)`` does not increase in ``\\theta``. The last line is a first-order step from ``\\boldsymbol{w}_t`` and its projection onto the simplex together, one scalar root. As ``\\alpha \\to 1``, ``D_{\\Psi_\\alpha}`` tends to the relative entropy of [`EntropicProjection`](@ref). As ``\\alpha \\to 0``, it tends to the Itakura–Saito divergence of [`LogBarrierProjection`](@ref), as Section 3.2 of [zimmertseldin2021](@cite) shows for the potentials. On a [`ProgrammeAllocationSet`](@ref) a weight that the set forces to zero stays zero, as under [`EntropicProjection`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TsallisProjection(; alpha::Real = 0.5) -> TsallisProjection

Keywords correspond to the struct's fields. The default is the ``\\alpha = 1/2`` of Section 4.1 of [zimmertseldin2021](@cite).

## Validation

  - `0 < alpha < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> TsallisProjection()
TsallisProjection
  alpha ┴ Float64: 0.5
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`LogBarrierProjection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:abernethy2015]) Section 3, Theorem 3.1, Corollary 3.2.
  - $(ref_dict[:zimmertseldin2021]) Section 3.2. Section 4.1, Theorem 1.
"""
@concrete struct TsallisProjection <: AbstractProjectionGeometry
    """
    The power of the potential, in `(0, 1)`.
    """
    alpha
    function TsallisProjection(alpha::Real)
        @argcheck(zero(alpha) < alpha < one(alpha),
                  DomainError(alpha, "alpha must be in (0, 1)"))
        return new{typeof(alpha)}(alpha)
    end
end
function TsallisProjection(; alpha::Real = 0.5)::TsallisProjection
    return TsallisProjection(alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

Projects the raw step onto the Allocation Set in the Itakura–Saito divergence, the Bregman divergence of the Burg entropy.

In Section 7 of [orseau2017](@cite), the Burg entropy is the natural regulariser of mirror descent on the log loss, the alternative to the Soft-Bayes step of [orseau2017](@cite). [orseau2017](@citet) state without a proof that its regret is ``O(\\sqrt{N T \\log (T / N)})`` and does not depend on the largest gradient. For a portfolio, this means that the bound needs no lower bound on the price relatives. The geometry cannot set a positive entry to zero, and a zero entry stays zero. The geometry refuses a negative lower bound, because the logarithm is undefined below zero.

# Mathematical definition

```math
\\begin{align}
\\Psi(\\boldsymbol{w}) &= -\\sum_i \\log w_i\\,, \\\\
D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q}) &= \\sum_i \\left( \\frac{w_i}{q_i} - 1 - \\log \\frac{w_i}{q_i} \\right)\\,, \\\\
\\mathrm{Proj}^{\\mathrm{IS}}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q})\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(\\frac{1}{1 / q_i + \\theta},\\, l_i\\right),\\, u_i\\right)\\,, \\\\
w_{t+1, i} &= \\frac{1}{1 / w_{t, i} + \\eta_t g_{t, i} + \\theta}\\,.
\\end{align}
```

Where:

  - ``\\Psi``: Burg entropy, the potential of this geometry.
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:g_t_loss])

The mirror image is ``\\nabla \\Psi(\\boldsymbol{w})_i = -1 / w_i``, so the projection sets ``\\nabla \\Psi(\\boldsymbol{w}) - \\nabla \\Psi(\\boldsymbol{q})`` to one constant over the free entries. On a [`BoundedAllocationSet`](@ref) the projection is ``\\boldsymbol{w}(\\theta)`` at the root. A base ``1 / q_i + \\theta`` at or below zero puts its asset at ``u_i``. The budget ``\\sum_i w_i(\\theta)`` does not increase in ``\\theta``. The last line is a first-order step from ``\\boldsymbol{w}_t`` and its projection onto the simplex together, one scalar root. The Hessian of ``\\Psi`` is ``\\mathrm{diag}(1 / w_i^2)``, so the step moves a small weight in proportion to its square. On a [`ProgrammeAllocationSet`](@ref) a weight that the set forces to zero stays zero, as under [`EntropicProjection`](@ref).

# Examples

```jldoctest
julia> LogBarrierProjection()
LogBarrierProjection()
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`TsallisProjection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:orseau2017]) Section 7, Alternative approaches.
"""
struct LogBarrierProjection <: AbstractProjectionGeometry end
"""
    mirror_step(proj::EuclideanProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::EntropicProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::TsallisProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::LogBarrierProjection, u::AbstractVector, s::AbstractVector)

Takes the unconstrained mirror step from the iterate `u` along the scaled gradient `s`, in the potential of the geometry.

The answer is the raw step that [`project`](@ref) then puts onto the Allocation Set. A barrier geometry has a step only while every base is positive. A base at or below zero is a step to an unbounded allocation, and the function refuses it and does not clip it. Under the plain log-wealth gradient every rate below one has a step. A transformed gradient changes that condition. The first step of [`AdaptiveMomentGradient`](@ref) with `gamma1 = 0.9` and `gamma2 = 0.999` scales every entry of the gradient to about `-3.16`, so at the rate `0.9` the log-barrier step fails when one asset holds more than 0.352 of the iterate.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{q} &= \\nabla \\Psi^{*}\\left(\\nabla \\Psi(\\boldsymbol{u}) - \\boldsymbol{s}\\right)\\,, \\\\
q^{\\mathrm{E}}_i &= u_i - s_i\\,, \\\\
q^{\\mathrm{KL}}_i &= u_i \\exp(-s_i)\\,, \\\\
q^{\\alpha}_i &= \\left( u_i^{\\alpha - 1} + (1 - \\alpha) s_i \\right)^{1 / (\\alpha - 1)}\\,, \\\\
q^{\\mathrm{IS}}_i &= \\frac{1}{1 / u_i + s_i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:q_raw])
  - $(math_dict[:Psi_pot])
  - ``\\nabla \\Psi^{*}``: Inverse of the mirror map ``\\nabla \\Psi``.
  - ``\\boldsymbol{u}``: The iterate the step starts from.
  - ``\\boldsymbol{s}``: The scaled gradient, ``\\eta_t \\boldsymbol{g}_t`` for a plain gradient.
  - ``q^{\\mathrm{E}}``, ``q^{\\mathrm{KL}}``, ``q^{\\alpha}``, ``q^{\\mathrm{IS}}``: The raw step under [`EuclideanProjection`](@ref), [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref).
  - ``\\alpha``: Power of the Tsallis potential, in ``(0, 1)``.
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:g_t_loss])
  - $(math_dict[:x_t_rel])

Under the log-wealth loss with a plain gradient, ``\\boldsymbol{s} = -\\eta_t \\boldsymbol{x}_t / \\langle \\boldsymbol{u}, \\boldsymbol{x}_t \\rangle``. The log-barrier base ``1 / u_i + s_i`` is then positive exactly while ``\\eta_t \\hat{u}_i < 1`` for every asset, with ``\\hat{\\boldsymbol{u}} = \\boldsymbol{u} \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{u}, \\boldsymbol{x}_t \\rangle`` the Price-Adjusted Allocation of the iterate. Each ``\\hat{u}_i`` is at most one, so a rate below one always has a step. The Tsallis base is positive exactly while ``(1 - \\alpha) \\eta_t \\hat{u}_i u_i^{-\\alpha} < 1`` for every asset.

# Arguments

  - `proj`: The Projection Geometry.
  - `u`: The iterate the step starts from.
  - `s`: The scaled gradient.

# Validation

  - Under [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): every base is positive. A `DomainError` is thrown otherwise ([`assert_mirror_base`](@ref)).

# Returns

  - `q::AbstractVector`: The raw step.

# Related

  - [`MirrorDescent`](@ref)
  - [`project`](@ref)
"""
function mirror_step(::EuclideanProjection, u::AbstractVector, s::AbstractVector)
    return u .- s
end
function mirror_step(::EntropicProjection, u::AbstractVector, s::AbstractVector)
    return u .* exp.(-s)
end
function mirror_step(proj::TsallisProjection, u::AbstractVector, s::AbstractVector)
    a = proj.alpha
    base = u .^ (a - 1) .+ (1 - a) .* s
    assert_mirror_base(base, s)
    return base .^ inv(a - 1)
end
function mirror_step(::LogBarrierProjection, u::AbstractVector, s::AbstractVector)
    base = inv.(u) .+ s
    assert_mirror_base(base, s)
    return inv.(base)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a barrier mirror step that has a base at or below zero.

The error names the scaled gradient `s` that gave the base.

# Validation

  - `all(> 0, base)`. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`mirror_step`](@ref)
"""
function assert_mirror_base(base::AbstractVector, s::AbstractVector)::Nothing
    @argcheck(all(x -> x > zero(x), base),
              DomainError(s,
                          "the mirror step leaves the geometry's domain: a base of the barrier potential is not positive, so the unconstrained step is unbounded in some asset. Lower the learning rate. Under the log barrier with the plain log-wealth gradient, the step exists while `eta * w_i * x_i / <w, x> < 1` for every asset."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Projects a raw step onto a bounded set in a barrier geometry, through the scalar root of the budget.

The root is the budget multiplier `mu` at which `Σ_i clip(phi(b_i + mu), lb_i, ub_i)` is one. Here `b` is the mirror image of the raw step and `phi` is the inverse mirror map. A base at or below zero puts its asset at its cap. `phi` decreases on positive bases and has no bound as the base falls to zero, so the clipped budget does not increase in `mu` and has no jump. A zero raw entry has the base `Inf`, and it stays at its floor. [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref) state the root as ``\\boldsymbol{w}(\\theta)``.

# Algorithm

 1. Check that `Σ lb ≤ 1 ≤ Σ ub` with [`assert_feasible_bounds`](@ref).
 2. When the floors sum to one, return `lb`.
 3. Collect the finite entries of `b` in `finite`, and refuse an empty `finite`.
 4. Write the clipped budget `f(mu)`. An asset whose base `b_i + mu` is at or below zero takes `ub_i`, and every other asset takes `clamp(phi(b_i + mu), lb_i, ub_i)`.
 5. Set the lower end `lo = -maximum(finite)`, at which every finite entry is at its cap. Refuse `f(lo) < 1`.
 6. Find the upper end `hi` with [`barrier_upper_bracket`](@ref).
 7. Bisect `[lo, hi]` for `mu` with [`bounded_root`](@ref).
 8. Return the clipped allocation at `mu`.

# Arguments

  - `b`: The mirror image of the raw step, `Inf` at a zero entry.
  - `phi`: The inverse mirror map of a positive base.
  - `wb`: The resolved bounds.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub`. An `ArgumentError` is thrown otherwise.
  - At least one raw entry is positive. A `DomainError` is thrown otherwise.
  - The caps of the positive entries and the floors of the zero entries reach the budget. A `DomainError` is thrown otherwise, because a zero entry stays at its floor under a barrier.

# Returns

  - `w'::Vector`: The projected allocation, a new vector.

# Related

  - [`project`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
  - [`bounded_root`](@ref)
"""
function barrier_projection(b::AbstractVector, phi, wb::WeightBounds)
    assert_feasible_bounds(wb)
    lb, ub = wb.lb, wb.ub
    if sum(lb) >= one(eltype(lb))
        return collect(lb)
    end
    finite = filter(isfinite, b)
    @argcheck(!isempty(finite),
              DomainError(b,
                          "a barrier projection needs a positive raw entry: a raw step of zeros has no projection, because a zero stays zero under the potential"))
    clipped = (m, l, u) -> m > zero(m) ? clamp(phi(m), l, u) : u
    f = mu -> sum(i -> clipped(b[i] + mu, lb[i], ub[i]), eachindex(b, lb, ub))
    lo = -maximum(finite)
    @argcheck(f(lo) >= one(lo),
              DomainError(b,
                          "the zeros of the raw step stay zero under a barrier potential, and the caps of the remaining assets do not reach the budget"))
    mu = bounded_root(f, lo, barrier_upper_bracket(f, lo))
    return clipped.(b .+ mu, lb, ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds the upper end of the bracket of a barrier root, a positive multiplier above `lo` at which the clipped budget is at most one.

The loop stops, because the clipped budget tends to `Σ lb < 1` as the multiplier grows. [`barrier_projection`](@ref) returns `lb` before it calls this function when `Σ lb = 1`.

# Algorithm

 1. Start from `hi = max(lo, 0) + 1`.
 2. Double `hi` while `f(hi) > 1`.
 3. Return `hi`.

# Arguments

  - `f`: The clipped budget as a function of the multiplier, non-increasing.
  - `lo`: The lower end of the bracket.

# Returns

  - `hi`: The upper end of the bracket.

# Related

  - [`barrier_projection`](@ref)
"""
function barrier_upper_bracket(f, lo)
    hi = max(lo, zero(lo)) + one(lo)
    while f(hi) > one(hi)
        hi *= 2
    end
    return hi
end
"""
    project(proj::TsallisProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::LogBarrierProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)

Projects the raw step `q` onto a [`BoundedAllocationSet`](@ref) in the Tsallis or the log-barrier geometry. These are the barrier arms of the Constrained Update, and `w` is the Price-Adjusted Allocation.

Neither potential has a closed-form projection onto the simplex, so the arms use the scalar root of [`barrier_projection`](@ref) on every bound, the simplex included. A root whose allocation misses the budget in floating point gives the Held Step ([`budget_or_held_step`](@ref)). This happens, for example, under the log barrier when a raw entry is `1e-20`. Its base is `1e20`, and the multiplier that brings the entry to a share of the budget cancels against that base.

# Algorithm

 1. Refuse a negative or non-finite entry of `q` and a negative floor, with [`assert_barrier_raw_step`](@ref).
 2. Map `q` to its mirror image `b`, `q .^ (alpha - 1)` under the Tsallis geometry and `inv.(q)` under the log barrier. A zero entry maps to `Inf`.
 3. Find the allocation `wn` with [`barrier_projection`](@ref), on `b`, the inverse mirror map and the bounds of `set`.
 4. Return `wn`, or the Held Step `w` when `wn` misses the budget in floating point, with [`budget_or_held_step`](@ref).

# Arguments

  - `proj`: The barrier geometry.
  - `set`: The bounded set, its bounds resolved.
  - `q`: The raw step.
  - `w`: The Price-Adjusted Allocation the step trades from, the answer of a Held Step.

# Validation

  - `all(>= 0, q)` and every entry finite. A `DomainError` is thrown otherwise.
  - `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise.
  - Everything [`barrier_projection`](@ref) refuses.

# Returns

  - `w'::Vector`: The projected allocation, or `w` copied on a Held Step.

# Related

  - [`project`](@ref)
  - [`barrier_projection`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function project(proj::TsallisProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    assert_barrier_raw_step(q, set.wb)
    a = proj.alpha
    return budget_or_held_step(barrier_projection(q .^ (a - 1), m -> m^inv(a - 1), set.wb),
                               w, proj, set)
end
function project(proj::LogBarrierProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    assert_barrier_raw_step(q, set.wb)
    return budget_or_held_step(barrier_projection(inv.(q), inv, set.wb), w, proj, set)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses, for a barrier geometry, a raw step with a negative or non-finite entry, and a bound with a negative floor.

The potential of a barrier geometry is undefined below zero.

# Validation

  - Every entry of `q` is finite and non-negative. A `DomainError` is thrown otherwise.
  - `all(>= 0, wb.lb)`. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`project`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function assert_barrier_raw_step(q::AbstractVector, wb::WeightBounds)::Nothing
    @argcheck(all(x -> isfinite(x) && x >= zero(x), q),
              DomainError(q,
                          "a barrier projection is defined on finite non-negative raw steps alone: the potential is undefined below zero"))
    @argcheck(all(x -> x >= zero(x), wb.lb),
              DomainError(wb.lb,
                          "a barrier projection admits no negative lower bound: the potential is undefined below zero"))
    return nothing
end
"""
    assert_geometry_admits_set(proj::AbstractProjectionGeometry, set::AbstractAllocationSet)

Refuses a negative lower bound under a geometry whose potential is undefined below zero.

The head calls this function at its construction, with the geometry of its rule and its set. The entropic, Tsallis and log-barrier geometries refuse a negative bound. This function refuses a bound that is a value, through [`assert_geometry_admits_bounds`](@ref). The projection refuses a bound that an estimator resolves. The Euclidean, Gram and diagonal geometries admit a negative bound, so a long-short rule needs no other set.

# Validation

  - With a [`WeightBounds`](@ref) in `set.wb`: the rules of [`assert_geometry_admits_bounds`](@ref).

# Returns

  - `nothing`.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`assert_geometry_admits_bounds`](@ref)
  - [`project`](@ref)
"""
function assert_geometry_admits_set(proj::AbstractProjectionGeometry,
                                    set::AbstractAllocationSet)::Nothing
    wb = set.wb
    if isa(wb, WeightBounds)
        assert_geometry_admits_bounds(proj, wb)
    end
    return nothing
end
"""
    assert_geometry_admits_bounds(proj::AbstractProjectionGeometry, wb::WeightBounds)

Refuses a resolved bound with a negative floor under a geometry whose potential is undefined below zero.

[`assert_geometry_admits_set`](@ref) calls it at the construction of the head, on a bound that is a value. [`projection_programme`](@ref) calls it at each step, on the bound that the step resolves, before the solve. The entropic, Tsallis and log-barrier geometries refuse a negative floor, and every other geometry admits it.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) or [`LogBarrierProjection`](@ref) with an `lb` that is a number or a vector: `all(>= 0, lb)`. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`assert_geometry_admits_set`](@ref)
  - [`projection_programme`](@ref)
  - [`EntropicProjection`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function assert_geometry_admits_bounds(::AbstractProjectionGeometry,
                                       ::WeightBounds)::Nothing
    return nothing
end
function assert_geometry_admits_bounds(proj::Union{EntropicProjection, TsallisProjection,
                                                   LogBarrierProjection},
                                       wb::WeightBounds)::Nothing
    if !isnothing(wb.lb)
        @argcheck(all(x -> x >= zero(x), wb.lb),
                  DomainError(wb.lb,
                              "a `$(nameof(typeof(proj)))` admits no negative lower bound: its potential is undefined below zero. Use a Euclidean rule, or a non-negative bound."))
    end
    return nothing
end
export ProgrammeAllocationSet, TsallisProjection, LogBarrierProjection
public mirror_step
