"""
$(DocStringExtensions.TYPEDEF)

Holds the Result of the resolution of a JuMP head: the prior result, the fees, the return terms, the scalariser, the Investable Mask and the Resolved Constraint Block.

[`resolve_jump_optimiser`](@ref) makes the bundle once for each `optimise` call, and [`assemble_jump_model!`](@ref) reads it. So each builder of the model reads a result, and no builder resolves an estimator again. The constraints of the feasible set are in the [`ResolvedConstraintBlock`](@ref) in `cbr`. The owner keeps the fees, the return terms and the scalariser, because the objective reads them.

`cbr` holds the block Result of the head, or a vector of block Results when the head has more than one block. The block of the head is then the first, and [`owner_block`](@ref) gives it.

A named forward rule keeps the reads of the resolved constraints on the bundle, so `jor.wb` reads `owner_block(jor.cbr).wb`. The rule forwards `wb`, `lt`, `st`, `lcsr`, `ctr`, `gcardr`, `sgcardr`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn` and `plr`, the constraint fields that the bundle held before it held the block. Every other field of the block is read through `cbr`.

Every JuMP optimisation Result holds the bundle. So the types of `ret` and `sca` are not type parameters of the bundle, and the Results of two returns estimators or of two scalarisers have one type. A verb that reads them compiles one time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ResolvedJuMPOptimiser(;
        pr::AbstractPriorResult, fees::Option{<:Fees}, ret::JRE_VecJRE, sca::Option{<:Scalariser},
        imsk::Option{<:BitVector} = nothing, cbr::Rcb_VecRcb
    ) -> ResolvedJuMPOptimiser

Keywords correspond to the struct's fields. Make the bundle with [`resolve_jump_optimiser`](@ref). The constructor checks the types of the fields and no value, because the resolution checked the values.

`ret` is a returns estimator and not a result, because a returns estimator has no result form. Before the bundle held the block, it held one field for each resolved constraint. Those fields are now fields of `cbr`, and the forward rule reads them.

# Examples

```jldoctest
julia> pr = prior(EmpiricalPrior(),
                  ReturnsResult(; nx = [\"a\", \"b\"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1]));

julia> cbr = PortfolioOptimisers.resolve_constraint_block(ConstraintBlock(), pr,
                                                          ReturnsResult(; nx = [\"a\", \"b\"],
                                                                        X = pr.X); slv = nothing);

julia> jor = ResolvedJuMPOptimiser(; pr = pr, fees = nothing, ret = ArithmeticReturn(),
                                   sca = nothing, cbr = cbr);

julia> jor.wb === jor.cbr.wb
true
```

# Related

  - [`ResolvedConstraintBlock`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`MeanRisk`](@ref)
"""
@concrete struct ResolvedJuMPOptimiser <: AbstractResolved
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:feesr])
    """
    fees
    """
    $(field_dict[:ret_jmp])
    """
    ret::JRE_VecJRE
    """
    $(field_dict[:sca_res])
    """
    sca::Option{<:Scalariser}
    """
    $(field_dict[:imsk])
    """
    imsk
    """
    $(field_dict[:cbr])
    """
    cbr
    function ResolvedJuMPOptimiser(pr::AbstractPriorResult, fees::Option{<:Fees},
                                   ret::JRE_VecJRE, sca::Option{<:Scalariser},
                                   imsk::Option{<:BitVector}, cbr::Rcb_VecRcb)
        return new{typeof(pr), typeof(fees), typeof(imsk), typeof(cbr)}(pr, fees, ret, sca,
                                                                        imsk, cbr)
    end
end
function ResolvedJuMPOptimiser(; pr::AbstractPriorResult, fees::Option{<:Fees},
                               ret::JRE_VecJRE, sca::Option{<:Scalariser},
                               imsk::Option{<:BitVector} = nothing,
                               cbr::Rcb_VecRcb)::ResolvedJuMPOptimiser
    return ResolvedJuMPOptimiser(pr, fees, ret, sca, imsk, cbr)
end
# The bundle held one field for each resolved constraint before it held the block, and the
# tests, the examples and the methods of a custom term read them, such as `jor.wb` (ADR 0237).
# `cbr` holds a vector when the head has more than one block, and a path of fields cannot
# read the first entry of a vector, so each rule reads the block of the owner by a function.
@forward_properties ResolvedJuMPOptimiser begin
    compute(wb, x -> owner_block(getfield(x, :cbr)).wb)
    compute(lt, x -> owner_block(getfield(x, :cbr)).lt)
    compute(st, x -> owner_block(getfield(x, :cbr)).st)
    compute(lcsr, x -> owner_block(getfield(x, :cbr)).lcsr)
    compute(ctr, x -> owner_block(getfield(x, :cbr)).ctr)
    compute(gcardr, x -> owner_block(getfield(x, :cbr)).gcardr)
    compute(sgcardr, x -> owner_block(getfield(x, :cbr)).sgcardr)
    compute(smtx, x -> owner_block(getfield(x, :cbr)).smtx)
    compute(sgmtx, x -> owner_block(getfield(x, :cbr)).sgmtx)
    compute(slt, x -> owner_block(getfield(x, :cbr)).slt)
    compute(sst, x -> owner_block(getfield(x, :cbr)).sst)
    compute(sglt, x -> owner_block(getfield(x, :cbr)).sglt)
    compute(sgst, x -> owner_block(getfield(x, :cbr)).sgst)
    compute(tn, x -> owner_block(getfield(x, :cbr)).tn)
    compute(plr, x -> owner_block(getfield(x, :cbr)).plr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Restricts an optimiser head and its returns data to the assets that an Investable Mask keeps.

[`resolve_jump_optimiser`](@ref) restricts the prior result and every constraint result of the bundle. It cannot restrict the values that the head holds outside the bundle, such as an initial weight vector, a risk measure with data for each asset, a tracking estimator or a custom constraint. These go from the head to [`assemble_jump_model!`](@ref) directly. So each head calls this function on itself and on `rd` before it builds a model.

The function does not report the reduction, because [`investable_reduction`](@ref) reports it once for the bundle.

# Algorithm

 1. Check that every [`UniverseSets`](@ref) in the tree of the head names the columns of `rd`, in order, with [`assert_universe_axis_order`](@ref). The bundle checks the universes of the [`JuMPOptimiser`](@ref) of the head. This step reaches the universes that the head holds outside it, such as the universe of a risk budget.
 2. When `imsk` is `nothing`, every asset is investable. Return `optimiser` and `rd` unchanged, with no allocation.
 3. Take the returns matrix `X` from `rd.X`, or from `pr.X` when `rd` holds no matrix. When the head holds a fitted prior, [`port_opt_view`](@ref) reads the matrix of that prior and ignores `X`, so the two choices give the same view.
 4. Find the positions `idx = findall(imsk)` of the investable assets. Every other caller of [`port_opt_view`](@ref) passes an index vector, so the function passes one too.
 5. View `optimiser` at `idx` with [`port_opt_view`](@ref). Then declare the names that the mask leaves out on the Non-Investable Axis of the sets that the head holds, with [`non_investable_universe`](@ref). The generic method returns a head with no such sets unchanged. A risk budget keyed by name resolves after this step.
 6. View `rd` at `idx`, and return the two views.

# Arguments

  - `optimiser::JuMPOptimisationEstimator`: The optimiser head to view.
  - $(arg_dict[:rd])
  - $(arg_dict[:pr])
  - $(arg_dict[:imsk])

# Validation

  - The rules of [`assert_universe_axis_order`](@ref).

# Returns

  - `(optimiser, rd)`: Both restricted to the investable assets, or both unchanged.

# Related

  - [`investable_mask`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`port_opt_view`](@ref)
"""
function investable_view(optimiser::JuMPOptimisationEstimator, rd::ReturnsResult,
                         ::AbstractPriorResult, ::Nothing)
    assert_universe_axis_order(optimiser, rd)
    return optimiser, rd
end
function investable_view(optimiser::JuMPOptimisationEstimator, rd::ReturnsResult,
                         pr::AbstractPriorResult, imsk::BitVector)
    assert_universe_axis_order(optimiser, rd)
    X = isnothing(rd.X) ? pr.X : rd.X
    idx = findall(imsk)
    # The head takes the view that `investable_reduction` took of the optimiser for the
    # bundle, so it declares the Non-Investable Axis on the sets that it holds. It reports
    # nothing, because `investable_reduction` reported the departure.
    return non_investable_universe(port_opt_view(optimiser, idx, X),
                                   non_investable_names(rd.nx, imsk)),
           port_opt_view(rd, idx)
end
function expand_investable_weights(::Nothing, sol::JuMPOptSol_VecJuMPOptSol)
    return sol
end
function expand_investable_weights(imsk::BitVector, sol::JuMPOptimisationSolution)
    return JuMPOptimisationSolution(; w = expand_investable_weights(imsk, sol.w))
end
function expand_investable_weights(imsk::BitVector, sol::VecJuMPOptSol)
    return [expand_investable_weights(imsk, s) for s in sol]
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the fields that every JuMP optimisation result shares.

Each concrete JuMP result holds it as its first field, `jr`, as each JuMP optimiser holds a [`JuMPOptimiser`](@ref) as `opt`. The concrete result adds its own fields and the last field `fb`.

The result holds the weights on the full asset universe of the caller. The solver solves over the investable assets alone, and `model` keeps that reduced problem when the head saves the model. The property `w` reads the weights of `sol`, and a vector of solutions gives a vector of weight vectors. A property name that is neither `w` nor a field reads the field of `jor` of the same name.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimisationResult(;
        jor::ResolvedJuMPOptimiser,
        retcode::OptRetCode_VecOptRetCode,
        sol::JuMPOptSol_VecJuMPOptSol,
        model::Option{<:JuMP.Model}
    ) -> JuMPOptimisationResult

Keywords correspond to the struct's fields. The keyword constructor expands the weights of `sol` from the investable assets to the full asset universe with [`expand_investable_weights`](@ref), at the mask `jor.imsk`. An asset outside the mask gets the weight zero. When `jor.imsk` is `nothing`, `sol` is kept unchanged.

## Validation

  - When `jor.imsk` is a mask, each weight vector of `sol` holds one weight for each `true` entry of the mask. Otherwise the constructor raises a `DimensionMismatch`.

# Related

  - [`BaseJuMPOptimisationResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`set_retcode`](@ref): Rebuilds the result with a new return code, and does not expand the weights again.
"""
@concrete struct JuMPOptimisationResult <: BaseJuMPOptimisationResult
    """
    $(field_dict[:jor])
    """
    jor
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:sol])
    """
    sol
    """
    $(field_dict[:model])
    """
    model
    # The type is declared here and not beside its supertype `BaseJuMPOptimisationResult`, so
    # that this constructor can bind `ResolvedJuMPOptimiser` and the solution types.
    function JuMPOptimisationResult(jor::ResolvedJuMPOptimiser,
                                    retcode::OptRetCode_VecOptRetCode,
                                    sol::JuMPOptSol_VecJuMPOptSol,
                                    model::Option{<:JuMP.Model})
        return new{typeof(jor), typeof(retcode), typeof(sol), typeof(model)}(jor, retcode,
                                                                             sol, model)
    end
end
function JuMPOptimisationResult(; jor::ResolvedJuMPOptimiser,
                                retcode::OptRetCode_VecOptRetCode,
                                sol::JuMPOptSol_VecJuMPOptSol,
                                model::Option{<:JuMP.Model})::JuMPOptimisationResult
    # Every JuMP family builds its result through this constructor, so the weight expansion
    # is here. `MeanRisk`, `RiskBudgeting`, `RelaxedRiskBudgeting`, `FactorRiskContribution`
    # and `NearOptimalCentering` all call it. The expansion is not in the inner constructor,
    # because a new value of `sol` there makes the inferred field types wider.
    return JuMPOptimisationResult(jor, retcode, expand_investable_weights(jor.imsk, sol),
                                  model)
end
# The JuMP families carry the mask on the processed attribute bundle, and every concrete JuMP
# result embeds the shared core as `jr`, so two methods cover the whole side.
function result_investable_mask(res::JuMPOptimisationResult)
    return res.jor.imsk
end
function result_investable_mask(res::Union{RiskJuMPOptimisationResult,
                                           NonRiskJuMPOptimisationResult})
    return result_investable_mask(res.jr)
end
"""
    set_retcode(res::JuMPOptimisationResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`JuMPOptimisationResult`](@ref) with a different return code.

The rebuild calls the positional constructor, not the keyword constructor. The keyword constructor expands the weights from the investable assets to the full asset universe. The weights of `res.sol` are expanded already, and a second expansion raises a `DimensionMismatch` when the mask leaves out an asset.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Returns

  - [`JuMPOptimisationResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
function set_retcode(res::JuMPOptimisationResult, retcode::OptRetCode_VecOptRetCode)
    return JuMPOptimisationResult(res.jor, retcode, res.sol, res.model)
end
# Virtual property `:w` extracts portfolio weights from `sol` (a single solution or a vector
# of them, hence the broadcast); unknown properties forward to `jor` (see [`@forward_properties`](@ref)).
@forward_properties JuMPOptimisationResult begin
    compute(w, sol.w; broadcast)
    forward(jor)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the static defaults of the [`JuMPOptimiser`](@ref) fields that can hold a [`TimeDependent`](@ref).

The constructor reads them when it checks each entry of a schedule, and [`time_dependent_field_defaults`](@ref) returns them. So the value of a field outside every fold loop is declared once. A field whose static default is `nothing` is not in the tuple. A slot of the [`ConstraintBlock`](@ref) in `cbe` takes the default of [`constraint_block_td_defaults`](@ref), under its flat name.

# Returns

  - `NamedTuple`: The defaults of `pe`, `bgt`, `ret` and `sca`, and of the slot `wb` of the block.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
  - [`constraint_block_td_defaults`](@ref)
"""
function jump_optimiser_td_defaults()::NamedTuple
    return merge((; pe = EmpiricalPrior(), bgt = 1.0, ret = ArithmeticReturn(),
                  sca = nothing), constraint_block_td_defaults())
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the problem definition and the solver settings that every JuMP optimiser shares.

It holds the prior estimator, the solvers, the net budget, the fees, the return term, the scalariser and the execution settings. The [`ConstraintBlock`](@ref) in its field `cbe` holds the feasible set of the weights: the bounds, the leverage caps, the constraints, the tracking and the regularisation. Give it to a JuMP optimiser such as [`MeanRisk`](@ref) or [`RiskBudgeting`](@ref), which adds the objective and the risk measure.

Each slot of the block is a property of the optimiser, so `opt.wb` reads `opt.cbe.wb`. `Accessors.set(opt, @optic(_.wb), v)` and a search key such as `"opt.wb"` replace that slot of the block, and a [`Pipeline`](@ref) routes a target such as `:wb` into the block, see [`pipe_block_field`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimiser(;
        pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
        slv::Slv_VecSlv,
        cbe::Option{<:ConstraintBlock} = nothing,
        bgt::TD_Option{<:Num_BgtCE} = 1.0,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        ret::TD{<:JRE_VecJRE} = ArithmeticReturn(),
        sca::TD_Option{<:NonHierarchicalScalariser} = nothing,
        sc::Number = 1,
        so::Number = 1,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing,
        slots...,
    ) -> JuMPOptimiser

Keywords correspond to the struct's fields and to the slots of [`ConstraintBlock`](@ref). `slots...` takes each keyword of the block with its type and its default: `wb`, `sbgt`, `gbgt`, `lt`, `st`, `lcse`, `cte`, `gcarde`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn`, `sets`, `tr`, `ple`, `ccnt`, `cobj`, `ss`, `card`, `scard`, `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp`, `linf` and `xbgt`. The constructor builds the block in `cbe` from these flat keywords, with the defaults of the block. When the caller gives `cbe`, a flat keyword replaces that slot of the given block, and a slot that the caller does not name keeps its value, see [`constraint_block`](@ref). So the default of a flat keyword never overwrites a slot of a given block. A name that is neither a field nor a slot raises a `MethodError`.

A field or a slot typed [`TD_Option`](@ref) or [`TD`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value. A cross-validation fold loop resolves the schedule at each fold, and an `optimise` call outside a fold loop reads the static default of the field. So the prior estimator, the returns model, the scalariser, the asset sets and the constraints can change from fold to fold. The execution settings `slv`, `sc`, `so` and `strict` are static.

`lcse`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn` and `ple` take a vector that mixes the members of their element type, such as `[Turnover(…), TurnoverEstimator(…)]`. Such a literal infers an element type that the positional constructor refuses, so the keyword constructor of the block narrows it through [`narrow_mixed_vector`](@ref), and the caller writes no [`concrete_typed_array`](@ref).

The positional constructor takes the fields in the order of the struct: `pe`, `slv`, `cbe`, `bgt`, `fees`, `ret`, `sca`, `sc`, `so`, `strict` and `cache`. Before the block, it took one argument for each slot.

## Validation

The constructor of [`ConstraintBlock`](@ref) checks each slot on its own. This constructor checks the fields and the rules that read a field and a slot together.

  - If `slv` is a vector: `!isempty(slv)`.
  - If `bgt` is a number: `isfinite(bgt)`.
  - If `bgt` is a [`BudgetCostEstimator`](@ref): `isnothing(sbgt)`. Otherwise the constructor raises a `ConflictingArgumentError`.
  - If `sbgt` or `gbgt` is a number: it is finite and `>= 0`.
  - If `gbgt` is not `nothing`: `bgt` and `sbgt` are not both numbers, see [`assert_gross_budget_admissible`](@ref). Two numbers fix the gross budget of the model at `bgt + 2 * sbgt`, which is the row that `gbgt` writes. The realised gross exposure `norm(w, 1)` equals this budget when `xbgt` is `true`, and it can be smaller when `xbgt` is `false`.
  - If `gbgt` is not `nothing` and `wb` is a [`WeightBounds`](@ref): `wb` admits a negative weight. The check of an [`AbstractWeightBoundsEstimator`](@ref) waits for the solve.
  - If `gbgt` has an upper end: it is non-negative, and it is at least the least magnitude that `bgt` admits, see [`assert_gross_budget_reaches_net`](@ref). The gross exposure is at least `abs(sum(w))`, so a smaller gross budget leaves no feasible weights. An `ArgumentError` is thrown otherwise.
  - If `lcse`, `cte`, `smtx`, `sgmtx`, `tn`, `tr`, `ple`, `l2`, `lp` or `lpc` is a vector: `!isempty`. An empty `l2`, `lp` or `lpc` builds no term, which is what `nothing` states.
  - If `card`, `l2c`, `linfc`, `l1` or `linf` is a number: it is finite and `> 0`. `l1` and `linf` also take an ambiguity-radius rule, and `l2c` and `linfc` a norm-ceiling rule. A rule states no number at construction, so [`assemble_jump_model!`](@ref) checks the number that the rule returns.
  - The rule in each [`LpRegularisation`](@ref) agrees with the field that holds it. `lp` is a penalty, so it refuses a norm-ceiling rule, and `lpc` is a constraint, so it refuses an ambiguity-radius rule. The term has one field for both readings, so the constructor of the block is the first point that knows the reading. The constructors of [`L2Regularisation`](@ref) and [`LpRegularisation`](@ref) check the other values of `l2`, `lp` and `lpc`.
  - The shapes of `scard`, `smtx`, `slt` and `sst` agree, see [`assert_subgroup_mip_fields`](@ref). The shapes of `sgcarde`, `sgmtx`, `sglt` and `sgst` agree, see [`assert_subgrouped_mip_fields`](@ref).
  - If a slot holds an estimator keyed by name ([`name_keyed`](@ref)), or `fees` is a [`FeesEstimator`](@ref): `!isnothing(sets)`.
  - If `tr` holds a [`DataBenchmarkTracking`](@ref): `pe` does not state `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).
  - If a field or a slot holds a [`TimeDependent`](@ref): the constructor runs again once for each entry of the schedule, with the other schedules at their static defaults, see [`assert_time_dependent_substitution`](@ref). So a type error or a conflict between two fields shows at construction. A check that couples a schedule to a static field runs again when each fold rebuilds the optimiser.

A value outside its bound raises an `IsEmptyError`, an `IsNonFiniteError`, a `DomainError`, an `IsNothingError`, an `ArgumentError` or a `DimensionMismatch`, as the helper that checks it states.

# Related

  - [`BaseJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
"""
@concrete struct JuMPOptimiser <: BaseJuMPOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:cbe])
    """
    cbe
    """
    $(field_dict[:bgt])
    """
    bgt
    """
    $(field_dict[:fees_jmp])
    """
    fees
    """
    $(field_dict[:ret_jmp])
    """
    ret
    """
    $(field_dict[:sca])
    """
    sca
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:strict_opt]) Under `true` the fit also refuses an asset with a gap in the rows that a plain graph reader reads, which it otherwise drops with a warning, see [`graph_reader_mask`](@ref). It also refuses an asset that a distance reader of the fit cannot estimate, which it otherwise drops with a warning, see [`distance_reader_refusal_mask`](@ref).
    """
    strict
    """
    $(field_dict[:cache_opt])
    """
    cache
    function JuMPOptimiser(pe::Onl{<:TD{<:PrE_Pr}}, slv::Slv_VecSlv, cbe::ConstraintBlock,
                           bgt::TD_Option{<:Num_BgtCE}, fees::TD_Option{<:FeesE_Fees},
                           ret::TD{<:JRE_VecJRE},
                           sca::TD_Option{<:NonHierarchicalScalariser}, sc::Number,
                           so::Number, strict::Bool, cache::Option{<:ReturnsBufferState})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        if isa(bgt, Number)
            @argcheck(isfinite(bgt), IsNonFiniteError("bgt must be finite, got $bgt"))
        elseif isa(bgt, BudgetCostEstimator)
            @argcheck(isnothing(cbe.sbgt),
                      ConflictingArgumentError("sbgt must be nothing when bgt is a BudgetCostEstimator, got sbgt = $(cbe.sbgt)"))
        end
        # The checks below read a field of the optimiser and a slot of its block. The
        # constructor of the block checks each slot on its own.
        assert_gross_budget_admissible(bgt, cbe.sbgt, cbe.gbgt, cbe.wb)
        assert_data_benchmark_prior(cbe.tr, pe, :tr, :JuMPOptimiser)
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(cbe.sets),
                      IsNothingError("sets cannot be nothing when estimator-type fields are provided"))
        end
        # The slots go in as flat keywords, so each call of the check builds a static block.
        assert_time_dependent_substitution(JuMPOptimiser,
                                           merge((; pe, slv, bgt, fees, ret, sca, sc, so,
                                                  strict), constraint_block_slots(cbe)),
                                           jump_optimiser_td_defaults())
        return new{typeof(pe), typeof(slv), typeof(cbe), typeof(bgt), typeof(fees),
                   typeof(ret), typeof(sca), typeof(sc), typeof(so), typeof(strict),
                   typeof(cache)}(pe, slv, cbe, bgt, fees, ret, sca, sc, so, strict, cache)
    end
end
function JuMPOptimiser(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(), slv::Slv_VecSlv,
                       cbe::Option{<:ConstraintBlock} = nothing,
                       bgt::TD_Option{<:Num_BgtCE} = 1.0,
                       fees::TD_Option{<:FeesE_Fees} = nothing,
                       ret::TD{<:JRE_VecJRE} = ArithmeticReturn(),
                       sca::TD_Option{<:NonHierarchicalScalariser} = nothing,
                       sc::Number = 1, so::Number = 1, strict::Bool = false,
                       cache::Option{<:ReturnsBufferState} = nothing,
                       slots...)::JuMPOptimiser
    return JuMPOptimiser(pe, slv, constraint_block(cbe; slots...), bgt, fees, ret, sca, sc,
                         so, strict, cache)
end
# Each slot of the block is a property of the optimiser, and a patch of `Accessors.set` can
# set it, so `opt.wb` and the search key `"opt.wb"` keep their meaning (ADR 0237).
@forward_properties JuMPOptimiser begin
    forward(cbe, wb, sbgt, gbgt, lt, st, lcse, cte, gcarde, sgcarde, smtx, sgmtx, slt, sst,
            sglt, sgst, tn, sets, tr, ple, ccnt, cobj, ss, card, scard, l2c, lpc, linfc, l1,
            l2, lp, linf, xbgt; settable)
end
function time_dependent_candidate_fields(opt::JuMPOptimiser)::Tuple
    return (typed_time_dependent_fields(opt)..., typed_time_dependent_fields(opt.cbe)...)
end
function time_dependent_slot(opt::JuMPOptimiser, f::Symbol)
    return getproperty(opt, f)
end
function schedule_reads_previous_weights(opt::JuMPOptimiser)::Bool
    return any(f -> schedule_reads_previous_weights(time_dependent_slot(opt, f)),
               time_dependent_candidate_fields(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns `true` if a field of `opt` reads the portfolio weights of the previous fold.

The function asks `bgt` and `fees`, and the [`ConstraintBlock`](@ref) in `cbe` through its own method, which asks `tn`, `tr`, `ccnt` and `cobj`. It also asks every field and every slot that holds a [`TimeDependent`](@ref) that the fold loop resolves, see [`time_dependent_fields`](@ref). A schedule reads the previous weights when it holds a [`PreviousWeightsFunction`](@ref) or an entry that reads them.

# Arguments

  - `opt::JuMPOptimiser`: The JuMP optimiser.

# Returns

  - `Bool`: `true` if one of these fields reads the previous weights, and `false` otherwise.

# Examples

```jldoctest
julia> PortfolioOptimisers.needs_previous_weights(JuMPOptimiser(;
                                                                slv = Solver(; solver = nothing)))
false
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(opt::JuMPOptimiser)
    return (needs_previous_weights(opt.bgt) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.cbe) ||
            any(f -> needs_previous_weights(time_dependent_slot(opt, f)),
                time_dependent_fields(opt)))
end
function time_dependent_field_defaults(::JuMPOptimiser)::NamedTuple
    return jump_optimiser_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `opt` whose fields that read previous weights hold the weights `w`.

The function calls `factory(x, w)` on `bgt` and `fees`, and on the [`ConstraintBlock`](@ref) in `cbe`, whose method rebuilds `tn`, `tr`, `ccnt` and `cobj`. It keeps every other field. So a cost budget on `bgt` measures its cost from `w`, unless it is `fixed`, see [`factory(bgt::BudgetCosts, w::VecNum)`](@ref). A [`TimeDependent`](@ref) in one of these fields applies `factory` to each of its entries. The keyword constructor of [`JuMPOptimiser`](@ref) checks the new fields.

# Arguments

  - `opt::JuMPOptimiser`: The JuMP optimiser.
  - `w::AbstractVector`: The new portfolio weights.

# Returns

  - `JuMPOptimiser`: The optimiser with the two fields and the block rebuilt for `w`.

# Examples

```jldoctest
julia> opt = JuMPOptimiser(; slv = Solver(; solver = nothing));

julia> PortfolioOptimisers.factory(opt, fill(0.1, 10)) isa JuMPOptimiser
true
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::JuMPOptimiser, w::AbstractVector)::JuMPOptimiser
    return rebuild_estimator(opt,
                             (; bgt = factory(opt.bgt, w), fees = factory(opt.fees, w),
                              cbe = factory(opt.cbe, w)))
end
"""
    obs_weights_view(opt::JuMPOptimiser, i) -> JuMPOptimiser

Return a copy of `opt` whose fields hold their observation weights cut to the observations `i`.

The optimiser is not [`@propagatable`](@ref), so it needs this method of [`obs_weights_view`](@ref). Its prior, its phylogeny and centrality estimators, its tracking measure and its returns estimator all read the observations of the sample, and each can hold fixed observation weights. A cut of the observations of a solve, such as the drop of a Shared Series gap, must reach each of them, see [`shared_series_estimator`](@ref).

# Algorithm

 1. Cut every field of `opt` with [`obs_weights_view`](@ref), and rebuild the optimiser with its keyword constructor, through [`obs_weights_fields`](@ref). A field that holds no estimator, such as a solver, a bound or a set of names, comes back with no change, and a partial-fit state in `cache` comes back as `nothing`. The constructor checks the cut fields again.

# Arguments

  - `opt::JuMPOptimiser`: The JuMP optimiser.
  - `i`: The observations to keep: a mask of the full sample, or their positions.

# Validation

  - The rules of [`obs_weights_index`](@ref) for each weights field in the tree of `opt`.

# Returns

  - `JuMPOptimiser`: The optimiser at the kept observations.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`obs_weights_view`](@ref)
  - [`shared_series_estimator`](@ref)
"""
function obs_weights_view(opt::JuMPOptimiser, i)::JuMPOptimiser
    return obs_weights_fields(opt, i)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `opt` restricted to the assets at the indices `i`.

A hierarchical optimiser calls it to build the optimiser of one cluster, and the entry of a fit calls it to restrict the optimiser to the Investable Mask.

# Algorithm

 1. When `opt.pe` is a prior result, replace `X` with `opt.pe.X`.
 2. View at `i` the fields that can hold a value for each asset: `pe`, `bgt`, `ret` and `cache`.
 3. View `fees` at `i` with `X`. A fee finds the assets outside `i`, which its liquidation charges read, from the width of `X`.
 4. View the [`ConstraintBlock`](@ref) in `cbe` at `i` with `X`, through its own method, `port_opt_view(cbe::ConstraintBlock, i, X)`. It views the slots that hold a value for each asset, and its tracking errors read `X`.
 5. Keep every other field. They hold no value for each asset.
 6. Rebuild the optimiser with its keyword constructor, which checks the viewed fields again.

# Arguments

  - `opt::JuMPOptimiser`: The JuMP optimiser.
  - `i`: The indices of the assets that the view keeps.
  - `X::MatNum`: The returns matrix of the full universe, with one column for each asset. A prior result in `opt.pe` gives its own `X` in its place.

# Returns

  - `JuMPOptimiser`: The optimiser restricted to the assets at `i`.

# Examples

```jldoctest
julia> opt = JuMPOptimiser(; slv = Solver(; solver = nothing));

julia> X = rand(50, 5);

julia> PortfolioOptimisers.port_opt_view(opt, 1:3, X) isa JuMPOptimiser
true
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(opt::JuMPOptimiser, i, X::MatNum, args...)::JuMPOptimiser
    X = isa(opt.pe, AbstractPriorResult) ? opt.pe.X : X
    # A fee has two axes. Its five fields for each asset are on the kept assets, and its two
    # liquidation charges are on the other assets. Its view finds the other assets from the
    # width of the unreduced `X`, so it needs the matrix, as the tracking errors of the
    # block do.
    return rebuild_estimator(opt,
                             (; pe = port_opt_view(opt.pe, i),
                              cbe = port_opt_view(opt.cbe, i, X),
                              bgt = port_opt_view(opt.bgt, i),
                              fees = port_opt_view(opt.fees, i, X),
                              ret = port_opt_view(opt.ret, i),
                              cache = port_opt_view(opt.cache, i)))
end
function non_investable_universe(opt::JuMPOptimiser, ni::VecStr)::JuMPOptimiser
    return rebuild_estimator(opt, (; sets = non_investable_sets(opt.sets, ni)))
end

export ResolvedJuMPOptimiser, JuMPOptimiser
