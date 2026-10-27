"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the values that [`set_risk_budgeting_constraints!`](@ref) returns.

A [`RiskBudgetingResult`](@ref) holds one of them in its field `prb`, so a caller can read the budget that the model used.

# Related

  - [`ProcessedAssetRiskBudgetingAttributes`](@ref)
  - [`ProcessedFactorRiskBudgetingAttributes`](@ref)
  - [`RiskBudgetingResult`](@ref)
"""
abstract type ProcessedRiskBudgetingAttributes <: ProcessedAttributes end
"""
$(DocStringExtensions.TYPEDEF)

Holds the factor budget, the factor basis and the loadings of a [`RiskBudgeting`](@ref) solve with [`FactorRiskBudgeting`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProcessedFactorRiskBudgetingAttributes(;
        rkb::RiskBudget, b1::MatNum, rr::AbstractLoadingsRegressionResult
    ) -> ProcessedFactorRiskBudgetingAttributes

Keywords correspond to the struct's fields.

# Related

  - [`RiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
"""
@concrete struct ProcessedFactorRiskBudgetingAttributes <: ProcessedRiskBudgetingAttributes
    """
    The [`RiskBudget`](@ref) of the factors, one entry for each column of `rr.L`.
    """
    rkb
    """
    The factor basis ``(\\mathbf{B}^\\intercal)^{+}``, of size ``N \\times N_{f}``. The asset weights are `b1 * w1`, plus the off-factor block when `flag = true`.
    """
    b1
    """
    The loadings regression that `b1` comes from. Its `L` is ``\\mathbf{B}``.
    """
    rr
    function ProcessedFactorRiskBudgetingAttributes(rkb::RiskBudget, b1::MatNum,
                                                    rr::AbstractLoadingsRegressionResult)
        return new{typeof(rkb), typeof(b1), typeof(rr)}(rkb, b1, rr)
    end
end
function ProcessedFactorRiskBudgetingAttributes(; rkb::RiskBudget, b1::MatNum,
                                                rr::AbstractLoadingsRegressionResult)::ProcessedFactorRiskBudgetingAttributes
    return ProcessedFactorRiskBudgetingAttributes(rkb, b1, rr)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the asset budget of a [`RiskBudgeting`](@ref) solve with [`AssetRiskBudgeting`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProcessedAssetRiskBudgetingAttributes(; rkb::RiskBudget) -> ProcessedAssetRiskBudgetingAttributes

Keywords correspond to the struct's fields.

# Related

  - [`RiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
"""
@concrete struct ProcessedAssetRiskBudgetingAttributes <: ProcessedRiskBudgetingAttributes
    """
    The [`RiskBudget`](@ref) of the assets, one entry for each asset.
    """
    rkb
    function ProcessedAssetRiskBudgetingAttributes(rkb::RiskBudget)
        return new{typeof(rkb)}(rkb)
    end
end
function ProcessedAssetRiskBudgetingAttributes(;
                                               rkb::RiskBudget)::ProcessedAssetRiskBudgetingAttributes
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the solution, the resolved risk measures, the budget and the fallback chain of a Risk Budgeting optimisation.

[`optimise`](@ref) returns it for a [`RiskBudgeting`](@ref). A name that is not a field resolves through `prb` first, so `res.rkb` is the budget of the model. Every other name, and the property `w`, then resolves through `jr`, the [`JuMPOptimisationResult`](@ref) that the result holds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudgetingResult(;
        jr::JuMPOptimisationResult,
        r::BaseRM_VecBaseRM,
        prb::Union{ProcessedAssetRiskBudgetingAttributes,
                   ProcessedFactorRiskBudgetingAttributes},
        fb::Option{<:OptE_Opt_FbChain}
    ) -> RiskBudgetingResult

Keywords correspond to the struct's fields.

# Related

  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
@concrete struct RiskBudgetingResult <: RiskJuMPOptimisationResult
    """
    The [`JuMPOptimisationResult`](@ref) that holds the processed attributes, the return code, the solution and the model.
    """
    jr
    """
    $(field_dict[:r_res])
    """
    r
    """
    $(field_dict[:prb])
    """
    prb
    """
    $(field_dict[:fb_res])
    """
    fb
    function RiskBudgetingResult(jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                 prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                            ProcessedFactorRiskBudgetingAttributes},
                                 fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(jr), typeof(r), typeof(prb), typeof(fb)}(jr, r, prb, fb)
    end
end
function RiskBudgetingResult(; jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                             prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                        ProcessedFactorRiskBudgetingAttributes},
                             fb::Option{<:OptE_Opt_FbChain})::RiskBudgetingResult
    return RiskBudgetingResult(jr, r, prb, fb)
end
"""
    set_retcode(res::RiskBudgetingResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`RiskBudgetingResult`](@ref) with a different return code.

`retcode` is not a field of this result. It resolves through `jr`, the [`JuMPOptimisationResult`](@ref) that the result holds. So the method rebuilds `jr` with the new return code, and keeps `r`, `prb` and `fb` unchanged.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Returns

  - [`RiskBudgetingResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`RiskBudgetingResult`](@ref)
"""
function set_retcode(res::RiskBudgetingResult, retcode::OptRetCode_VecOptRetCode)
    return RiskBudgetingResult(set_retcode(res.jr, retcode), res.r, res.prb, res.fb)
end
# Unique field `prb` resolves directly; unknown properties forward into `prb` first, then
# into the embedded [`JuMPOptimisationResult`](@ref) `jr` (the virtual `:w` and `pa` fall-through).
@forward_properties RiskBudgetingResult begin
    forward(prb)
    forward(jr)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the formulations of the risk budgeting condition that [`AssetRiskBudgeting`](@ref) selects.

The `# Mathematical definition` of [`RiskBudgeting`](@ref) states each formulation, and [`set_risk_budgeting_constraints!`](@ref) states the rows of each.

# Related

  - [`LogRiskBudgeting`](@ref)
  - [`MixedIntegerRiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
"""
abstract type RiskBudgetingFormulation <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects the log-barrier formulation of the risk budgeting condition, in one orthant of the weights.

Equation 9 of [bruderroncalli2012](@cite) states it in the positive orthant, and problem 2.13 of [bai2016](@cite) states it in any orthant. The logarithm needs a positive argument, so the solution lies in the orthant that `z` states, asset by asset. A stated orthant costs no binary variable, but the signs must be known in advance. [`MixedIntegerRiskBudgeting`](@ref) searches every orthant.

The solution in the orthant of `z` must have a positive sum. Otherwise the budget row needs a negative ``k``, which a short budget or a weight bound forbids. The solve then returns a portfolio of that orthant that misses the budget, or it fails. Lemma 2.2 of [bai2016](@cite) states that the portfolio of that orthant, normalised to a positive budget, lies in the complementary orthant, so `-z` gives it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogRiskBudgeting(;
        z::Option{<:VecInt} = nothing
    ) -> LogRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - If `z` is provided: `!isempty(z)` and `all(x -> abs(x) == 1, z)`.

# Related

  - [`RiskBudgetingFormulation`](@ref)
  - [`MixedIntegerRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`set_risk_budgeting_constraints!`](@ref)

# References

  - $(ref_dict[:bruderroncalli2012]) Equation 9, and Section 2.4 for a general convex risk measure.
  - $(ref_dict[:maillard2008]) Equation 7 and Appendix A.2, for equal budgets.
  - $(ref_dict[:bai2016]) Problem 2.13 and Lemma 2.2, for equal budgets in any orthant.
  - $(ref_dict[:mosek2023c]) Section 9.3.2, Problem 9.7, for a general budget in any orthant.
  - $(ref_dict[:cajas2025]) Section 10.1.3, Equations 10.8 to 10.10, for the variable ``k`` that scales the weights.
"""
@concrete struct LogRiskBudgeting{T} <: RiskBudgetingFormulation
    """
    Orthant vector, one entry of `1` or `-1` for each asset. Entry ``i`` is the sign of weight ``i``. `nothing` puts every weight in the positive orthant.
    """
    z::T
    function LogRiskBudgeting(z::Option{<:VecInt})
        if !isnothing(z)
            @argcheck(!isempty(z), IsEmptyError("z cannot be empty"))
            @argcheck(all(x -> abs(x) == 1, z),
                      ArgumentError("all elements of z must be ±1"))
        end
        return new{typeof(z)}(z)
    end
end
function LogRiskBudgeting(; z::Option{<:VecInt} = nothing)::LogRiskBudgeting
    return LogRiskBudgeting(z)
end
"""
    port_opt_view(alg::LogRiskBudgeting{<:VecInt}, i, args...) -> LogRiskBudgeting

Return a [`LogRiskBudgeting`](@ref) whose orthant vector is the view of `alg.z` at the assets `i`.

A [`LogRiskBudgeting`](@ref) with no orthant vector holds no data, and the universal method of [`port_opt_view`](@ref) returns it unchanged.

# Arguments

  - `alg`: The formulation.
  - `i`: Indices of the assets that the view keeps.
  - `args...`: Not read.

# Related

  - [`LogRiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(alg::LogRiskBudgeting{<:VecInt}, i, args...)
    return LogRiskBudgeting(; z = view(alg.z, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the mixed-integer formulation of the risk budgeting condition, which searches every orthant of the weights.

Problem 9.9 of [mosek2023c](@cite) states it. The weights are the difference of a long part and a short part, the logarithm reads their sum, and one binary variable for each asset holds one of the two parts at zero. The cost is one binary variable for each asset, and a mixed-integer conic solver.

The solve gives the orthant whose solution has the least risk at the scale ``\\boldsymbol{b}^\\intercal \\ln \\lvert \\boldsymbol{y} \\rvert = 0``. Problem 9.9 writes the logarithm as a penalty in the objective, and this formulation writes it as a row, as [`LogRiskBudgeting`](@ref) does. In each orthant the two forms have the same solution up to a positive factor, and the risk measure is positively homogeneous, so both choose the same orthant. That orthant need not have the least risk after the normalisation to the budget, as Section 9.3.3 of [mosek2023c](@cite) states.

The parts are the weights, so the budget rows bound them exactly. With a long budget and no short budget, the long parts sum to the budget, and the book stays long. A long-short solution needs `bgt = nothing`, or a short budget that leaves the short exposure free.

!!! warning

    The search stops when the bound of the solver is near the objective, so a small objective can stop it in the wrong orthant. Measured on five assets with [`Variance`](@ref), Pajarito with HiGHS and Clarabel at their default tolerances, and `so = 1`: the solve stopped in an orthant with 5.8 % more variance than the best one, and with a largest gap of 0.17 between a share of the risk and its budget. At `so = 10_000`, or with [`StandardDeviation`](@ref), it found the best orthant, and the largest gap was below `2e-5`. Scale the objective through the `so` field of [`JuMPOptimiser`](@ref).

# Related

  - [`RiskBudgetingFormulation`](@ref)
  - [`LogRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`set_rb_mip_w!`](@ref)
  - [`set_mip_constraints!`](@ref)

# References

  - $(ref_dict[:mosek2023c]) Section 9.3.3, Problem 9.9.
  - $(ref_dict[:cajas2025]) Section 10.1.4, Equation 10.11, for the logarithm as a row and one binary variable for each asset.
"""
struct MixedIntegerRiskBudgeting <: RiskBudgetingFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the choice between a budget over the assets and a budget over the factors.

# Related

  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
"""
abstract type RiskBudgetingAlgorithm <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects a risk budget over the assets, and the formulation that solves for it.

Each asset takes the share of the risk that its entry of `rkb` states. `rkb = nothing` gives every asset the share ``1/N``. A [`RiskBudgetEstimator`](@ref) states the shares by asset or group names, and `sets` resolves the names. `alg` selects the formulation, see [`RiskBudgetingFormulation`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetRiskBudgeting(;
        rkb::Option{<:RkbE_Rkb} = nothing,
        sets::Option{<:UniverseSets} = nothing,
        alg::RiskBudgetingFormulation = LogRiskBudgeting()
    ) -> AssetRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - If `rkb` is a `RiskBudgetEstimator`: `!isnothing(sets)`.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `rkb`: Recursively viewed through [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `alg`: Recursively viewed through [`port_opt_view`](@ref).

# Related

  - [`RiskBudgetingAlgorithm`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RiskBudgetEstimator`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct AssetRiskBudgeting <: RiskBudgetingAlgorithm
    """
    $(field_dict[:rkb])
    """
    @vprop rkb
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:rba])
    """
    @vprop alg
    function AssetRiskBudgeting(rkb::Option{<:RkbE_Rkb}, sets::Option{<:UniverseSets},
                                alg::RiskBudgetingFormulation)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        return new{typeof(rkb), typeof(sets), typeof(alg)}(rkb, sets, alg)
    end
end
function AssetRiskBudgeting(; rkb::Option{<:RkbE_Rkb} = nothing,
                            sets::Option{<:UniverseSets} = nothing,
                            alg::RiskBudgetingFormulation = LogRiskBudgeting())::AssetRiskBudgeting
    return AssetRiskBudgeting(rkb, sets, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects a risk budget over the factors of a loadings regression.

Each factor takes the share of the risk that its entry of `rkb` states, and `rkb = nothing` gives every factor the same share. Problem 8 of [roncalliweisang2012](@cite) states the formulation, and the `# Mathematical definition` of [`RiskBudgeting`](@ref) states it in the notation of the library. `flag = false` keeps the weights in the span of the loadings. `flag = true` adds the off-factor weights, which the solve sets freely, and `hedge = true` holds their marginal variance at zero.

A [`RiskBudgetEstimator`](@ref) states the budget by **factor** names. The names resolve against the factor axis that `re` names: [`factor_axis_key`](@ref) reads `sets.tfkey` for a time-series regression and `sets.cfkey` for a cross-sectional one. That axis must name the columns of `rr.L` in order, see [`risk_budget_universe_key`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorRiskBudgeting(;
        re::RegE_Reg = StepwiseRegression(),
        rkb::Option{<:RkbE_Rkb} = nothing,
        sets::Option{<:UniverseSets} = nothing,
        flag::Bool = true,
        hedge::Bool = false
    ) -> FactorRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - If `rkb` is a `RiskBudgetEstimator`: `!isnothing(sets)`.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `re`: Loadings sliced to the selected assets through [`port_opt_view`](@ref).
  - `sets`: Sliced through [`port_opt_view`](@ref), which keeps the selected entries of the **asset** axis and leaves the factor entries unchanged. The field takes a view because it can hold both axes. `rkb` takes no view, because a factor budget has no asset index.

# Related

  - [`RiskBudgetingAlgorithm`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
  - [`risk_budget_universe_key`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:roncalliweisang2012]) Equation 2, Theorem 1, Problem 8, and Section 4.1.1 for the separation principle.
  - $(ref_dict[:cajas2025]) Section 10.2.2, Equation 10.21, for the variable ``k`` that scales the weights.
"""
@propagatable @concrete struct FactorRiskBudgeting <: RiskBudgetingAlgorithm
    """
    $(field_dict[:re])
    """
    @vprop re
    """
    $(field_dict[:rkb])
    """
    rkb
    """
    $(field_dict[:sets_frb])
    """
    @vprop sets
    """
    $(field_dict[:flag])
    """
    flag
    """
    $(field_dict[:hedge_frb])
    """
    hedge
    function FactorRiskBudgeting(re::RegE_Reg, rkb::Option{<:RkbE_Rkb},
                                 sets::Option{<:UniverseSets}, flag::Bool, hedge::Bool)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when rkb is a RiskBudgetEstimator: the budget is written in factor names and is resolved against the factor axis `re` names, `sets.dict[sets.tfkey]` or `sets.dict[sets.cfkey]`"))
        end
        return new{typeof(re), typeof(rkb), typeof(sets), typeof(flag), typeof(hedge)}(re,
                                                                                       rkb,
                                                                                       sets,
                                                                                       flag,
                                                                                       hedge)
    end
end
function FactorRiskBudgeting(; re::RegE_Reg = StepwiseRegression(),
                             rkb::Option{<:RkbE_Rkb} = nothing,
                             sets::Option{<:UniverseSets} = nothing, flag::Bool = true,
                             hedge::Bool = false)::FactorRiskBudgeting
    return FactorRiskBudgeting(re, rkb, sets, flag, hedge)
end
"""
    reader_frames(rba::FactorRiskBudgeting, path::AbstractString, refit::Bool) -> ReturnsFrames

Return the frames of the regression reader of `rba`, at `"path.re"`, for the frame walk of a [`RiskBudgeting`](@ref) head.

# Arguments

  - `rba`: The factor risk budgeting algorithm.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of `rba.re`.

# Related

  - [`FactorRiskBudgeting`](@ref)
  - [`reader_frames`](@ref)
"""
function reader_frames(rba::FactorRiskBudgeting, path::AbstractString, refit::Bool)
    return reader_frames(rba.re, join_path(path, :re), refit)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`RiskBudgeting`](@ref) fields that can hold a [`TimeDependent`](@ref).

The constructor reads these defaults when it checks each schedule with [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) returns them. So the value of each field outside a fold loop is declared once. A field whose static default is `nothing` is not in the result.

# Returns

  - `(; r = Variance(), rba = AssetRiskBudgeting())`.

# Related

  - [`RiskBudgeting`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function risk_budgeting_td_defaults()::NamedTuple
    return (; r = Variance(), rba = AssetRiskBudgeting())
end
"""
$(DocStringExtensions.TYPEDEF)

Optimises the portfolio whose assets, or whose factors, take stated shares of the risk.

`rba` selects the budget and its formulation, `r` gives the risk measures, and `opt` gives the prior, the solver and every other constraint. A row of `opt` that binds at the solution changes the optimality conditions, so the shares then miss the budget. With `bgt = nothing` no row fixes ``k``, and the scale of the weights is the scale that the solver returns.

# Mathematical definition

The risk of a positively homogeneous risk measure splits into the contributions of the assets, which is the Euler decomposition. The risk budgeting condition states the share of each contribution:

```math
\\begin{align}
RC_{i}(\\boldsymbol{w}) &= w_{i} \\frac{\\partial R(\\boldsymbol{w})}{\\partial w_{i}}\\,, \\quad \\frac{RC_{i}(\\boldsymbol{w})}{\\sum_{j=1}^{N} RC_{j}(\\boldsymbol{w})} = b_{i}\\,, \\quad \\sum_{i=1}^{N} b_{i} = 1\\,, \\quad b_{i} \\geq 0\\,.
\\end{align}
```

The condition is a system of equations that is not convex. [`LogRiskBudgeting`](@ref) solves a convex programme in its place, equation 9 of [bruderroncalli2012](@cite) in the orthant of ``\\boldsymbol{z}``:

```math
\\begin{align}
\\underset{\\boldsymbol{y},\\, k}{\\min} \\; &R(\\boldsymbol{y}) \\\\
\\text{s.t.} \\quad &\\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{z} \\odot \\boldsymbol{y}) \\geq 0\\,, \\\\
&\\boldsymbol{z} \\odot \\boldsymbol{y} \\geq \\boldsymbol{0}\\,, \\\\
&\\boldsymbol{y} \\in k\\, \\mathcal{W}\\,, \\quad \\boldsymbol{w} = \\boldsymbol{y} / k\\,.
\\end{align}
```

When no row of ``\\mathcal{W}`` binds, the optimality conditions give ``y_{i}\\, \\partial R(\\boldsymbol{y}) / \\partial y_{i} = \\lambda b_{i}``, which is the risk budgeting condition. The solution cannot be zero, so the row of the logarithm binds, and it fixes the scale of ``\\boldsymbol{y}``. The right-hand side sets that scale alone, so the library uses `0` where the source writes a free constant. The budget row of ``\\mathcal{W}``, ``\\boldsymbol{1}^\\intercal \\boldsymbol{y} = b_{g} k``, then sets ``k``.

[`MixedIntegerRiskBudgeting`](@ref) searches every orthant, after problem 9.9 of [mosek2023c](@cite):

```math
\\begin{align}
\\underset{\\boldsymbol{y}^{+},\\, \\boldsymbol{y}^{-},\\, \\boldsymbol{g},\\, k}{\\min} \\; &R(\\boldsymbol{y}) \\\\
\\text{s.t.} \\quad &\\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}^{+} + \\boldsymbol{y}^{-}) \\geq 0\\,, \\quad \\boldsymbol{y} = \\boldsymbol{y}^{+} - \\boldsymbol{y}^{-}\\,, \\\\
&\\boldsymbol{y}^{+} \\leq M \\boldsymbol{g}\\,, \\quad \\boldsymbol{y}^{-} \\leq M (\\boldsymbol{1} - \\boldsymbol{g})\\,, \\\\
&\\boldsymbol{y}^{+},\\, \\boldsymbol{y}^{-} \\geq \\boldsymbol{0}\\,, \\quad \\boldsymbol{g} \\in \\{0, 1\\}^{N}\\,, \\\\
&\\boldsymbol{y} \\in k\\, \\mathcal{W}\\,, \\quad \\boldsymbol{w} = \\boldsymbol{y} / k\\,.
\\end{align}
```

[`FactorRiskBudgeting`](@ref) states the budget over the factor exposures ``\\boldsymbol{y}_{f} = \\mathbf{B}^\\intercal \\boldsymbol{y}``, after problem 8 of [roncalliweisang2012](@cite). The weights are written on the factor axis, and the row of the logarithm reads the factor exposures:

```math
\\begin{align}
\\boldsymbol{y} &= (\\mathbf{B}^\\intercal)^{+} \\boldsymbol{y}_{f} + (\\tilde{\\mathbf{B}}^\\intercal)^{+} \\tilde{\\boldsymbol{y}}_{af}\\,, \\quad \\boldsymbol{b}_{f}^\\intercal \\ln(\\boldsymbol{y}_{f}) \\geq 0\\,, \\\\
RC_{j}^{f}(\\boldsymbol{w}) &= \\left[ \\mathbf{B}^\\intercal \\boldsymbol{w} \\right]_{j} \\left[ \\frac{\\partial R(\\boldsymbol{w})}{\\partial \\boldsymbol{w}} (\\mathbf{B}^\\intercal)^{+} \\right]_{j}\\,.
\\end{align}
```

The off-factor term is present when `flag = true`. The solve sets ``\\tilde{\\boldsymbol{y}}_{af}`` freely, so at the optimum the marginal risk of each off-factor direction is zero, and the factor contributions ``RC_{j}^{f}`` take the shares ``\\boldsymbol{b}_{f}``. That is the separation principle of Section 4.1.1 of [roncalliweisang2012](@cite). With `hedge = true` the rows ``((\\tilde{\\mathbf{B}}^\\intercal)^{+})^\\intercal \\mathbf{\\Sigma} \\boldsymbol{y} = \\boldsymbol{0}`` hold that condition for the variance, also when a row of ``\\mathcal{W}`` binds.

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{y}``: Unscaled weights, the weight variable of the model.
  - $(math_dict[:k_budget]) The model declares it free.
  - $(math_dict[:R_w]) With many measures it is the scalarised risk of the model.
  - ``RC_{i}(\\boldsymbol{w})``: Risk contribution of asset ``i``.
  - ``\\boldsymbol{b}``: Budget of the assets, `rba.rkb`.
  - ``\\boldsymbol{z}``: Orthant vector of [`LogRiskBudgeting`](@ref), a vector of ones when `z = nothing`.
  - $(math_dict[:W_feas]) It holds the weight bounds, the budgets and every other row of `opt`.
  - ``b_{g}``: The budget `opt.bgt`.
  - ``\\lambda``: Multiplier of the row of the logarithm.
  - ``\\boldsymbol{y}^{+}``, ``\\boldsymbol{y}^{-}``: Long and short parts of the unscaled weights.
  - ``\\boldsymbol{g}``: Binary vector. ``g_{i} = 1`` lets asset ``i`` be long, and ``g_{i} = 0`` lets it be short.
  - ``M``: Big-M constant, `opt.ss`.
  - $(math_dict[:B_loadings])
  - ``\\tilde{\\mathbf{B}}``: Basis of the kernel of ``\\mathbf{B}^\\intercal``, the ``N - N_{f}`` directions that the loadings do not span.
  - ``\\boldsymbol{y}_{f}``, ``\\tilde{\\boldsymbol{y}}_{af}``: Factor exposures and off-factor weights, both unscaled.
  - ``RC_{j}^{f}(\\boldsymbol{w})``: Risk contribution of factor ``j``.
  - ``\\boldsymbol{b}_{f}``: Budget of the factors, `rba.rkb`.
  - ``\\mathbf{\\Sigma}``: Covariance matrix of the prior.
  - $(math_dict[:N])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudgeting(;
        opt::JuMPOptimiser,
        r::TD{<:RM_VecRM} = Variance(),
        rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
        wi::TD_Option{<:VecNum} = nothing,
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> RiskBudgeting

Keywords correspond to the struct's fields. A field of the type [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value. The risk measures, the budgeting algorithm with its budget, the initial weights and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. An `optimise` outside a fold loop uses the static default of each field. The default of `wi` and of `fb` is `nothing`.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is provided: `!isempty(wi)`.
  - A schedule in `fb`: `bind !== :nearest`.
  - The constructor refuses a risk expression that is identically zero, because every portfolio that meets the row of the logarithm is then optimal. A [`NoRisk`](@ref) measure gives such an expression, and so does `settings.rke = false` on every measure.
  - If `r` holds a [`DataBenchmarkTracking`](@ref): `opt.pe` must not state `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).
  - A part that states no source and reads the raw returns, beside a part that states `x_src = ExcessSource()`, logs one warning that names its path, see [`warn_raw_readers`](@ref). The constructor then builds the estimator as configured.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `opt`: Recursively updated through [`factory`](@ref).
  - `r`: Recursively updated through [`factory`](@ref).
  - `fb`: Recursively updated through [`factory`](@ref).

## View parameters

`RiskBudgeting` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` holds a prior result, the method uses `opt.pe.X` in place of `X`. So the views of the children read the observations of the prior, and not the matrix of the caller.
  - `opt` and `r` recurse through [`port_opt_view`](@ref) with that matrix. `rba` recurses with the indices alone.
  - `fb` recurses through [`view_child`](@ref) with that matrix.
  - `wi` is sliced to the selected assets.

# Related

  - [`optimise`](@ref)
  - [`RiskBudgetingResult`](@ref)
  - [`scalarise_risk_expression!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:bruderroncalli2012]) Equation 9, and Section 2.4 for a general convex risk measure.
  - $(ref_dict[:maillard2008]) Equation 7 and Appendix A.2, for equal budgets.
  - $(ref_dict[:bai2016]) Problem 2.13, for equal budgets in any orthant.
  - $(ref_dict[:mosek2023c]) Section 9.3.2, Problem 9.7, and Section 9.3.3, Problem 9.9.
  - $(ref_dict[:roncalliweisang2012]) Equation 2, Theorem 1, Problem 8, and Section 4.1.1.
  - $(ref_dict[:cajas2025]) Sections 10.1.3, 10.1.4 and 10.2.2, for the variable ``k`` that scales the weights.
"""
@propagatable @concrete struct RiskBudgeting <: RiskJuMPOptimisationEstimator
    """
    $(field_dict[:opt_jmp])
    """
    @fprop opt
    """
    $(field_dict[:r_opt])
    """
    @fprop r
    """
    $(field_dict[:rba])
    """
    rba
    """
    $(field_dict[:wi])
    """
    wi
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function RiskBudgeting(opt::JuMPOptimiser, r::TD{<:RM_VecRM},
                           rba::TD{<:RiskBudgetingAlgorithm}, wi::TD_Option{<:VecNum},
                           fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :RiskBudgeting)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        assert_risk_measure_required(r, :RiskBudgeting; flag = zero_risk_expression_flag)
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        assert_time_dependent_substitution(RiskBudgeting, (; opt, r, rba, wi, fb),
                                           risk_budgeting_td_defaults())
        assert_data_benchmark_prior(r, opt.pe, :r, :RiskBudgeting, "opt.pe")
        est = new{typeof(opt), typeof(r), typeof(rba), typeof(wi), typeof(fb)}(opt, r, rba,
                                                                               wi, fb)
        warn_raw_readers(est)
        return est
    end
end
function RiskBudgeting(; opt::JuMPOptimiser, r::TD{<:RM_VecRM} = Variance(),
                       rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
                       wi::TD_Option{<:VecNum} = nothing,
                       fb::TDO_Option{<:OptE_Opt} = nothing)::RiskBudgeting
    return RiskBudgeting(opt, r, rba, wi, fb)
end
function time_dependent_field_defaults(::RiskBudgeting)::NamedTuple
    return risk_budgeting_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a part of `opt` needs the previous portfolio weights.

The parts are the schedules of the time-dependent fields, the [`JuMPOptimiser`](@ref) `opt.opt`, the risk measures `opt.r` and the fallback `opt.fb`. A turnover measure or a fee on the turnover is a part that needs them.

# Returns

  - `Bool`: `true` when any part answers `true`.

# Related

  - [`needs_previous_weights`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`RiskBudgeting`](@ref)
"""
function needs_previous_weights(opt::RiskBudgeting)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the view of a [`RiskBudgeting`](@ref) at the assets `i`.

The `## View parameters` subsection of [`RiskBudgeting`](@ref) states how the method takes the view of each field.

# Arguments

  - `rb`: The optimiser.
  - `i`: Indices of the assets that the view keeps.
  - `X::MatNum`: Returns matrix of the caller. The method uses the matrix of the prior in its place when `rb.opt.pe` holds a prior result.
  - `args...`: Not read.

# Returns

  - `RiskBudgeting`: A new optimiser, built through the keyword constructor.

# Related

  - [`port_opt_view`](@ref)
  - [`view_child`](@ref)
  - [`RiskBudgeting`](@ref)
"""
function port_opt_view(rb::RiskBudgeting, i, X::MatNum, args...)::RiskBudgeting
    X = isa(rb.opt.pe, AbstractPriorResult) ? rb.opt.pe.X : X
    opt = port_opt_view(rb.opt, i, X)
    r = port_opt_view(rb.r, i, X)
    rba = port_opt_view(rb.rba, i)
    wi = nothing_scalar_array_view(rb.wi, i)
    return RiskBudgeting(; opt = opt, r = r, rba = rba, wi = wi,
                         fb = view_child(rb.fb, i, X))
end
"""
    risk_budget_universe_key(rba::RiskBudgetingAlgorithm,
                             N::Integer) -> Option{<:AbstractString}

Return the key of the universe a named risk budget resolves against, or `nothing` for the asset frame.

The budget has one entry for each variable that it is *over*, so the algorithm, and not the sets, decides the universe that names it. [`AssetRiskBudgeting`](@ref) budgets the asset weights and takes the default axis. [`FactorRiskBudgeting`](@ref) budgets the factor weights `w1` and takes the factor axis that its `re` names, which [`factor_axis_key`](@ref) reads.

The method reads the axis only when `rba.rkb` is a [`RiskBudgetEstimator`](@ref). A [`RiskBudget`](@ref) holds its own vector and resolves no names, so the method does not check an axis that it does not read. Every other reader of the declared axis does the same. When the method reads the axis, [`factor_universe`](@ref) checks it against `N`, the number of factor weights, which is `size(rr.L, 2)`. Under a [`DimensionReductionRegression`](@ref) that is the size of the reduced basis, not the number of columns of `F`. So the method refuses a budget that names the original factors, before the solve meets a `DimensionMismatch`.

# Arguments

  - `rba`: Risk budgeting algorithm.
  - `N`: Number of budgeted variables, `length(w)` at the call site.

# Returns

  - `key::Option{<:AbstractString}`: The universe key, or `nothing` to use `sets.xkey`.

# Related

  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`factor_universe`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_universe_key(::AssetRiskBudgeting, ::Integer)
    return nothing
end
function risk_budget_universe_key(::FactorRiskBudgeting, ::Integer)
    return nothing
end
function risk_budget_universe_key(rba::FactorRiskBudgeting{<:Any, <:RiskBudgetEstimator},
                                  N::Integer)
    key = factor_axis_key(rba.sets, rba.re)
    factor_universe(rba.sets, key, N, "a $(FactorRiskBudgeting) risk budget", "rr.L")
    return key
end
"""
    _set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting,
                                     w::VecJuMPScalar; strict::Bool = false) -> RiskBudget

Add the row of the logarithm, ``\\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{x}) \\geq 0``, and the variable ``k``, to the model of a [`RiskBudgeting`](@ref).

Every method of [`set_risk_budgeting_constraints!`](@ref) calls it, with the vector ``\\boldsymbol{x}`` that its formulation reads. The row fixes the scale of the weights, so the function also records a unit budget with [`set_unit_budget!`](@ref). A builder that reads [`effective_k`](@ref) then uses `1` in place of ``k``, so it bounds the unscaled weights whose risk the model minimises.

# JuMP formulation

## Variables

  - `k`: ``k``, free. It is not bounded below by zero and it is not the number `1`.
  - `log_w`: ``\\boldsymbol{t}``, one entry for each entry of `w`.

## Constraints

  - `clog_w[i]`: ``(t_{i}, 1, x_{i}) \\in \\mathcal{K}_{\\exp}``, which is ``t_{i} \\leq \\ln x_{i}`` with ``x_{i} > 0``.
  - `crkb`: ``\\boldsymbol{b}^\\intercal \\boldsymbol{t} \\geq 0``.

Where:

  - ``\\boldsymbol{x}``: The vector `w`.
  - ``\\boldsymbol{t}``: Lower bounds of the logarithms of ``\\boldsymbol{x}``.
  - ``\\boldsymbol{b}``: The budget, resolved from `rb.rba.rkb`.
  - $(math_dict[:K_exp])

# Arguments

  - $(arg_dict[:model])
  - `rb`: The optimiser.
  - `w`: The vector that the logarithm reads: the weights, the weights times the orthant vector, the sum of the long and short parts, or the factor weights.
  - `strict`: Passed on to [`risk_budget_constraints`](@ref).

# Validation

  - The budget has one entry for each entry of `w`. Otherwise a `DimensionMismatch` names both lengths.

# Returns

  - `rkb::RiskBudget`: The budget that the row reads.

# Related

  - [`set_risk_budgeting_constraints!`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`risk_budget_universe_key`](@ref)
  - [`set_unit_budget!`](@ref)
  - [`get_k`](@ref)
"""
function _set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting,
                                          w::VecJuMPScalar; strict::Bool = false)
    N = length(w)
    rkb = risk_budget_constraints(rb.rba.rkb, rb.rba.sets,
                                  risk_budget_universe_key(rb.rba, N); N = N,
                                  strict = strict)
    b = rkb.val
    @argcheck(length(b) == N, DimensionMismatch("rb ($(length(b))) must match N ($N)"))
    sc = get_constraint_scale(model)
    # This head is the one producer of `k` that is not
    # `set_maximum_ratio_factor_variables!`: the barrier below pins the scale, so `k` is a
    # *free* variable here rather than `k >= 0` or the literal `1`. `get_k`'s error message
    # names both routes.
    JuMP.@variable(model, k)
    JuMP.@variable(model, log_w[1:N])
    JuMP.@constraints(model,
                      begin
                          clog_w[i = 1:N],
                          [sc * log_w[i], sc, sc * w[i]] in JuMP.MOI.ExponentialCone()
                          crkb, sc * LinearAlgebra.dot(b, log_w) >= 0
                      end)
    # The log-barrier normalisation above pins the scale, so downstream builders may use 1
    # in place of the free variable `k`.
    set_unit_budget!(model)
    return rkb
end
function non_investable_universe(rba::AssetRiskBudgeting, ni::VecStr)::AssetRiskBudgeting
    return rebuild_estimator(rba, (; sets = non_investable_sets(rba.sets, ni)))
end
function non_investable_universe(rb::RiskBudgeting, ni::VecStr)::RiskBudgeting
    return rebuild_estimator(rb, (; rba = non_investable_universe(rb.rba, ni)))
end
"""
    set_risk_budgeting_constraints!(model, rb, pr, wb, args...)
    set_risk_budgeting_constraints!(model, rb, pr, wb, rd::ReturnsResult)

Add the weights, the row of the logarithm and the weight constraints of a [`RiskBudgeting`](@ref) to the model.

The method dispatches on the formulation of `rb.rba`. Each method registers the weights, calls [`_set_risk_budgeting_constraints!`](@ref) with the vector that its formulation reads, and adds the weight bounds and the budgets with [`set_weight_constraints!`](@ref). The `# Mathematical definition` of [`RiskBudgeting`](@ref) states each programme.

# JuMP formulation

## Variables

  - `w`: ``\\boldsymbol{y}``, registered by [`set_w!`](@ref) for [`LogRiskBudgeting`](@ref).
  - `lw`, `sw`: ``\\boldsymbol{y}^{+}``, ``\\boldsymbol{y}^{-}``, registered by [`set_rb_mip_w!`](@ref) for [`MixedIntegerRiskBudgeting`](@ref). `w` is then the expression ``\\boldsymbol{y}^{+} - \\boldsymbol{y}^{-}``.
  - `w1`, `w2`: ``\\boldsymbol{y}_{f}``, ``\\tilde{\\boldsymbol{y}}_{af}``, registered by [`set_factor_risk_contribution_constraints!`](@ref) for [`FactorRiskBudgeting`](@ref). `w` is then an expression of them.

## Constraints

  - The rows of [`_set_risk_budgeting_constraints!`](@ref), on ``\\boldsymbol{y}`` for [`LogRiskBudgeting`](@ref) with no orthant vector, on ``\\boldsymbol{z} \\odot \\boldsymbol{y}`` with an orthant vector, on ``\\boldsymbol{y}^{+} + \\boldsymbol{y}^{-}`` for [`MixedIntegerRiskBudgeting`](@ref), and on ``\\boldsymbol{y}_{f}`` for [`FactorRiskBudgeting`](@ref).
  - `orthcrkb`: ``\\boldsymbol{z} \\odot \\boldsymbol{y} \\geq \\boldsymbol{0}``, for [`LogRiskBudgeting`](@ref) with an orthant vector.
  - The rows of [`set_weight_constraints!`](@ref). [`LogRiskBudgeting`](@ref) with no orthant vector calls it with `long = true`.
  - For [`MixedIntegerRiskBudgeting`](@ref), the binary variables and the rows ``\\boldsymbol{y}^{+} \\leq M \\boldsymbol{g}`` and ``\\boldsymbol{y}^{-} \\leq M (\\boldsymbol{1} - \\boldsymbol{g})`` come later, from [`set_mip_constraints!`](@ref), which reads the [`WeightsFromParts`](@ref) contract that [`set_rb_mip_w!`](@ref) records.

Where:

  - ``\\boldsymbol{y}``: Unscaled weights.
  - ``\\boldsymbol{z}``: Orthant vector `rb.rba.alg.z`.
  - ``\\boldsymbol{y}^{+}``, ``\\boldsymbol{y}^{-}``: Long and short parts of ``\\boldsymbol{y}``.
  - ``\\boldsymbol{y}_{f}``, ``\\tilde{\\boldsymbol{y}}_{af}``: Factor exposures and off-factor weights.
  - ``\\boldsymbol{g}``: Binary vector of the long gates.
  - ``M``: Big-M constant, `rb.opt.ss`.

# Arguments

  - $(arg_dict[:model])
  - `rb`: The optimiser.
  - $(arg_dict[:pr])
  - $(arg_dict[:wb_arg])
  - `rd`: Returns data that [`FactorRiskBudgeting`](@ref) reads when it fits the loadings. The asset methods do not read it.

# Validation

  - [`LogRiskBudgeting`](@ref) with no orthant vector: no weight bound is negative. Otherwise [`set_weight_constraints!`](@ref) throws an `ArgumentError`.
  - [`LogRiskBudgeting`](@ref) with an orthant vector: `z` has one entry for each asset. Otherwise a `DimensionMismatch` names both lengths.

# Returns

  - [`ProcessedAssetRiskBudgetingAttributes`](@ref) for [`AssetRiskBudgeting`](@ref), and [`ProcessedFactorRiskBudgetingAttributes`](@ref) for [`FactorRiskBudgeting`](@ref).

# Related

  - [`RiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`_set_risk_budgeting_constraints!`](@ref)
"""
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:LogRiskBudgeting{Nothing}},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
    set_w!(model, pr.X, rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, get_w(model); strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt, true)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:LogRiskBudgeting{<:VecInt}},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
    set_w!(model, pr.X, rb.wi)
    z = rb.rba.alg.z
    @argcheck(length(z) == length(get_w(model)),
              DimensionMismatch("z ($(length(z))) must match w ($(length(get_w(model)))))"))
    w = z .* get_w(model)
    rkb = _set_risk_budgeting_constraints!(model, rb, w; strict = rb.opt.strict)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, orthcrkb, sc * w >= 0)
    set_weight_constraints!(model, wb, rb.opt)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:FactorRiskBudgeting, <:Any},
                                         pr::AbstractPriorResult, wb::WeightBounds,
                                         rd::ReturnsResult)
    b1, _, rr = set_factor_risk_contribution_constraints!(model, rb.rba.re, rd, pr,
                                                          rb.rba.flag, rb.wi;
                                                          hedge = rb.rba.hedge)
    rkb = _set_risk_budgeting_constraints!(model, rb, shared_get(model, :w1);
                                           strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt)
    return ProcessedFactorRiskBudgetingAttributes(; rkb = rkb, b1 = b1, rr = rr)
end
"""
    set_rb_mip_w!(model::JuMP.Model, X::MatNum)

Register the long and short parts of the weights of [`MixedIntegerRiskBudgeting`](@ref), and the weights as their difference.

The weights are an expression of the parts, so the function records the [`WeightsFromParts`](@ref) contract with [`set_decomposition_contract!`](@ref). [`set_mip_constraints!`](@ref) reads the contract, and gates the parts with one binary variable for each asset.

# JuMP formulation

## Variables

  - `lw`: ``\\boldsymbol{y}^{+} \\geq \\boldsymbol{0}``.
  - `sw`: ``\\boldsymbol{y}^{-} \\geq \\boldsymbol{0}``.

## Expressions

  - `w`: ``\\boldsymbol{y}^{+} - \\boldsymbol{y}^{-}``, the unscaled weights.
  - `w_obj`: ``\\boldsymbol{y}^{+} + \\boldsymbol{y}^{-}``, the vector that the row of the logarithm reads.

# Arguments

  - `model::JuMP.Model`: The model.
  - `X::MatNum`: Returns matrix. The function reads its number of columns.

# Returns

  - `nothing`.

# Related

  - [`set_risk_budgeting_constraints!`](@ref)
  - [`set_decomposition_contract!`](@ref)
  - [`WeightsFromParts`](@ref)
  - [`RiskBudgeting`](@ref)
"""
function set_rb_mip_w!(model::JuMP.Model, X::MatNum)
    N = size(X, 2)
    JuMP.@variables(model, begin
                        lw[1:N] >= 0
                        sw[1:N] >= 0
                    end)
    JuMP.@expressions(model, begin
                          w, lw - sw
                          w_obj, lw + sw
                      end)
    set_decomposition_contract!(model, WeightsFromParts())
    return nothing
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:MixedIntegerRiskBudgeting},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
    set_rb_mip_w!(model, pr.X)
    rkb = _set_risk_budgeting_constraints!(model, rb, shared_get(model, :w_obj);
                                           strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
"""
    _optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult(); str_names::Bool = false,
              save::Bool = true, kwargs...) -> RiskBudgetingResult

Build the model of a [`RiskBudgeting`](@ref), solve it, and return its result.

[`optimise`](@ref) reaches this method through [`shared_series_solve`](@ref), which drops the observations of a Shared Series gap before the solve.

# Algorithm

 1. Replace each schedule of `rb` with its static default through [`reset_time_dependent_estimator`](@ref).
 2. Process the attributes of `rb.opt` against `rd` with [`processed_jump_optimiser_attributes`](@ref), giving `attrs`. It holds the prior, the weight bounds, the fees, the return terms and the Investable Mask.
 3. Take the view of `rb` and of `rd` at the investable assets with [`investable_view`](@ref). Both are unchanged when every asset is investable.
 4. Bind the benchmark of each [`DataBenchmarkTracking`](@ref) measure to `rd` with [`bind_data_benchmark`](@ref).
 5. Build a new model. Register the scales with [`set_model_scales!`](@ref) and the number of observations with [`set_model_observations!`](@ref).
 6. Add the weights, the row of the logarithm and the weight constraints with [`set_risk_budgeting_constraints!`](@ref), giving `prb`.
 7. Add the constraints, the return terms and the risk measures with [`assemble_jump_model!`](@ref), under [`MinimumRisk`](@ref).
 8. Set the objective [`MinimumRisk`](@ref), and solve with [`optimise_JuMP_model!`](@ref), giving `retcode` and `sol`. [`process_model`](@ref) divides the unscaled weights by ``k``.
 9. Build the result. It holds `attrs`, `retcode`, `sol`, the model when `save` is `true`, the risk measures resolved against the prior, and `prb`.

# Arguments

  - `rb::RiskBudgeting`: The optimiser.
  - $(arg_dict[:rd])
  - `str_names::Bool`: `true` gives each variable and each row of the model its string name.
  - `save::Bool`: `true` keeps the model in the result.
  - `kwargs...`: Passed on to [`processed_jump_optimiser_attributes`](@ref).

# Returns

  - [`RiskBudgetingResult`](@ref): The result, with `fb = nothing`. The method of [`optimise`](@ref) for an optimiser with a fallback runs the fallback chain.

# Related

  - [`optimise`](@ref)
  - [`set_risk_budgeting_constraints!`](@ref)
  - [`RiskBudgetingResult`](@ref)
"""
function _optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult();
                   str_names::Bool = false, save::Bool = true, kwargs...)
    rb = reset_time_dependent_estimator(rb)
    attrs = processed_jump_optimiser_attributes(rb.opt, rd; kwargs...)
    # The bundle reduced what it carries. The head carries the rest — an initial weight
    # vector, a risk measure holding per-asset data, tracking, a custom term — and hands
    # them to `assemble_jump_model!` itself, so it takes the same view of itself and of
    # `rd`. Both are unchanged when every asset is investable.
    rb, rd = investable_view(rb, rd, attrs.pr, attrs.imsk)
    rb = bind_data_benchmark(rb, rd)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    set_model_observations!(model, size(attrs.pr.X, 1))
    prb = set_risk_budgeting_constraints!(model, rb, attrs.pr, attrs.wb, rd)
    assemble_jump_model!(model, rb, rb.opt, attrs, rd, rb.r, MinimumRisk())
    set_portfolio_objective_function!(model, MinimumRisk(), rb, attrs)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(attrs.pr.X))
    return RiskBudgetingResult(;
                               jr = JuMPOptimisationResult(; pa = attrs, retcode = retcode,
                                                           sol = sol,
                                                           model = ifelse(save, model,
                                                                          nothing)),
                               r = factory(rb.r, attrs.pr, rb.opt.slv), prb = prb,
                               fb = nothing)
end
"""
    optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; str_names::Bool = false, save::Bool = true, kwargs...) -> RiskBudgetingResult

Optimise a [`RiskBudgeting`](@ref) that has no fallback, and return its result.

This method serves an optimiser whose `fb` is `nothing`. It solves through [`shared_series_solve`](@ref), and [`_optimise`](@ref) states the steps of the solve.

# Arguments

  - `rb`: The optimiser.
  - $(arg_dict[:rd]) When `rb.opt.pe` is a prior result, the solve does not read `rd`, except to fit the loadings of [`FactorRiskBudgeting`](@ref) when no other source holds them. A fallback or a clustering step can still read it.
  - `str_names`: `true` gives each variable and each row of the model its string name.
  - `save`: `true` keeps the model in the result.
  - `kwargs`: Passed on to [`_optimise`](@ref).

# Validation

  - No field in the tree of `rb` holds an [`Online`](@ref). Otherwise [`assert_batch_entry`](@ref) throws an `ArgumentError` that names the field. A plain `optimise` is a batch fit, and the fold loop resolves an `Online` only in its online branch.

# Returns

  - [`RiskBudgetingResult`](@ref): The result of the solve.

# Related

  - [`RiskBudgeting`](@ref)
  - [`RiskBudgetingResult`](@ref)
  - [`_optimise`](@ref)
"""
function optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  str_names::Bool = false, save::Bool = true, kwargs...)
    assert_batch_entry(rb, "`optimise`")
    return shared_series_solve(rb, rd; str_names = str_names, save = save, kwargs...)
end

@pipe_delegates RiskBudgeting opt
@pipe_route_rkb RiskBudgeting
@pipe_route_sigma_ucs RiskBudgeting
export AssetRiskBudgeting, FactorRiskBudgeting, RiskBudgeting, RiskBudgetingResult,
       LogRiskBudgeting, MixedIntegerRiskBudgeting
