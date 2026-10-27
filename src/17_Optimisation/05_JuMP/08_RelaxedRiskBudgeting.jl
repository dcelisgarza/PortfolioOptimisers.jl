"""
$(DocStringExtensions.TYPEDEF)

Supertype of the three models of [`RelaxedRiskBudgeting`](@ref).

Each subtype selects the cones that bound the variable ``\\psi`` of the model. The models are Models (A), (B) and (C) of [gambetakwon2020](@cite), and the `# Mathematical definition` of [`RelaxedRiskBudgeting`](@ref) states them.

# Related

  - [`BasicRelaxedRiskBudgeting`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
  - [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`set_relaxed_risk_budgeting_alg_constraints!`](@ref)

# References

  - $(ref_dict[:gambetakwon2020]) Section 3.
"""
abstract type RelaxedRiskBudgetingAlgorithm <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects Model (A) of [`RelaxedRiskBudgeting`](@ref), where the standard deviation of the portfolio alone bounds ``\\psi``.

The objective of this model is zero exactly when the contributions meet the budget, so the model gives the risk budgeting portfolio when no other row binds. Model (A) of [gambetakwon2020](@cite) is the second-order cone model of [mausserromanko2014](@cite) with a target return.

# Related

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
  - [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)

# References

  - $(ref_dict[:gambetakwon2020]) Model (0), Equations 5 to 11, and Model (A), Equations 12 to 17. $(ref_dict[:no_original_source])
  - $(ref_dict[:mausserromanko2014])
"""
struct BasicRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects Model (B) of [`RelaxedRiskBudgeting`](@ref), which adds a regulating term ``\\rho`` that the standard deviation of the portfolio bounds.

The term raises the price of the standard deviation in the objective, so the model trades the budget for a lower standard deviation. The objective is not zero at the budget, so the solution can leave the budget also when no other row binds.

# Related

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`BasicRelaxedRiskBudgeting`](@ref)
  - [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)

# References

  - $(ref_dict[:gambetakwon2020]) Section 3.2, Model (B), Equations 21 to 27.
"""
struct RegularisedRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects Model (C) of [`RelaxedRiskBudgeting`](@ref), where the variances of the single assets, times the penalty `p`, bound the regulating term ``\\rho``.

The term reads the diagonal of the covariance matrix, and not the whole matrix as Model (B) does. So it limits the weight of an asset of high variance. Section 3.3 of [gambetakwon2020](@cite) uses this to keep the contributions nearer to the budget than Model (B) does. A larger `p` gives the term more strength. The objective is not zero at the budget, so the solution can leave the budget also when no other row binds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegularisedPenalisedRelaxedRiskBudgeting(;
        p::Number = 1.0
    ) -> RegularisedPenalisedRelaxedRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(p)` and `p > 0`. Otherwise a `DomainError` names `p`.

# Related

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`BasicRelaxedRiskBudgeting`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)

# References

  - $(ref_dict[:gambetakwon2020]) Section 3.3, Equations 28 and 30, and Section 3.4, Model (C), Equations 31 to 37.
"""
@concrete struct RegularisedPenalisedRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm
    """
    Penalty ``\\lambda`` of the regulating term. It multiplies ``\\boldsymbol{w}^\\intercal \\mathbf{\\Theta} \\boldsymbol{w}``, the variances of the single assets times the squares of their weights.
    """
    p
    function RegularisedPenalisedRelaxedRiskBudgeting(p::Number)
        @argcheck(isfinite(p) && p > zero(p), DomainError(p, "p must be finite and > 0"))
        return new{typeof(p)}(p)
    end
end
function RegularisedPenalisedRelaxedRiskBudgeting(;
                                                  p::Number = 1.0)::RegularisedPenalisedRelaxedRiskBudgeting
    return RegularisedPenalisedRelaxedRiskBudgeting(p)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the fields of [`RelaxedRiskBudgeting`](@ref) that can hold a [`TimeDependent`](@ref) schedule.

The constructor reads them when it checks a schedule through [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) returns them. So the value of each field outside a fold loop is stated in one place. A field whose static default is `nothing` is not in the result.

# Returns

  - `NamedTuple`: `(; rba = AssetRiskBudgeting())`.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function relaxed_risk_budgeting_td_defaults()::NamedTuple
    return (; rba = AssetRiskBudgeting())
end
"""
$(DocStringExtensions.TYPEDEF)

Optimises the portfolio whose assets, or whose factors, take stated shares of the variance, through a model of second-order cones.

`rba` selects the budget, `alg` selects the model, and `opt` gives the prior, the solver and every other constraint. The model reads the covariance matrix alone and resolves no risk measure, so its result carries no `r`. Model (A) gives the risk budgeting portfolio when no row of `opt` binds at that portfolio. A row that binds, such as a target return, moves the contributions away from the budget. Models (B) and (C) can leave the budget also when no row binds. [`RiskBudgeting`](@ref) states the budget as a row of the logarithm, and it reads any risk measure.

# Mathematical definition

The risk budgeting condition asks that each asset contribute the share ``b_{i}`` of the variance, ``w_{i} (\\mathbf{\\Sigma} \\boldsymbol{w})_{i} = b_{i} \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}``. [mausserromanko2014](@cite) writes the condition for equal shares as a second-order cone programme, and [gambetakwon2020](@cite) adds a target return and a regulating term to it. The library writes Models (A), (B) and (C) of [gambetakwon2020](@cite) with the shares of a general budget:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\, \\psi,\\, \\gamma,\\, \\boldsymbol{\\zeta}}{\\min} \\quad & \\psi - \\gamma \\\\
\\text{s.t.} \\quad & \\boldsymbol{\\zeta} = \\mathbf{\\Sigma} \\boldsymbol{w}\\,, \\quad \\boldsymbol{\\zeta} \\geq \\boldsymbol{0}\\,, \\\\
& w_{i} \\zeta_{i} \\geq b_{i} \\gamma^{2}\\,, \\quad i = 1, \\ldots, n\\,, \\\\
& \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

The model in `alg` adds the cones that bound ``\\psi``. Model (A), [`BasicRelaxedRiskBudgeting`](@ref), adds one:

```math
\\begin{align}
\\psi &\\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_{2}\\,.
\\end{align}
```

Model (B), [`RegularisedRelaxedRiskBudgeting`](@ref), and Model (C), [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref), add a regulating term ``\\rho``:

```math
\\begin{align}
\\psi^{2} &\\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_{2}^{2} + \\rho^{2}\\,, \\\\
\\rho^{2} &\\geq n\\, \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} \\quad \\text{in Model (B)}\\,, \\\\
\\rho^{2} &\\geq n \\lambda\\, \\boldsymbol{w}^\\intercal \\mathbf{\\Theta} \\boldsymbol{w}\\,, \\quad \\mathbf{\\Theta} = \\mathrm{diag}(\\mathbf{\\Sigma})\\,, \\quad \\text{in Model (C)}\\,.
\\end{align}
```

The variables ``\\psi``, ``\\gamma`` and ``\\rho`` are ``\\sqrt{n}`` times the variables of [gambetakwon2020](@cite), so the objective is ``\\sqrt{n}`` times its objective and the weights are the same. With ``b_{i} = 1/n`` the rows are its Equations 13 to 15, 22 to 25 and 32 to 35. Its target return, Equations 16, 26 and 36, is the lower bound of the return that `opt.ret` can state. It is then a row of ``\\mathcal{W}``.

In Model (A), for a budget whose shares sum to one, the rows give the chain ``\\gamma^{2} = \\gamma^{2} \\sum_{i} b_{i} \\leq \\sum_{i} w_{i} \\zeta_{i} = \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} \\leq \\psi^{2}``. So the objective is not below zero, and it is zero exactly when ``w_{i} \\zeta_{i} = b_{i} \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}`` for every ``i``, which is the risk budgeting condition. In Models (B) and (C) the regulating term is at its bound at the optimum, so ``\\psi`` is larger than the standard deviation, and the objective is not zero at the budget. The optimum then trades the budget for a lower ``\\psi``.

Where:

  - $(math_dict[:w_port])
  - ``\\psi``: Upper bound of the standard deviation of the portfolio. In Models (B) and (C) it also bounds the regulating term.
  - ``\\gamma``: Lower bound of ``\\sqrt{w_{i} \\zeta_{i} / b_{i}}`` over the assets.
  - ``\\boldsymbol{\\zeta}``: Marginal risks, ``\\mathbf{\\Sigma} \\boldsymbol{w}``. So ``w_{i} \\zeta_{i}`` is the contribution of asset ``i`` to the variance.
  - ``b_{i}``: Share of asset ``i`` in the budget `rba.rkb`, its entry divided by the sum of the entries. So a budget of `1:5` and a budget of `(1:5) / 15` give the same portfolio.
  - ``n``: Number of entries of the budget, which is the number of assets, or the number of factors of a [`FactorRiskBudgeting`](@ref).
  - $(math_dict[:W_feas]) It holds the weight bounds, the budget, the target return and every other row of `opt`. The budgeted weights are not negative, because ``\\boldsymbol{\\zeta} \\geq \\boldsymbol{0}`` and the cones give ``w_{i} \\geq 0``.
  - $(math_dict[:G_cov_factor]) It is the `chol` of the prior, or the transpose of the square root of ``\\mathbf{\\Sigma}`` that [`matrix_square_root`](@ref) takes under `mtx_sqrt`.
  - ``\\mathbf{\\Sigma}``: Covariance matrix of the prior.
  - ``\\rho``: Regulating term of Models (B) and (C).
  - ``\\lambda``: Penalty `p` of [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref).
  - ``\\mathbf{\\Theta}``: Diagonal matrix of the variances of the single assets.

## Factor risk budgeting

With a [`FactorRiskBudgeting`](@ref) the budget is over the factor weights ``\\boldsymbol{w}_{1}``. So the cones of the budget pair ``[\\boldsymbol{w}_{1}]_{j}`` with ``\\zeta_{j}`` and read the share of factor ``j``. The marginal risks and the cones on ``\\psi`` read the whole portfolio:

```math
\\begin{align}
\\boldsymbol{w} &= \\begin{cases} \\mathbf{B}_{1} \\boldsymbol{w}_{1} & \\text{if } \\texttt{flag} = \\texttt{false}\\,, \\\\ \\mathbf{B}_{1} \\boldsymbol{w}_{1} + \\mathbf{B}_{2} \\boldsymbol{w}_{2} & \\text{if } \\texttt{flag} = \\texttt{true}\\,, \\end{cases} \\\\
\\boldsymbol{\\zeta} &= \\mathbf{B}_{1}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,, \\qquad \\psi \\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_{2} \\quad \\text{in Model (A)}\\,.
\\end{align}
```

So ``[\\boldsymbol{w}_{1}]_{j} \\zeta_{j}`` is the Euler contribution of factor ``j`` to ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}``, the value that [`factor_risk_contribution`](@ref) reports. With `flag = true` the variance splits into the factor contributions and the off-factor contribution ``c_{2} = \\boldsymbol{w}_{2}^\\intercal \\mathbf{B}_{2}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}``, which takes no budget. The chain of Model (A) then gives ``\\gamma^{2} \\leq \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} - c_{2}``. So the objective can fall below zero by the term ``c_{2}``, and in general its optimum is not the optimum of [`RiskBudgeting`](@ref). The penalty of Model (C) reads the decision vector, ``\\boldsymbol{w}_{1}``, or ``\\boldsymbol{w}_{1}`` followed by ``\\boldsymbol{w}_{2}``, with the covariance ``\\mathbf{P}^\\intercal \\mathbf{\\Sigma} \\mathbf{P}`` of its basis ``\\mathbf{P}``.

With `hedge = true` the model adds the rows ``\\mathbf{B}_{2}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} = \\boldsymbol{0}``. They are the first-order condition of the variance in ``\\boldsymbol{w}_{2}``, so ``c_{2} = 0``, and the model is the model above on the factor covariance ``(\\mathbf{B}^\\intercal \\mathbf{\\Sigma}^{-1} \\mathbf{B})^{-1}``. Model (A) is then exact: when no other row binds, its optimum is the optimum of [`RiskBudgeting`](@ref) under [`Variance`](@ref). The cost is that the weights lie in a subspace of dimension ``N_{f}``, so a weight bound that the free off-factor weights meet can make the model infeasible.

Where:

  - ``\\boldsymbol{w}_{1}``: Factor weights, one entry for each of the ``N_{f}`` factors.
  - $(math_dict[:w_2_off_factor])
  - $(math_dict[:B_loadings]) It is `rr.L`.
  - ``\\mathbf{B}_{1}``: Factor basis, the pseudoinverse of ``\\mathbf{B}^\\intercal``.
  - ``\\mathbf{B}_{2}``: Off-factor basis, an orthonormal basis of the null space of ``\\mathbf{B}^\\intercal``.
  - ``\\mathbf{P}``: Basis of the decision vector, ``\\mathbf{B}_{1}``, or ``[\\mathbf{B}_{1} \\; \\mathbf{B}_{2}]`` when `flag = true`.
  - ``c_{2}``: Off-factor contribution to the variance.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelaxedRiskBudgeting(;
        opt::JuMPOptimiser,
        rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
        wi::TD_Option{<:VecNum} = nothing,
        alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot()
    ) -> RelaxedRiskBudgeting

Keywords correspond to the struct's fields. A field of the type [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value. The budgeting algorithm with its budget, the initial weights and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. An `optimise` outside a fold loop uses the static default of each field. The default of `wi` and of `fb` is `nothing`. The model `alg` and the square-root algorithm `mtx_sqrt` control the formulation, and they cannot hold a schedule.

## Validation

  - If `wi` is provided: `!isempty(wi)`. Otherwise an `IsEmptyError`.
  - A schedule in `fb`: `bind !== :nearest`.
  - A part that states no source and reads the raw returns, beside a part that states `x_src = ExcessSource()`, logs one warning that names its path, see [`warn_raw_readers`](@ref). The constructor then builds the estimator as configured.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `opt`: Recursively updated through [`factory`](@ref).
  - `fb`: Recursively updated through [`factory`](@ref).

## View parameters

`RelaxedRiskBudgeting` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` holds a prior result, the method uses `opt.pe.X` in place of `X`. So the views of the children read the observations of the prior, and not the matrix of the caller.
  - `opt` recurses through [`port_opt_view`](@ref) with that matrix. `rba` recurses with the indices alone.
  - `fb` recurses through [`view_child`](@ref) with that matrix.
  - `wi` is sliced to the selected assets.
  - `alg` and `mtx_sqrt` pass unchanged.

# Related

  - [`optimise`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:gambetakwon2020]) Model (0), Equations 5 to 11, and Model (A), Equations 12 to 17. $(ref_dict[:no_original_source])
  - $(ref_dict[:gambetakwon2020]) Model (B), Equations 21 to 27, and Model (C), Equations 31 to 37.
  - $(ref_dict[:mausserromanko2014])
  - $(ref_dict[:cajas2025]) Section 10.1.2, Equations 10.5 and 10.6, for the rotated cone with a general budget.
  - $(ref_dict[:roncalliweisang2012]) Problem 8 and Section 4.1.1, for the budget over the factors.
"""
@propagatable @concrete struct RelaxedRiskBudgeting <: JuMPOptimisationEstimator
    """
    $(field_dict[:opt_jmp])
    """
    @fprop opt
    """
    $(field_dict[:rba])
    """
    rba
    """
    $(field_dict[:wi])
    """
    wi
    """
    Model of the relaxation, a [`RelaxedRiskBudgetingAlgorithm`](@ref).
    """
    alg
    """
    $(field_dict[:fb])
    """
    @fprop fb
    """
    Square-root algorithm of the covariance matrix that the cones read, when the prior carries no `chol`, or `nothing` for the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. The default takes the square root of the eigendecomposition of a singular positive semidefinite matrix. [`matrix_square_root`](@ref) states each algorithm.
    """
    mtx_sqrt
    function RelaxedRiskBudgeting(opt::JuMPOptimiser, rba::TD{<:RiskBudgetingAlgorithm},
                                  wi::TD_Option{<:VecNum},
                                  alg::RelaxedRiskBudgetingAlgorithm,
                                  fb::TDO_Option{<:OptE_Opt},
                                  mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :RelaxedRiskBudgeting)
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        assert_time_dependent_substitution(RelaxedRiskBudgeting, (; opt, rba, wi, alg, fb),
                                           relaxed_risk_budgeting_td_defaults())
        est = new{typeof(opt), typeof(rba), typeof(wi), typeof(alg), typeof(fb),
                  typeof(mtx_sqrt)}(opt, rba, wi, alg, fb, mtx_sqrt)
        warn_raw_readers(est)
        return est
    end
end
function RelaxedRiskBudgeting(; opt::JuMPOptimiser,
                              rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
                              wi::TD_Option{<:VecNum} = nothing,
                              alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
                              fb::TDO_Option{<:OptE_Opt} = nothing,
                              mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())::RelaxedRiskBudgeting
    return RelaxedRiskBudgeting(opt, rba, wi, alg, fb, mtx_sqrt)
end
function time_dependent_field_defaults(::RelaxedRiskBudgeting)::NamedTuple
    return relaxed_risk_budgeting_td_defaults()
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the solution, the budget and the fallback chain of a relaxed risk budgeting optimisation.

[`optimise`](@ref) returns it for a [`RelaxedRiskBudgeting`](@ref). The model reads the covariance matrix of the prior and resolves no risk measure, so the result has no `r`, and it is a [`NonRiskJuMPOptimisationResult`](@ref). A name that is not a field resolves through `prb` first, so `res.rkb` is the budget of the model. Every other name, and the property `w`, then resolves through `jr`, the [`JuMPOptimisationResult`](@ref) that the result holds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelaxedRiskBudgetingResult(;
        jr::JuMPOptimisationResult,
        prb::Union{ProcessedAssetRiskBudgetingAttributes,
                   ProcessedFactorRiskBudgetingAttributes},
        fb::Option{<:OptE_Opt_FbChain}
    ) -> RelaxedRiskBudgetingResult

Keywords correspond to the struct's fields.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`RiskBudgetingResult`](@ref)
  - [`NonRiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
@concrete struct RelaxedRiskBudgetingResult <: NonRiskJuMPOptimisationResult
    """
    The [`JuMPOptimisationResult`](@ref) that holds the processed attributes, the return code, the solution and the model.
    """
    jr
    """
    $(field_dict[:prb])
    """
    prb
    """
    $(field_dict[:fb_res])
    """
    fb
    function RelaxedRiskBudgetingResult(jr::JuMPOptimisationResult,
                                        prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                                   ProcessedFactorRiskBudgetingAttributes},
                                        fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(jr), typeof(prb), typeof(fb)}(jr, prb, fb)
    end
end
function RelaxedRiskBudgetingResult(; jr::JuMPOptimisationResult,
                                    prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                               ProcessedFactorRiskBudgetingAttributes},
                                    fb::Option{<:OptE_Opt_FbChain})::RelaxedRiskBudgetingResult
    return RelaxedRiskBudgetingResult(jr, prb, fb)
end
"""
    set_retcode(res::RelaxedRiskBudgetingResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`RelaxedRiskBudgetingResult`](@ref) with a different return code.

`retcode` is not a field of this result. It resolves through `jr`, the [`JuMPOptimisationResult`](@ref) that the result holds. So the method rebuilds `jr` with the new return code, and keeps `prb` and `fb` unchanged.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Returns

  - [`RelaxedRiskBudgetingResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
"""
function set_retcode(res::RelaxedRiskBudgetingResult, retcode::OptRetCode_VecOptRetCode)
    return RelaxedRiskBudgetingResult(set_retcode(res.jr, retcode), res.prb, res.fb)
end
# Unique field `prb` resolves directly; unknown properties forward into `prb` first, then
# into the embedded [`JuMPOptimisationResult`](@ref) `jr` (the virtual `:w` and `pa` fall-through).
@forward_properties RelaxedRiskBudgetingResult begin
    forward(prb)
    forward(jr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a part of `opt` needs the previous portfolio weights.

The parts are the schedules of the time-dependent fields, the [`JuMPOptimiser`](@ref) `opt.opt` and the fallback `opt.fb`. A turnover constraint or a fee on the turnover is a part that needs them.

# Returns

  - `Bool`: `true` when any part answers `true`.

# Related

  - [`needs_previous_weights`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
"""
function needs_previous_weights(opt::RelaxedRiskBudgeting)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the view of a [`RelaxedRiskBudgeting`](@ref) at the assets `i`.

The `## View parameters` subsection of [`RelaxedRiskBudgeting`](@ref) states how the method takes the view of each field.

# Arguments

  - `rrb`: The optimiser.
  - `i`: Indices of the assets that the view keeps.
  - `X::MatNum`: Returns matrix of the caller. The method uses the matrix of the prior in its place when `rrb.opt.pe` holds a prior result.
  - `args...`: Not read.

# Returns

  - `RelaxedRiskBudgeting`: A new optimiser, built through the keyword constructor.

# Related

  - [`port_opt_view`](@ref)
  - [`view_child`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
"""
function port_opt_view(rrb::RelaxedRiskBudgeting, i, X::MatNum,
                       args...)::RelaxedRiskBudgeting
    X = isa(rrb.opt.pe, AbstractPriorResult) ? rrb.opt.pe.X : X
    opt = port_opt_view(rrb.opt, i, X)
    rba = port_opt_view(rrb.rba, i)
    wi = nothing_scalar_array_view(rrb.wi, i)
    return RelaxedRiskBudgeting(; opt = opt, rba = rba, wi = wi, alg = rrb.alg,
                                fb = view_child(rrb.fb, i, X), mtx_sqrt = rrb.mtx_sqrt)
end
function non_investable_universe(rrb::RelaxedRiskBudgeting,
                                 ni::VecStr)::RelaxedRiskBudgeting
    return rebuild_estimator(rrb, (; rba = non_investable_universe(rrb.rba, ni)))
end
"""
    set_relaxed_risk_budgeting_alg_constraints!(alg, model::JuMP.Model, n::Integer,
                                                Gw::VecJuMPScalar, z::VecJuMPScalar,
                                                sigma_z::MatNum) -> Nothing

Add the cones that bound ``\\psi`` in the model that `alg` selects.

The method dispatches on `alg`. Model (A) adds one cone. Models (B) and (C) add the regulating term ``\\rho`` and two cones. The `# Mathematical definition` of [`RelaxedRiskBudgeting`](@ref) states the three models.

# JuMP formulation

## Variables

  - `psi`: ``\\psi``, read from the model.
  - `rho`: ``\\rho \\geq 0``, created by Models (B) and (C).

## Constraints

  - `cbasic_rrp`: ``s_{c} (\\psi, \\mathbf{G} \\boldsymbol{w}) \\in \\mathcal{K}_{2}``, for [`BasicRelaxedRiskBudgeting`](@ref).
  - `creg_rrp_soc_1`: ``s_{c} (\\psi, \\mathbf{G} \\boldsymbol{w}, \\rho) \\in \\mathcal{K}_{2}``, for [`RegularisedRelaxedRiskBudgeting`](@ref).
  - `creg_rrp_soc_2`: ``s_{c} (\\rho, \\sqrt{n}\\, \\mathbf{G} \\boldsymbol{w}) \\in \\mathcal{K}_{2}``, for [`RegularisedRelaxedRiskBudgeting`](@ref).
  - `creg_pen_rrp_soc_1`: ``s_{c} (\\psi, \\mathbf{G} \\boldsymbol{w}, \\rho) \\in \\mathcal{K}_{2}``, for [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref).
  - `creg_pen_rrp_soc_2`: ``s_{c} (\\rho, \\sqrt{n \\lambda\\, \\mathrm{diag}(\\mathbf{\\Sigma}_{z})} \\odot \\boldsymbol{z}) \\in \\mathcal{K}_{2}``, for [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref).

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])
  - ``n``: Number of entries of the budget.
  - ``\\mathbf{G} \\boldsymbol{w}``: The vector `Gw`, whose norm is the standard deviation of the portfolio.
  - ``\\boldsymbol{z}``, ``\\mathbf{\\Sigma}_{z}``: The decision vector `z` and its covariance matrix `sigma_z`.
  - ``\\lambda``: Penalty `alg.p`.

# Arguments

  - `alg`: The model, a [`RelaxedRiskBudgetingAlgorithm`](@ref).
  - $(arg_dict[:model])
  - `n`: Number of entries of the budget.
  - `Gw`: The covariance factor times the weights. Its norm is the standard deviation of the portfolio.
  - `z`: The decision vector that the penalty of Model (C) reads. It is the weights for [`AssetRiskBudgeting`](@ref). For [`FactorRiskBudgeting`](@ref) it is the factor weights, followed by the off-factor weights when `flag = true`.
  - `sigma_z`: Covariance matrix of `z`.

# Returns

  - `nothing`.

# Related

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`_set_relaxed_risk_budgeting_constraints!`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
"""
function set_relaxed_risk_budgeting_alg_constraints!(::BasicRelaxedRiskBudgeting,
                                                     model::JuMP.Model, n::Integer,
                                                     Gw::VecJuMPScalar, z::VecJuMPScalar,
                                                     sigma_z::MatNum)
    sc = get_constraint_scale(model)
    psi = shared_get(model, :psi)
    JuMP.@constraint(model, cbasic_rrp, [sc * psi; sc * Gw] in JuMP.SecondOrderCone())
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(::RegularisedRelaxedRiskBudgeting,
                                                     model::JuMP.Model, n::Integer,
                                                     Gw::VecJuMPScalar, z::VecJuMPScalar,
                                                     sigma_z::MatNum)
    sc = get_constraint_scale(model)
    psi = shared_get(model, :psi)
    JuMP.@variable(model, rho >= 0)
    JuMP.@constraints(model,
                      begin
                          creg_rrp_soc_1,
                          [sc * psi; sc * Gw; sc * rho] in JuMP.SecondOrderCone()
                          creg_rrp_soc_2,
                          [sc * rho; sc * sqrt(n) * Gw] in JuMP.SecondOrderCone()
                      end)
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(alg::RegularisedPenalisedRelaxedRiskBudgeting,
                                                     model::JuMP.Model, n::Integer,
                                                     Gw::VecJuMPScalar, z::VecJuMPScalar,
                                                     sigma_z::MatNum)
    sc = get_constraint_scale(model)
    psi = shared_get(model, :psi)
    theta = sqrt.(n * alg.p * LinearAlgebra.diag(sigma_z))
    JuMP.@variable(model, rho >= 0)
    JuMP.@constraints(model,
                      begin
                          creg_pen_rrp_soc_1,
                          [sc * psi; sc * Gw; sc * rho] in JuMP.SecondOrderCone()
                          creg_pen_rrp_soc_2,
                          [sc * rho; sc * (theta .* z)] in JuMP.SecondOrderCone()
                      end)
    return nothing
end
"""
    _set_relaxed_risk_budgeting_constraints!(model, rrb, x, A, w, sigma, chol = nothing,
                                             z = w, sigma_z = sigma)

Add the variables, the cones of the budget and the cones on ``\\psi`` of a [`RelaxedRiskBudgeting`](@ref) to the model.

Each budgeted weight ``x_{i}`` takes one cone with its marginal risk ``\\zeta_{i} = [\\mathbf{A} \\boldsymbol{w}]_{i}``, so ``x_{i} \\zeta_{i}`` is its contribution to the variance. The function then takes the covariance factor ``\\mathbf{G}`` of `sigma`, and adds the cones of `rrb.alg` through [`set_relaxed_risk_budgeting_alg_constraints!`](@ref). The callers pass these values:

| Caller                                        | `x`            | `A`                                                          | `w`            | `sigma`                                                      |
|:--------------------------------------------- |:-------------- |:------------------------------------------------------------ |:-------------- |:------------------------------------------------------------ |
| [`AssetRiskBudgeting`](@ref)                  | asset weights  | ``\\mathbf{\\Sigma}``                                        | asset weights  | ``\\mathbf{\\Sigma}``                                        |
| [`FactorRiskBudgeting`](@ref), `flag = false` | factor weights | ``\\mathbf{B}_1^\\intercal \\mathbf{\\Sigma} \\mathbf{B}_1`` | factor weights | ``\\mathbf{B}_1^\\intercal \\mathbf{\\Sigma} \\mathbf{B}_1`` |
| [`FactorRiskBudgeting`](@ref), `flag = true`  | factor weights | ``\\mathbf{B}_1^\\intercal \\mathbf{\\Sigma}``               | asset weights  | ``\\mathbf{\\Sigma}``                                        |

# JuMP formulation

## Variables

  - `psi`: ``\\psi \\geq 0``.
  - `gamma`: ``\\gamma \\geq 0``.
  - `zeta`: ``\\boldsymbol{\\zeta} \\geq \\boldsymbol{0}``, one entry for each entry of `x`.

## Expressions

  - `risk`: ``\\psi - \\gamma``, which the objective [`MinimumRisk`](@ref) minimises.

## Constraints

  - `crrp`: ``s_{c} (\\boldsymbol{\\zeta} - \\mathbf{A} \\boldsymbol{w}) = \\boldsymbol{0}``.
  - `crrp_soc[i]`: ``s_{c} (x_{i} + \\zeta_{i}, 2 \\gamma \\sqrt{b_{i}}, x_{i} - \\zeta_{i}) \\in \\mathcal{K}_{2}``, which is ``x_{i} \\zeta_{i} \\geq b_{i} \\gamma^{2}`` with ``x_{i} + \\zeta_{i} \\geq 0``.
  - The rows of [`set_relaxed_risk_budgeting_alg_constraints!`](@ref).

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])
  - ``\\boldsymbol{x}``, ``\\mathbf{A}``, ``\\boldsymbol{w}``: The arguments `x`, `A` and `w`.
  - ``n``: Number of entries of the budget, `length(x)`.
  - ``b_{i}``: Share of entry ``i`` in the budget, its entry divided by the sum of the entries. The chain of the `# Mathematical definition` of [`RelaxedRiskBudgeting`](@ref) needs shares that sum to one, and a [`RiskBudget`](@ref) built by hand keeps its own sum.

## Relaxation

$(val_dict[:relax])

  - The row `crrp_soc[i]` holds `gamma` at or **below** ``\\min_{i} \\sqrt{x_{i} \\zeta_{i} / b_{i}}``, and the cones of `rrb.alg` hold `psi` at or **above** the standard deviation of the portfolio. So `risk` lies at or above the gap between the standard deviation and the least scaled contribution.
  - For [`BasicRelaxedRiskBudgeting`](@ref), with no off-factor weights, the bound is tight, and `risk` is zero, exactly when ``x_{i} \\zeta_{i} = b_{i} \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}`` for every ``i``. The minimum reaches it when no other row of the model binds at that portfolio.
  - For the regularised models, `psi` also bounds the regulating term, so the bound is not tight. At a portfolio that meets the budget, `risk` is at least ``(\\sqrt{n + 1} - 1)`` times the standard deviation in Model (B).

# Arguments

  - $(arg_dict[:model])
  - `rrb::RelaxedRiskBudgeting`: The optimiser. The function reads its budget, `alg`, `mtx_sqrt` and `opt.strict`.
  - `x::VecJuMPScalar`: The budgeted weights, one entry for each entry of the budget.
  - `A::MatNum`: The matrix that maps `w` to the marginal risks of `x`.
  - `w::VecJuMPScalar`: The weights whose standard deviation the cones on ``\\psi`` read.
  - `sigma::MatNum`: Covariance matrix of `w`.
  - `chol::Option{<:MatNum}`: Covariance factor of `sigma`, with `chol' * chol == sigma`. When it is `nothing`, the function takes the transpose of the square root of `sigma` that [`matrix_square_root`](@ref) gives under `rrb.mtx_sqrt`.
  - `z::VecJuMPScalar`: The decision vector that the penalty of [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref) reads: `w`, or the factor weights followed by the off-factor weights when `flag = true`.
  - `sigma_z::MatNum`: Covariance matrix of `z`.

# Validation

  - The budget has one entry for each entry of `x`. Otherwise a `DimensionMismatch` names both lengths.

# Returns

  - `rkb::RiskBudget`: The budget that the cones read.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`set_relaxed_risk_budgeting_constraints!`](@ref)
  - [`set_relaxed_risk_budgeting_alg_constraints!`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function _set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                  rrb::RelaxedRiskBudgeting,
                                                  x::VecJuMPScalar, A::MatNum,
                                                  w::VecJuMPScalar, sigma::MatNum,
                                                  chol::Option{<:MatNum} = nothing,
                                                  z::VecJuMPScalar = w,
                                                  sigma_z::MatNum = sigma)
    N = length(x)
    rkb = risk_budget_constraints(rrb.rba.rkb, rrb.rba.sets,
                                  risk_budget_universe_key(rrb.rba, N); N = N,
                                  strict = rrb.opt.strict)
    # The chain `gamma^2 <= sum_i x_i zeta_i <= psi^2` needs a budget that sums to one, and a
    # hand-built `RiskBudget` keeps its own sum, so the cones read its shares.
    rb = rkb.val / sum(rkb.val)
    @argcheck(length(rb) == N, DimensionMismatch("rb ($(length(rb))) must match N ($N)"))
    sc = get_constraint_scale(model)
    JuMP.@variables(model, begin
                        psi >= 0
                        gamma >= 0
                        zeta[1:N] >= 0
                    end)
    JuMP.@expression(model, risk, psi - gamma)
    # RRB constraints.
    JuMP.@constraints(model,
                      begin
                          crrp, sc * (zeta - A * w) == 0
                          crrp_soc[i = 1:N],
                          [sc * (x[i] + zeta[i])
                           sc * (2 * gamma * sqrt(rb[i]))
                           sc * (x[i] - zeta[i])] in JuMP.SecondOrderCone()
                      end)
    G = isnothing(chol) ? transpose(matrix_square_root(rrb.mtx_sqrt, sigma)) : chol
    set_relaxed_risk_budgeting_alg_constraints!(rrb.alg, model, N, G * w, z, sigma_z)
    return rkb
end
"""
    set_relaxed_risk_budgeting_constraints!(model, rrb, pr, wb, args...)
    set_relaxed_risk_budgeting_constraints!(model, rrb, pr, wb, rd::ReturnsResult)

Add the weights, the cones of the model and the weight constraints of a [`RelaxedRiskBudgeting`](@ref) to the model.

The method dispatches on `rrb.rba`. Each method registers the weights, calls [`_set_relaxed_risk_budgeting_constraints!`](@ref) with the vectors and the matrices that its budget reads, and adds the weight bounds and the budgets with [`set_weight_constraints!`](@ref). The table of [`_set_relaxed_risk_budgeting_constraints!`](@ref) states the values that each method passes.

# JuMP formulation

## Variables

  - `w`: ``\\boldsymbol{w}``, registered by [`set_w!`](@ref) for [`AssetRiskBudgeting`](@ref).
  - `w1`, `w2`: ``\\boldsymbol{w}_{1}``, ``\\boldsymbol{w}_{2}``, registered by [`set_factor_risk_contribution_constraints!`](@ref) for [`FactorRiskBudgeting`](@ref). `w` is then an expression of them.

## Constraints

  - The rows of [`_set_relaxed_risk_budgeting_constraints!`](@ref), with the budget on ``\\boldsymbol{w}`` for [`AssetRiskBudgeting`](@ref), and on ``\\boldsymbol{w}_{1}`` for [`FactorRiskBudgeting`](@ref).
  - The rows of [`set_weight_constraints!`](@ref). [`AssetRiskBudgeting`](@ref) calls it with `long = true`.

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_{1}``: Factor weights.
  - $(math_dict[:w_2_off_factor])

# Arguments

  - $(arg_dict[:model])
  - `rrb`: The optimiser.
  - $(arg_dict[:pr])
  - $(arg_dict[:wb_arg])
  - `rd`: Returns data that [`FactorRiskBudgeting`](@ref) reads when it fits the loadings. The asset method does not read it.

# Validation

  - [`AssetRiskBudgeting`](@ref): no weight bound is negative. Otherwise [`set_weight_constraints!`](@ref) throws an `ArgumentError`.

# Returns

  - [`ProcessedAssetRiskBudgetingAttributes`](@ref) for [`AssetRiskBudgeting`](@ref), and [`ProcessedFactorRiskBudgetingAttributes`](@ref) for [`FactorRiskBudgeting`](@ref).

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`_set_relaxed_risk_budgeting_constraints!`](@ref)
  - [`set_relaxed_risk_budgeting_alg_constraints!`](@ref)
"""
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting{<:Any,
                                                                           <:FactorRiskBudgeting,
                                                                           <:Any, <:Any},
                                                 pr::AbstractPriorResult, wb::WeightBounds,
                                                 rd::ReturnsResult)
    b1, b2, rr = set_factor_risk_contribution_constraints!(model, rrb.rba.re, rd, pr,
                                                           rrb.rba.flag, rrb.wi;
                                                           hedge = rrb.rba.hedge)
    w1 = shared_get(model, :w1)
    rkb = if isnothing(b2)
        sigma_f = Matrix(LinearAlgebra.Symmetric(rr.L \ pr.sigma * b1))
        _set_relaxed_risk_budgeting_constraints!(model, rrb, w1, sigma_f, w1, sigma_f)
    else
        # The off-factor weights move the risk too, so the marginal risks and the risk cone
        # read the asset weights `w = b1 * w1 + b2 * w2`, not the factor part alone (#1351).
        # A penalty reads the whole decision vector `z = [w1; w2]`, so the off-factor weights
        # cannot take up the variance that the penalty moves off the factors.
        P = hcat(b1, b2)
        _set_relaxed_risk_budgeting_constraints!(model, rrb, w1, transpose(b1) * pr.sigma,
                                                 get_w(model), pr.sigma, pr.chol,
                                                 vcat(w1, shared_get(model, :w2)),
                                                 Matrix(LinearAlgebra.Symmetric(transpose(P) *
                                                                                pr.sigma *
                                                                                P)))
    end
    set_weight_constraints!(model, wb, rrb.opt)
    return ProcessedFactorRiskBudgetingAttributes(; rkb = rkb, b1 = b1, rr = rr)
end
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting{<:Any,
                                                                           <:AssetRiskBudgeting,
                                                                           <:Any, <:Any},
                                                 pr::AbstractPriorResult, wb::WeightBounds,
                                                 args...)
    set_w!(model, pr.X, rrb.wi)
    set_weight_constraints!(model, wb, rrb.opt, true)
    w = get_w(model)
    rkb = _set_relaxed_risk_budgeting_constraints!(model, rrb, w, pr.sigma, w, pr.sigma,
                                                   pr.chol)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
"""
    _optimise(rrb::RelaxedRiskBudgeting, rd::ReturnsResult = ReturnsResult();
              str_names::Bool = false, save::Bool = true, kwargs...) -> RelaxedRiskBudgetingResult

Build the model of a [`RelaxedRiskBudgeting`](@ref), solve it, and return its result.

[`optimise`](@ref) reaches this method through [`shared_series_solve`](@ref), which drops the observations of a Shared Series gap before the solve.

# Algorithm

 1. Replace each schedule of `rrb` with its static default through [`reset_time_dependent_estimator`](@ref).
 2. Process the attributes of `rrb.opt` against `rd` with [`processed_jump_optimiser_attributes`](@ref), giving `attrs`. It holds the prior, the weight bounds, the fees, the return terms and the Investable Mask.
 3. Take the view of `rrb` and of `rd` at the investable assets with [`investable_view`](@ref). Both are unchanged when every asset is investable.
 4. Bind the benchmark of each [`DataBenchmarkTracking`](@ref) term to `rd` with [`bind_data_benchmark`](@ref).
 5. Build a new model. Register the scales with [`set_model_scales!`](@ref), the number of observations with [`set_model_observations!`](@ref), and the expression `k = 1` with [`set_maximum_ratio_factor_variables!`](@ref), because [`MinimumRisk`](@ref) does not scale the weights.
 6. Add the weights, the cones and the weight constraints with [`set_relaxed_risk_budgeting_constraints!`](@ref), giving `prb`.
 7. Add the constraints and the return terms of `rrb.opt` with [`assemble_jump_model!`](@ref).
 8. Set the objective [`MinimumRisk`](@ref), which minimises `risk`, and solve with [`optimise_JuMP_model!`](@ref), giving `retcode` and `sol`.
 9. Build the result. It holds `attrs`, `retcode`, `sol`, the model when `save` is `true`, and `prb`.

# Arguments

  - `rrb::RelaxedRiskBudgeting`: The optimiser.
  - $(arg_dict[:rd])
  - `str_names::Bool`: `true` gives each variable and each row of the model its string name.
  - `save::Bool`: `true` keeps the model in the result.
  - `kwargs...`: Passed on to [`processed_jump_optimiser_attributes`](@ref).

# Returns

  - [`RelaxedRiskBudgetingResult`](@ref): The result, with `fb = nothing`. The method of [`optimise`](@ref) for an optimiser with a fallback runs the fallback chain.

# Related

  - [`optimise`](@ref)
  - [`set_relaxed_risk_budgeting_constraints!`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
"""
function _optimise(rrb::RelaxedRiskBudgeting, rd::ReturnsResult = ReturnsResult();
                   str_names::Bool = false, save::Bool = true, kwargs...)
    rrb = reset_time_dependent_estimator(rrb)
    attrs = processed_jump_optimiser_attributes(rrb.opt, rd; kwargs...)
    # The bundle reduced what it carries. The head carries the rest — an initial weight
    # vector, a risk measure holding per-asset data, tracking, a custom term — and hands
    # them to `assemble_jump_model!` itself, so it takes the same view of itself and of
    # `rd`. Both are unchanged when every asset is investable.
    rrb, rd = investable_view(rrb, rd, attrs.pr, attrs.imsk)
    rrb = bind_data_benchmark(rrb, rd)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rrb.opt.sc, rrb.opt.so)
    set_model_observations!(model, size(attrs.pr.X, 1))
    set_maximum_ratio_factor_variables!(model, MinimumRisk())
    prb = set_relaxed_risk_budgeting_constraints!(model, rrb, attrs.pr, attrs.wb, rd)
    assemble_jump_model!(model, rrb, rrb.opt, attrs, rd)
    set_portfolio_objective_function!(model, MinimumRisk(), rrb, attrs)
    retcode, sol = optimise_JuMP_model!(model, rrb, eltype(attrs.pr.X))
    jr = JuMPOptimisationResult(; pa = attrs, retcode = retcode, sol = sol,
                                model = ifelse(save, model, nothing))
    return RelaxedRiskBudgetingResult(; jr = jr, prb = prb, fb = nothing)
end
"""
    optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; str_names::Bool = false, save::Bool = true, kwargs...) -> RelaxedRiskBudgetingResult

Optimise a [`RelaxedRiskBudgeting`](@ref) that has no fallback, and return its result.

This method serves an optimiser whose `fb` is `nothing`. It solves through [`shared_series_solve`](@ref), and [`_optimise`](@ref) states the steps of the solve.

# Arguments

  - `rrb`: The optimiser.
  - $(arg_dict[:rd]) When `rrb.opt.pe` is a prior result, the solve does not read `rd`, except to fit the loadings of [`FactorRiskBudgeting`](@ref) when no other source holds them. A fallback or a clustering step can still read it.
  - `str_names`: `true` gives each variable and each row of the model its string name.
  - `save`: `true` keeps the model in the result.
  - `kwargs`: Passed on to [`_optimise`](@ref).

# Validation

  - No field in the tree of `rrb` holds an [`Online`](@ref). Otherwise [`assert_batch_entry`](@ref) throws an `ArgumentError` that names the field. A plain `optimise` is a batch fit, and the fold loop resolves an `Online` only in its online branch.

# Returns

  - [`RelaxedRiskBudgetingResult`](@ref): The result of the solve.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
  - [`_optimise`](@ref)
"""
function optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; str_names::Bool = false, save::Bool = true, kwargs...)
    assert_batch_entry(rrb, "`optimise`")
    return shared_series_solve(rrb, rd; str_names = str_names, save = save, kwargs...)
end

@pipe_delegates RelaxedRiskBudgeting opt
@pipe_route_rkb RelaxedRiskBudgeting
export BasicRelaxedRiskBudgeting, RegularisedRelaxedRiskBudgeting,
       RegularisedPenalisedRelaxedRiskBudgeting, RelaxedRiskBudgeting,
       RelaxedRiskBudgetingResult
