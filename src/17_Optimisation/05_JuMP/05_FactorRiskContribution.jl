"""
$(DocStringExtensions.TYPEDEF)

Result of a [`FactorRiskContribution`](@ref) optimisation.

# Fields

$(DocStringExtensions.FIELDS)

A property that is not a field resolves in `rr` first, and then in `jr`. So `res.L` reads the loadings, and `res.w` reads the asset weights through the virtual `:w` of [`JuMPOptimisationResult`](@ref).

# Constructors

    FactorRiskContributionResult(;
        jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM, rr::AbstractLoadingsRegressionResult,
        frc_plr::Option{<:PlC_VecPlC}, fb::Option{<:OptE_Opt_FbChain}
    ) -> FactorRiskContributionResult

Keywords correspond to the struct's fields.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
@concrete struct FactorRiskContributionResult <: RiskJuMPOptimisationResult
    """
    Shared JuMP result core, see [`JuMPOptimisationResult`](@ref).
    """
    jr
    """
    $(field_dict[:r_res])
    """
    r
    """
    $(field_dict[:reg_rr])
    """
    rr
    """
    Phylogeny constraints on the factor weights, resolved on the factor returns. `nothing` when the estimator holds none.
    """
    frc_plr
    """
    $(field_dict[:fb_res])
    """
    fb
    function FactorRiskContributionResult(jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                          rr::AbstractLoadingsRegressionResult,
                                          frc_plr::Option{<:PlC_VecPlC},
                                          fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(jr), typeof(r), typeof(rr), typeof(frc_plr), typeof(fb)}(jr, r,
                                                                                   rr,
                                                                                   frc_plr,
                                                                                   fb)
    end
end
function FactorRiskContributionResult(; jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                      rr::AbstractLoadingsRegressionResult,
                                      frc_plr::Option{<:PlC_VecPlC},
                                      fb::Option{<:OptE_Opt_FbChain})::FactorRiskContributionResult
    return FactorRiskContributionResult(jr, r, rr, frc_plr, fb)
end
"""
    set_retcode(res::FactorRiskContributionResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`FactorRiskContributionResult`](@ref) with a different return code.

`retcode` is a field of the [`JuMPOptimisationResult`](@ref) in `jr`, not of this result. So the method rebuilds `jr` and keeps every other field.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Returns

  - [`FactorRiskContributionResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`FactorRiskContributionResult`](@ref)
"""
function set_retcode(res::FactorRiskContributionResult, retcode::OptRetCode_VecOptRetCode)
    return FactorRiskContributionResult(set_retcode(res.jr, retcode), res.r, res.rr,
                                        res.frc_plr, res.fb)
end
# Unique fields resolve directly; unknown properties forward into `rr` first, then into the
# embedded [`JuMPOptimisationResult`](@ref) `jr` (the virtual `:w` and `pa` fall-through).
@forward_properties FactorRiskContributionResult begin
    forward(rr)
    forward(jr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`FactorRiskContribution`](@ref) fields that can hold a [`TimeDependent`](@ref).

The constructor reads them to check each schedule, and [`time_dependent_field_defaults`](@ref) returns them, so the value of a field outside a fold is written once. A field whose static default is `nothing` is not in the tuple.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function factor_risk_contribution_td_defaults()::NamedTuple
    return (; re = StepwiseRegression(), r = Variance(), obj = MinimumRisk())
end
"""
$(DocStringExtensions.TYPEDEF)

Solves a mean-risk problem whose decision variable is the vector of factor weights, not the vector of asset weights.

The asset weights are a linear map of the factor weights through the factor loadings, so a constraint on the decision variable is a constraint on a factor. `FactorRiskContribution` changes the basis and sets **no** risk budget of its own. A target contribution of a factor is a row of the `rc` constraints of the risk measure, as the target of an asset is.

Under [`Variance`](@ref) the `rc` rows sit on the semidefinite relaxation of Formulation 16 of [sdprp](@cite), over the factor weights. With `flag = true` the lift covers the off-factor weights ``\\boldsymbol{w}_2`` too. The variance is then the variance of the returned weights, and each row states the Euler contribution of a factor as a share of the whole variance, as [`factor_risk_contribution`](@ref) reports it. [`set_risk_constraints!`](@ref) states the formulation. A solve can report success while the factor shares of the returned portfolio miss the rows, so compare [`factor_risk_contribution`](@ref) of the result with them. The `## Risk contribution constraints` subsection of [`Variance`](@ref) states the relaxation and the condition under which the rows bind.

A [`Variance`](@ref) always takes this semidefinite formulation here, with or without rows. Under [`MaximumRatio`](@ref) the `ratio_alg` of the variance selects the degree of its objective term, and so the ratio. The default gives the Sharpe ratio, and [`LinearRatio`](@ref) gives the excess return per unit of variance. The `# Mathematical definition` of [`MaximumRatio`](@ref) states the rule, and [`set_sdp_frc_objective_lift!`](@ref) states the lift.

The optimiser solves its model once. A number in `settings.ub` of a risk measure logs a warning, and the model ignores the bound. A [`Frontier`](@ref) or a vector in `settings.ub` raises an `ArgumentError`, see [`set_risk_upper_bound!`](@ref). A return term with a [`Frontier`](@ref) or a vector floor raises an `ArgumentError`, see [`assert_frontier_sweep_owner`](@ref). [`MeanRisk`](@ref) sweeps a frontier.

# Mathematical definition

The factor model of the loadings, which `re` fits or the prior carries:

```math
\\begin{align}
\\mathbf{R} &= \\mathbf{F} \\mathbf{B}^\\intercal + \\mathbf{E}\\,.
\\end{align}
```

The asset weights are a linear map of the decision variables. The first form is the model when `flag = false`, and the second is the model when `flag = true`:

```math
\\begin{align}
\\boldsymbol{w} &= \\mathbf{B}_1 \\boldsymbol{w}_1\\,, \\\\
\\boldsymbol{w} &= \\mathbf{B}_1 \\boldsymbol{w}_1 + \\mathbf{B}_2 \\boldsymbol{w}_2\\,, \\\\
\\mathbf{B}_1 &= (\\mathbf{B}^\\intercal)^{+}\\,.
\\end{align}
```

The columns of ``\\mathbf{B}_2`` are an orthonormal basis of the null space of ``\\mathbf{B}^\\intercal``, so ``\\mathbf{B}^\\intercal \\mathbf{B}_2 = \\mathbf{0}``. When ``\\mathbf{B}`` has full column rank, ``\\mathbf{B}^\\intercal \\mathbf{B}_1`` is the identity, and the factor exposures of the portfolio are ``\\mathbf{B}^\\intercal \\boldsymbol{w} = \\boldsymbol{w}_1``. The first form keeps ``\\boldsymbol{w}`` in the column space of ``\\mathbf{B}``. The second form adds the directions that the loadings do not span, so ``\\boldsymbol{w}`` can be any vector of size ``N``.

The problem is the mean-risk problem of [`MeanRisk`](@ref) over the decision variables:

```math
\\begin{align}
\\underset{\\boldsymbol{w}_1,\\, \\boldsymbol{w}_2}{\\min} \\; f(\\boldsymbol{w}) \\quad \\text{s.t.} \\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

The risk contribution of factor ``j``, which [`factor_risk_contribution`](@ref) reports, is the Euler decomposition in this basis. The last entry is the contribution of the off-factor part, and the entries sum to ``\\boldsymbol{w}^\\intercal \\nabla R(\\boldsymbol{w})``, which is ``R(\\boldsymbol{w})`` for a risk measure that is homogeneous of degree one:

```math
\\begin{align}
RC_j &= \\left[ \\mathbf{B}^\\intercal \\boldsymbol{w} \\right]_j \\left[ \\mathbf{B}_1^\\intercal \\nabla R(\\boldsymbol{w}) \\right]_j\\,, \\quad j = 1, \\ldots, N_f\\,, \\\\
RC_{N_f + 1} &= \\left( \\mathbf{B}_2^\\intercal \\boldsymbol{w} \\right)^\\intercal \\mathbf{B}_2^\\intercal \\nabla R(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``\\mathbf{R}``: Asset returns matrix, ``T \\times N``.
  - ``\\mathbf{F}``: Factor returns matrix, ``T \\times N_f``.
  - $(math_dict[:B_loadings])
  - ``\\mathbf{E}``: Residual matrix, ``T \\times N``.
  - $(math_dict[:w_port])
  - $(math_dict[:w_1_factor])
  - $(math_dict[:w_2_off_factor])
  - ``\\mathbf{B}_1``: The factor basis, ``N \\times N_f``.
  - ``\\mathbf{B}_2``: The off-factor basis, ``N \\times (N - \\operatorname{rank}\\mathbf{B})``.
  - ``(\\cdot)^{+}``: Moore-Penrose pseudoinverse.
  - ``f(\\boldsymbol{w})``: The objective, which `obj` selects, see [`ObjectiveFunction`](@ref).
  - $(math_dict[:W_feas])
  - $(math_dict[:R_w])
  - ``RC_j``: Risk contribution of factor ``j``.
  - $(math_dict[:N])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorRiskContribution(;
        opt::JuMPOptimiser,
        re::TD{<:RegE_Reg} = StepwiseRegression(),
        r::TD{<:RM_VecRM} = Variance(),
        obj::TD{<:ObjectiveFunction} = MinimumRisk(),
        frc_ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wi::TD_Option{<:VecNum} = nothing,
        flag::Bool = false,
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> FactorRiskContribution

Keywords correspond to the struct's fields. A field typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value. These fields define the problem: the factor model, the risk measure, the objective, the factor phylogeny constraints, the asset sets, the warm start and the fallback. A cross-validation fold loop resolves each schedule for its fold, and an `optimise` outside a fold uses the static default of each field. `flag` controls the execution and is always static.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is a vector: `!isempty(wi)`.
  - `fb` schedules: `bind !== :nearest`.
  - The risk expression is not identically zero. The problem is stated in the factor basis,
    so a model with no risk term bounds nothing. A [`NoRisk`](@ref) measure and
    `settings.rke = false` on every measure are the two ways to get such a model.
  - If `r` holds a [`DataBenchmarkTracking`](@ref): `opt.pe` must not state `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).
  - Each estimator of `frc_ple` states no source, because it reads the factor returns that the head picks, see [`assert_inner_reader`](@ref).
  - A reader that states no source and reads the raw returns data, beside a reader that states `x_src = ExcessSource()`, logs one warning that names its path, see [`warn_raw_readers`](@ref). The head then builds as configured.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `opt`: Recursively updated through [`factory`](@ref).
  - `r`: Recursively updated through [`factory`](@ref).
  - `fb`: Recursively updated through [`factory`](@ref).

## View parameters

`FactorRiskContribution` has its own [`port_opt_view`](@ref) method. It does not derive the view from the tags of the fields.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` holds a prior **result**, the method replaces `X` with `opt.pe.X`, so the children are viewed against the observations of the prior and not against the matrix of the caller.
  - `opt` and `r` recurse through [`port_opt_view`](@ref) with that matrix. `re` recurses with the index alone, which slices its loadings to the selected assets and keeps every factor.
  - `fb` goes through [`view_child`](@ref), which views a fallback estimator and keeps a fallback result.
  - `wi` is not changed, because it holds start values of the **factor** weights. The asset selection does not index the factor axis.
  - `obj`, `frc_ple`, `sets` and `flag` are not changed.

# Related

  - [`optimise`](@ref)
  - [`FactorRiskContributionResult`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:roncalliweisang2012]) Section 3, Equation 2 and Theorem 1. The matrix ``\\mathbf{A}`` of that work is ``\\mathbf{B}`` here.
  - $(ref_dict[:meucci2007]) Section 4, Equations 20 and 21. The completion of the basis by the null space.
  - $(ref_dict[:sdprp]) Section 3.2, Formulation 16. The semidefinite relaxation of the `rc` rows under [`Variance`](@ref).
"""
@propagatable @concrete struct FactorRiskContribution <: RiskJuMPOptimisationEstimator
    """
    $(field_dict[:opt_jmp])
    """
    @fprop opt
    """
    $(field_dict[:re])
    """
    re
    """
    $(field_dict[:r_opt])
    """
    @fprop r
    """
    $(field_dict[:obj])
    """
    obj
    """
    Phylogeny constraints on the factor weights, an estimator or a result, or a vector of them. A semidefinite entry adds rows through [`set_sdp_frc_phylogeny_constraints!`](@ref), and an integer entry through [`set_frc_iplg_constraints!`](@ref), which needs a MIP solver and finite asset weight bounds.
    """
    frc_ple
    """
    $(field_dict[:sets])
    """
    sets
    """
    $(field_dict[:wi])
    """
    wi
    """
    $(field_dict[:flag])
    """
    flag
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function FactorRiskContribution(opt::JuMPOptimiser, re::TD{<:RegE_Reg},
                                    r::TD{<:RM_VecRM}, obj::TD{<:ObjectiveFunction},
                                    frc_ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC},
                                    sets::TD_Option{<:UniverseSets},
                                    wi::TD_Option{<:VecNum}, flag::Bool,
                                    fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :FactorRiskContribution)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        assert_risk_measure_required(r, :FactorRiskContribution;
                                     flag = zero_risk_expression_flag)
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        assert_time_dependent_substitution(FactorRiskContribution,
                                           (; opt, re, r, obj, frc_ple, sets, wi, flag, fb),
                                           factor_risk_contribution_td_defaults())
        assert_data_benchmark_prior(r, opt.pe, :r, :FactorRiskContribution, "opt.pe")
        # The factor phylogeny reads the factor returns that the head picks, so each of its
        # estimators is an inner reader.
        foreach(x -> assert_inner_reader(x, :frc_ple, :FactorRiskContribution),
                vcat(frc_ple))
        est = new{typeof(opt), typeof(re), typeof(r), typeof(obj), typeof(frc_ple),
                  typeof(sets), typeof(wi), typeof(flag), typeof(fb)}(opt, re, r, obj,
                                                                      frc_ple, sets, wi,
                                                                      flag, fb)
        warn_raw_readers(est)
        return est
    end
end
function FactorRiskContribution(; opt::JuMPOptimiser,
                                re::TD{<:RegE_Reg} = StepwiseRegression(),
                                r::TD{<:RM_VecRM} = Variance(),
                                obj::TD{<:ObjectiveFunction} = MinimumRisk(),
                                frc_ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
                                sets::TD_Option{<:UniverseSets} = nothing,
                                wi::TD_Option{<:VecNum} = nothing, flag::Bool = false,
                                fb::TDO_Option{<:OptE_Opt} = nothing)::FactorRiskContribution
    return FactorRiskContribution(opt, re, r, obj, frc_ple, sets, wi, flag, fb)
end
function time_dependent_field_defaults(::FactorRiskContribution)::NamedTuple
    return factor_risk_contribution_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a part of `opt` needs the previous portfolio weights.

The parts are the schedules of the time-dependent fields, the [`JuMPOptimiser`](@ref) in `opt.opt`, the risk measure `opt.r` and the fallback `opt.fb`.

# Related

  - [`needs_previous_weights`](@ref)
  - [`FactorRiskContribution`](@ref)
"""
function needs_previous_weights(opt::FactorRiskContribution)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the view of a [`FactorRiskContribution`](@ref) at the assets `i`, for the returns matrix `X`.

The `## View parameters` subsection of [`FactorRiskContribution`](@ref) states how each field is viewed.

# Related

  - [`port_opt_view`](@ref)
  - [`FactorRiskContribution`](@ref)
"""
function port_opt_view(frc::FactorRiskContribution, i, X::MatNum,
                       args...)::FactorRiskContribution
    X = isa(frc.opt.pe, AbstractPriorResult) ? frc.opt.pe.X : X
    opt = port_opt_view(frc.opt, i, X)
    re = port_opt_view(frc.re, i)
    r = port_opt_view(frc.r, i, X)
    return FactorRiskContribution(; opt = opt, re = re, r = r, obj = frc.obj,
                                  frc_ple = frc.frc_ple, sets = frc.sets, wi = frc.wi,
                                  flag = frc.flag, fb = view_child(frc.fb, i, X))
end
"""
    set_factor_risk_contribution_constraints!(model, re, rd, pr, flag, wi; hedge = false)

Create the factor weights of a model, and register the asset weights `w` as their image in the asset basis.

The `# Mathematical definition` of [`FactorRiskContribution`](@ref) states the two forms of the map, which `flag` selects.

When `flag` and `hedge` are both `true`, the rows ``\\mathbf{B}_2^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} = \\boldsymbol{0}`` fix the off-factor weights at the minimum-variance hedge of the factor exposures ``\\boldsymbol{w}_1``. They are the first-order condition of ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}`` in ``\\boldsymbol{w}_2``. So the off-factor weights take no risk contribution under the variance, and ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} = \\boldsymbol{w}_1^\\intercal (\\mathbf{B}^\\intercal \\mathbf{\\Sigma}^{-1} \\mathbf{B})^{-1} \\boldsymbol{w}_1``. The weights then lie in a subspace of dimension ``N_f``, so a bound that the free off-factor weights meet can make the model infeasible.

[`resolve_factor_regression`](@ref) picks the loadings, in the order that the value-level [`factor_risk_contribution`](@ref) uses: a precomputed [`Regression`](@ref) in `re`, then the `rr` of the prior, then a fit on `rd`. The prior comes before the fit, so the basis of the decision variable is the basis through which the prior projected its moments.

!!! warning

    A prior that carries loadings comes before a regression **estimator** in `re`. To replace the loadings of a factor prior, pass them as a precomputed [`Regression`](@ref) in `re`.

# Algorithm

 1. Pick the loadings `rr` with [`resolve_factor_regression`](@ref).
 2. Form `Bt`, the transpose of the loadings, and `b1`, the pseudoinverse of `Bt`. `Nf` is the number of columns of `b1`.
 3. When `flag` is `true`, form `b2` from an orthonormal basis of the null space of `Bt`, create `w1` and `w2`, and register `w`. When `hedge` is also `true`, add the hedge rows.
 4. When `flag` is `false`, set `b2 = nothing`, create `w1`, and register `w`.
 5. Set the start values of `w1` from `wi` with [`set_initial_w!`](@ref).

# JuMP formulation

## Variables

  - `w1`: Created, ``N_f`` entries.
  - `w2`: Created when `flag = true`, one entry for each column of ``\\mathbf{B}_2``.

## Expressions

  - `w`: ``\\mathbf{B}_1 \\boldsymbol{w}_1``, or ``\\mathbf{B}_1 \\boldsymbol{w}_1 + \\mathbf{B}_2 \\boldsymbol{w}_2`` when `flag = true`.

## Constraints

  - `cfrb_hedge`: ``s_c \\mathbf{B}_2^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} = \\boldsymbol{0}``, when `flag = true` and `hedge = true`.

Where:

  - $(math_dict[:w_1_factor])
  - $(math_dict[:w_2_off_factor])
  - $(math_dict[:w_port])
  - ``\\mathbf{B}_1``: The factor basis, ``(\\mathbf{B}^\\intercal)^{+}``.
  - ``\\mathbf{B}_2``: The off-factor basis, an orthonormal basis of the null space of ``\\mathbf{B}^\\intercal``, ``N \\times (N - \\operatorname{rank}\\mathbf{B})``.
  - $(math_dict[:B_loadings])
  - ``\\mathbf{\\Sigma}``: The covariance matrix of the prior, `pr.sigma`.
  - $(math_dict[:sc_scale])

# Arguments

  - `model`: JuMP model.
  - `re`: Regression result or estimator ([`RegE_Reg`](@ref)).
  - `rd`: Returns result that carries `X` and `F`. The method reads it only to fit the loadings.
  - `pr`: Prior result. The method reads its loadings, and `sigma` for the hedge rows.
  - `flag`: `true` adds the off-factor weights.
  - `wi`: Start values of the factor weights, or `nothing`.
  - `hedge`: `true` adds the hedge rows. The method reads it only when `flag` is `true`, and then `pr` must carry `sigma`.

# Returns

  - `b1, b2, rr`: The factor basis, the off-factor basis, and the loadings of both. `b2` is `nothing` when `flag` is `false`.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`resolve_factor_regression`](@ref)
  - [`RegE_Reg`](@ref)
"""
function set_factor_risk_contribution_constraints!(model::JuMP.Model, re::RegE_Reg,
                                                   rd::ReturnsResult,
                                                   pr::Option{<:AbstractPriorResult},
                                                   flag::Bool, wi::Option{<:VecNum};
                                                   hedge::Bool = false)
    rr = resolve_factor_regression(re, rd, pr)
    Bt = transpose(rr.L)
    b1 = LinearAlgebra.pinv(Bt)
    Nf = size(b1, 2)
    if flag
        b2 = LinearAlgebra.pinv(transpose(LinearAlgebra.nullspace(Bt)))
        # The null space has `N - rank(B)` columns, which is more than `N - Nf` when the
        # loadings have a lower rank, so `w2` takes its size from `b2`.
        JuMP.@variables(model, begin
                            w1[1:Nf]
                            w2[1:size(b2, 2)]
                        end)
        JuMP.@expression(model, w, b1 * w1 + b2 * w2)
        if hedge
            sc = get_constraint_scale(model)
            JuMP.@constraint(model, cfrb_hedge, sc * (transpose(b2) * pr.sigma * w) == 0)
        end
    else
        b2 = nothing
        JuMP.@variable(model, w1[1:Nf])
        JuMP.@expression(model, w, b1 * w1)
    end
    set_initial_w!(w1, wi)
    return b1, b2, rr
end
"""
    _optimise(frc::FactorRiskContribution, rd::ReturnsResult = ReturnsResult();
              str_names::Bool = false, save::Bool = true, kwargs...) -> FactorRiskContributionResult

Build the model of a [`FactorRiskContribution`](@ref), solve it, and return its result.

[`optimise`](@ref) reaches this method through [`shared_series_solve`](@ref), which drops the observations of a Shared Series gap before the solve.

# Algorithm

 1. Replace each schedule of `frc` with its static default through [`reset_time_dependent_estimator`](@ref).
 2. Process the attributes of `frc.opt` against `rd` with [`processed_jump_optimiser_attributes`](@ref), giving `attrs`.
 3. Take the view of `frc` and of `rd` at the investable assets with [`investable_view`](@ref). Both are unchanged when every asset is investable.
 4. Bind the benchmark of each [`DataBenchmarkTracking`](@ref) measure to `rd` with [`bind_data_benchmark`](@ref).
 5. Build a new model. Register the scales with [`set_model_scales!`](@ref), the number of observations with [`set_model_observations!`](@ref), the homogenisation variables of a ratio with [`set_maximum_ratio_factor_variables!`](@ref) and the corner of a lifted variance with [`set_sdp_lift_corner!`](@ref).
 6. Create the factor weights and register `w` with [`set_factor_risk_contribution_constraints!`](@ref), giving `b1`, `b2` and `rr`. `b` is `b1`, or `[b1 b2]` when `flag = true`, the basis of the whole decision vector.
 7. Add the weight bounds with [`set_weight_constraints!`](@ref).
 8. Add the constraints, the return terms and the risk measures with [`assemble_jump_model!`](@ref), which reads `b` and adds no asset phylogeny of the semidefinite kind.
 9. Resolve `frc.frc_ple` on the factor returns with [`phylogeny_constraints`](@ref), giving `frc_plr`. Add its rows with [`set_sdp_frc_phylogeny_constraints!`](@ref) and [`set_frc_iplg_constraints!`](@ref).
10. Set the objective with [`set_portfolio_objective_function!`](@ref), and solve with [`optimise_JuMP_model!`](@ref), giving `retcode` and `sol`.
11. Build the result. It holds `attrs`, `retcode`, `sol`, the model when `save` is `true`, the risk measures resolved against the prior, `rr` and `frc_plr`.

# Arguments

  - `frc::FactorRiskContribution`: The optimiser.
  - $(arg_dict[:rd])
  - `str_names::Bool`: `true` gives each variable and each row of the model its string name.
  - `save::Bool`: `true` keeps the model in the result.
  - `kwargs...`: Passed on to [`processed_jump_optimiser_attributes`](@ref).

# Returns

  - [`FactorRiskContributionResult`](@ref): The result, with `fb = nothing`. The method of [`optimise`](@ref) for an optimiser with a fallback runs the fallback chain.

# Related

  - [`optimise`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
  - [`FactorRiskContributionResult`](@ref)
"""
function _optimise(frc::FactorRiskContribution, rd::ReturnsResult = ReturnsResult();
                   str_names::Bool = false, save::Bool = true, kwargs...)
    frc = reset_time_dependent_estimator(frc)
    attrs = processed_jump_optimiser_attributes(frc.opt, rd; kwargs...)
    # The bundle reduced what it carries. The head carries the rest — an initial weight
    # vector, a risk measure holding per-asset data, tracking, a custom term — and hands
    # them to `assemble_jump_model!` itself, so it takes the same view of itself and of
    # `rd`. Both are unchanged when every asset is investable.
    frc, rd = investable_view(frc, rd, attrs.pr, attrs.imsk)
    frc = bind_data_benchmark(frc, rd)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, frc.opt.sc, frc.opt.so)
    set_model_observations!(model, size(attrs.pr.X, 1))
    set_maximum_ratio_factor_variables!(model, frc.obj)
    set_sdp_lift_corner!(model, frc.obj, attrs.pr)
    b1, b2, rr = set_factor_risk_contribution_constraints!(model, frc.re, rd, attrs.pr,
                                                           frc.flag, frc.wi)
    # The risk builders read the basis of the whole decision vector, so a variance under
    # `flag = true` prices the off-factor weights too (#1350).
    b = isnothing(b2) ? b1 : hcat(b1, b2)
    set_weight_constraints!(model, attrs.wb, frc.opt)
    assemble_jump_model!(model, frc, frc.opt, attrs, rd, frc.r, frc.obj, b, false)
    # After the model is assembled, so the factor phylogeny reads the marks that the
    # variance builders and `mark_risk_minimised!` write, as the asset phylogeny does.
    # A factor phylogeny reads the factor returns, so it drops their gaps as the loadings do.
    frc_plr = phylogeny_constraints(frc.frc_ple, shared_series_view(rd, nothing, true).F)
    set_sdp_frc_phylogeny_constraints!(model, frc_plr)
    set_frc_iplg_constraints!(model, frc_plr, attrs.wb, transpose(rr.L), frc.opt.ss)
    set_portfolio_objective_function!(model, frc.obj, frc, attrs)
    retcode, sol = optimise_JuMP_model!(model, frc, eltype(attrs.pr.X))
    return FactorRiskContributionResult(;
                                        jr = JuMPOptimisationResult(; pa = attrs,
                                                                    retcode = retcode,
                                                                    sol = sol,
                                                                    model = ifelse(save,
                                                                                   model,
                                                                                   nothing)),
                                        r = factory(frc.r, attrs.pr, frc.opt.slv), rr = rr,
                                        frc_plr = frc_plr, fb = nothing)
end
"""
    optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                      <:Any, <:Any, Nothing
                  },
             rd::ReturnsResult; str_names::Bool = false, save::Bool = true, kwargs...) -> FactorRiskContributionResult

Optimise a [`FactorRiskContribution`](@ref) that has no fallback, and return its result.

This method serves an optimiser whose `fb` is `nothing`. It solves through [`shared_series_solve`](@ref), and [`_optimise`](@ref) states the steps of the solve.

# Arguments

  - `frc`: The optimiser.
  - $(arg_dict[:rd]) The solve reads it to fit the loadings when neither `re` nor the prior holds them, and reads its factor returns to resolve `frc_ple`. A fallback or a clustering step can also read it.
  - `str_names`: `true` gives each variable and each row of the model its string name.
  - `save`: `true` keeps the model in the result.
  - `kwargs`: Passed on to [`_optimise`](@ref).

# Validation

  - No field in the tree of `frc` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise, through [`assert_batch_entry`](@ref): a plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the fold loop's online arm.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`FactorRiskContributionResult`](@ref)
"""
function optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  str_names::Bool = false, save::Bool = true, kwargs...)
    assert_batch_entry(frc, "`optimise`")
    return shared_series_solve(frc, rd; str_names = str_names, save = save, kwargs...)
end

@pipe_delegates FactorRiskContribution opt
@pipe_route_sigma_ucs FactorRiskContribution
export FactorRiskContribution, FactorRiskContributionResult
