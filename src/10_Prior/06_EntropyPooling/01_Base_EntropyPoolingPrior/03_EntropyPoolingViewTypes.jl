"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that hold a group of entropy pooling views and the settings of those views.

A significance level is a property of a view, not of the estimator that holds it. The value at risk at 1% and the value at risk at 10% are different statistics of the same series. An estimator of this family pairs a group of view equations with their settings, so one entropy pooling estimator can hold views at several levels.

# Related

  - [`ValueatRiskView`](@ref)
  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
abstract type AbstractEntropyPoolingViewEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

A group of value at risk views, with their significance level.

A value at risk view is linear in the posterior probabilities, and a tail view is not. [`ep_var_views!`](@ref) writes it as rows of the constraint set, so it needs no auxiliary variable and has no choice of formulation. Both [`OptimEntropyPooling`](@ref) and [`JuMPEntropyPooling`](@ref) accept it. For this reason, the estimator holds a level and no other setting.

[meucci2008](@cite) has no value at risk view. It states a view on a quantile in Equation 11, and it writes a view on the median as a constraint on a posterior mass in Equation 66. The library applies the same mass constraint at the level `alpha`.

The views of this estimator accept only `==` and `>=`, with one asset for each view, a unit coefficient and a non-negative target. [`replace_prior_views`](@ref) replaces a `prior(...)` reference inside `views` by the prior value at risk at the `alpha` of this group. A view stated against the prior therefore changes with the level.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05
    ) -> ValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.

# Examples

```jldoctest
julia> ValueatRiskView(; alpha = 0.01, views = LinearConstraintEstimator(; val = \"A >= 0.05\"))
ValueatRiskView
  views ┼ LinearConstraintEstimator
        │   val ┼ String: "A >= 0.05"
        │   key ┴ nothing
  alpha ┴ Float64: 0.01
```

# Related

  - [`AbstractEntropyPoolingViewEstimator`](@ref)
  - [`ep_var_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:meucci2008]) Section 2, Equation 11, and Appendix A.2, Equations 62 to 64 and 66.
"""
@concrete struct ValueatRiskView <: AbstractEntropyPoolingViewEstimator
    """
    $(field_dict[:ep_vv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    function ValueatRiskView(views::LinearConstraintEstimator, alpha::Number)
        assert_unit_interval(alpha, :alpha)
        return new{typeof(views), typeof(alpha)}(views, alpha)
    end
end
function ValueatRiskView(; views::LinearConstraintEstimator,
                         alpha::Number = 0.05)::ValueatRiskView
    return ValueatRiskView(views, alpha)
end
"""
    const VV_VecVV = Union{<:ValueatRiskView, <:AbstractVector{<:ValueatRiskView}}

Alias for the shapes a `var_views` field accepts: one [`ValueatRiskView`](@ref), or a vector of them read under their own significance levels.

# Related

  - [`ValueatRiskView`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
const VV_VecVV = Union{<:ValueatRiskView, <:AbstractVector{<:ValueatRiskView}}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations that express a tail view inside an entropy pooling problem.

A tail view constrains a quantile risk measure of the posterior distribution. A mean, variance or correlation view is linear in the posterior probabilities, and a tail view is not. So each measure has more than one way to write it as a program that a solver can solve. The concrete subtypes name those ways.

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Sections 2.2 and 2.3.
  - $(ref_dict[:EPRLVaR]) Section 3.2.
"""
abstract type AbstractEntropyPoolingViewFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations of a conditional value-at-risk view.

# Related

  - [`AbstractEntropyPoolingViewFormulation`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equations 2 to 6.
"""
abstract type AbstractConditionalValueatRiskViewFormulation <:
              AbstractEntropyPoolingViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations of an entropic value-at-risk view.

# Related

  - [`AbstractEntropyPoolingViewFormulation`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equations 7 to 11.
  - $(ref_dict[:EPRLVaR]) Section 2, Equation 1.
"""
abstract type AbstractEntropicValueatRiskViewFormulation <:
              AbstractEntropyPoolingViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations of a relativistic value-at-risk view.

# Related

  - [`AbstractEntropyPoolingViewFormulation`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 3.2, Equation 7.
"""
abstract type AbstractRelativisticValueatRiskViewFormulation <:
              AbstractEntropyPoolingViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Linear formulation of a conditional value-at-risk view [EPTail](@cite).

`LinearConditionalValueatRiskView` writes the view through the dual representation of CVaR. It adds ``T`` continuous variables and no integer variable, so it is the cheapest of the three CVaR formulations. It is exact.

It accepts the operators `>=` and `==`, over any number of assets whose coefficients have one sign. An equality view needs a target at or above the prior CVaR of the left hand side of the view. For a target below the prior CVaR, the prior already meets the constraint, so the minimisation of the divergence keeps the prior and does not meet the view. For such a target, use [`IntegerConditionalValueatRiskView`](@ref) or [`SequentialConditionalValueatRiskView`](@ref). A relative view, whose coefficients have both signs, needs one of the same two formulations.

# Mathematical definition

The view ``\\mathrm{CVaR}_{\\alpha}(X) \\geq \\bar{c}`` on one asset is written as:

```math
\\begin{align}
&\\nu_{j} \\geq 0\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\nu_{j} \\leq \\dfrac{w_{j}}{\\alpha}\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cvar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:cvar_target])
  - $(math_dict[:ep_tail_nu])

The box and the simplex describe every reweighting of the sample that gives no observation more than ``1/\\alpha`` times its posterior probability. The largest expected loss over such reweightings is the CVaR. So the constraint set is feasible if and only if ``\\mathrm{CVaR}_{\\alpha}(X) \\geq \\bar{c}``, and a lower-bound view is exact.

A view over several assets, ``\\sum_{i} \\gamma_{i} \\mathrm{CVaR}_{\\alpha}(X_{i}) \\geq \\bar{c}`` with every ``\\gamma_{i} > 0``, takes one block of the first three rows for each asset, and the last row reads ``\\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} \\geq \\bar{c}``. The CVaR is concave in the probabilities, so a positive combination of CVaRs is concave and its upper level set is convex. Each block attains the CVaR of its asset on its own, so the encoding stays exact. The upper level set of a view with coefficients of both signs is not convex in general, and this formulation refuses such a view. [EPTail](@cite) states the rows for a view on one asset. The view over several assets is the library's own extension.

# Examples

```jldoctest
julia> LinearConditionalValueatRiskView()
LinearConditionalValueatRiskView()
```

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equation 6.
"""
struct LinearConditionalValueatRiskView <: AbstractConditionalValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Integer formulation of a conditional value-at-risk view [EPTail](@cite).

`IntegerConditionalValueatRiskView` writes the view through the ordered weights representation of CVaR, and a monotone binary vector selects the tail of the posterior. It expresses every comparison operator and any linear combination of the CVaRs of single assets. The cost is `sbar` binary variables for each asset that the view names. It needs a solver that supports mixed-integer exponential cone programs.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The conditional value at risk is the mean loss of the tail of the posterior that has the mass ``\\alpha``. This formulation states it over the ``\\bar{s}`` largest losses only:

```math
\\begin{align}
&y_{j} \\leq y_{j+1}\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}-1\\\\
&q_{j} \\leq y_{j}\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}\\\\
&q_{j} \\leq w_{[j]}\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}\\\\
&q_{j} \\geq w_{[j]} - (1 - y_{j-1})\\,, &\\forall\\, j = 2,\\ldots,\\bar{s}\\\\
&q_{j} \\geq 0\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}\\\\
&\\alpha = \\sum_{j=1}^{\\bar{s}} q_{j}\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{\\bar{s}}\\\\
&\\mathrm{CVaR}_{\\alpha}(X) = \\dfrac{1}{\\alpha} \\sum_{j=1}^{\\bar{s}} q_{j} x_{[j]}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cvar_stat])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - ``x_{[1]} \\leq x_{[2]} \\leq \\ldots \\leq x_{[\\bar{s}]}``: The ``\\bar{s}`` largest losses of the asset, sorted ascending, so the largest loss is last.
  - ``w_{[j]}``: Posterior probability of the observation in position ``j``.
  - ``\\bar{s}``: Number of largest losses the formulation reads.
  - ``\\boldsymbol{y}``: ``\\bar{s} \\times 1`` binary vector that marks the observations in the tail.
  - ``\\boldsymbol{q}``: ``\\bar{s} \\times 1`` auxiliary vector that holds the tail mass of each observation of the window.

The monotonicity constraint makes the marked set a suffix of the ascending order. For this reason, the expression is the CVaR and not the mean of an arbitrary subset of probability ``\\alpha``. An observation is in the tail in full when the observation below it is also marked, so every marked observation except the lowest one has ``q_{j} = w_{[j]}``. The lowest one has a part of its probability, and the sum condition sets that part to ``\\alpha`` less the mass above it. That observation is the value at risk observation of the posterior, so the expression is exactly the posterior CVaR.

[EPTail](@citet) writes the bound on ``q_{j}`` through ``y_{j}``. This forces ``q_{j} = w_{[j]}`` for every marked observation, so the marked mass equals ``\\alpha`` only when whole observations sum to it. The library sorts the losses in ascending order and writes the bound through ``y_{j-1}``, so the lowest marked observation can have a part of its probability.

The window is the only restriction. If the tail of mass ``\\alpha`` of a posterior goes below the ``\\bar{s}`` largest losses, that posterior is outside the feasible set. So the result is the posterior of least divergence among the posteriors whose tail stays inside the window. An upper-bound view moves mass down the order, and needs a wider window than a lower-bound view.

[`entropy_pooling`](@ref) warns when the window binds, which happens when the window holds no more than the tail mass ``\\alpha``. Then increase `sbar`. An `sbar` equal to the number of observations restricts nothing, and costs one binary variable for each observation.

# Constructors

    IntegerConditionalValueatRiskView(;
        sbar::Option{<:Number} = nothing
    ) -> IntegerConditionalValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - If `sbar` is an `Integer`, `sbar >= 1`.
  - If `sbar` is not an `Integer`, `0 < sbar < 1`. Use an `Integer` to name the whole sample.

# Examples

```jldoctest
julia> IntegerConditionalValueatRiskView()
IntegerConditionalValueatRiskView
  sbar ┴ nothing
```

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equations 2 to 5.
"""
@concrete struct IntegerConditionalValueatRiskView <:
                 AbstractConditionalValueatRiskViewFormulation
    """
    $(field_dict[:sbar])
    """
    sbar
    function IntegerConditionalValueatRiskView(sbar::Option{<:Number})
        if isa(sbar, Integer)
            @argcheck(sbar >= one(sbar), DomainError(sbar, "sbar must be >= 1"))
        elseif !isnothing(sbar)
            assert_unit_interval(sbar, :sbar)
        end
        return new{typeof(sbar)}(sbar)
    end
end
function IntegerConditionalValueatRiskView(;
                                           sbar::Option{<:Number} = nothing)::IntegerConditionalValueatRiskView
    return IntegerConditionalValueatRiskView(sbar)
end
"""
$(DocStringExtensions.TYPEDEF)

Sequential convex formulation of a conditional value-at-risk view.

[EPTail](@cite) states no sequential formulation. The sequential formulation is the library's own. Its dual representation rows are the rows of Equation 6 of [EPTail](@cite).

`SequentialConditionalValueatRiskView` writes the views that [`LinearConditionalValueatRiskView`](@ref) cannot write, with no integer variable. These views are an upper bound, an equality below the prior CVaR, and a relative view whose coefficients have both signs. The formulation replaces the CVaR of each asset on the wrong side of the inequality by a linear upper bound, and solves the convex problem that results. It then reads the bound again at the posterior and solves again, until the bound is tight. The view holds at each posterior of this sequence, and the divergence of each posterior is at most the divergence of the posterior before it.

The posterior is a local minimiser of the divergence. The feasible set of an upper-bound or relative CVaR view is not convex, so no convex program describes it exactly. The sequence stops at a fixed point, which is not always the posterior of least divergence. [`IntegerConditionalValueatRiskView`](@ref) finds the posterior of least divergence when its window holds the tail of the posterior. Its cost is binary variables and a solver that supports mixed-integer exponential cone programs. A lower-bound view whose coefficients are all positive is convex, and the formulation is then the linear one, with no re-solve.

# Mathematical definition

Orient the view as a lower bound. Where its operator is `<=`, multiply both sides by minus one. Write ``\\mathcal{P}`` for the assets whose coefficient is then positive, and ``\\mathcal{N}`` for the assets whose coefficient is negative. The conditional value at risk is concave in the observation probabilities, so the assets of ``\\mathcal{P}`` take the exact dual representation of [`LinearConditionalValueatRiskView`](@ref). Each asset of ``\\mathcal{N}`` takes the primal representation at a fixed value ``\\eta_{i}`` of its value at risk. This representation is linear in the probabilities, and it bounds the measure from above:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(X_{i}) &\\leq \\eta_{i} + \\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} \\left(x_{i,\\,j} - \\eta_{i}\\right)^{+}\\,, &\\forall\\, i \\in \\mathcal{N}\\\\
\\bar{c} &\\leq \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} + \\sum_{i \\in \\mathcal{N}} \\gamma_{i} \\left(\\eta_{i} + \\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} \\left(x_{i,\\,j} - \\eta_{i}\\right)^{+}\\right)\\,.
\\end{align}
```

The bound holds with equality where ``\\eta_{i}`` is the value at risk of ``X_{i}`` under ``\\boldsymbol{w}``, so the row is tight at the probabilities at which the formulation reads it. Each re-solve reads ``\\eta_{i}`` at the last posterior. That posterior stays feasible for the new row, so the divergence cannot increase. The formulation writes an equality view as the bound that the prior violates, and the minimisation of the divergence makes that bound tight.

Where:

  - ``x_{i,\\,j}``: The loss of asset ``i`` at observation ``j``, which is the negated return.
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:cvar_target])
  - ``\\boldsymbol{\\nu}_{i}``: ``T \\times 1`` vector of weights that attains the CVaR of asset ``i``, the variable of its dual representation.
  - ``\\gamma_{i}``: The coefficient that the view gives asset ``i``.
  - ``\\eta_{i}``: The value at risk of asset ``i`` under the probabilities of the row.
  - ``\\mathcal{P}``, ``\\mathcal{N}``: The assets whose coefficient is positive, and the assets whose coefficient is negative, after the orientation of the view as a lower bound.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialConditionalValueatRiskView(;
        iters::Integer = 20,
        tol::Number = 1e-8
    ) -> SequentialConditionalValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `iters >= 0`.
  - `tol > 0`.

# Examples

```jldoctest
julia> SequentialConditionalValueatRiskView()
SequentialConditionalValueatRiskView
  iters ┼ Int64: 20
    tol ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equation 6.
"""
@concrete struct SequentialConditionalValueatRiskView <:
                 AbstractConditionalValueatRiskViewFormulation
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
    function SequentialConditionalValueatRiskView(iters::Integer, tol::Number)
        @argcheck(iters >= zero(iters), DomainError(iters, "iters must be >= 0"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be > 0"))
        return new{typeof(iters), typeof(tol)}(iters, tol)
    end
end
function SequentialConditionalValueatRiskView(; iters::Integer = 20,
                                              tol::Number = 1e-8)::SequentialConditionalValueatRiskView
    return SequentialConditionalValueatRiskView(iters, tol)
end
"""
$(DocStringExtensions.TYPEDEF)

Exponential cone formulation of an entropic value-at-risk view [EPRLVaR](@cite).

[EPRLVaR](@cite) states the dual of the entropic value at risk for non-uniform probabilities in Equation 1. Its Section 3.1 says that a view of the form `>=` or `==` reduces to linear and exponential cone constraints. It gives [EPTail](@cite) as the source of that reduction, but the version of that work on file does not state it.

`ConicEntropicValueatRiskView` writes the view through the dual representation of EVaR. It adds ``T`` continuous variables and one relative entropy cone, and it is exact.

It accepts the operators `>=` and `==`, over any number of assets whose coefficients have one sign. An equality view needs a target at or above the prior EVaR of the left hand side of the view. For a target below the prior EVaR, use [`GridEntropicValueatRiskView`](@ref) or [`SequentialEntropicValueatRiskView`](@ref). A relative view, whose coefficients have both signs, needs [`SequentialEntropicValueatRiskView`](@ref).

# Mathematical definition

The view ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}`` on one asset is written as:

```math
\\begin{align}
&0 \\leq \\nu_{j} \\leq 1\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} \\ln\\left(\\dfrac{\\nu_{j}}{w_{j}}\\right) \\leq \\ln\\left(\\dfrac{1}{\\alpha}\\right)\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{e}\\,.
\\end{align}
```

Where:

  - $(math_dict[:evar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:evar_target])
  - $(math_dict[:ep_tail_nu])

The relative entropy budget is the dual description of EVaR, so the constraint set is feasible if and only if ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}``.

A view over several assets, ``\\sum_{i} \\gamma_{i} \\mathrm{EVaR}_{\\alpha}(X_{i}) \\geq \\bar{e}`` with every ``\\gamma_{i} > 0``, takes one block of the first three rows for each asset, and the last row reads ``\\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} \\geq \\bar{e}``. The EVaR is concave in the probabilities, so a positive combination of EVaRs is concave and its upper level set is convex. Each block attains the EVaR of its asset on its own, so the encoding stays exact. The upper level set of a view with coefficients of both signs is not convex in general, and this formulation refuses such a view.

# Examples

```jldoctest
julia> ConicEntropicValueatRiskView()
ConicEntropicValueatRiskView()
```

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 1, and Section 3.1. $(ref_dict[:no_original_source])
  - $(ref_dict[:EPTail])
"""
struct ConicEntropicValueatRiskView <: AbstractEntropicValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Grid formulation of an entropic value-at-risk view [EPTail](@cite).

`GridEntropicValueatRiskView` writes the view on a grid of values of the EVaR dual variable. For a lower-bound view, the centre of the grid is the value that attains the prior EVaR of the asset. For an upper-bound or equality view, it is the value that [`ep_evar_anchor`](@ref) finds. A lower-bound view is a set of linear constraints and needs no integer variable. An upper-bound or equality view selects one grid point with a binary vector and a big-``M`` relaxation, and needs a solver that supports mixed-integer exponential cone programs.

The formulation divides each row of the upper-bound block by its bound, so the row reads against one at every grid point. At its own scale, the row of a small dual variable is smaller than the feasibility tolerance of a solver, and the solver accepts any posterior for it. The rows of the lower-bound block keep their own scale. A posterior that meets a lower bound moves mass toward the largest loss, so the slack of a row with a small bound is much larger than the tolerance. The big-M constant of each row is the smallest constant that releases the row, the data set it, and `M` multiplies it.

The result is approximate in both directions. A lower-bound view holds at the grid points and can fall short between them. An upper-bound view holds at one grid point and can be conservative. Increase `pct` or `K` when the posterior value misses the target. Use [`ConicEntropicValueatRiskView`](@ref) when it accepts the view.

It accepts `==`, `>=` and `<=`, with one asset for each view. The library divides the view by its coefficient, so the coefficient becomes one. This division flips the operator when the coefficient is negative, so this formulation restricts neither the operator nor the sign.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The sample EVaR is the value of a scalar minimisation:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(X) &= \\underset{z > 0}{\\min} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
\\end{align}
```

So ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}`` holds exactly when the objective is at or above ``\\bar{e}`` at every ``z``, and ``\\mathrm{EVaR}_{\\alpha}(X) \\leq \\bar{e}`` holds when the objective is at or below ``\\bar{e}`` at one or more ``z``. At a fixed ``z``, the objective is linear in ``\\boldsymbol{w}``, so each grid point gives one row. On a grid ``\\bar{z}_{1},\\ldots,\\bar{z}_{K}``, a lower-bound view gives these rows:

```math
\\begin{align}
&\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/\\bar{z}_{k})}{\\exp(\\bar{e}/\\bar{z}_{k})} \\geq \\alpha\\,, &\\forall\\, k = 1,\\ldots,K
\\end{align}
```

An upper-bound view gives the rows below. ``\\boldsymbol{y}`` is a binary selector, and the formulation divides each row by its bound ``\\alpha``, so that the row reads against one:

```math
\\begin{align}
&\\boldsymbol{1}^{\\intercal} \\boldsymbol{y} = 1\\\\
&\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/\\bar{z}_{k})}{\\alpha \\exp(\\bar{e}/\\bar{z}_{k})} \\leq 1 + M M_{k} (1 - y_{k})\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&M_{k} = \\max_{j} \\dfrac{\\exp(x_{j}/\\bar{z}_{k})}{\\alpha \\exp(\\bar{e}/\\bar{z}_{k})} - 1\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{K}\\,.
\\end{align}
```

Where:

  - $(math_dict[:evar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:evar_target])
  - ``z > 0``: Dual variable of the entropic value at risk.
  - ``\\bar{z}_{k}``: Dual variable of the ``k``-th grid point.
  - ``K``: Number of grid points.
  - ``\\boldsymbol{y}``: ``K \\times 1`` binary selector, with one entry for each grid point.
  - ``M_{k}``: Smallest big-M constant that releases the row of the ``k``-th grid point. The weights sum to one, so the left hand side of the row is never larger than its largest coefficient.
  - ``M``: Big-M multiplier.

An equality view has both blocks. The rows are the rows of Equations 9 to 11 of [EPTail](@cite). [EPTail](@citet) bounds the upper-bound row at ``\\alpha + M (1 - y_{k})`` with one constant ``M``. The library divides each row by its bound and gives each row its own smallest constant ``M_{k}``, which ``M`` multiplies.

# Constructors

    GridEntropicValueatRiskView(;
        pct::Number = 0.5,
        K::Integer = 11,
        M::Number = 1,
        iters::Integer = 50,
        tol::Number = 1e-10,
        tilt_iters::Integer = 200
    ) -> GridEntropicValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < pct < 1`.
  - $(val_dict[:ep_gridK])
  - `M >= 1`.
  - `iters >= 1`.
  - `tol >= 0`.
  - `tilt_iters >= 1`.

# Examples

```jldoctest
julia> GridEntropicValueatRiskView()
GridEntropicValueatRiskView
         pct ┼ Float64: 0.5
           K ┼ Int64: 11
           M ┼ Int64: 1
       iters ┼ Int64: 50
         tol ┼ Float64: 1.0e-10
  tilt_iters ┴ Int64: 200
```

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`ep_evar_anchor`](@ref)
  - [`ep_evar_grid`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equations 7 to 11.
"""
@concrete struct GridEntropicValueatRiskView <: AbstractEntropicValueatRiskViewFormulation
    """
    $(field_dict[:zpct])
    """
    pct
    """
    $(field_dict[:zK])
    """
    K
    """
    $(field_dict[:bigM])
    """
    M
    """
    $(field_dict[:ep_grid_iters])
    """
    iters
    """
    $(field_dict[:ep_grid_tol])
    """
    tol
    """
    $(field_dict[:ep_grid_tilt_iters])
    """
    tilt_iters
    function GridEntropicValueatRiskView(pct::Number, K::Integer, M::Number, iters::Integer,
                                         tol::Number, tilt_iters::Integer)
        assert_unit_interval(pct, :pct)
        assert_ep_grid_size(K)
        @argcheck(M >= one(M), DomainError(M, "M must be >= 1"))
        @argcheck(iters >= one(iters), DomainError(iters, "iters must be >= 1"))
        @argcheck(tol >= zero(tol), DomainError(tol, "tol must be >= 0"))
        @argcheck(tilt_iters >= one(tilt_iters),
                  DomainError(tilt_iters, "tilt_iters must be >= 1"))
        return new{typeof(pct), typeof(K), typeof(M), typeof(iters), typeof(tol),
                   typeof(tilt_iters)}(pct, K, M, iters, tol, tilt_iters)
    end
end
function GridEntropicValueatRiskView(; pct::Number = 0.5, K::Integer = 11, M::Number = 1,
                                     iters::Integer = 50, tol::Number = 1e-10,
                                     tilt_iters::Integer = 200)::GridEntropicValueatRiskView
    return GridEntropicValueatRiskView(pct, K, M, iters, tol, tilt_iters)
end
"""
$(DocStringExtensions.TYPEDEF)

Sequential convex formulation of an entropic value-at-risk view.

[EPTail](@cite) and [EPRLVaR](@cite) state no sequential formulation. The sequential formulation is the library's own. It uses the sample EVaR of Equation 7 of [EPTail](@cite) and the dual representation of Equation 1 of [EPRLVaR](@cite).

`SequentialEntropicValueatRiskView` writes the views that [`ConicEntropicValueatRiskView`](@ref) cannot write, with no integer variable and no grid. These views are an upper bound, an equality below the prior EVaR, and a relative view whose coefficients have both signs. The formulation replaces the EVaR of each asset on the wrong side of the inequality by a linear upper bound, and solves the convex problem that results. It then reads the bound again at the posterior and solves again, until the bound is tight. The view holds at each posterior of this sequence, and the divergence of each posterior is at most the divergence of the posterior before it.

The posterior is a local minimiser of the divergence. The feasible set of an upper-bound or relative EVaR view is not convex, so no convex program describes it exactly. The sequence stops at a fixed point, which is not always the posterior of least divergence. [`GridEntropicValueatRiskView`](@ref) uses binary variables to search a grid of dual variables instead, and the view then holds only at the grid points. A lower-bound view whose coefficients are all positive is convex, and the formulation is then the conic one, with no re-solve.

# Mathematical definition

Orient the view as a lower bound. Where its operator is `<=`, multiply both sides by minus one. Write ``\\mathcal{P}`` for the assets whose coefficient is then positive, and ``\\mathcal{N}`` for the assets whose coefficient is negative. The entropic value at risk is concave in the observation probabilities, so the assets of ``\\mathcal{P}`` take the exact dual representation of [`ConicEntropicValueatRiskView`](@ref). Each asset of ``\\mathcal{N}`` takes the primal representation at a fixed dual variable ``z_{i}``. This representation bounds the measure from above, and it is concave in the probabilities. So its tangent at the probabilities ``\\boldsymbol{w}^{0}`` of the row bounds it from above again:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(X_{i}) &\\leq z_{i} \\ln\\left(\\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} e^{x_{i,\\,j}/z_{i}}\\right) \\leq \\mathrm{EVaR}_{\\alpha}(X_{i};\\, \\boldsymbol{w}^{0}) - z_{i} + \\sum_{j=1}^{T} w_{j} \\dfrac{z_{i} e^{x_{i,\\,j}/z_{i}}}{\\sum_{k=1}^{T} w^{0}_{k} e^{x_{i,\\,k}/z_{i}}}\\,, &\\forall\\, i \\in \\mathcal{N}\\\\
\\bar{e} &\\leq \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} + \\sum_{i \\in \\mathcal{N}} \\gamma_{i} \\left(\\mathrm{EVaR}_{\\alpha}(X_{i};\\, \\boldsymbol{w}^{0}) - z_{i} + \\sum_{j=1}^{T} w_{j} \\dfrac{z_{i} e^{x_{i,\\,j}/z_{i}}}{\\sum_{k=1}^{T} w^{0}_{k} e^{x_{i,\\,k}/z_{i}}}\\right)\\,.
\\end{align}
```

Both bounds hold with equality at ``\\boldsymbol{w} = \\boldsymbol{w}^{0}`` when ``z_{i}`` attains the EVaR there, so the row is tight at the probabilities at which the formulation reads it. Each re-solve reads ``z_{i}`` and the tangent at the last posterior. That posterior stays feasible for the new row, so the divergence cannot increase. The formulation writes an equality view as the bound that the prior violates, and the minimisation of the divergence makes that bound tight.

Where:

  - ``x_{i,\\,j}``: The loss of asset ``i`` at observation ``j``, which is the negated return.
  - $(math_dict[:rlvar_probs])
  - ``\\boldsymbol{w}^{0}``: The probabilities at which the formulation reads the row. They are the prior for the first solve, and the last posterior for each re-solve.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:evar_target])
  - ``\\boldsymbol{\\nu}_{i}``: ``T \\times 1`` vector of weights that attains the EVaR of asset ``i``, the variable of its dual representation.
  - ``\\gamma_{i}``: The coefficient that the view gives asset ``i``.
  - ``z_{i}``: The dual variable that attains the EVaR of asset ``i`` under ``\\boldsymbol{w}^{0}``, from [`ep_evar`](@ref).
  - ``\\mathcal{P}``, ``\\mathcal{N}``: The assets whose coefficient is positive, and the assets whose coefficient is negative, after the orientation of the view as a lower bound.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialEntropicValueatRiskView(;
        iters::Integer = 20,
        tol::Number = 1e-8
    ) -> SequentialEntropicValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `iters >= 0`.
  - `tol > 0`.

# Examples

```jldoctest
julia> SequentialEntropicValueatRiskView()
SequentialEntropicValueatRiskView
  iters ┼ Int64: 20
    tol ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`ep_evar`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equation 7.
  - $(ref_dict[:EPRLVaR]) Section 2, Equation 1.
"""
@concrete struct SequentialEntropicValueatRiskView <:
                 AbstractEntropicValueatRiskViewFormulation
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
    function SequentialEntropicValueatRiskView(iters::Integer, tol::Number)
        @argcheck(iters >= zero(iters), DomainError(iters, "iters must be >= 0"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be > 0"))
        return new{typeof(iters), typeof(tol)}(iters, tol)
    end
end
function SequentialEntropicValueatRiskView(; iters::Integer = 20,
                                           tol::Number = 1e-8)::SequentialEntropicValueatRiskView
    return SequentialEntropicValueatRiskView(iters, tol)
end
"""
$(DocStringExtensions.TYPEDEF)

Power cone formulation of a relativistic value-at-risk view [EPRLVaR](@cite).

`ConicRelativisticValueatRiskView` writes the view through the dual representation of RLVaR. It adds ``3T`` continuous variables and ``2T`` power cones, and it is exact.

It accepts the operators `>=` and `==`, over any number of assets whose coefficients have one sign. An equality view needs a target at or above the prior RLVaR of the left hand side of the view. For a target below the prior RLVaR, use [`GridRelativisticValueatRiskView`](@ref) or [`SequentialRelativisticValueatRiskView`](@ref). A relative view, whose coefficients have both signs, needs [`SequentialRelativisticValueatRiskView`](@ref). The solver must support the power cone, and also the exponential cone that the entropy pooling objective needs.

A conic solver can stop on this programme before it finds an optimal solution. A long sample, a small `alpha`, a small `kappa` or many of these views in one model make this more probable. Then give `opt` a vector of solver configurations, use a shorter sample, or state the view with [`GridRelativisticValueatRiskView`](@ref), whose rows are linear in the posterior probabilities.

# Mathematical definition

For one asset, the formulation writes the view ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) \\geq \\bar{\\vartheta}`` as:

```math
\\begin{align}
&0 \\leq \\nu_{j} \\leq 1\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\dfrac{\\tau_{j} - \\varsigma_{j}}{2\\kappa} \\leq \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\\\
&\\left(\\tau_{j},\\, T w_{j},\\, \\nu_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\left(\\dfrac{1}{1+\\kappa}\\right)\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\left(\\nu_{j},\\, T w_{j},\\, \\varsigma_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{\\vartheta}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_target])
  - ``\\boldsymbol{\\nu}``: ``T \\times 1`` vector of weights that attains the RLVaR.
  - ``\\boldsymbol{\\tau}``, ``\\boldsymbol{\\varsigma}``: ``T \\times 1`` vectors that carry the Kaniadakis entropy budget of ``\\boldsymbol{\\nu}``.
  - $(math_dict[:K_pow])

The budget is the dual description of RLVaR, so the constraint set is feasible if and only if ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) \\geq \\bar{\\vartheta}``.

A view over several assets, ``\\sum_{i} \\gamma_{i} \\mathrm{RLVaR}_{\\alpha,\\kappa}(X_{i}) \\geq \\bar{\\vartheta}`` with every ``\\gamma_{i} > 0``, takes one block of the first five rows for each asset, and the last row reads ``\\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} \\geq \\bar{\\vartheta}``. The RLVaR is concave in the probabilities, so a positive combination of RLVaRs is concave and its upper level set is convex. Each block attains the RLVaR of its asset on its own, so the encoding stays exact. The upper level set of a view with coefficients of both signs is not convex in general, and this formulation refuses such a view.

# Examples

```jldoctest
julia> ConicRelativisticValueatRiskView()
ConicRelativisticValueatRiskView()
```

# Related

  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`kappa_log`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 3.2, Equation 7, and Section 2, Equation 4.
"""
struct ConicRelativisticValueatRiskView <: AbstractRelativisticValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Grid formulation of a relativistic value-at-risk view.

[EPRLVaR](@cite) states no grid for the relativistic value at risk. The grid is the library's own, and it uses the primal programme of Equation 5 of [EPRLVaR](@cite).

`GridRelativisticValueatRiskView` writes the view on a grid of points of the primal programme of RLVaR. The centre of the grid is the point that a posterior attains when it meets the view. A lower-bound view is a set of linear constraints and needs no integer variable. An upper-bound or equality view selects one grid point with a binary vector and a big-``M`` relaxation, and needs a solver that supports mixed-integer exponential cone programs.

The formulation divides each row of the upper-bound block by its bound, so the row reads against one at every grid point. At its own scale, the row of a small dual variable is smaller than the feasibility tolerance of a solver, and the solver accepts any posterior for it. The rows of the lower-bound block keep their own scale. A posterior that meets a lower bound moves mass toward the largest loss, so the slack of a row with a small bound is much larger than the tolerance. The big-M constant of each row is the smallest constant that releases the row, the data set it, and `M` multiplies it.

The upper-bound block drops a grid point whose bound is at or below zero, because its row holds at no posterior. The lower-bound block keeps such a point, because its row holds at every posterior.

It accepts `==`, `>=` and `<=`, with one asset for each view. The library divides the view by its coefficient, so the coefficient becomes one. This division flips the operator when the coefficient is negative, so this formulation restricts neither the operator nor the sign.

When `kappa` goes near one, the RLVaR goes near the largest loss, and [`ep_rlvar_tail`](@ref) overflows at the dual variable that attains it. The formulation drops each grid point at which the tail function overflows, and raises an error when it keeps no point. The iteration that finds the centre of the grid reads the same tail function, so it does not converge there either. The grid then has the dual variable of the prior as its centre, and the posterior falls short of the target. Use a smaller `kappa`, or use [`ConicRelativisticValueatRiskView`](@ref) when it accepts the operator.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

[`ep_rlvar_tail`](@ref) minimises out the pair of power cones of each observation in closed form, so the sample RLVaR is the value of a minimisation over two variables:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) &= \\underset{t,\\, z > 0}{\\min} \\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,,
\\end{align}
```

where ``\\varphi_{\\kappa}(u, z)`` is the smallest ``\\psi + \\theta`` that the two power cones of one observation allow. Its closed form is:

```math
\\begin{align}
\\varphi_{\\kappa}(u, z) &= \\dfrac{\\kappa}{1+\\kappa} \\left(\\dfrac{2\\kappa}{(1+\\kappa) z}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma - u}{2}\\right)^{\\frac{1+\\kappa}{\\kappa}} + \\kappa (1-\\kappa)^{\\frac{1-\\kappa}{\\kappa}} \\left(\\dfrac{z}{2\\kappa}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma + u}{2}\\right)^{-\\frac{1-\\kappa}{\\kappa}}\\,,\\\\
\\sigma &= \\sqrt{u^{2} + \\dfrac{(1 - \\kappa^{2}) z^{2}}{\\kappa^{2}}}\\,.
\\end{align}
```

At a fixed ``t`` and ``z``, the objective is linear in ``\\boldsymbol{w}``, so each grid point gives one row. On a grid ``(\\bar{t}_{1}, \\bar{z}_{1}),\\ldots,(\\bar{t}_{K}, \\bar{z}_{K})``, a lower-bound view gives these rows:

```math
\\begin{align}
&T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) \\geq \\bar{\\vartheta} - \\bar{t}_{k} - \\bar{z}_{k} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\,, &\\forall\\, k = 1,\\ldots,K
\\end{align}
```

An upper-bound view gives the rows below. ``\\boldsymbol{y}`` is a binary selector, and the formulation divides each row by its bound ``b_{k}``, so that the row reads against one:

```math
\\begin{align}
&\\boldsymbol{1}^{\\intercal} \\boldsymbol{y} = 1\\\\
&b_{k} = \\bar{\\vartheta} - \\bar{t}_{k} - \\bar{z}_{k} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&\\dfrac{T}{b_{k}} \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) \\leq 1 + M M_{k} (1 - y_{k})\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&M_{k} = \\max_{j} \\dfrac{T}{b_{k}} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) - 1\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{K}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_target])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:rlvar_u])
  - $(math_dict[:rlvar_sigma])
  - $(math_dict[:rlvar_phi])
  - ``\\psi``, ``\\theta``: The two tail variables of one observation, whose smallest sum is ``\\varphi_{\\kappa}``.
  - ``\\bar{t}_{k}``, ``\\bar{z}_{k}``: The shift and the dual variable of the ``k``-th grid point.
  - ``K``: The number of grid points.
  - ``\\boldsymbol{y}``: The ``K \\times 1`` binary selector, with one entry for each grid point.
  - ``b_{k}``: The bound of the row of the ``k``-th grid point. The upper-bound block drops a point whose bound is at or below zero.
  - ``M_{k}``: The smallest big-M constant that releases the row of the ``k``-th grid point. The weights sum to one, so the left hand side of the row is never larger than its largest coefficient.
  - ``M``: The big-M multiplier.

An equality view has both blocks. Each grid point is a feasible point of the primal programme, so a posterior that meets the upper-bound block also meets the view. The block can only be tighter than the view. The lower-bound block holds at the grid points and can fall short between them, so use [`ConicRelativisticValueatRiskView`](@ref) when it accepts the view.

# Constructors

    GridRelativisticValueatRiskView(;
        pct::Number = 0.5,
        K::Integer = 11,
        M::Number = 1,
        iters::Integer = 50,
        tol::Number = 1e-10,
        tilt_iters::Integer = 200
    ) -> GridRelativisticValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < pct < 1`.
  - $(val_dict[:ep_gridK])
  - `M >= 1`.
  - `iters >= 1`.
  - `tol >= 0`.
  - `tilt_iters >= 1`.

# Examples

```jldoctest
julia> GridRelativisticValueatRiskView()
GridRelativisticValueatRiskView
         pct ┼ Float64: 0.5
           K ┼ Int64: 11
           M ┼ Int64: 1
       iters ┼ Int64: 50
         tol ┼ Float64: 1.0e-10
  tilt_iters ┴ Int64: 200
```

# Related

  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`ep_rlvar_anchor`](@ref)
  - [`ep_rlvar_grid`](@ref)
  - [`ep_rlvar_tail`](@ref)
  - [`kappa_log`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
@concrete struct GridRelativisticValueatRiskView <:
                 AbstractRelativisticValueatRiskViewFormulation
    """
    $(field_dict[:rlvar_zpct])
    """
    pct
    """
    $(field_dict[:rlvar_zK])
    """
    K
    """
    $(field_dict[:rlvar_bigM])
    """
    M
    """
    $(field_dict[:ep_grid_iters])
    """
    iters
    """
    $(field_dict[:ep_grid_tol])
    """
    tol
    """
    $(field_dict[:ep_grid_tilt_iters])
    """
    tilt_iters
    function GridRelativisticValueatRiskView(pct::Number, K::Integer, M::Number,
                                             iters::Integer, tol::Number,
                                             tilt_iters::Integer)
        assert_unit_interval(pct, :pct)
        assert_ep_grid_size(K)
        @argcheck(M >= one(M), DomainError(M, "M must be >= 1"))
        @argcheck(iters >= one(iters), DomainError(iters, "iters must be >= 1"))
        @argcheck(tol >= zero(tol), DomainError(tol, "tol must be >= 0"))
        @argcheck(tilt_iters >= one(tilt_iters),
                  DomainError(tilt_iters, "tilt_iters must be >= 1"))
        return new{typeof(pct), typeof(K), typeof(M), typeof(iters), typeof(tol),
                   typeof(tilt_iters)}(pct, K, M, iters, tol, tilt_iters)
    end
end
function GridRelativisticValueatRiskView(; pct::Number = 0.5, K::Integer = 11,
                                         M::Number = 1, iters::Integer = 50,
                                         tol::Number = 1e-10,
                                         tilt_iters::Integer = 200)::GridRelativisticValueatRiskView
    return GridRelativisticValueatRiskView(pct, K, M, iters, tol, tilt_iters)
end
"""
$(DocStringExtensions.TYPEDEF)

Sequential convex formulation of a relativistic value-at-risk view.

[EPRLVaR](@cite) states no sequential formulation. The sequential formulation is the library's own. It uses the dual representation of Equation 7 and the primal programme of Equation 5 of [EPRLVaR](@cite).

`SequentialRelativisticValueatRiskView` writes the views that [`ConicRelativisticValueatRiskView`](@ref) cannot write, with no integer variable and no grid. These views are an upper bound, an equality below the prior RLVaR, and a relative view whose coefficients have both signs. The formulation replaces the RLVaR of each asset on the wrong side of the inequality by a linear upper bound, and solves the convex problem that results. It then reads the bound again at the posterior and solves again, until the bound is tight. The view holds at each posterior of this sequence, and the divergence of each posterior is at most the divergence of the posterior before it.

The posterior is a local minimiser of the divergence. The feasible set of an upper-bound or relative RLVaR view is not convex, so no convex program describes it exactly. The sequence stops at a fixed point, which is not always the posterior of least divergence. [`GridRelativisticValueatRiskView`](@ref) uses binary variables to search a grid of primal points instead, and the view then holds only at the grid points. A lower-bound view whose coefficients are all positive is convex, and the formulation is then the conic one, with no re-solve.

# Mathematical definition

Orient the view as a lower bound. Where its operator is `<=`, multiply both sides by minus one. Write ``\\mathcal{P}`` for the assets whose coefficient is then positive, and ``\\mathcal{N}`` for the assets whose coefficient is negative. The relativistic value at risk is concave in the observation probabilities, so the assets of ``\\mathcal{P}`` take the exact dual representation of [`ConicRelativisticValueatRiskView`](@ref). Each asset of ``\\mathcal{N}`` takes the primal representation at a fixed pair ``(t_{i}, z_{i})``. This representation is linear in the probabilities, and it bounds the measure from above:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X_{i}) &\\leq t_{i} + z_{i} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t_{i} - x_{i,\\,j},\\, z_{i})\\,, &\\forall\\, i \\in \\mathcal{N}\\\\
\\bar{\\vartheta} &\\leq \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} + \\sum_{i \\in \\mathcal{N}} \\gamma_{i} \\left(t_{i} + z_{i} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t_{i} - x_{i,\\,j},\\, z_{i})\\right)\\,.
\\end{align}
```

The bound holds with equality where ``(t_{i}, z_{i})`` attains the RLVaR of ``X_{i}`` under ``\\boldsymbol{w}``, so the row is tight at the probabilities at which the formulation reads it. Each re-solve reads the pair at the last posterior. That posterior stays feasible for the new row, so the divergence cannot increase. The formulation writes an equality view as the bound that the prior violates, and the minimisation of the divergence makes that bound tight.

The solver must support the power cone and the exponential cone, as [`ConicRelativisticValueatRiskView`](@ref) states, but only when the view has an asset in ``\\mathcal{P}``. A view whose assets are all in ``\\mathcal{N}`` is one linear row. An upper bound whose coefficients are all positive is such a view.

Where:

  - ``x_{i,\\,j}``: The loss of asset ``i`` at observation ``j``, which is the negated return.
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_target])
  - ``\\boldsymbol{\\nu}_{i}``: ``T \\times 1`` vector of weights that attains the RLVaR of asset ``i``, the variable of its dual representation.
  - ``\\gamma_{i}``: The coefficient that the view gives asset ``i``.
  - ``(t_{i}, z_{i})``: The shift and the dual variable that attain the RLVaR of asset ``i`` under the probabilities of the row, from [`ep_rlvar`](@ref).
  - ``\\mathcal{P}``, ``\\mathcal{N}``: The assets whose coefficient is positive, and the assets whose coefficient is negative, after the orientation of the view as a lower bound.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialRelativisticValueatRiskView(;
        iters::Integer = 20,
        tol::Number = 1e-8
    ) -> SequentialRelativisticValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `iters >= 0`.
  - `tol > 0`.

# Examples

```jldoctest
julia> SequentialRelativisticValueatRiskView()
SequentialRelativisticValueatRiskView
  iters ┼ Int64: 20
    tol ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)
  - [`RelativisticValueatRiskView`](@ref)
  - [`ep_rlvar`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 3.2, Equation 7, and Section 2, Equation 5.
"""
@concrete struct SequentialRelativisticValueatRiskView <:
                 AbstractRelativisticValueatRiskViewFormulation
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
    function SequentialRelativisticValueatRiskView(iters::Integer, tol::Number)
        @argcheck(iters >= zero(iters), DomainError(iters, "iters must be >= 0"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be > 0"))
        return new{typeof(iters), typeof(tol)}(iters, tol)
    end
end
function SequentialRelativisticValueatRiskView(; iters::Integer = 20,
                                               tol::Number = 1e-8)::SequentialRelativisticValueatRiskView
    return SequentialRelativisticValueatRiskView(iters, tol)
end
"""
    const CVaRVF_VecCVaRVF = Union{<:AbstractConditionalValueatRiskViewFormulation,
                                   <:AbstractVector{<:AbstractConditionalValueatRiskViewFormulation}}

Alias for a union of a single conditional value-at-risk view formulation or a vector of them.

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
"""
const CVaRVF_VecCVaRVF = Union{<:AbstractConditionalValueatRiskViewFormulation,
                               <:AbstractVector{<:AbstractConditionalValueatRiskViewFormulation}}
"""
    const EVaRVF_VecEVaRVF = Union{<:AbstractEntropicValueatRiskViewFormulation,
                                   <:AbstractVector{<:AbstractEntropicValueatRiskViewFormulation}}

Alias for a union of a single entropic value-at-risk view formulation or a vector of them.

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
"""
const EVaRVF_VecEVaRVF = Union{<:AbstractEntropicValueatRiskViewFormulation,
                               <:AbstractVector{<:AbstractEntropicValueatRiskViewFormulation}}
"""
    const RLVaRVF_VecRLVaRVF = Union{<:AbstractRelativisticValueatRiskViewFormulation,
                                     <:AbstractVector{<:AbstractRelativisticValueatRiskViewFormulation}}

Alias for a union of a single relativistic value-at-risk view formulation or a vector of them.

# Related

  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
"""
const RLVaRVF_VecRLVaRVF = Union{<:AbstractRelativisticValueatRiskViewFormulation,
                                 <:AbstractVector{<:AbstractRelativisticValueatRiskViewFormulation}}
"""
$(DocStringExtensions.TYPEDEF)

Spans of the two searches that read a relativistic value at risk.

[`ep_rlvar_shift`](@ref) minimises over the shift of the primal programme, and [`ep_rlvar`](@ref) minimises over the logarithm of the dual variable. No proof shows that a bracket holds the minimiser. Each bracket is a margin that is wide enough for the data on which the library was measured. If the minimiser is at an end of a bracket, make that bracket wider. `Optim` reports a minimiser at an end as converged, so read the minimiser and do not trust the flag. [EPRLVaR](@cite) states the primal programme in Equation 5 and states no span for a search, so the spans are the library's own.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskViewBracket(;
        tspan::Number = 2,
        log_zlo::Number = -20,
        log_zhi::Number = 10
    ) -> RelativisticValueatRiskViewBracket

Keywords correspond to the struct's fields.

## Validation

  - `tspan > 0`.
  - `log_zlo < log_zhi`.

# Examples

```jldoctest
julia> RelativisticValueatRiskViewBracket()
RelativisticValueatRiskViewBracket
    tspan ┼ Int64: 2
  log_zlo ┼ Int64: -20
  log_zhi ┴ Int64: 10
```

# Related

  - [`RelativisticValueatRiskView`](@ref)
  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_shift`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
@concrete struct RelativisticValueatRiskViewBracket <: AbstractAlgorithm
    """
    $(field_dict[:ep_bracket_rlvar_tspan])
    """
    tspan
    """
    $(field_dict[:ep_bracket_rlvar_log_zlo])
    """
    log_zlo
    """
    $(field_dict[:ep_bracket_rlvar_log_zhi])
    """
    log_zhi
    function RelativisticValueatRiskViewBracket(tspan::Number, log_zlo::Number,
                                                log_zhi::Number)
        @argcheck(tspan > zero(tspan), DomainError(tspan, "tspan must be > 0"))
        @argcheck(log_zlo < log_zhi,
                  DomainError((log_zlo, log_zhi), "log_zlo must be < log_zhi"))
        return new{typeof(tspan), typeof(log_zlo), typeof(log_zhi)}(tspan, log_zlo, log_zhi)
    end
end
function RelativisticValueatRiskViewBracket(; tspan::Number = 2, log_zlo::Number = -20,
                                            log_zhi::Number = 10)::RelativisticValueatRiskViewBracket
    return RelativisticValueatRiskViewBracket(tspan, log_zlo, log_zhi)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that hold a group of tail views and the settings of those views.

A significance level is a property of a view, not of the estimator that holds it. The conditional value at risk at 1% and the conditional value at risk at 10% are different statistics of the same series. An estimator of this family pairs a group of view equations with their level and their formulation, so one [`EntropyPoolingPrior`](@ref) can hold views at several levels.

# Related

  - [`AbstractEntropyPoolingViewEstimator`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Sections 2.2 and 2.3.
  - $(ref_dict[:EPRLVaR]) Section 3.2.
"""
abstract type AbstractEntropyPoolingTailViewEstimator <: AbstractEntropyPoolingViewEstimator end
"""
$(DocStringExtensions.TYPEDEF)

A group of conditional value at risk views, with their significance level and their formulation.

[`replace_prior_views`](@ref) replaces a `prior(...)` reference inside `views` by the prior conditional value at risk at the `alpha` of this group. A view stated against the prior therefore changes with the level.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        alg::Option{<:CVaRVF_VecCVaRVF} = nothing
    ) -> ConditionalValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `alg` is a vector, `!isempty(alg)`.

# Examples

```jldoctest
julia> ConditionalValueatRiskView(; alpha = 0.01,
                                  views = LinearConstraintEstimator(; val = \"A >= 0.07\"))
ConditionalValueatRiskView
  views ┼ LinearConstraintEstimator
        │   val ┼ String: "A >= 0.07"
        │   key ┴ nothing
  alpha ┼ Float64: 0.01
    alg ┴ nothing
```

# Related

  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equations 2 to 6.
"""
@concrete struct ConditionalValueatRiskView <: AbstractEntropyPoolingTailViewEstimator
    """
    $(field_dict[:ep_tv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    """
    $(field_dict[:ep_tv_alg])
    """
    alg
    function ConditionalValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                        alg::Option{<:CVaRVF_VecCVaRVF})
        assert_unit_interval(alpha, :alpha)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        return new{typeof(views), typeof(alpha), typeof(alg)}(views, alpha, alg)
    end
end
function ConditionalValueatRiskView(; views::LinearConstraintEstimator,
                                    alpha::Number = 0.05,
                                    alg::Option{<:CVaRVF_VecCVaRVF} = nothing)::ConditionalValueatRiskView
    return ConditionalValueatRiskView(views, alpha, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

A group of entropic value at risk views, with their significance level and their formulation.

[`replace_prior_views`](@ref) replaces a `prior(...)` reference inside `views` by the prior entropic value at risk at the `alpha` of this group. A view stated against the prior therefore changes with the level.

The field `alg` holds the settings of the grid of dual variables and the big-M multiplier. A [`GridEntropicValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`, so views at different significance levels can use different grids.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropicValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        alg::Option{<:EVaRVF_VecEVaRVF} = nothing,
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        zlo_frac::Option{<:Number} = nothing
    ) -> EntropicValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `alg` is a vector, `!isempty(alg)`.
  - If `zlo_frac` is a number, `0 < zlo_frac < 1`.

# Examples

```jldoctest
julia> EntropicValueatRiskView(; alpha = 0.01,
                               views = LinearConstraintEstimator(; val = \"A <= 0.09\"),
                               alg = GridEntropicValueatRiskView(; pct = 0.8, K = 21))
EntropicValueatRiskView
     views ┼ LinearConstraintEstimator
           │   val ┼ String: "A <= 0.09"
           │   key ┴ nothing
     alpha ┼ Float64: 0.01
       alg ┼ GridEntropicValueatRiskView
           │          pct ┼ Float64: 0.8
           │            K ┼ Int64: 21
           │            M ┼ Int64: 1
           │        iters ┼ Int64: 50
           │          tol ┼ Float64: 1.0e-10
           │   tilt_iters ┴ Int64: 200
      args ┼ Tuple{}: ()
    kwargs ┼ @NamedTuple{}: NamedTuple()
  zlo_frac ┴ nothing
```

# Related

  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equations 7 to 11.
  - $(ref_dict[:EPRLVaR]) Section 2, Equation 1.
"""
@concrete struct EntropicValueatRiskView <: AbstractEntropyPoolingTailViewEstimator
    """
    $(field_dict[:ep_tv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    """
    $(field_dict[:ep_tv_alg])
    """
    alg
    """
    $(field_dict[:optargs]) They go to each [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/) call that these views make. These calls are the search for the sample EVaR in [`ep_evar`](@ref), and the search for the centre of the grid in [`ep_evar_anchor`](@ref). Each call is a bracketed scalar minimisation, so an empty `args` gives `Optim.Brent()`.
    """
    args
    """
    $(field_dict[:optkwargs]) They go to the same searches as `args`.
    """
    kwargs
    """
    $(field_dict[:ep_tv_evar_zlo_frac])
    """
    zlo_frac
    function EntropicValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                     alg::Option{<:EVaRVF_VecEVaRVF}, args::Tuple,
                                     kwargs::NamedTuple, zlo_frac::Option{<:Number})
        assert_unit_interval(alpha, :alpha)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        if !isnothing(zlo_frac)
            @argcheck(zero(zlo_frac) < zlo_frac < one(zlo_frac),
                      DomainError(zlo_frac, "zlo_frac must be in (0, 1)"))
        end
        return new{typeof(views), typeof(alpha), typeof(alg), typeof(args), typeof(kwargs),
                   typeof(zlo_frac)}(views, alpha, alg, args, kwargs, zlo_frac)
    end
end
function EntropicValueatRiskView(; views::LinearConstraintEstimator, alpha::Number = 0.05,
                                 alg::Option{<:EVaRVF_VecEVaRVF} = nothing,
                                 args::Tuple = (), kwargs::NamedTuple = (;),
                                 zlo_frac::Option{<:Number} = nothing)::EntropicValueatRiskView
    return EntropicValueatRiskView(views, alpha, alg, args, kwargs, zlo_frac)
end
"""
$(DocStringExtensions.TYPEDEF)

A group of relativistic value at risk views, with their significance level, their deformation parameter and their formulation.

[`replace_prior_views`](@ref) replaces a `prior(...)` reference inside `views` by the prior relativistic value at risk at the `alpha` and the `kappa` of this group. A view stated against the prior therefore changes with both.

The field `alg` holds the settings of the grid of primal points and the big-M multiplier. A [`GridRelativisticValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`, so views at different significance levels can use different grids.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        alg::Option{<:RLVaRVF_VecRLVaRVF} = nothing,
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing
    ) -> RelativisticValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.
  - If `alg` is a vector, `!isempty(alg)`.

# Examples

```jldoctest
julia> RelativisticValueatRiskView(; alpha = 0.01, kappa = 0.5,
                                   views = LinearConstraintEstimator(; val = \"A >= 0.09\"))
RelativisticValueatRiskView
    views ┼ LinearConstraintEstimator
          │   val ┼ String: "A >= 0.09"
          │   key ┴ nothing
    alpha ┼ Float64: 0.01
    kappa ┼ Float64: 0.5
      alg ┼ nothing
     args ┼ Tuple{}: ()
   kwargs ┼ @NamedTuple{}: NamedTuple()
  bracket ┴ nothing
```

# Related

  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 3.2, Equation 7.
"""
@concrete struct RelativisticValueatRiskView <: AbstractEntropyPoolingTailViewEstimator
    """
    $(field_dict[:ep_tv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    """
    $(field_dict[:ep_tv_kappa])
    """
    kappa
    """
    $(field_dict[:ep_tv_alg])
    """
    alg
    """
    $(field_dict[:optargs]) They go to each [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/) call that these views make. These calls are the two searches of [`ep_rlvar`](@ref), the search for the shift in [`ep_rlvar_shift`](@ref), and the search for the centre of the grid in [`ep_rlvar_anchor`](@ref). Each call is a bracketed scalar minimisation, so an empty `args` gives `Optim.Brent()`.
    """
    args
    """
    $(field_dict[:optkwargs]) They go to the same searches as `args`.
    """
    kwargs
    """
    $(field_dict[:ep_tv_bracket])
    """
    bracket
    function RelativisticValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                         kappa::Number, alg::Option{<:RLVaRVF_VecRLVaRVF},
                                         args::Tuple, kwargs::NamedTuple,
                                         bracket::Option{<:RelativisticValueatRiskViewBracket})
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(kappa, :kappa)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        return new{typeof(views), typeof(alpha), typeof(kappa), typeof(alg), typeof(args),
                   typeof(kwargs), typeof(bracket)}(views, alpha, kappa, alg, args, kwargs,
                                                    bracket)
    end
end
function RelativisticValueatRiskView(; views::LinearConstraintEstimator,
                                     alpha::Number = 0.05, kappa::Number = 0.3,
                                     alg::Option{<:RLVaRVF_VecRLVaRVF} = nothing,
                                     args::Tuple = (), kwargs::NamedTuple = (;),
                                     bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)::RelativisticValueatRiskView
    return RelativisticValueatRiskView(views, alpha, kappa, alg, args, kwargs, bracket)
end
"""
    const CVV_VecCVV = Union{<:ConditionalValueatRiskView,
                             <:AbstractVector{<:ConditionalValueatRiskView}}

Alias for the shapes a `cvar_views` field accepts: one [`ConditionalValueatRiskView`](@ref), or a vector of them read under their own significance levels and formulations.

# Related

  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const CVV_VecCVV = Union{<:ConditionalValueatRiskView,
                         <:AbstractVector{<:ConditionalValueatRiskView}}
"""
    const EVV_VecEVV = Union{<:EntropicValueatRiskView,
                             <:AbstractVector{<:EntropicValueatRiskView}}

Alias for the shapes an `evar_views` field accepts: one [`EntropicValueatRiskView`](@ref), or a vector of them read under their own significance levels and formulations.

# Related

  - [`EntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const EVV_VecEVV = Union{<:EntropicValueatRiskView,
                         <:AbstractVector{<:EntropicValueatRiskView}}
"""
    const RVV_VecRVV = Union{<:RelativisticValueatRiskView,
                             <:AbstractVector{<:RelativisticValueatRiskView}}

Alias for the shapes an `rlvar_views` field accepts: one [`RelativisticValueatRiskView`](@ref), or a vector of them read under their own significance levels, deformation parameters and formulations.

# Related

  - [`RelativisticValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const RVV_VecRVV = Union{<:RelativisticValueatRiskView,
                         <:AbstractVector{<:RelativisticValueatRiskView}}

export ValueatRiskView, ConditionalValueatRiskView, EntropicValueatRiskView,
       LinearConditionalValueatRiskView, IntegerConditionalValueatRiskView,
       ConicEntropicValueatRiskView, GridEntropicValueatRiskView,
       RelativisticValueatRiskView, ConicRelativisticValueatRiskView,
       GridRelativisticValueatRiskView, RelativisticValueatRiskViewBracket,
       SequentialConditionalValueatRiskView, SequentialEntropicValueatRiskView,
       SequentialRelativisticValueatRiskView
