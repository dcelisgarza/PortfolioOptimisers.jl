"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that carry a group of entropy pooling views together with the settings those views are read under.

A significance level is a property of a view, not of the estimator that holds it: the value at risk at 1% and at 10% are different statistics of the same series. An estimator of this family pairs a group of view equations with the settings they are read under, so one entropy pooling estimator can hold views stated at several levels.

# Related

  - [`ValueatRiskView`](@ref)
  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
abstract type AbstractEntropyPoolingViewEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

A group of **value at risk** views, with the significance level they are read under.

Unlike a conditional or entropic value at risk view, a value at risk view is linear in the posterior probabilities: it reduces to rows of the constraint set through [`add_ep_constraint!`](@ref), so it needs no auxiliary variable, admits no choice of formulation, and reaches [`OptimEntropyPooling`](@ref) as readily as [`JuMPEntropyPooling`](@ref). That is why this estimator carries a level and nothing else.

The views this estimator holds accept `==` and `>=` alone, one asset per view, with a unit coefficient and a non-negative target. A `prior(...)` reference inside `views` is replaced by the prior value at risk at this group's `alpha`, so a view stated against the prior moves with the level.

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

  - $(ref_dict[:meucci2008])
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

A tail view constrains a quantile-based risk measure of the posterior distribution. Unlike a mean, variance or correlation view, it is not a linear function of the posterior probabilities, so each measure admits more than one way of writing it as a solvable program. The concrete subtypes name those ways.

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
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

  - $(ref_dict[:EPTail])
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

  - $(ref_dict[:EPTail])
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

  - $(ref_dict[:EPRLVaR])
"""
abstract type AbstractRelativisticValueatRiskViewFormulation <:
              AbstractEntropyPoolingViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Linear formulation of a conditional value-at-risk view [EPTail](@cite).

`LinearConditionalValueatRiskView` writes the view through the dual representation of CVaR. It adds ``T`` continuous variables and no integer variable, so it is the cheapest of the two CVaR formulations, and it is exact.

It accepts the operators `>=` and `==`, over any number of assets whose coefficients share one sign. An equality view needs a target greater than or equal to the prior CVaR of the view's left hand side. Below it the constraint is slack at the prior, so the entropy minimiser leaves the prior untouched and the view is not met. Use [`IntegerConditionalValueatRiskView`](@ref) or [`SequentialConditionalValueatRiskView`](@ref) there, and for a relative view whose coefficients carry both signs.

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

The box and the simplex describe every reweighting of the sample that no observation gives more than ``1/\\alpha`` times its posterior probability, and the largest loss such a reweighting attains is the CVaR. So the constraint set is feasible if and only if ``\\mathrm{CVaR}_{\\alpha}(X) \\geq \\bar{c}``, and a lower-bound view is exact.

A view over several assets, ``\\sum_{i} \\gamma_{i} \\mathrm{CVaR}_{\\alpha}(X_{i}) \\geq \\bar{c}`` with every ``\\gamma_{i} > 0``, takes one block of the first three rows per asset, and the last row reads ``\\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} \\geq \\bar{c}``. The CVaR is concave in the probabilities, so a positive combination of CVaRs is concave and its lower level set is convex. Each block attains its asset's CVaR on its own, so the encoding stays exact. A view with coefficients of both signs has no convex lower level set, and this formulation refuses it.

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

  - $(ref_dict[:EPTail])
"""
struct LinearConditionalValueatRiskView <: AbstractConditionalValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Integer formulation of a conditional value-at-risk view [EPTail](@cite).

`IntegerConditionalValueatRiskView` writes the view through the ordered weights representation of CVaR, selecting the tail of the posterior with a monotone binary vector. It expresses every comparison operator and any linear combination of per-asset CVaRs, at the cost of `sbar` binary variables per asset named by the view. It needs a solver that handles mixed-integer exponential cone programs.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The conditional value at risk is the mean of the ``\\alpha`` heaviest tail mass of the posterior, and this formulation states it over the ``\\bar{s}`` largest losses alone:

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
  - ``\\boldsymbol{y}``: ``\\bar{s} \\times 1`` binary vector that marks the observations entering the tail.
  - ``\\boldsymbol{q}``: ``\\bar{s} \\times 1`` auxiliary vector that carries the tail mass of each observation of the window.

The monotonicity constraint makes the marked set a suffix of the ascending order, which is what makes the expression the CVaR rather than the mean of an arbitrary subset of probability ``\\alpha``. An observation enters the tail in full when the observation below it is also marked, so every marked observation except the lowest one carries ``q_{j} = w_{[j]}``. The lowest one carries a part of its probability, and the sum condition fixes that part at ``\\alpha`` less the mass above it. That is the value at risk observation of the posterior, so the expression is the posterior CVaR exactly.

The window is the one restriction. A posterior whose tail of mass ``\\alpha`` reaches below the ``\\bar{s}`` largest losses is outside the feasible set, so the posterior is the one of least divergence among those whose tail stays inside the window. An upper-bound view moves mass down the order and needs a wider window than a lower-bound view. [`entropy_pooling`](@ref) warns when the window binds, which it does where the window holds no more than the tail mass ``\\alpha``. Raise `sbar` there. An `sbar` equal to the number of observations restricts nothing, at the cost of one binary variable per observation.

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

  - $(ref_dict[:EPTail])
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

`SequentialConditionalValueatRiskView` writes every view [`LinearConditionalValueatRiskView`](@ref) cannot, with no integer variable: an upper bound, an equality below the prior CVaR, and a relative view whose coefficients carry both signs. It replaces the CVaR of every asset on the wrong side of the inequality by a linear upper bound, solves the convex problem that results, and re-solves with the bound re-read at the posterior until the bound is tight. The view holds on every posterior of that sequence, and the divergence of each is at most that of the one before it.

The posterior is a local minimiser of the divergence. The feasible set of an upper-bound or relative CVaR view is not convex, so no convex program describes it exactly, and the sequence stops at a fixed point rather than at the posterior of least divergence. [`IntegerConditionalValueatRiskView`](@ref) reaches the latter when its window holds the tail of the posterior, at the cost of binary variables and a solver that handles mixed-integer exponential cone programs. A view of one asset with a lower-bound operator is convex, and the formulation then reduces to the linear one with no re-solve.

# Mathematical definition

Orient the view as a lower bound, negating both sides where its operator is `<=`, and write ``\\mathcal{P}`` for the assets whose coefficient is then positive and ``\\mathcal{N}`` for those whose coefficient is negative. The conditional value at risk is concave in the observation probabilities, so the assets of ``\\mathcal{P}`` take the dual representation of [`LinearConditionalValueatRiskView`](@ref), which is exact, and each asset of ``\\mathcal{N}`` takes the primal representation at a fixed value ``\\eta_{i}`` of its value at risk, which bounds the measure from above:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(X_{i}) &\\leq \\eta_{i} + \\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} \\left(x_{i,\\,j} - \\eta_{i}\\right)^{+}\\,, &\\forall\\, i \\in \\mathcal{N}\\\\
\\bar{c} &\\leq \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} + \\sum_{i \\in \\mathcal{N}} \\gamma_{i} \\left(\\eta_{i} + \\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} \\left(x_{i,\\,j} - \\eta_{i}\\right)^{+}\\right)\\,.
\\end{align}
```

The bound holds with equality where ``\\eta_{i}`` is the value at risk of ``X_{i}`` under ``\\boldsymbol{w}``, so the row is tight at the probabilities it was read at. Each re-solve reads ``\\eta_{i}`` at the last posterior, which stays feasible for the row that results, and that is why the divergence cannot rise. An equality view is written as the bound the prior violates, and the entropy minimiser makes it tight.

Where:

  - ``x_{i,\\,j}``: Loss of asset ``i`` at observation ``j``, the negated return.
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:cvar_target])
  - ``\\boldsymbol{\\nu}_{i}``: ``T \\times 1`` vector of weights that attains the CVaR of asset ``i``, the variable of its dual representation.
  - ``\\gamma_{i}``: Coefficient the view gives asset ``i``.
  - ``\\eta_{i}``: Value at risk of asset ``i`` under the probabilities the row was read at.
  - ``\\mathcal{P}``, ``\\mathcal{N}``: Assets whose coefficient is positive and negative once the view is oriented as a lower bound.

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

  - $(ref_dict[:EPTail])
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

Exponential cone formulation of an entropic value-at-risk view [EPTail](@cite).

`ConicEntropicValueatRiskView` writes the view through the dual representation of EVaR. It adds ``T`` continuous variables and one relative entropy cone, and it is exact.

It accepts the operators `>=` and `==`, over any number of assets whose coefficients share one sign. An equality view needs a target greater than or equal to the prior EVaR of the view's left hand side. Use [`GridEntropicValueatRiskView`](@ref) or [`SequentialEntropicValueatRiskView`](@ref) below it, and the latter for a relative view whose coefficients carry both signs.

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

A view over several assets, ``\\sum_{i} \\gamma_{i} \\mathrm{EVaR}_{\\alpha}(X_{i}) \\geq \\bar{e}`` with every ``\\gamma_{i} > 0``, takes one block of the first three rows per asset, and the last row reads ``\\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} \\geq \\bar{e}``. The EVaR is concave in the probabilities, so a positive combination of EVaRs is concave and its lower level set is convex. Each block attains its asset's EVaR on its own, so the encoding stays exact. A view with coefficients of both signs has no convex lower level set, and this formulation refuses it.

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

  - $(ref_dict[:EPTail])
"""
struct ConicEntropicValueatRiskView <: AbstractEntropicValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Grid formulation of an entropic value-at-risk view [EPTail](@cite).

`GridEntropicValueatRiskView` writes the view on a grid of values of the EVaR dual variable, built around the value that attains the prior EVaR of the asset. A lower-bound view is a set of linear constraints and needs no integer variable. An upper-bound or equality view selects one grid point with a binary vector and a big-``M`` relaxation, and needs a solver that handles mixed-integer exponential cone programs.

Each row of the upper-bound block reaches the model divided by its bound, so it reads against one at every grid point. Read at its own scale, the row of a small dual variable is smaller than a solver's feasibility tolerance, and the solver takes it as met by any posterior. The rows of the lower-bound block keep their own scale: a posterior that meets a lower bound moves mass toward the largest loss, so a row with a small bound is slack by far more than the tolerance. The big-M constant of each row is the smallest that releases it, which the data fix, and `M` multiplies it.

The answer is approximate in both directions. A lower-bound view holds at the grid points and may fall short between them, and an upper-bound view holds at one grid point and may be conservative. Widen `pct` or raise `K` when the posterior value misses the target, and prefer [`ConicEntropicValueatRiskView`](@ref) whenever the view admits it.

It accepts `==`, `>=` and `<=`, one asset per view. The view is normalised so its coefficient is one, which flips the operator when the coefficient is negative, so this formulation restricts neither the operator nor the sign.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The sample EVaR is the value of a scalar minimisation:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(X) &= \\underset{z > 0}{\\min} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
\\end{align}
```

So ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}`` holds exactly when the objective is at or above ``\\bar{e}`` at *every* ``z``, and ``\\mathrm{EVaR}_{\\alpha}(X) \\leq \\bar{e}`` holds when it is at or below ``\\bar{e}`` at *some* ``z``. The objective is linear in ``\\boldsymbol{w}`` once ``z`` is fixed, which is what makes a grid point a row. On a grid ``\\bar{z}_{1},\\ldots,\\bar{z}_{K}`` that gives, for a lower-bound view:

```math
\\begin{align}
&\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/\\bar{z}_{k})}{\\exp(\\bar{e}/\\bar{z}_{k})} \\geq \\alpha\\,, &\\forall\\, k = 1,\\ldots,K
\\end{align}
```

and for an upper-bound view, with ``\\boldsymbol{y}`` a binary selector, and each row divided by its bound ``\\alpha`` so that it reads against one:

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
  - ``\\boldsymbol{y}``: ``K \\times 1`` binary selector, one entry per grid point.
  - ``M_{k}``: Smallest big-M constant that releases the row of the ``k``-th grid point. The weights sum to one, so the left hand side of the row never exceeds its largest coefficient.
  - ``M``: Big-M multiplier.

An equality view carries both blocks.

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

  - $(ref_dict[:EPTail])
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

`SequentialEntropicValueatRiskView` writes every view [`ConicEntropicValueatRiskView`](@ref) cannot, with no integer variable and no grid: an upper bound, an equality below the prior EVaR, and a relative view whose coefficients carry both signs. It replaces the EVaR of every asset on the wrong side of the inequality by a linear upper bound, solves the convex problem that results, and re-solves with the bound re-read at the posterior until the bound is tight. The view holds on every posterior of that sequence, and the divergence of each is at most that of the one before it.

The posterior is a local minimiser of the divergence. The feasible set of an upper-bound or relative EVaR view is not convex, so no convex program describes it exactly, and the sequence stops at a fixed point rather than at the posterior of least divergence. [`GridEntropicValueatRiskView`](@ref) searches a grid of dual variables with binary variables instead, and holds the view only at the grid points. A view of one asset with a lower-bound operator is convex, and the formulation then reduces to the conic one with no re-solve.

# Mathematical definition

Orient the view as a lower bound, negating both sides where its operator is `<=`, and write ``\\mathcal{P}`` for the assets whose coefficient is then positive and ``\\mathcal{N}`` for those whose coefficient is negative. The entropic value at risk is concave in the observation probabilities, so the assets of ``\\mathcal{P}`` take the dual representation of [`ConicEntropicValueatRiskView`](@ref), which is exact. Each asset of ``\\mathcal{N}`` takes the primal representation at a fixed dual variable ``z_{i}``, which bounds the measure from above and is itself concave in the probabilities, so its tangent at the probabilities ``\\boldsymbol{w}^{0}`` it was read at bounds it again:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(X_{i}) &\\leq z_{i} \\ln\\left(\\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} e^{x_{i,\\,j}/z_{i}}\\right) \\leq \\mathrm{EVaR}_{\\alpha}(X_{i};\\, \\boldsymbol{w}^{0}) - z_{i} + \\sum_{j=1}^{T} w_{j} \\dfrac{z_{i} e^{x_{i,\\,j}/z_{i}}}{\\sum_{k=1}^{T} w^{0}_{k} e^{x_{i,\\,k}/z_{i}}}\\,, &\\forall\\, i \\in \\mathcal{N}\\\\
\\bar{e} &\\leq \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} + \\sum_{i \\in \\mathcal{N}} \\gamma_{i} \\left(\\mathrm{EVaR}_{\\alpha}(X_{i};\\, \\boldsymbol{w}^{0}) - z_{i} + \\sum_{j=1}^{T} w_{j} \\dfrac{z_{i} e^{x_{i,\\,j}/z_{i}}}{\\sum_{k=1}^{T} w^{0}_{k} e^{x_{i,\\,k}/z_{i}}}\\right)\\,.
\\end{align}
```

Both bounds hold with equality at ``\\boldsymbol{w} = \\boldsymbol{w}^{0}`` where ``z_{i}`` attains the EVaR there, so the row is tight at the probabilities it was read at. Each re-solve reads ``z_{i}`` and the tangent at the last posterior, which stays feasible for the row that results, and that is why the divergence cannot rise. An equality view is written as the bound the prior violates, and the entropy minimiser makes it tight.

Where:

  - ``x_{i,\\,j}``: Loss of asset ``i`` at observation ``j``, the negated return.
  - $(math_dict[:rlvar_probs])
  - ``\\boldsymbol{w}^{0}``: Probabilities the row was read at, the prior for the first solve and the last posterior for each re-solve.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:evar_target])
  - ``\\boldsymbol{\\nu}_{i}``: ``T \\times 1`` vector of weights that attains the EVaR of asset ``i``, the variable of its dual representation.
  - ``\\gamma_{i}``: Coefficient the view gives asset ``i``.
  - ``z_{i}``: Dual variable that attains the EVaR of asset ``i`` under ``\\boldsymbol{w}^{0}``, from [`ep_evar`](@ref).
  - ``\\mathcal{P}``, ``\\mathcal{N}``: Assets whose coefficient is positive and negative once the view is oriented as a lower bound.

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

  - $(ref_dict[:EPTail])
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

It accepts the operators `>=` and `==`, over any number of assets whose coefficients share one sign. An equality view needs a target greater than or equal to the prior RLVaR of the view's left hand side. Use [`GridRelativisticValueatRiskView`](@ref) or [`SequentialRelativisticValueatRiskView`](@ref) below it, and the latter for a relative view whose coefficients carry both signs. The solver must handle the power cone alongside the exponential cone the entropy pooling objective needs.

The programme is a demanding solve. A long sample, a small `alpha`, a small `kappa` or several of these views in one model can make a conic solver stop short of a solution. Give `opt` a vector of solver configurations, shorten the sample, or state the view under [`GridRelativisticValueatRiskView`](@ref), whose rows are linear in the posterior probabilities.

# Mathematical definition

The view ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) \\geq \\bar{\\vartheta}`` on one asset is written as:

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

A view over several assets, ``\\sum_{i} \\gamma_{i} \\mathrm{RLVaR}_{\\alpha,\\kappa}(X_{i}) \\geq \\bar{\\vartheta}`` with every ``\\gamma_{i} > 0``, takes one block of the first five rows per asset, and the last row reads ``\\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} \\geq \\bar{\\vartheta}``. The RLVaR is concave in the probabilities, so a positive combination of RLVaRs is concave and its lower level set is convex. Each block attains its asset's RLVaR on its own, so the encoding stays exact. A view with coefficients of both signs has no convex lower level set, and this formulation refuses it.

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

  - $(ref_dict[:EPRLVaR])
"""
struct ConicRelativisticValueatRiskView <: AbstractRelativisticValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Grid formulation of a relativistic value-at-risk view.

`GridRelativisticValueatRiskView` writes the view on a grid of points of the primal programme of RLVaR, centred on the point a posterior that meets the view attains. A lower-bound view is a set of linear constraints and needs no integer variable. An upper-bound or equality view selects one grid point with a binary vector and a big-``M`` relaxation, and needs a solver that handles mixed-integer exponential cone programs.

Each row of the upper-bound block reaches the model divided by its bound, so it reads against one at every grid point. Read at its own scale, the row of a small dual variable is smaller than a solver's feasibility tolerance, and the solver takes it as met by any posterior. The rows of the lower-bound block keep their own scale: a posterior that meets a lower bound moves mass toward the largest loss, so a row with a small bound is slack by far more than the tolerance. The big-M constant of each row is the smallest that releases it, which the data fix, and `M` multiplies it. The upper-bound block drops a grid point whose bound is at or below zero, because its row holds at no posterior. The lower-bound block keeps it: its row holds at every posterior.

It accepts `==`, `>=` and `<=`, one asset per view. The view is normalised so its coefficient is one, which flips the operator when the coefficient is negative, so this formulation restricts neither the operator nor the sign.

As `kappa` approaches one the RLVaR approaches the largest loss, and [`ep_rlvar_tail`](@ref) overflows at the dual variable that attains it. The points it overflows at are dropped, and a grid that keeps none of them raises. The centre of the grid is found by an iteration that reads the same tail function, so it too stops converging there and the grid falls back to the prior's dual variable, which lands short of the target. Prefer a smaller `kappa`, or [`ConicRelativisticValueatRiskView`](@ref) where the operator admits it.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The sample RLVaR is the value of a two-variable minimisation, in which the pair of power cones of each observation is already minimised out:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) &= \\underset{t,\\, z > 0}{\\min} \\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,,
\\end{align}
```

where ``\\varphi_{\\kappa}(u, z)`` is the smallest ``\\psi + \\theta`` the two power cones of one observation allow, and has the closed form:

```math
\\begin{align}
\\varphi_{\\kappa}(u, z) &= \\dfrac{\\kappa}{1+\\kappa} \\left(\\dfrac{2\\kappa}{(1+\\kappa) z}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma - u}{2}\\right)^{\\frac{1+\\kappa}{\\kappa}} + \\kappa (1-\\kappa)^{\\frac{1-\\kappa}{\\kappa}} \\left(\\dfrac{z}{2\\kappa}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma + u}{2}\\right)^{-\\frac{1-\\kappa}{\\kappa}}\\,,\\\\
\\sigma &= \\sqrt{u^{2} + \\dfrac{(1 - \\kappa^{2}) z^{2}}{\\kappa^{2}}}\\,.
\\end{align}
```

The objective is linear in ``\\boldsymbol{w}`` once ``t`` and ``z`` are fixed, which is what makes a grid point a row. On a grid ``(\\bar{t}_{1}, \\bar{z}_{1}),\\ldots,(\\bar{t}_{K}, \\bar{z}_{K})`` that gives, for a lower-bound view:

```math
\\begin{align}
&T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) \\geq \\bar{\\vartheta} - \\bar{t}_{k} - \\bar{z}_{k} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\,, &\\forall\\, k = 1,\\ldots,K
\\end{align}
```

and for an upper-bound view, with ``\\boldsymbol{y}`` a binary selector, and each row divided by its bound ``b_{k}`` so that it reads against one:

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
  - ``\\bar{t}_{k}``, ``\\bar{z}_{k}``: Shift and dual variable of the ``k``-th grid point.
  - ``K``: Number of grid points.
  - ``\\boldsymbol{y}``: ``K \\times 1`` binary selector, one entry per grid point.
  - ``b_{k}``: Bound of the row of the ``k``-th grid point. The upper-bound block drops a point whose bound is at or below zero.
  - ``M_{k}``: Smallest big-M constant that releases the row of the ``k``-th grid point. The weights sum to one, so the left hand side of the row never exceeds its largest coefficient.
  - ``M``: Big-M multiplier.

An equality view carries both blocks. Every grid point is a feasible point of the primal programme, so the upper-bound block is never violated: it can only be tighter than the view asks. The lower-bound block holds at the grid points and may fall short between them, so prefer [`ConicRelativisticValueatRiskView`](@ref) whenever the view admits it.

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

  - $(ref_dict[:EPRLVaR])
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

`SequentialRelativisticValueatRiskView` writes every view [`ConicRelativisticValueatRiskView`](@ref) cannot, with no integer variable and no grid: an upper bound, an equality below the prior RLVaR, and a relative view whose coefficients carry both signs. It replaces the RLVaR of every asset on the wrong side of the inequality by a linear upper bound, solves the convex problem that results, and re-solves with the bound re-read at the posterior until the bound is tight. The view holds on every posterior of that sequence, and the divergence of each is at most that of the one before it.

The posterior is a local minimiser of the divergence. The feasible set of an upper-bound or relative RLVaR view is not convex, so no convex program describes it exactly, and the sequence stops at a fixed point rather than at the posterior of least divergence. [`GridRelativisticValueatRiskView`](@ref) searches a grid of primal points with binary variables instead, and holds the view only at the grid points. A view of one asset with a lower-bound operator is convex, and the formulation then reduces to the conic one with no re-solve.

# Mathematical definition

Orient the view as a lower bound, negating both sides where its operator is `<=`, and write ``\\mathcal{P}`` for the assets whose coefficient is then positive and ``\\mathcal{N}`` for those whose coefficient is negative. The relativistic value at risk is concave in the observation probabilities, so the assets of ``\\mathcal{P}`` take the dual representation of [`ConicRelativisticValueatRiskView`](@ref), which is exact, and each asset of ``\\mathcal{N}`` takes the primal representation at a fixed pair ``(t_{i}, z_{i})``, which is linear in the probabilities and bounds the measure from above:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X_{i}) &\\leq t_{i} + z_{i} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t_{i} - x_{i,\\,j},\\, z_{i})\\,, &\\forall\\, i \\in \\mathcal{N}\\\\
\\bar{\\vartheta} &\\leq \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} + \\sum_{i \\in \\mathcal{N}} \\gamma_{i} \\left(t_{i} + z_{i} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t_{i} - x_{i,\\,j},\\, z_{i})\\right)\\,.
\\end{align}
```

The bound holds with equality where ``(t_{i}, z_{i})`` attains the RLVaR of ``X_{i}`` under ``\\boldsymbol{w}``, so the row is tight at the probabilities it was read at. Each re-solve reads the pair at the last posterior, which stays feasible for the row that results, and that is why the divergence cannot rise. An equality view is written as the bound the prior violates, and the entropy minimiser makes it tight.

The solver must handle the power cone alongside the exponential cone, as [`ConicRelativisticValueatRiskView`](@ref) states, and only where the view carries an asset in ``\\mathcal{P}``. A view whose assets are all in ``\\mathcal{N}``, which is every upper bound on a group, is one linear row.

Where:

  - ``x_{i,\\,j}``: Loss of asset ``i`` at observation ``j``, the negated return.
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_target])
  - ``\\boldsymbol{\\nu}_{i}``: ``T \\times 1`` vector of weights that attains the RLVaR of asset ``i``, the variable of its dual representation.
  - ``\\gamma_{i}``: Coefficient the view gives asset ``i``.
  - ``(t_{i}, z_{i})``: Shift and dual variable that attain the RLVaR of asset ``i`` under the probabilities the row was read at, from [`ep_rlvar`](@ref).
  - ``\\mathcal{P}``, ``\\mathcal{N}``: Assets whose coefficient is positive and negative once the view is oriented as a lower bound.

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

  - $(ref_dict[:EPRLVaR])
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

[`ep_rlvar_shift`](@ref) minimises over the shift of the primal programme, and [`ep_rlvar`](@ref) minimises over the logarithm of the dual variable. Neither bracket is a proof: each is a margin wide enough for the data this library was measured on. Widen one where the minimiser lands on an end of it. `Optim` reports an end as converged, so read the minimiser rather than trust the flag.

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

  - $(ref_dict[:EPRLVaR])
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

Abstract supertype for the estimators that carry a group of tail views together with the settings those views are read under.

A significance level is a property of a view, not of the estimator that holds it: the conditional value at risk at 1% and at 10% are different statistics of the same series. An estimator of this family pairs a group of view equations with the level and the formulation they take, so one [`EntropyPoolingPrior`](@ref) can hold views stated at several levels.

# Related

  - [`AbstractEntropyPoolingViewEstimator`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
abstract type AbstractEntropyPoolingTailViewEstimator <: AbstractEntropyPoolingViewEstimator end
"""
$(DocStringExtensions.TYPEDEF)

A group of **conditional value at risk** views, with the significance level and formulation they are read under.

A `prior(...)` reference inside `views` is replaced by the prior conditional value at risk at this group's `alpha`, so a view stated against the prior moves with the level.

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

  - $(ref_dict[:EPTail])
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

A group of **entropic value at risk** views, with the significance level and formulation they are read under.

A `prior(...)` reference inside `views` is replaced by the prior entropic value at risk at this group's `alpha`, so a view stated against the prior moves with the level.

`alg` is where the grid of dual variables and the big-M multiplier live: a [`GridEntropicValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`, so views at different significance levels can take different grids.

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

  - $(ref_dict[:EPTail])
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
    $(field_dict[:optargs]) It reaches every [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) call these views make: the sample EVaR of [`ep_evar`](@ref), and the centre of the grid of [`ep_evar_anchor`](@ref). Each is a bracketed scalar minimisation, so left empty it takes `Optim.Brent()`.
    """
    args
    """
    $(field_dict[:optkwargs]) They reach the same searches `args` does.
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

A group of **relativistic value at risk** views, with the significance level, the deformation parameter and the formulation they are read under.

A `prior(...)` reference inside `views` is replaced by the prior relativistic value at risk at this group's `alpha` and `kappa`, so a view stated against the prior moves with both.

`alg` is where the grid of primal points and the big-M multiplier live: a [`GridRelativisticValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`, so views at different significance levels can take different grids.

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

  - $(ref_dict[:EPRLVaR])
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
    $(field_dict[:optargs]) It reaches every [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) call these views make: the two searches of [`ep_rlvar`](@ref), the shift of [`ep_rlvar_shift`](@ref), and the centre of the grid of [`ep_rlvar_anchor`](@ref). Each is a bracketed scalar minimisation, so left empty it takes `Optim.Brent()`.
    """
    args
    """
    $(field_dict[:optkwargs]) They reach the same searches `args` does.
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
