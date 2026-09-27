"""
    const Sd_Var = Union{<:StandardDeviation, <:Variance}

Groups the two risk measures that a Schur complement bundle accepts.

The allocation reads the risk of each half from the augmented covariance block of that half, so it accepts only a measure that is a function of a covariance matrix alone.

# Related

  - [`StandardDeviation`](@ref)
  - [`Variance`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`naive_portfolio_risk`](@ref): Computes the risk of a block under either measure.
"""
const Sd_Var = Union{<:StandardDeviation, <:Variance}
"""
$(DocStringExtensions.TYPEDEF)

Holds the weights of a Schur complement allocation, with the prior, the clustering and the parameters that produced them.

[`SchurComplementHierarchicalRiskParity`](@ref) returns it. It also holds the resolved weight bounds, the resolved fees, the resolved risk measure, the value of ``\\gamma`` the allocation ran at, the return code and the fallback.

The result belongs to [`HierarchicalOptimisationResult`](@ref), because its estimator holds a [`HierarchicalOptimiser`](@ref). It does not embed a [`HierarchicalResult`](@ref) as its two siblings do. It keeps a flat block of fields, because it also holds `gamma`.

The result holds no scalariser. [`SchurComplementParams`](@ref) bounds its measure to [`Sd_Var`](@ref), so a bundle has one standard deviation or one variance and no vector of measures to combine.

!!! warning

    With a vector of bundles, `r` and `gamma` are vectors with one entry per bundle, and the allocation blends the portfolios of the bundles, not their risks. `expected_risk(res.r, res.w, res.pr)` then adds the scaled risk of each measure at the blended weights. The allocation never computes that number. With one bundle, which is the default, the same call returns the risk of the allocation under its measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SchurComplementHierarchicalRiskParityResult(;
        pr::Option{<:AbstractPriorResult},
        wb::Option{<:WeightBounds},
        clr::Option{<:AbstractClusteringResult},
        fees::Option{<:Fees},
        r::Union{<:Sd_Var, <:VecBaseRM},
        gamma::Union{<:Number, <:VecNum},
        retcode::OptimisationReturnCode,
        w::Option{<:VecNum},
        imsk::Option{<:BitVector} = nothing,
        fb::Option{<:OptE_Opt_FbChain}
    ) -> SchurComplementHierarchicalRiskParityResult

Keywords correspond to the struct's fields.

`_optimise` builds the result with the keyword constructor alone, so that constructor expands the solved weights back onto the full asset universe with [`expand_investable_weights`](@ref). The positional constructor does not expand. A rebuild calls it, and a second expansion of expanded weights is wrong.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`investable_reduction`](@ref)
  - [`expand_investable_weights`](@ref)
"""
@concrete struct SchurComplementHierarchicalRiskParityResult <:
                 HierarchicalOptimisationResult
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:clr])
    """
    clr
    """
    $(field_dict[:fees])
    """
    fees
    """
    $(field_dict[:r_res_schur])
    """
    r
    """
    $(field_dict[:gamma_schur_res])
    """
    gamma
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:pw])
    """
    w
    """
    $(field_dict[:imsk])
    """
    imsk
    """
    $(field_dict[:fb_res])
    """
    fb
    function SchurComplementHierarchicalRiskParityResult(pr::Option{<:AbstractPriorResult},
                                                         wb::Option{<:WeightBounds},
                                                         clr::Option{<:AbstractClusteringResult},
                                                         fees::Option{<:Fees},
                                                         r::Union{<:Sd_Var, <:VecBaseRM},
                                                         gamma::Union{<:Number, <:VecNum},
                                                         retcode::OptimisationReturnCode,
                                                         w::Option{<:VecNum},
                                                         imsk::Option{<:BitVector},
                                                         fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(pr), typeof(wb), typeof(clr), typeof(fees), typeof(r),
                   typeof(gamma), typeof(retcode), typeof(w), typeof(imsk), typeof(fb)}(pr,
                                                                                        wb,
                                                                                        clr,
                                                                                        fees,
                                                                                        r,
                                                                                        gamma,
                                                                                        retcode,
                                                                                        w,
                                                                                        imsk,
                                                                                        fb)
    end
end
function SchurComplementHierarchicalRiskParityResult(; pr::Option{<:AbstractPriorResult},
                                                     wb::Option{<:WeightBounds},
                                                     clr::Option{<:AbstractClusteringResult},
                                                     fees::Option{<:Fees},
                                                     r::Union{<:Sd_Var, <:VecBaseRM},
                                                     gamma::Union{<:Number, <:VecNum},
                                                     retcode::OptimisationReturnCode,
                                                     w::Option{<:VecNum},
                                                     imsk::Option{<:BitVector} = nothing,
                                                     fb::Option{<:OptE_Opt_FbChain})::SchurComplementHierarchicalRiskParityResult
    return SchurComplementHierarchicalRiskParityResult(pr, wb, clr, fees, r, gamma, retcode,
                                                       expand_investable_weights(imsk, w),
                                                       imsk, fb)
end
# The Schur family carries the mask on the result itself, so the fold reads it directly.
function result_investable_mask(res::SchurComplementHierarchicalRiskParityResult)
    return res.imsk
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that choose the Schur complement parameter ``\\gamma``.

A subtype decides whether [`SchurComplementParams`](@ref)`.gamma` is the value to use or the upper end of a range to search.

# Interfaces

To implement a new way to choose ``\\gamma``, subtype `SchurComplementAlgorithm` and implement the following method:

## `schur_complement_weights`

  - `schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt, wb::WeightBounds, params::SchurComplementParams{<:Any, <:Any, <:Any, <:MySchurComplementAlgorithm, <:Any}) -> Tuple`: The weights of the bundle `params`, at the value of ``\\gamma`` that the new algorithm chooses.

### Arguments

  - `pr`: Prior result.
  - `items`: The leaf orders to split, one entry that holds the whole leaf order of the dendrogram.
  - `wb`: Resolved weight bounds.
  - `params`: The bundle, whose `alg` field holds the new algorithm.

### Returns

  - `(w, gamma, r)::Tuple`: The weights, the value of ``\\gamma`` the allocation ran at, and the measure that [`factory`](@ref) resolved. `w` is `nothing` when the allocation fails. The usual implementation chooses ``\\gamma`` and calls the [`NonMonotonicSchurComplement`](@ref) method at it.

### Examples

An algorithm that runs the allocation at half of `params.gamma`:

```jldoctest
julia> struct MyHalfGamma <: PortfolioOptimisers.SchurComplementAlgorithm end

julia> function PortfolioOptimisers.schur_complement_weights(pr::PortfolioOptimisers.AbstractPriorResult,
                                                             items::PortfolioOptimisers.VecVecInt,
                                                             wb::WeightBounds,
                                                             params::SchurComplementParams{<:Any,
                                                                                           <:Any,
                                                                                           <:Any,
                                                                                           <:MyHalfGamma,
                                                                                           <:Any})
           nm = SchurComplementParams(; r = params.r, gamma = params.gamma / 2, pdm = params.pdm,
                                      alg = NonMonotonicSchurComplement(), flag = params.flag)
           return PortfolioOptimisers.schur_complement_weights(pr, items, wb, nm)
       end

julia> pr = LowOrderPrior(; X = zeros(2, 4), mu = zeros(4),
                          sigma = [4.0 1 1 0; 1 3 0 1; 1 0 2 0; 0 1 0 1] / 100);

julia> w, gamma, _ = PortfolioOptimisers.schur_complement_weights(pr, [collect(1:4)],
                                                                  WeightBounds(; lb = zeros(4),
                                                                               ub = ones(4)),
                                                                  SchurComplementParams(;
                                                                                        gamma = 0.8,
                                                                                        alg = MyHalfGamma()));

julia> gamma
0.4

julia> round.(w; digits = 4)
4-element Vector{Float64}:
 0.0872
 0.0956
 0.2616
 0.5557
```

# Related

  - [`SchurComplementParams`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
  - [`MonotonicSchurComplement`](@ref)
  - [`schur_complement_weights`](@ref)
"""
abstract type SchurComplementAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects the allocation at the value of ``\\gamma`` that the caller gives, with no search.

The augmentation uses [`SchurComplementParams`](@ref)`.gamma` as it is. The portfolio variance is not monotonic in ``\\gamma``, so a larger value does not always give a portfolio of lower variance. [`MonotonicSchurComplement`](@ref) searches for a value that does.

# Related

  - [`SchurComplementAlgorithm`](@ref)
  - [`MonotonicSchurComplement`](@ref)
  - [`SchurComplementParams`](@ref)
"""
struct NonMonotonicSchurComplement <: SchurComplementAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Caps ``\\gamma`` at the first turning point of the portfolio variance in ``[0, \\gamma]``.

The portfolio variance is not monotonic in the Schur complement parameter. It falls, then it can rise again. The allocation runs at the value that this tag finds, which is at most the [`SchurComplementParams`](@ref)`.gamma` that the caller gives.

# Mathematical definition

```math
\\begin{align}
v(g) &= \\boldsymbol{w}(g)^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}(g)\\,,\\\\
\\gamma^{\\star} &= \\sup \\left\\{ g \\in [0, \\gamma] : v \\textrm{ is non-increasing on } [0, g] \\right\\}\\,.
\\end{align}
```

Where:

  - ``v(g)``: Portfolio variance of the allocation at ``g``. It is ``+\\infty`` when an augmented block at ``g`` is not positive definite.
  - ``\\boldsymbol{w}(g)``: Weights of the allocation at ``g``, with no repair of an augmented block.
  - ``\\mathbf{\\Sigma}``: Covariance matrix of the measure of the bundle. It is the covariance matrix of the prior unless the measure holds its own.
  - $(math_dict[:gamma_sch])
  - ``\\gamma^{\\star}``: The value that the allocation runs at.

``\\gamma^{\\star}`` is the first turning point, not the minimiser of ``v`` over ``[0, \\gamma]``. Past the turning point the variance can fall again to a lower value, but a monotone path from [`HierarchicalRiskParity`](@ref) does not reach a portfolio there.

The objective is the variance for both measures. A [`StandardDeviation`](@ref) measure changes the split factors of the allocation, but not the quantity that the search compares.

# Algorithm

The branch of [`schur_complement_weights`](@ref) that this tag selects runs these steps.

 1. When `gamma` is zero, run the allocation at zero with the `flag` of the bundle, and return it.
 2. Make `nm_params`, a [`NonMonotonicSchurComplement`](@ref) copy of the bundle with `flag = false`. The `objective` of a value runs the allocation of `nm_params` at that value and returns the weights and ``v``. A failed allocation scores the largest value of the weight type.
 3. Make `gammas`, `N` evenly spaced values from zero to `gamma`.
 4. Evaluate `objective` at each entry of `gammas` in order. At the first entry `gammas[i]` whose variance is not lower than the variance at `gammas[i - 1]`, bisect the bracket with [`schur_complement_binary_search`](@ref) and return its result. The bracket is `gammas[i - 2]` to `gammas[i]`, or `gammas[1]` to `gammas[2]` when `i` is 2.
 5. When the scan finds no rise, evaluate `objective` at `gamma - tol`. If the variance at `gamma` is not higher, return `gamma` and its weights.
 6. Otherwise, bisect the bracket from `gammas[N - 1]` to `gamma`.

The scan sees a turning point only to the resolution of `gammas`. A rise and a fall again between two neighbouring entries stay unseen. The probe at `gamma - tol` in step 5, and the probe at `mgamma - tol` in the bisection, can evaluate a negative value. The formula of the augmentation has a value there, and the probe estimates the slope of ``v`` from the left.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MonotonicSchurComplement(;
        N::Integer = 10,
        tol::Number = 1e-4,
        iter::Option{<:Integer} = nothing,
        strict::Bool = false
    ) -> MonotonicSchurComplement

Keywords correspond to the struct's fields. `iter` defaults to `nothing`, which means the bisection derives its own budget from the bracket and `tol`, as `ceil(Int, log2((hgamma - lgamma) / tol) * 4 + 10)`.

## Validation

  - `N > 1`. The scan needs both ends of the range.
  - `tol > 0`.
  - If `iter` is given: `iter > 0`.

# Related

  - [`SchurComplementAlgorithm`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`schur_complement_binary_search`](@ref)
"""
@concrete struct MonotonicSchurComplement <: SchurComplementAlgorithm
    """
    $(field_dict[:N_msc])
    """
    N
    """
    $(field_dict[:tol])
    """
    tol
    """
    $(field_dict[:iter])
    """
    iter
    """
    $(field_dict[:strict_conv])
    """
    strict
    function MonotonicSchurComplement(N::Integer, tol::Number, iter::Option{<:Integer},
                                      strict::Bool)
        @argcheck(N > 1, DomainError(N, "N must be > 1"))
        @argcheck(tol > 0, DomainError(tol, "tol must be > 0"))
        if !isnothing(iter)
            @argcheck(iter > 0, DomainError(iter, "iter must be > 0"))
        end
        return new{typeof(N), typeof(tol), typeof(iter), typeof(strict)}(N, tol, iter,
                                                                         strict)
    end
end
function MonotonicSchurComplement(; N::Integer = 10, tol::Number = 1e-4,
                                  iter::Option{<:Integer} = nothing,
                                  strict::Bool = false)::MonotonicSchurComplement
    return MonotonicSchurComplement(N, tol, iter, strict)
end
"""
$(DocStringExtensions.TYPEDEF)

Collects the risk measure, the interpolation parameter ``\\gamma`` and the two algorithms of one Schur complement bundle.

[`SchurComplementHierarchicalRiskParity`](@ref) holds one bundle or a vector of bundles. A vector runs one allocation per bundle, and blends the portfolios of the bundles in proportion to the `r.settings.scale` of each bundle.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SchurComplementParams(;
        r::Sd_Var = Variance(),
        gamma::Number = 0.5,
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        alg::SchurComplementAlgorithm = MonotonicSchurComplement(),
        flag::Bool = true
    ) -> SchurComplementParams

Keywords correspond to the struct's fields. [`Sd_Var`](@ref) bounds `r`, because the allocation reads the risk of a half from the augmented covariance block of that half.

## Validation

  - `0 <= gamma <= 1`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementAlgorithm`](@ref)
  - [`MonotonicSchurComplement`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
  - [`Sd_Var`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cotton2024])
"""
@concrete struct SchurComplementParams <: AbstractAlgorithm
    """
    $(field_dict[:r])
    """
    r
    """
    $(field_dict[:gamma_schur])
    """
    gamma
    """
    $(field_dict[:pdm])
    """
    pdm
    """
    $(field_dict[:schalg])
    """
    alg
    """
    $(field_dict[:flag_schur])
    """
    flag
    function SchurComplementParams(r::Sd_Var, gamma::Number,
                                   pdm::Option{<:AbstractPosdefEstimator},
                                   alg::SchurComplementAlgorithm, flag::Bool)
        @argcheck(one(gamma) >= gamma >= zero(gamma),
                  DomainError(gamma, "gamma must be in [0, 1]"))
        return new{typeof(r), typeof(gamma), typeof(pdm), typeof(alg), typeof(flag)}(r,
                                                                                     gamma,
                                                                                     pdm,
                                                                                     alg,
                                                                                     flag)
    end
end
function SchurComplementParams(; r::Sd_Var = Variance(), gamma::Number = 0.5,
                               pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                               alg::SchurComplementAlgorithm = MonotonicSchurComplement(),
                               flag::Bool = true)::SchurComplementParams
    return SchurComplementParams(r, gamma, pdm, alg, flag)
end
"""
    const VecScP = AbstractVector{<:SchurComplementParams}

Groups the vectors of Schur complement bundles.

A vector of bundles runs one allocation per bundle and blends the portfolios. The alias is the vector half of [`ScP_VecScP`](@ref).

# Related

  - [`SchurComplementParams`](@ref)
  - [`ScP_VecScP`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
const VecScP = AbstractVector{<:SchurComplementParams}
"""
    const ScP_VecScP = Union{<:SchurComplementParams, <:VecScP}

Groups one Schur complement bundle and a vector of bundles.

The `params` field of the estimator takes either form, so its bound is this alias.

# Related

  - [`SchurComplementParams`](@ref)
  - [`VecScP`](@ref)
"""
const ScP_VecScP = Union{<:SchurComplementParams, <:VecScP}
"""
    port_opt_view(sp::SchurComplementParams, i, X::MatNum, args...) -> SchurComplementParams

Return the view of a Schur complement bundle on the assets `i`.

A view of the estimator on a subset of assets calls it, for example when an outer optimiser runs the estimator on one cluster. `gamma`, `pdm`, `alg` and `flag` hold no per-asset data, so they pass through unchanged.

# Arguments

  - `sp`: Bundle to view.
  - `i`: Asset indices of the view.
  - `X`: Returns matrix, which the risk measure's own [`port_opt_view`](@ref) method reads.
  - `args`: Ignored.

# Returns

  - `sp::SchurComplementParams`: The bundle with `r` replaced by its view.

# Related

  - [`SchurComplementParams`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function port_opt_view(sp::SchurComplementParams, i, X::MatNum, args...)
    r = port_opt_view(sp.r, i, X)
    return SchurComplementParams(; r = r, gamma = sp.gamma, pdm = sp.pdm, alg = sp.alg,
                                 flag = sp.flag)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`SchurComplementHierarchicalRiskParity`](@ref) fields that may hold a [`TimeDependent`](@ref).

The substitution check of the constructor, [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) both read it, so the code declares the value of a field outside a fold once. The tuple leaves out a field whose static default is `nothing`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function schur_complement_hrp_td_defaults()::NamedTuple
    return (; params = SchurComplementParams())
end
"""
$(DocStringExtensions.TYPEDEF)

Runs the hierarchical risk parity recursion on covariance blocks that a Schur complement augments with the information of the cross-cluster block.

The parameter ``\\gamma`` interpolates. At `gamma = 0` with a [`Variance`](@ref) measure, the allocation equals that of [`HierarchicalRiskParity`](@ref) with the same `opt` and a [`Variance`](@ref) measure. A larger value moves the allocation towards the minimum variance portfolio.

# Mathematical definition

The recursion is that of [`HierarchicalRiskParity`](@ref). It splits the dendrogram's leaf order in half, and divides the weight of the part between the two halves in inverse proportion to their risks. Schur changes only the covariance block that gives the risk of each half. Partition the covariance matrix of the part over its two halves ``C_1`` and ``C_2``:

```math
\\begin{align}
\\mathbf{\\Sigma} &= \\begin{pmatrix} \\mathbf{\\Sigma}_{11} & \\mathbf{\\Sigma}_{12} \\\\ \\mathbf{\\Sigma}_{21} & \\mathbf{\\Sigma}_{22} \\end{pmatrix}\\,,\\\\
\\mathbf{A} &= \\mathbf{\\Sigma}_{11} - \\gamma \\, \\mathbf{\\Sigma}_{12} \\mathbf{\\Sigma}_{22}^{-1} \\mathbf{\\Sigma}_{21}\\,,\\\\
\\mathbf{R} &= \\mathbf{I} - \\gamma \\, \\mathbf{\\Sigma}_{12} \\mathbf{\\Sigma}_{22}^{-1} \\mathbf{M}^\\intercal\\,,\\\\
\\hat{\\mathbf{\\Sigma}}_{11} &= \\frac{1}{2}\\left(\\mathbf{R}^{-1}\\mathbf{A} + \\left(\\mathbf{R}^{-1}\\mathbf{A}\\right)^\\intercal\\right)\\,.
\\end{align}
```

``\\hat{\\mathbf{\\Sigma}}_{22}`` follows when the two halves exchange their roles. The risk of a half then comes from its augmented block, with the naive risk parity weights of that block:

```math
\\begin{align}
\\tilde{w}_i &= \\frac{\\left(\\hat{\\mathbf{\\Sigma}}_{11}\\right)_{ii}^{-1}}{\\sum_{j} \\left(\\hat{\\mathbf{\\Sigma}}_{11}\\right)_{jj}^{-1}}\\,,\\\\
\\tilde{\\rho}(C_1) &= \\tilde{\\boldsymbol{w}}^\\intercal \\hat{\\mathbf{\\Sigma}}_{11} \\tilde{\\boldsymbol{w}}\\,,\\\\
\\alpha &= \\frac{\\tilde{\\rho}(C_2)}{\\tilde{\\rho}(C_1) + \\tilde{\\rho}(C_2)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_halves_hier])
  - $(math_dict[:gamma_sch])
  - $(math_dict[:Sigma_blocks_sch])
  - $(math_dict[:A_sch])
  - $(math_dict[:M_step_up_sch])
  - $(math_dict[:I_identity])
  - $(math_dict[:R_sch])
  - $(math_dict[:Sigma_hat_sch])
  - $(math_dict[:w_naive_sch])
  - $(math_dict[:rho_naive_sch])
  - $(math_dict[:alpha_split_hier])

The weights are the inverse variances of the augmented block for both measures. So at ``\\gamma = 0`` a [`Variance`](@ref) measure gives the allocation of [`HierarchicalRiskParity`](@ref) under a [`Variance`](@ref) measure. A [`StandardDeviation`](@ref) measure gives a different allocation from [`HierarchicalRiskParity`](@ref) under a [`StandardDeviation`](@ref) measure, because that optimiser takes inverse volatility weights.

A part of two or three leaves has a half ``C_1`` of one leaf, and the recursion augments neither of its halves. The resolved weight bounds clamp ``\\alpha``, as in [`HierarchicalRiskParity`](@ref). ``C_1`` and ``C_2`` receive ``\\alpha`` and ``1 - \\alpha`` of the weight of the part in every case, so the weights sum to one.

# Algorithm

The fit of [`optimise`](@ref) runs these steps.

 1. Resolve every [`TimeDependent`](@ref) field to its static default, and pick the returns `rd` that `opt.brt` selects.
 2. Compute the prior `pr` with `opt.pe`, and the investable mask `imsk` of the prior.
 3. Resolve the fees `fees` on the full asset universe.
 4. Reduce `pr`, the estimator and `rd` to the investable universe.
 5. Cluster the reduced prior with `opt.cle`, giving `clr`. The leaf order `clr.res.order` is the input of the recursion.
 6. Resolve the weight bounds `wb` on the investable universe.
 7. Run [`schur_complement_weights`](@ref) with `params`, giving the weights `w`, the value `gamma` it ran at, and the resolved measure `r`. When `params` is a vector, run it once per bundle, add the weights of each bundle times the `r.settings.scale` of the bundle into `w`, and divide `w` by its sum.
 8. Check that the recursion produced weights, with [`assert_schur_weights`](@ref).
 9. Apply the weight finaliser `opt.wf` to `w` with `wb`, giving the return code `retcode` and the final weights.
10. Build the result, which expands the weights back onto the full asset universe.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SchurComplementHierarchicalRiskParity(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        params::TD{<:ScP_VecScP} = SchurComplementParams(),
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> SchurComplementHierarchicalRiskParity

Keywords correspond to the struct's fields. A field typed [`TD`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule with one value per fold. The bundles and the fallback define the problem, so a cross-validation fold loop resolves them once per fold. A plain `optimise` with no fold runs each at its static default, which is `nothing` for `fb`.

## Validation

  - If `params` is a vector: `!isempty(params)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`SchurComplementHierarchicalRiskParity` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` already holds a prior result, the method replaces `X` with `opt.pe.X`, so it views the children against the observations of the prior and not against the matrix of the caller.
  - `opt` and `params` recurse through [`port_opt_view`](@ref) with that matrix.
  - The method views `fb` through [`view_child`](@ref), which views an estimator fallback at `i` and keeps a precomputed result as it is.

# Examples

```jldoctest
julia> SchurComplementHierarchicalRiskParity()
SchurComplementHierarchicalRiskParity
     opt ┼ HierarchicalOptimiser
         │       pe ┼ EmpiricalPrior
         │          │           ce ┼ PortfolioOptimisersCovariance
         │          │              │   ce ┼ Covariance
         │          │              │      │    me ┼ SimpleExpectedReturns
         │          │              │      │       │   w ┴ nothing
         │          │              │      │    ce ┼ GeneralCovariance
         │          │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │          │              │      │       │    w ┴ nothing
         │          │              │      │   alg ┼ FullMoment()
         │          │              │      │     w ┴ nothing
         │          │              │   mp ┼ MatrixProcessing
         │          │              │      │     pdm ┼ Posdef
         │          │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │          │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │          │              │      │      dn ┼ nothing
         │          │              │      │      dt ┼ nothing
         │          │              │      │     alg ┼ nothing
         │          │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │          │           me ┼ SimpleExpectedReturns
         │          │              │   w ┴ nothing
         │          │      horizon ┼ nothing
         │          │   fill_limit ┴ nothing
         │      cle ┼ ClustersEstimator
         │          │    ce ┼ PortfolioOptimisersCovariance
         │          │       │   ce ┼ Covariance
         │          │       │      │    me ┼ SimpleExpectedReturns
         │          │       │      │       │   w ┴ nothing
         │          │       │      │    ce ┼ GeneralCovariance
         │          │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │          │       │      │       │    w ┴ nothing
         │          │       │      │   alg ┼ FullMoment()
         │          │       │      │     w ┴ nothing
         │          │       │   mp ┼ MatrixProcessing
         │          │       │      │     pdm ┼ Posdef
         │          │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │          │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │          │       │      │      dn ┼ nothing
         │          │       │      │      dt ┼ nothing
         │          │       │      │     alg ┼ nothing
         │          │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │          │    de ┼ Distance
         │          │       │   power ┼ nothing
         │          │       │     alg ┴ CanonicalDistance()
         │          │   alg ┼ HClustAlgorithm
         │          │       │   linkage ┴ Symbol: :ward
         │          │   onc ┼ OptimalNumberClusters
         │          │       │   max_k ┼ nothing
         │          │       │     alg ┼ SecondOrderDifference
         │          │       │         │   alg ┼ StandardisedValue
         │          │       │         │       │   mv ┼ MeanValue
         │          │       │         │       │      │   w ┴ nothing
         │          │       │         │       │   sv ┼ StdValue
         │          │       │         │       │      │           w ┼ nothing
         │          │       │         │       │      │   corrected ┴ Bool: true
         │      slv ┼ nothing
         │       wb ┼ WeightBounds
         │          │   lb ┼ Float64: 0.0
         │          │   ub ┴ Float64: 1.0
         │     fees ┼ nothing
         │     sets ┼ nothing
         │       wf ┼ IterativeWeightFinaliser
         │          │   iter ┴ Int64: 100
         │      brt ┼ Bool: false
         │    x_src ┼ Symbol: :prior
         │   strict ┴ Bool: false
  params ┼ SchurComplementParams
         │       r ┼ Variance
         │         │   settings ┼ RiskMeasureSettings
         │         │            │   scale ┼ Float64: 1.0
         │         │            │      ub ┼ nothing
         │         │            │     rke ┴ Bool: true
         │         │      sigma ┼ nothing
         │         │       chol ┼ nothing
         │         │         rc ┼ nothing
         │         │        alg ┴ SquaredSOCRiskExpr()
         │   gamma ┼ Float64: 0.5
         │     pdm ┼ Posdef
         │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │     alg ┼ MonotonicSchurComplement
         │         │        N ┼ Int64: 10
         │         │      tol ┼ Float64: 0.0001
         │         │     iter ┼ nothing
         │         │   strict ┴ Bool: false
         │    flag ┴ Bool: true
      fb ┴ nothing
```

# Related

  - [`optimise`](@ref)
  - [`SchurComplementHierarchicalRiskParityResult`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`symmetric_step_up_matrix`](@ref)
  - [`schur_augmentation`](@ref)
  - [`naive_portfolio_risk`](@ref)
  - [`schur_complement_weights`](@ref): Runs the recursion of one bundle.
  - [`split_factor_weight_constraints`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`view_child`](@ref)

# References

  - $(ref_dict[:cotton2024])
"""
@propagatable @concrete struct SchurComplementHierarchicalRiskParity <:
                               ClusteringOptimisationEstimator
    """
    $(field_dict[:opt_hier])
    """
    @fprop opt
    """
    $(field_dict[:params])
    """
    params
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function SchurComplementHierarchicalRiskParity(opt::HierarchicalOptimiser,
                                                   params::TD{<:ScP_VecScP},
                                                   fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb,
                                                  :SchurComplementHierarchicalRiskParity)
        if isa(params, AbstractVector)
            @argcheck(!isempty(params), IsEmptyError("params cannot be empty"))
        end
        assert_time_dependent_substitution(SchurComplementHierarchicalRiskParity,
                                           (; opt, params, fb),
                                           schur_complement_hrp_td_defaults())
        return new{typeof(opt), typeof(params), typeof(fb)}(opt, params, fb)
    end
end
function SchurComplementHierarchicalRiskParity(;
                                               opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                               params::TD{<:ScP_VecScP} = SchurComplementParams(),
                                               fb::TDO_Option{<:OptE_Opt} = nothing)::SchurComplementHierarchicalRiskParity
    return SchurComplementHierarchicalRiskParity(opt, params, fb)
end
function time_dependent_field_defaults(::SchurComplementHierarchicalRiskParity)::NamedTuple
    return schur_complement_hrp_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`SchurComplementHierarchicalRiskParity`](@ref) needs previous portfolio weights.

It needs them when a [`TimeDependent`](@ref) field, `opt` or `fb` needs them.

# Related

  - [`needs_previous_weights`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function needs_previous_weights(opt::SchurComplementHierarchicalRiskParity)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`SchurComplementHierarchicalRiskParity`](@ref) `sh` sliced to asset indices `i`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(sh::SchurComplementHierarchicalRiskParity, i, X::MatNum,
                       args...)::SchurComplementHierarchicalRiskParity
    X = isa(sh.opt.pe, AbstractPriorResult) ? sh.opt.pe.X : X
    opt = port_opt_view(sh.opt, i, X)
    params = port_opt_view(sh.params, i, X)
    return SchurComplementHierarchicalRiskParity(; opt = opt, params = params,
                                                 fb = view_child(sh.fb, i, X))
end
function non_investable_universe(sh::SchurComplementHierarchicalRiskParity, ni::VecStr)
    return rebuild_estimator(sh, (; opt = non_investable_universe(sh.opt, ni)))
end
"""
    symmetric_step_up_matrix(n1::Integer, n2::Integer) -> AbstractMatrix

Build the matrix that carries a Schur complement between two halves of nearly equal size.

The augmentation subtracts a term that the other half shapes, so the result must come back to the size of the half that is augmented. This matrix is that map.

# Mathematical definition

```math
\\begin{align}
\\mathbf{M}_{n_1, n_2} &= \\begin{cases} \\mathbf{I}_{n_1} & n_1 = n_2\\,,\\\\ \\dfrac{1}{n_1} \\sum_{k=1}^{n_1} \\mathbf{E}_k & n_1 = n_2 + 1\\,,\\\\ \\dfrac{n_1}{n_2} \\mathbf{M}_{n_2, n_1}^\\intercal & n_1 = n_2 - 1\\,,\\end{cases}\\\\
\\mathbf{E}_k &= \\begin{pmatrix} \\boldsymbol{e}_1 & \\cdots & \\boldsymbol{e}_{k-1} & \\frac{1}{n_2} \\mathbf{1} & \\boldsymbol{e}_k & \\cdots & \\boldsymbol{e}_{n_2} \\end{pmatrix}^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}_{n_1, n_2}``: Symmetric step-up matrix of ``n_1`` rows and ``n_2`` columns.
  - ``n_1``, ``n_2``: Sizes of the half that is augmented and of the other half.
  - ``\\mathbf{E}_k``: The identity matrix of size ``n_2`` with a uniform row inserted at row ``k``.
  - ``\\boldsymbol{e}_j``: The ``j``-th unit vector of length ``n_2``.
  - ``\\mathbf{1}``: The vector of ``n_2`` ones.

``\\mathbf{M}_{n_1, n_2}`` averages the ``n_1`` positions that an extra uniform row can take. Every row of it sums to one in all three cases, so ``\\mathbf{M}_{n_1, n_2} \\mathbf{1} = \\mathbf{1}``.

# Arguments

  - `n1`: Size of the half that is augmented, and the number of rows.
  - `n2`: Size of the other half, and the number of columns.

# Validation

  - `abs(n1 - n2) <= 1`. A bisection makes halves that differ by at most one, so no other shape reaches this method from the recursion.

# Returns

  - `m::AbstractMatrix`: An `n1` by `n2` matrix. It is `LinearAlgebra.I(n1)` when `n1 == n2`.

# Related

  - [`schur_augmentation`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function symmetric_step_up_matrix(n1::Integer, n2::Integer)
    @argcheck(abs(n1 - n2) <= 1,
              DomainError((n1, n2),
                          "`n1` is $n1 and `n2` is $n2. A bisection produces halves that differ by at most one, so `abs(n1 - n2) <= 1` must hold."))

    if n1 == n2
        return LinearAlgebra.I(n1)
    elseif n1 < n2
        return transpose(symmetric_step_up_matrix(n2, n1)) * n1 / n2
    end

    m = zeros(n1, n2)
    row = fill(inv(n2), n2)
    e = LinearAlgebra.I(n2)
    for i in axes(m, 1)
        view(m, 1:(i - 1), :) .+= view(e, 1:(i - 1), :) ./ n1
        view(m, i, :) .+= row ./ n1
        view(m, (i + 1):n1, :) .+= view(e, i:n2, :) ./ n1
    end
    return m
end
"""
    schur_augmentation(A::MatNum, B::MatNum, C::MatNum, gamma::Number) -> MatNum

Augment the covariance block of one half with the information of the cross block.

The result is not always positive definite. [`schur_complement_weights`](@ref) repairs it or abandons the allocation, as [`SchurComplementParams`](@ref)`.flag` selects. When either half holds one asset, the function returns ``\\mathbf{\\Sigma}_{11}`` unchanged and does not apply the formula below.

# Mathematical definition

```math
\\begin{align}
\\mathbf{A} &= \\mathbf{\\Sigma}_{11} - \\gamma \\, \\mathbf{\\Sigma}_{12} \\mathbf{\\Sigma}_{22}^{-1} \\mathbf{\\Sigma}_{21}\\,,\\\\
\\mathbf{R} &= \\mathbf{I} - \\gamma \\, \\mathbf{\\Sigma}_{12} \\mathbf{\\Sigma}_{22}^{-1} \\mathbf{M}^\\intercal\\,,\\\\
\\hat{\\mathbf{\\Sigma}}_{11} &= \\frac{1}{2}\\left(\\mathbf{R}^{-1}\\mathbf{A} + \\left(\\mathbf{R}^{-1}\\mathbf{A}\\right)^\\intercal\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_blocks_sch])
  - $(math_dict[:gamma_sch])
  - $(math_dict[:A_sch])
  - $(math_dict[:M_step_up_sch])
  - $(math_dict[:I_identity])
  - $(math_dict[:R_sch])
  - $(math_dict[:Sigma_hat_sch])

# Algorithm

 1. When `gamma` is zero, or when `A` or `C` has one row, return `A` unchanged.
 2. Compute `A_aug`, the matrix ``\\mathbf{A}``, with a linear solve against `C`.
 3. Build `m`, the [`symmetric_step_up_matrix`](@ref) of the sizes of `A` and `C`.
 4. Compute `r`, the matrix ``\\mathbf{R}``, with a linear solve against the transpose of `C`.
 5. Solve `r \\ A_aug`, and return the mean of the solution and its transpose.

# Arguments

  - `A`: Covariance block ``\\mathbf{\\Sigma}_{11}`` of the half that is augmented.
  - `B`: Cross block ``\\mathbf{\\Sigma}_{12}`` of the two halves, with the assets of `A` along the rows.
  - `C`: Covariance block ``\\mathbf{\\Sigma}_{22}`` of the other half.
  - `gamma`: Schur complement parameter in `[0, 1]`.

# Returns

  - `A_aug::MatNum`: The augmented block ``\\hat{\\mathbf{\\Sigma}}_{11}``. It is symmetric, and of the same size as `A`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`symmetric_step_up_matrix`](@ref)
  - [`naive_portfolio_risk`](@ref)
"""
function schur_augmentation(A::MatNum, B::MatNum, C::MatNum, gamma::Number)
    Na = size(A, 1)
    Nc = size(C, 1)
    if iszero(gamma) || isone(Na) || isone(Nc)
        return A
    end
    A_aug = A - gamma * B * (C \ transpose(B))
    m = symmetric_step_up_matrix(Na, Nc)
    r = LinearAlgebra.I - gamma * transpose(transpose(C) \ transpose(B)) * transpose(m)
    A_aug = r \ A_aug
    return (A_aug + transpose(A_aug)) / 2
end
"""
    naive_portfolio_risk(r::Sd_Var, sigma::MatNum) -> Number

Compute the risk of the naive risk parity portfolio of a covariance block.

The weights are the inverse variances of the block for both measures, and the measure changes only what the function does with the quadratic form. So the weights match the naive weights of [`HierarchicalRiskParity`](@ref) under a [`Variance`](@ref) measure, and differ from its inverse volatility weights under a [`StandardDeviation`](@ref) measure. `sigma` is usually an augmented block, so its diagonal is not a plain asset variance.

# Mathematical definition

```math
\\begin{align}
\\tilde{w}_i &= \\frac{\\Sigma_{ii}^{-1}}{\\sum_{j} \\Sigma_{jj}^{-1}}\\,,\\\\
\\tilde{\\rho} &= \\begin{cases} \\tilde{\\boldsymbol{w}}^\\intercal \\mathbf{\\Sigma} \\tilde{\\boldsymbol{w}} & \\textrm{for a variance}\\,,\\\\ \\sqrt{\\tilde{\\boldsymbol{w}}^\\intercal \\mathbf{\\Sigma} \\tilde{\\boldsymbol{w}}} & \\textrm{for a standard deviation}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}``: The block `sigma`, with entries ``\\Sigma_{ij}``.
  - $(math_dict[:w_naive_sch])
  - ``\\tilde{\\rho}``: The risk that the function returns, the ``\\tilde{\\rho}(C)`` of [`SchurComplementHierarchicalRiskParity`](@ref) when `sigma` is the augmented block of a half ``C``.

# Arguments

  - `r`: Risk measure, which selects the case of the definition.
  - `sigma`: Covariance block, usually the result of [`schur_augmentation`](@ref).

# Returns

  - `risk::Number`: The risk ``\\tilde{\\rho}`` under `r`.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`Sd_Var`](@ref)
  - [`schur_augmentation`](@ref)
  - [`schur_complement_weights`](@ref)
"""
function naive_portfolio_risk(::Variance, sigma::MatNum)
    w = inv.(LinearAlgebra.diag(sigma))
    w ./= sum(w)
    return LinearAlgebra.dot(w, sigma, w)
end
function naive_portfolio_risk(::StandardDeviation, sigma::MatNum)
    w = inv.(LinearAlgebra.diag(sigma))
    w ./= sum(w)
    return sqrt(LinearAlgebra.dot(w, sigma, w))
end
"""
    schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt,
                             wb::WeightBounds, params::SchurComplementParams,
                             gamma::Option{<:Number} = nothing) -> Tuple

Run the Schur complement recursion at one value of ``\\gamma``.

This method takes a [`NonMonotonicSchurComplement`](@ref) bundle. The [`MonotonicSchurComplement`](@ref) method searches over ``\\gamma`` and calls this method at each value it tries. [`SchurComplementHierarchicalRiskParity`](@ref) states the mathematics.

# Algorithm

 1. Resolve the measure `r` of the bundle against `pr` with [`factory`](@ref), and copy its covariance matrix into `sigma`.
 2. Take `gamma` from the argument, or from `params.gamma` when the argument is `nothing`.
 3. Set every entry of the weights `w` to one.
 4. Split each entry of `items` of more than one leaf into its halves, giving the new `items`. Stop when no entry has more than one leaf.
 5. For each pair of halves `lc` and `rc`, read the blocks `A` and `C` of `sigma`. When `lc` holds more than one leaf, augment both blocks with [`schur_augmentation`](@ref), giving `A_aug` and `C_aug`, and write them back into `sigma`. A later split then reads the augmented blocks. Otherwise `A_aug` and `C_aug` are `A` and `C`.
 6. When `params.flag` is `true`, repair `A_aug` and `C_aug` with `params.pdm`. The repaired copies give the risks of step 7, and `sigma` keeps the blocks of step 5. When `params.flag` is `false` and either block is not positive definite, return `nothing` as the weights.
 7. Compute the risks `lrisk` and `rrisk` of the two blocks with [`naive_portfolio_risk`](@ref), and the split factor `alpha` from them.
 8. Clamp `alpha` to `wb` with [`split_factor_weight_constraints`](@ref). Multiply the weights of `lc` by `alpha`, and the weights of `rc` by `1 - alpha`.
 9. Go back to step 4.

# Arguments

  - `pr`: Prior result. Its covariance matrix is the default of the measure, and its `X` gives the number of assets and the number type of the weights.
  - `items`: The leaf orders to split. The recursion starts from one entry, the whole leaf order of the dendrogram.
  - `wb`: Resolved weight bounds, which clamp each split factor.
  - `params`: The bundle. The method reads `r`, `gamma`, `pdm` and `flag`.
  - `gamma`: A value that overrides `params.gamma`, or `nothing` to use the field. The monotonic search passes each value it tries in this argument.

# Validation

  - With `params.flag` true, a repair that throws, for example on a negative diagonal entry, is rethrown as an `ArgumentError` that names `gamma`. A repair that only warns lets the recursion continue.

# Returns

  - `(w, gamma, r)::Tuple`: The weights, the value of ``\\gamma`` the recursion ran at, and the measure that [`factory`](@ref) resolved. The weights sum to one. `w` is `nothing` when `params.flag` is `false` and an augmented block is not positive definite. [`assert_schur_weights`](@ref) turns that `nothing` into an error for a caller that keeps the weights.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`schur_augmentation`](@ref)
  - [`naive_portfolio_risk`](@ref)
  - [`split_factor_weight_constraints`](@ref)
  - [`assert_schur_weights`](@ref)
"""
function schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt,
                                  wb::WeightBounds,
                                  params::SchurComplementParams{<:Any, <:Any, <:Any,
                                                                <:NonMonotonicSchurComplement,
                                                                <:Any},
                                  gamma::Option{<:Number} = nothing)
    r = factory(params.r, pr)
    sigma = ismutable(r.sigma) ? copy(r.sigma) : Matrix(r.sigma)
    gamma = isnothing(gamma) ? params.gamma : gamma
    X = pr.X
    # A split factor is a quotient of two risks, so an integer sample takes float weights.
    w = ones(float_if_integer(eltype(X)), size(X, 2))
    pdm = params.pdm
    flag = params.flag
    while length(items) > 0
        items = [i[j:k] for i in items
                 for (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i in 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            # Copies, not views: the write-backs into `sigma` below (and `posdef!`) must
            # not alias the blocks being augmented.
            A = sigma[lc, lc]
            C = sigma[rc, rc]
            if length(lc) <= 1
                A_aug = A
                C_aug = C
            else
                B = sigma[lc, rc]
                A_aug = schur_augmentation(A, B, C, gamma)
                C_aug = schur_augmentation(C, transpose(B), A, gamma)
                sigma[lc, lc] = A_aug
                sigma[rc, rc] = C_aug
            end
            if flag
                try
                    posdef!(pdm, A_aug)
                    posdef!(pdm, C_aug)
                catch e
                    throw(ArgumentError("Augmented matrix could not be made positive definite. Use `MonotonicSchurComplement()` or reduce gamma: $gamma. Original error: $(sprint(showerror, e))"))
                end
            else
                if !LinearAlgebra.isposdef(A_aug) || !LinearAlgebra.isposdef(C_aug)
                    # Three values, like every other return of this method, and the real
                    # `gamma` so a caller can name the value that failed. The monotonic
                    # search reads `[1]` and treats `nothing` as an infinite risk; a caller
                    # that destructures three names must not get a `BoundsError` instead.
                    return nothing, gamma, r
                end
            end
            lrisk = naive_portfolio_risk(r, A_aug)
            rrisk = naive_portfolio_risk(r, C_aug)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    # The resolved measure is threaded out beside `gamma`. It is computed here and was
    # discarded before, so the result had no way to name what it optimised.
    return w, gamma, r
end
"""
    schur_complement_binary_search(objective::Function, lgamma::Number, hgamma::Number,
                                   lrisk::Number, lw::Option{<:VecNum},
                                   tol::Number = 1e-4,
                                   iter::Option{<:Integer} = nothing,
                                   strict::Bool = false) -> Tuple

Bisect a bracket that holds the turning point of the portfolio variance in ``\\gamma``.

The scan of [`MonotonicSchurComplement`](@ref) gives a bracket in which the portfolio variance stops falling. This method halves the bracket until it is at most `tol` wide. The returned weights are always the weights of the returned value of ``\\gamma``, because a midpoint that the test rejects never replaces the weights of the incumbent.

# Algorithm

 1. When `iter` is `nothing`, derive the budget `ceil(Int, log2((hgamma - lgamma) / tol) * 4 + 10)`. A bracket that is already narrow can derive a budget of zero or less.
 2. Stop when the bracket is at most `tol` wide.
 3. Evaluate `objective` at the midpoint `mgamma`, giving the weights `mw` and the variance `risk`, and at `mgamma - tol`, giving `hrisk`.
 4. When `risk` is not above `lrisk` and not above `hrisk`, the variance still falls at the midpoint, so `mgamma`, `risk` and `mw` become the incumbent `lgamma`, `lrisk` and `lw`. Otherwise `mgamma` becomes `hgamma`.
 5. Go back to step 2, at most `iter` times.
 6. When the bracket is still wider than `tol`, report it through [`strict_diagnostic`](@ref). Return `lw` and `lgamma`.

When no midpoint passes the test of step 4, the answer is the first `lgamma`. The scan measured that the variance falls up to that value.

# Arguments

  - `objective`: Takes a value of ``\\gamma`` and returns `(w, risk)`. `risk` is the largest value of its type when the allocation fails.
  - `lgamma`: Lower end of the bracket, and the incumbent.
  - `hgamma`: Upper end of the bracket.
  - `lrisk`: The variance at `lgamma`, which a midpoint must not exceed.
  - `lw`: The weights at `lgamma`.
  - `tol`: Width at which the bracket is narrow enough, and the step of the one-sided slope test.
  - `iter`: Budget of bisections. `nothing` derives one from the bracket and `tol`.
  - `strict`: Whether a bracket that stays wider than `tol` throws instead of a warning.

# Validation

  - With `strict = true`, a bracket that stays wider than `tol` after `iter` bisections throws an `ArgumentError`. With `strict = false` it logs a warning. The derived budget always narrows the bracket, because each bisection halves it.

# Returns

  - `(w, gamma)::Tuple`: The lower end of the last bracket, and the weights at it.

# Related

  - [`MonotonicSchurComplement`](@ref)
  - [`schur_complement_weights`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function schur_complement_binary_search(objective::Function, lgamma::Number, hgamma::Number,
                                        lrisk::Number, lw::Option{<:VecNum},
                                        tol::Number = 1e-4,
                                        iter::Option{<:Integer} = nothing,
                                        strict::Bool = false)
    if isnothing(iter)
        iter = ceil(Int, log2((hgamma - lgamma) / tol) * 4 + 10)
    end
    for _ in 1:iter
        # A bracket that is already narrow enough ends the search before a bisection. A
        # bracket narrower than about `0.18 * tol` derives a budget of zero or less, and
        # it must not be reported as unconverged.
        if hgamma - lgamma <= tol
            break
        end
        mgamma = (lgamma + hgamma) / 2
        mw, risk = objective(mgamma)
        hrisk = objective(mgamma - tol)[2]
        if risk <= lrisk && risk <= hrisk
            # The variance still falls at the midpoint: it becomes the incumbent.
            lgamma, lrisk, lw = mgamma, risk, mw
        else
            # The turning point lies below the midpoint. The incumbent keeps its weights.
            hgamma = mgamma
        end
    end
    if hgamma - lgamma > tol
        msg = "Binary search did not converge within the specified tolerance: tol => $tol"
        strict_diagnostic(msg, strict)
    end
    return lw, lgamma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Search ``[0, \\gamma]`` for the turning point of the portfolio variance, then allocate at it.

[`MonotonicSchurComplement`](@ref) states the definition of the value that the search finds and the steps of this method. Every evaluation calls the [`NonMonotonicSchurComplement`](@ref) method with the repair off, so a value that fails scores the largest value of the weight type. The returned weights are always the weights at the returned value of ``\\gamma``.

# Related

  - [`schur_complement_weights`](@ref)
  - [`MonotonicSchurComplement`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt,
                                  wb::WeightBounds,
                                  params::SchurComplementParams{<:Any, <:Any, <:Any,
                                                                <:MonotonicSchurComplement,
                                                                <:Any})
    max_gamma = params.gamma
    r = factory(params.r, pr)
    # The type of the weights, so a failed allocation scores the largest value of that type.
    T = float_if_integer(eltype(pr.X))
    if iszero(max_gamma)
        nm_params = SchurComplementParams(; r = r, gamma = max_gamma, pdm = params.pdm,
                                          alg = NonMonotonicSchurComplement(),
                                          flag = params.flag)
        wi, gi, _ = schur_complement_weights(pr, items, wb, nm_params)
        return wi, gi, r
    end
    nm_params = SchurComplementParams(; r = r, gamma = max_gamma, pdm = params.pdm,
                                      alg = NonMonotonicSchurComplement(), flag = false)
    # `wx`, not `w`: a closure that assigns a name the enclosing function also assigns
    # rebinds the enclosing variable, so each call would overwrite the scan's weights.
    function objective(x::Number)
        wx = schur_complement_weights(pr, items, wb, nm_params, x)[1]
        risk = isnothing(wx) ? typemax(T) : LinearAlgebra.dot(wx, r.sigma, wx)
        return wx, risk
    end
    (; tol, iter, strict) = params.alg
    gammas = range(zero(max_gamma), max_gamma; length = params.alg.N)
    # The two previous scan points, `i - 2` and `i - 1`, with their weights.
    w2, risk2 = w1, risk1 = objective(gammas[1])
    for i in 2:length(gammas)
        w, risk = objective(gammas[i])
        if risk >= risk1
            # The turning point lies in [gammas[i - 2], gammas[i]], or in
            # [gammas[1], gammas[2]] when the variance rises at the first step.
            lidx, lw, lrisk = i == 2 ? (1, w1, risk1) : (i - 2, w2, risk2)
            wi, gi = schur_complement_binary_search(objective, gammas[lidx], gammas[i],
                                                    lrisk, lw, tol, iter, strict)
            return wi, gi, r
        end
        w2, risk2 = w1, risk1
        w1, risk1 = w, risk
    end
    # No turning point in the scan: check the derivative at the last gamma.
    if risk1 <= objective(max_gamma - tol)[2]
        return w1, max_gamma, r
    end
    # The turning point lies between the last two gammas.
    wi, gi = schur_complement_binary_search(objective, gammas[end - 1], gammas[end], risk2,
                                            w2, tol, iter, strict)
    return wi, gi, r
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that [`schur_complement_weights`](@ref) produced a weight vector.

A [`SchurComplementParams`](@ref) with `flag = false` does not repair an augmented block that is not positive definite. The recursion then abandons the allocation and returns `nothing`, which the monotonic search reads as a failure. A caller that keeps the weights gets the reason as an error instead.

# Arguments

  - `w`: Weight vector returned by [`schur_complement_weights`](@ref), or `nothing`.
  - `gamma`: The value of `gamma` the allocation ran with.

# Validation

  - `!isnothing(w)`. Otherwise it throws an `ArgumentError` that names `gamma` and the three remedies.

# Returns

  - `nothing`.

# Related

  - [`schur_complement_weights`](@ref)
  - [`SchurComplementParams`](@ref)
"""
function assert_schur_weights(w::Option{<:VecNum}, gamma::Number)::Nothing
    @argcheck(!isnothing(w),
              ArgumentError("Augmented matrix is not positive definite at gamma = $gamma, and `flag = false` disables the positive-definite repair. Set `flag = true`, use `MonotonicSchurComplement()`, or reduce gamma."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a Schur complement allocation with one bundle.

[`optimise`](@ref) calls this method. [`SchurComplementHierarchicalRiskParity`](@ref) states its steps under `# Algorithm`. The weights, their bounds and the fee take the type of a quotient of two returns, so an integer returns matrix allocates in floating point and a `Float32` matrix stays in `Float32`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any},
                   rd::ReturnsResult = ReturnsResult(); kwargs...)
    sh = reset_time_dependent_estimator(sh)
    rd = returns_result_picker(rd, sh.opt.brt)
    pr = prior(sh.opt.pe, rd)
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`,
    # as `HierarchicalRiskParity` does. The allocation reads no fee: its measure is a
    # variance or a standard deviation, which a fee does not move. The fee rides on the
    # result, where the net returns and a fold's forced exit read it.
    imsk = investable_mask(pr)
    # A split factor is a quotient of two risks, so the weights, their bounds and the fee
    # take the type of the returns, widened to a float only when it is an integer. An
    # integer sample then allocates in floating point, and a `Float32` sample stays in
    # `Float32`.
    T = float_if_integer(eltype(pr.X))
    fees = investable_fees_view(fees_constraints(sh.opt.fees, sh.opt.sets;
                                                 strict = sh.opt.strict, datatype = T),
                                imsk, pr.X)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here: the
    # augmented matrix the Schur complement builds is finite, and the leaf permutation
    # indexes the investable universe. The weights are expanded back in
    # `SchurComplementHierarchicalRiskParityResult`.
    _, pr, sh, rd = investable_reduction(imsk, pr, sh, rd)
    X = pr.X
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(sh.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa,
                     x_src = sh.opt.x_src)
    assert_clustering_universe(clr, size(X, 2))
    items = [clr.res.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(X, 2),
                                   strict = sh.opt.strict, datatype = T)
    w, gamma, r = schur_complement_weights(pr, items, wb, sh.params)
    assert_schur_weights(w, gamma)
    retcode, w = finalise_weight_bounds(sh.opt.wf, wb, w)
    return SchurComplementHierarchicalRiskParityResult(; pr = pr, wb = wb, clr = clr,
                                                       fees = fees, r = r, gamma = gamma,
                                                       retcode = retcode, w = w,
                                                       imsk = imsk, fb = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a Schur complement allocation with a vector of bundles, and blend their portfolios.

[`optimise`](@ref) calls this method. It runs the steps of the single-bundle method, with step 7 of the `# Algorithm` of [`SchurComplementHierarchicalRiskParity`](@ref) once per bundle. The result holds one measure and one value of ``\\gamma`` per bundle.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w} &= \\frac{\\sum_{k=1}^{K} s_k \\boldsymbol{w}_k}{\\sum_{k=1}^{K} s_k}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: Blended weights, before the weight finaliser.
  - ``K``: Number of bundles.
  - ``s_k``: Scale of the measure of bundle ``k``, its `r.settings.scale`.
  - ``\\boldsymbol{w}_k``: Weights of the allocation of bundle ``k``, which sum to one.

The denominator is the sum of the blended weights, because each ``\\boldsymbol{w}_k`` sums to one.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:AbstractVector},
                   rd::ReturnsResult = ReturnsResult(); kwargs...)
    sh = reset_time_dependent_estimator(sh)
    rd = returns_result_picker(rd, sh.opt.brt)
    pr = prior(sh.opt.pe, rd)
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`,
    # as `HierarchicalRiskParity` does. The allocation reads no fee: its measure is a
    # variance or a standard deviation, which a fee does not move. The fee rides on the
    # result, where the net returns and a fold's forced exit read it.
    imsk = investable_mask(pr)
    # A split factor is a quotient of two risks, so the weights, their bounds and the fee
    # take the type of the returns, widened to a float only when it is an integer. An
    # integer sample then allocates in floating point, and a `Float32` sample stays in
    # `Float32`.
    T = float_if_integer(eltype(pr.X))
    fees = investable_fees_view(fees_constraints(sh.opt.fees, sh.opt.sets;
                                                 strict = sh.opt.strict, datatype = T),
                                imsk, pr.X)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here: the
    # augmented matrix the Schur complement builds is finite, and the leaf permutation
    # indexes the investable universe. The weights are expanded back in
    # `SchurComplementHierarchicalRiskParityResult`.
    _, pr, sh, rd = investable_reduction(imsk, pr, sh, rd)
    X = pr.X
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(sh.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa,
                     x_src = sh.opt.x_src)
    assert_clustering_universe(clr, size(X, 2))
    items = [clr.res.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(X, 2),
                                   strict = sh.opt.strict, datatype = T)
    params = sh.params
    # Each bundle returns the `gamma` it ran at, of that bundle's own type. The vector
    # literal below promotes them to one type.
    gammas = Vector{Any}(undef, length(params))
    rs = Vector{Any}(undef, length(params))
    w = zeros(T, size(X, 2))
    for (i, ps) in enumerate(params)
        wi, gamma, ri = schur_complement_weights(pr, items, wb, ps)
        assert_schur_weights(wi, gamma)
        w .+= ps.r.settings.scale * wi
        gammas[i] = gamma
        rs[i] = ri
    end
    retcode, w = finalise_weight_bounds(sh.opt.wf, wb, w / sum(w))
    return SchurComplementHierarchicalRiskParityResult(; pr = pr, wb = wb, clr = clr,
                                                       fees = fees, r = [rs...],
                                                       gamma = [gammas...],
                                                       retcode = retcode, w = w,
                                                       imsk = imsk, fb = nothing)
end
"""
    optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any, Nothing},
             rd::ReturnsResult; kwargs...) -> SchurComplementHierarchicalRiskParityResult

Run the Schur complement hierarchical risk parity optimisation.

Unlike [`HierarchicalEqualRiskContribution`](@ref) and [`NestedClustered`](@ref), this optimiser takes no `branchorder` keyword. Recursive bisection splits the leaf order of the dendrogram, so that order is the input of the algorithm, and the clustering always runs with the optimal order. `kwargs` absorbs a `branchorder` and the fit ignores it.

# Arguments

  - `sh`: The Schur complement hierarchical risk parity optimiser.
  - $(arg_dict[:rd]) When `sh.opt.pe` is a prior result, the fit itself does not read `rd`, but the clustering and a fallback can read it.
  - `kwargs`: Absorbed and ignored.

# Validation

  - No field in the tree of `sh` holds an [`Online`](@ref). Otherwise [`assert_batch_entry`](@ref) throws an `ArgumentError` that names the field. A plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the online arm of the fold loop.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementHierarchicalRiskParityResult`](@ref)
"""
function optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any, Nothing},
                  rd::ReturnsResult; kwargs...)
    assert_batch_entry(sh, "`optimise`")
    return _optimise(sh, rd; kwargs...)
end

export SchurComplementHierarchicalRiskParityResult, SchurComplementParams,
       SchurComplementHierarchicalRiskParity, NonMonotonicSchurComplement,
       MonotonicSchurComplement
public SchurComplementAlgorithm, schur_complement_weights
