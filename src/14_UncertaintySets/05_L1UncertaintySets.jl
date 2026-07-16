"""
$(DocStringExtensions.TYPEDEF)

Radius algorithm that calibrates the ``\\ell_1`` uncertainty radius to a target number of active assets.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ActiveAssetsUncertaintyAlgorithm(;
        active::Union{<:Integer, <:AbstractFloat} = 0.2
    ) -> ActiveAssetsUncertaintyAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - If `active` is an `Integer`: `active >= 1`.
  - If `active` is an `AbstractFloat`: `0 < active < 1`.

# Details

The radius ``\\epsilon`` of an ``\\ell_1`` uncertainty set has no natural scale — it is a sum of characteristic differences, so on daily returns it is of order ``10^{-3}``, and on annualised returns roughly ``250`` times larger. This algorithm converts the quantity a caller *can* reason about (how many assets should the portfolio hold?) into the radius that produces it, by inverting the closed forms of [quintile](@cite).

The map is exact, but **only for the bare problem** the closed forms assume: the budget and sign constraints, and nothing else. It is a *radius calibration*, not a cardinality constraint. Adding weight bounds, cardinality, or linear constraints may change the realised number of active assets, and this algorithm neither knows nor checks. For a hard bound on the number of holdings use the `card` field of [`JuMPOptimiser`](@ref).

# Related

  - [`AbstractUncertaintyEpsAlgorithm`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
@concrete struct ActiveAssetsUncertaintyAlgorithm <: AbstractUncertaintyEpsAlgorithm
    """
    $(field_dict[:active_ucs])
    """
    active
    function ActiveAssetsUncertaintyAlgorithm(active::Union{<:Integer, <:AbstractFloat})
        if isa(active, Integer)
            @argcheck(active >= one(active),
                      DomainError(active, "active must be >= 1 when given as a count"))
        else
            @argcheck(zero(active) < active < one(active),
                      DomainError(active,
                                  "active must be in (0, 1) when given as a fraction"))
        end
        return new{typeof(active)}(active)
    end
end
function ActiveAssetsUncertaintyAlgorithm(;
                                          active::Union{<:Integer, <:AbstractFloat} = 0.2)::ActiveAssetsUncertaintyAlgorithm
    return ActiveAssetsUncertaintyAlgorithm(active)
end
"""
$(DocStringExtensions.TYPEDEF)

``\\ell_1`` (cross-polytope) uncertainty set on the characteristic vector.

The set is ``\\mathcal{S} = \\{\\hat{\\boldsymbol{\\mu}} + \\boldsymbol{e} : \\lVert \\boldsymbol{e} \\oslash \\boldsymbol{\\sigma} \\rVert_1 \\leq \\epsilon\\}``, with a single error budget shared across every asset and both signs. Produced by [`CharacteristicUncertaintySet`](@ref) and consumed by [`ArithmeticReturn`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L1UncertaintySet(;
        eps::Number,
        sd::Option{<:VecNum} = nothing
    ) -> L1UncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(eps)` and `eps >= 0`.
  - If `sd` is provided: `!isempty(sd)` and `all(sd .> 0)`.

# Mathematical definition

The worst case of a linear characteristic over the set collapses to a scaled infinity norm (Lemmas 1 and 8 of [quintile](@cite)):

```math
\\underset{\\boldsymbol{\\mu} \\in \\mathcal{S}}{\\min}\\, \\boldsymbol{\\mu}^{\\intercal} \\boldsymbol{w} = \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w} - \\epsilon \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_{\\infty}\\,.
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated characteristic vector.
  - ``\\epsilon``: Radius of the set.
  - ``\\boldsymbol{\\sigma}``: Per-asset scaling (`sd`); ``\\boldsymbol{1}`` when `sd` is `nothing`.
  - ``\\odot``, ``\\oslash``: Element-wise product and division.

Because the right-hand side is concave and positively homogeneous, this is an LP once the infinity norm is epigraphed — no conic solver is needed.

# Notes

This set bounds a *mean/characteristic* vector. It has no covariance analogue, so [`sigma_ucs`](@ref) is not defined for the estimator that produces it.

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
@concrete struct L1UncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:eps_ucs])
    """
    eps
    """
    $(field_dict[:sd_ucs])
    """
    sd
    function L1UncertaintySet(eps::Number, sd::Option{<:VecNum})
        @argcheck(isfinite(eps) && eps >= zero(eps),
                  DomainError(eps, "eps must be finite and >= 0"))
        if isa(sd, VecNum)
            @argcheck(!isempty(sd), IsEmptyError("sd cannot be empty"))
            @argcheck(all(x -> x > zero(x), sd),
                      DomainError(sd, "all entries of sd must be > 0"))
        end
        return new{typeof(eps), typeof(sd)}(eps, sd)
    end
end
function L1UncertaintySet(; eps::Number, sd::Option{<:VecNum} = nothing)::L1UncertaintySet
    return L1UncertaintySet(eps, sd)
end
"""
$(DocStringExtensions.TYPEDEF)

Signed ``\\ell_1`` uncertainty set on the characteristic vector, with a separate error budget per sign.

The set is ``\\mathcal{A}_2 = \\{\\hat{\\boldsymbol{\\mu}} + \\boldsymbol{e} : \\boldsymbol{1}^{\\intercal} [\\boldsymbol{e} \\oslash \\boldsymbol{\\sigma}]_{+} \\leq \\epsilon_{+},\\, -\\boldsymbol{1}^{\\intercal} [\\boldsymbol{e} \\oslash \\boldsymbol{\\sigma}]_{-} \\leq \\epsilon_{-}\\}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignedL1UncertaintySet(;
        ep::Number,
        em::Number,
        sd::Option{<:VecNum} = nothing
    ) -> SignedL1UncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(ep)` and `ep >= 0`.
  - `isfinite(em)` and `em >= 0`.
  - If `sd` is provided: `!isempty(sd)` and `all(sd .> 0)`.

# Mathematical definition

```math
\\underset{\\boldsymbol{\\mu} \\in \\mathcal{A}_2}{\\min}\\, \\boldsymbol{\\mu}^{\\intercal} \\boldsymbol{w} = \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w} - \\epsilon_{+} \\left[\\underset{i}{\\max}\\, (-\\sigma_i w_i)\\right]_{+} - \\epsilon_{-} \\left[\\underset{i}{\\max}\\, (\\sigma_i w_i)\\right]_{+}\\,.
```

Where:

  - ``\\epsilon_{+}``, ``\\epsilon_{-}``: Radii of the positive- and negative-error sides (`ep`, `em`).
  - ``[\\cdot]_{+}``, ``[\\cdot]_{-}``: Element-wise positive and negative parts.

Still concave and LP-representable, with one epigraph variable per sign.

# Notes

This is **not** [`L1UncertaintySet`](@ref) with `ep == em`: the joint set shares one budget across both signs, giving ``\\max(t_{+}, t_{-})``, whereas this one spends a budget per sign, giving ``\\epsilon_{+} t_{+} + \\epsilon_{-} t_{-}``. The two agree only when ``\\boldsymbol{w}`` is single-signed — as it is under a long-only budget, where the joint set is the simpler choice.

[quintile](@cite) introduces this set in order to *decouple* the long-short problem into two independent problems (its equations 27 and 28), which its Remark 12 then recombines only when the two legs happen to have complementary support. Modelling the worst case above directly keeps the problem coupled, so that caveat does not arise.

# Related

  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
@concrete struct SignedL1UncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:ep_ucs])
    """
    ep
    """
    $(field_dict[:em_ucs])
    """
    em
    """
    $(field_dict[:sd_ucs])
    """
    sd
    function SignedL1UncertaintySet(ep::Number, em::Number, sd::Option{<:VecNum})
        @argcheck(isfinite(ep) && ep >= zero(ep),
                  DomainError(ep, "ep must be finite and >= 0"))
        @argcheck(isfinite(em) && em >= zero(em),
                  DomainError(em, "em must be finite and >= 0"))
        if isa(sd, VecNum)
            @argcheck(!isempty(sd), IsEmptyError("sd cannot be empty"))
            @argcheck(all(x -> x > zero(x), sd),
                      DomainError(sd, "all entries of sd must be > 0"))
        end
        return new{typeof(ep), typeof(em), typeof(sd)}(ep, em, sd)
    end
end
function SignedL1UncertaintySet(; ep::Number, em::Number,
                                sd::Option{<:VecNum} = nothing)::SignedL1UncertaintySet
    return SignedL1UncertaintySet(ep, em, sd)
end
"""
$(DocStringExtensions.TYPEDEF)

Shape algorithm selecting a joint ``\\ell_1`` uncertainty set.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L1UncertaintySetAlgorithm(;
        method::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
        scaled::Bool = false,
        paired::Bool = false
    ) -> L1UncertaintySetAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - If `method` is a `Number`: `isfinite(method)` and `method >= 0`.

# Details

`scaled` selects between the two uncertainty sets of [quintile](@cite): `false` gives ``\\mathcal{S}`` (its equation 5), which assumes every characteristic suffers the same estimation error and yields equally-weighted active assets; `true` gives ``\\mathcal{A}_1`` (its equation 18), which assumes assets with larger volatility suffer larger estimation error and yields inverse-volatility weights.

`paired` selects which closed form calibrates the radius when `method` is an [`AbstractUncertaintyEpsAlgorithm`](@ref), and is **inert when `method` is a number**. The number of assets a radius activates depends on the sign structure of the problem it is used in, which an uncertainty set cannot observe, so the caller must say:

  - `false`: the long-only ladder, for a problem with `w >= 0` and `bgt = 1` (Corollaries 4 and 11 of [quintile](@cite)).
  - `true`: the paired ladder, for a dollar-neutral problem with `bgt = 0` and `sbgt = 1/2`, where assets activate in long/short pairs (Corollary 7).

Using the wrong one mis-calibrates the radius; it does not make the optimisation incorrect.

# Related

  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
"""
@concrete struct L1UncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    """
    $(field_dict[:method_l1_ucs])
    """
    method
    """
    $(field_dict[:scaled_ucs])
    """
    scaled
    """
    `paired`: Whether to calibrate the radius against the paired (dollar-neutral) ladder rather than the long-only one. Inert when `method` is a number.
    """
    paired
    function L1UncertaintySetAlgorithm(method::Num_UcSEps, scaled::Bool, paired::Bool)
        if isa(method, Number)
            @argcheck(isfinite(method) && method >= zero(method),
                      DomainError(method, "method must be finite and >= 0"))
        end
        return new{typeof(method), typeof(scaled), typeof(paired)}(method, scaled, paired)
    end
end
function L1UncertaintySetAlgorithm(;
                                   method::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
                                   scaled::Bool = false,
                                   paired::Bool = false)::L1UncertaintySetAlgorithm
    return L1UncertaintySetAlgorithm(method, scaled, paired)
end
"""
$(DocStringExtensions.TYPEDEF)

Shape algorithm selecting a signed ``\\ell_1`` uncertainty set.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignedL1UncertaintySetAlgorithm(;
        mp::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
        mm::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
        scaled::Bool = true
    ) -> SignedL1UncertaintySetAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - If `mp`/`mm` is a `Number`: `isfinite` and `>= 0`.

# Details

`mp` yields ``\\epsilon_{+}`` and `mm` yields ``\\epsilon_{-}``. When calibrated by an [`AbstractUncertaintyEpsAlgorithm`](@ref) the ladders are fixed by the paper's construction and need no `paired` flag: ``\\epsilon_{-}`` governs the long leg and is calibrated against the top of the ranking, ``\\epsilon_{+}`` governs the short leg and is calibrated against the bottom (Corollary 13 of [quintile](@cite)).

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
@concrete struct SignedL1UncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    """
    $(field_dict[:mp_ucs])
    """
    mp
    """
    $(field_dict[:mm_ucs])
    """
    mm
    """
    $(field_dict[:scaled_ucs])
    """
    scaled
    function SignedL1UncertaintySetAlgorithm(mp::Num_UcSEps, mm::Num_UcSEps, scaled::Bool)
        if isa(mp, Number)
            @argcheck(isfinite(mp) && mp >= zero(mp),
                      DomainError(mp, "mp must be finite and >= 0"))
        end
        if isa(mm, Number)
            @argcheck(isfinite(mm) && mm >= zero(mm),
                      DomainError(mm, "mm must be finite and >= 0"))
        end
        return new{typeof(mp), typeof(mm), typeof(scaled)}(mp, mm, scaled)
    end
end
function SignedL1UncertaintySetAlgorithm(;
                                         mp::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
                                         mm::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
                                         scaled::Bool = true)::SignedL1UncertaintySetAlgorithm
    return SignedL1UncertaintySetAlgorithm(mp, mm, scaled)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator for ``\\ell_1`` uncertainty sets on the characteristic vector.

Fits the robust best-characteristic uncertainty sets of [quintile](@cite). Maximising the worst-case characteristic over one of these sets recovers the heuristic ``1/N``, quintile, and inverse-volatility portfolios as *exact solutions of a robust optimisation problem*, with the radius controlling how many assets are held — which is why this library ships no dedicated quintile optimiser. Compose the set with [`MeanRisk`](@ref) instead:

```julia
MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
         opt = JuMPOptimiser(; slv = slv, bgt = 1.0,
                             ret = ArithmeticReturn(; ucs = CharacteristicUncertaintySet())))
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CharacteristicUncertaintySet(;
        pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
        alg::AbstractUncertaintySetAlgorithm = L1UncertaintySetAlgorithm()
    ) -> CharacteristicUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `alg` must be an [`L1UncertaintySetAlgorithm`](@ref) or a [`SignedL1UncertaintySetAlgorithm`](@ref).

# Details

`pe` supplies both the characteristic vector ``\\hat{\\boldsymbol{\\mu}}`` and, when the shape algorithm is `scaled`, the per-asset scaling ``\\hat{\\boldsymbol{\\sigma}} = \\sqrt{\\mathrm{diag}(\\hat{\\mathbf{\\Sigma}})}``.

The characteristic need not be an expected return. [quintile](@cite) notes that any characteristic works, and the library already carries the machinery: a prior built on [`StandardDeviationExpectedReturns`](@ref) ranks on volatility (the Low Volatility factor), reproducing Table III of the paper.

# Notes

This estimator is **mean-only**. [`ucs`](@ref) and [`sigma_ucs`](@ref) are defined solely to throw an informative error: the ``\\ell_1`` ball bounds a characteristic vector, and the paper defines no covariance analogue.

# Related

  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`mu_ucs`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
"""
@concrete struct CharacteristicUncertaintySet <: AbstractUncertaintySetEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    `alg`: Shape algorithm — [`L1UncertaintySetAlgorithm`](@ref) or [`SignedL1UncertaintySetAlgorithm`](@ref).
    """
    alg
    function CharacteristicUncertaintySet(pe::AbstractLowOrderPriorEstimator,
                                          alg::AbstractUncertaintySetAlgorithm)
        @argcheck(isa(alg,
                      Union{<:L1UncertaintySetAlgorithm, <:SignedL1UncertaintySetAlgorithm}),
                  ArgumentError("alg must be an L1UncertaintySetAlgorithm or a SignedL1UncertaintySetAlgorithm, got $(typeof(alg))"))
        return new{typeof(pe), typeof(alg)}(pe, alg)
    end
end
function CharacteristicUncertaintySet(;
                                      pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
                                      alg::AbstractUncertaintySetAlgorithm = L1UncertaintySetAlgorithm())::CharacteristicUncertaintySet
    return CharacteristicUncertaintySet(pe, alg)
end
"""
    l1_activation_ladder(mu::VecNum, sd::Option{<:VecNum})

Return the vector `g` whose `k`-th entry is the radius at which the `k`-th asset becomes active in the long-only problem.

`mu` must be sorted in non-increasing order, with `sd` (when given) under the same permutation. Entry `k` is ``\\sum_{i=1}^{k} (\\hat{\\mu}_i - \\hat{\\mu}_k) / \\sigma_i``, the threshold of Lemma 2 (`sd === nothing`) or Lemma 9 (scaled) of [quintile](@cite). The sequence is non-decreasing, so a radius `eps` activates the largest `k` with `g[k] < eps`.

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function l1_activation_ladder(mu::VecNum, sd::Option{<:VecNum})
    N = length(mu)
    return [sum(i -> (mu[i] - mu[k]) / (isnothing(sd) ? one(eltype(mu)) : sd[i]), 1:k)
            for k in 1:N]
end
"""
    l1_active_count(active::Union{<:Integer, <:AbstractFloat}, N::Int)

Convert an `active` target — a count or a fraction of the universe — into an asset count in `1:N`.

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
"""
function l1_active_count(active::Union{<:Integer, <:AbstractFloat}, N::Int)
    q = isa(active, Integer) ? Int(active) : round(Int, active * N)
    return clamp(q, 1, N)
end
"""
    l1_eps_from_ladder(method, ladder::VecNum, N::Int)

Resolve a radius from a [`Num_UcSEps`](@ref) against an activation ladder.

A number passes through unchanged. An [`ActiveAssetsUncertaintyAlgorithm`](@ref) returns the midpoint of the open interval `(ladder[q], ladder[q+1])` that the closed forms of [quintile](@cite) require for exactly `q` active entries; at the top of the ladder it continues the final increment by a half step, since any radius above `ladder[end]` activates everything.

Throws when the target interval is empty, which happens when the characteristic has ties across the cut — the paper excludes this case by assumption, and it cannot be satisfied.

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`l1_activation_ladder`](@ref)
"""
function l1_eps_from_ladder(method::Number, args...)
    return method
end
function l1_eps_from_ladder(method::ActiveAssetsUncertaintyAlgorithm, ladder::VecNum,
                            q::Int)
    L = length(ladder)
    @argcheck(L >= 1, IsEmptyError("activation ladder cannot be empty"))
    q = clamp(q, 1, L)
    if q < L
        lo, hi = ladder[q], ladder[q + 1]
        @argcheck(hi > lo,
                  DomainError(method.active,
                              "cannot calibrate a radius for $(q) active entries: the characteristic has ties across the cut, so the interval ($(lo), $(hi)) required by the closed form is empty. Break the tie, or pass an explicit radius."))
        return (lo + hi) / 2
    end
    # Top of the ladder: any radius above ladder[end] activates everything.
    return L == 1 ? one(eltype(ladder)) : ladder[L] + (ladder[L] - ladder[L - 1]) / 2
end
"""
    l1_resolve_eps(method, mus::VecNum, sds::Option{<:VecNum}, paired::Bool)

Resolve a radius from a [`Num_UcSEps`](@ref) against the characteristic vector.

`mus` must be sorted in non-increasing order, with `sds` under the same permutation. A number passes through unchanged. An [`ActiveAssetsUncertaintyAlgorithm`](@ref) is calibrated against the long-only ladder when `paired` is `false`, and against the paired ladder — where the `i`-th best pairs with the `i`-th worst, so entries activate two assets at a time — when it is `true`.

# Related

  - [`l1_activation_ladder`](@ref)
  - [`l1_eps_from_ladder`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
"""
function l1_resolve_eps(method::Number, args...)
    return method
end
function l1_resolve_eps(method::ActiveAssetsUncertaintyAlgorithm, mus::VecNum,
                        sds::Option{<:VecNum}, paired::Bool)
    N = length(mus)
    if !paired
        return l1_eps_from_ladder(method, l1_activation_ladder(mus, sds),
                                  l1_active_count(method.active, N))
    end
    half = N ÷ 2
    @argcheck(half >= 1,
              ArgumentError("a paired radius calibration needs at least 2 assets, got $N"))
    ladder = (l1_activation_ladder(mus, sds) + l1_activation_ladder(reverse(-mus),
                                                                    if isnothing(sds)
                                                                        nothing
                                                                    else
                                                                        reverse(sds)
                                                                    end))[1:half]
    # `active` counts assets, but each paired ladder entry activates a long/short pair.
    q = if isa(method.active, Integer)
        max(Int(method.active) ÷ 2, 1)
    else
        clamp(round(Int, method.active * N / 2), 1, half)
    end
    return l1_eps_from_ladder(method, ladder, q)
end
"""
    mu_ucs(ue::CharacteristicUncertaintySet, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Construct an ``\\ell_1`` uncertainty set on the characteristic vector.

Computes the prior, takes ``\\hat{\\boldsymbol{\\mu}}`` from it (and ``\\hat{\\boldsymbol{\\sigma}} = \\sqrt{\\mathrm{diag}(\\hat{\\mathbf{\\Sigma}})}`` when the shape algorithm is `scaled`), then resolves the radius from the shape algorithm.

# Arguments

  - `ue`: Characteristic uncertainty set estimator.
  - `X`: Data matrix (e.g. returns).
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::Union{<:L1UncertaintySet, <:SignedL1UncertaintySet}`: The uncertainty set.

# Related

  - [`CharacteristicUncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
"""
function mu_ucs(ue::CharacteristicUncertaintySet{<:Any, <:L1UncertaintySetAlgorithm},
                X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    alg = ue.alg
    sd = alg.scaled ? sqrt.(LinearAlgebra.diag(pr.sigma)) : nothing
    idx = sortperm(pr.mu; rev = true)
    eps = l1_resolve_eps(alg.method, pr.mu[idx], isnothing(sd) ? nothing : sd[idx],
                         alg.paired)
    return L1UncertaintySet(; eps = eps, sd = sd)
end
function mu_ucs(ue::CharacteristicUncertaintySet{<:Any, <:SignedL1UncertaintySetAlgorithm},
                X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    alg = ue.alg
    sd = alg.scaled ? sqrt.(LinearAlgebra.diag(pr.sigma)) : nothing
    idx = sortperm(pr.mu; rev = true)
    mus = pr.mu[idx]
    sds = isnothing(sd) ? nothing : sd[idx]
    # em governs the long leg, calibrated against the top of the ranking; ep governs the
    # short leg, calibrated against the bottom — i.e. the top of the reversed, negated
    # ranking, which is the same ladder read from the other end (Corollary 13).
    em = l1_resolve_eps(alg.mm, mus, sds, false)
    ep = l1_resolve_eps(alg.mp, reverse(-mus), isnothing(sds) ? nothing : reverse(sds),
                        false)
    return SignedL1UncertaintySet(; ep = ep, em = em, sd = sd)
end
"""
    ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)
    sigma_ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)

Always throw. [`CharacteristicUncertaintySet`](@ref) is mean-only.

The ``\\ell_1`` set bounds a characteristic (mean) vector; [quintile](@cite) defines no covariance analogue, so there is nothing for these to return. Use [`NormalUncertaintySet`](@ref), [`DeltaUncertaintySet`](@ref), or [`ARCHUncertaintySet`](@ref) for a covariance uncertainty set.

These take the same `(X, F)` signature as the rest of the family rather than a catch-all, so that the [`ReturnsResult`](@ref) forwarders in the base reach them without ambiguity.

# Related

  - [`CharacteristicUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
"""
function ucs(::CharacteristicUncertaintySet, ::MatNum, ::Option{<:MatNum} = nothing;
             kwargs...)
    return throw(ArgumentError("CharacteristicUncertaintySet is mean-only: it bounds a characteristic vector and has no covariance analogue, so it cannot produce the (mu, sigma) pair `ucs` returns. Use `mu_ucs` for the mean set, and NormalUncertaintySet, DeltaUncertaintySet or ARCHUncertaintySet for a covariance set."))
end
function sigma_ucs(::CharacteristicUncertaintySet, ::MatNum, ::Option{<:MatNum} = nothing;
                   kwargs...)
    return throw(ArgumentError("CharacteristicUncertaintySet is mean-only: the l1 set bounds a characteristic vector, and no covariance analogue is defined for it. Use NormalUncertaintySet, DeltaUncertaintySet or ARCHUncertaintySet for a covariance uncertainty set."))
end

export ActiveAssetsUncertaintyAlgorithm, L1UncertaintySet, SignedL1UncertaintySet,
       L1UncertaintySetAlgorithm, SignedL1UncertaintySetAlgorithm,
       CharacteristicUncertaintySet
