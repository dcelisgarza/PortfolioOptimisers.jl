"""
$(DocStringExtensions.TYPEDEF)

Calibrates the ``\\ell_1`` uncertainty radius to a target number of active assets.

The radius ``\\epsilon`` of an ``\\ell_1`` uncertainty set has no natural scale — it is a sum of characteristic differences, so on daily returns it is of order ``10^{-3}``, and on annualised returns roughly ``250`` times larger. This algorithm converts the quantity a caller *can* reason about (how many assets should the portfolio hold?) into the radius that produces it, by inverting the closed forms of [quintile](@cite). The inversion is exact, and **only for the bare problem** those closed forms assume: the budget and sign constraints, and nothing else.

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

The type of `active` decides which rule applies, so the boundary is sharp: `active = 1` is a count of one asset and is accepted, while `active = 1.0` is the whole universe as a fraction and is rejected.

# Examples

```jldoctest
julia> ActiveAssetsUncertaintyAlgorithm()
ActiveAssetsUncertaintyAlgorithm
  active ┴ Float64: 0.2
```

# Related

  - [`AbstractUncertaintyEpsAlgorithm`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`l1_resolve_eps`](@ref): the function that reads this algorithm and returns the radius.
  - [`JuMPOptimiser`](@ref): its `card` field bounds the number of holdings outright. Adding weight bounds, cardinality or linear constraints may move the realised count away from the one calibrated here, and this algorithm neither knows nor checks.

# References

  - $(ref_dict[:quintile])
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

The set is ``\\mathcal{S} = \\{\\hat{\\boldsymbol{\\mu}} + \\boldsymbol{e} : \\lVert \\boldsymbol{e} \\oslash \\boldsymbol{\\sigma} \\rVert_1 \\leq \\epsilon\\}``, with a single error budget shared across every asset and both signs. Produced by [`CharacteristicUncertaintySet`](@ref) and consumed by [`ArithmeticReturn`](@ref). It bounds a *mean/characteristic* vector alone, and has no covariance analogue.

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

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L1UncertaintySet(;
        eps::Number,
        sd::Option{<:VecNum} = nothing,
        mu::Option{<:VecNum} = nothing
    ) -> L1UncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(eps)` and `eps >= 0`.
  - If `sd` is provided: `!isempty(sd)` and `all(sd .> 0)`.
  - If `mu` is provided: `!isempty(mu)` and `all(isfinite, mu)`.
  - If both `sd` and `mu` are provided: `length(mu) == length(sd)`.

# Examples

```jldoctest
julia> L1UncertaintySet(; eps = 0.1)
L1UncertaintySet
  eps ┼ Float64: 0.1
   sd ┼ nothing
   mu ┴ nothing
```

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref): the estimator that fits this set. It is mean-only, so [`sigma_ucs`](@ref) is not defined for it.
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`mu_ucs`](@ref): a set it builds carries in `mu` the characteristic vector its radius was calibrated on, so the consumer bounds that vector and not an unrelated one. See ADR 0050.
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:quintile])
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
    """
    $(field_dict[:mu_l1_ucs])
    """
    mu
    function L1UncertaintySet(eps::Number, sd::Option{<:VecNum}, mu::Option{<:VecNum})
        @argcheck(isfinite(eps) && eps >= zero(eps),
                  DomainError(eps, "eps must be finite and >= 0"))
        if isa(sd, VecNum)
            @argcheck(!isempty(sd), IsEmptyError("sd cannot be empty"))
            @argcheck(all(x -> x > zero(x), sd),
                      DomainError(sd, "all entries of sd must be > 0"))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu),
                      IsNonFiniteError("all elements of mu must be finite"))
            if isa(sd, VecNum)
                @argcheck(length(mu) == length(sd),
                          DimensionMismatch("mu ($(length(mu))) must match sd ($(length(sd)))"))
            end
        end
        return new{typeof(eps), typeof(sd), typeof(mu)}(eps, sd, mu)
    end
end
function L1UncertaintySet(eps::Number, sd::Option{<:VecNum})::L1UncertaintySet
    return L1UncertaintySet(eps, sd, nothing)
end
function L1UncertaintySet(; eps::Number, sd::Option{<:VecNum} = nothing,
                          mu::Option{<:VecNum} = nothing)::L1UncertaintySet
    return L1UncertaintySet(eps, sd, mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of an [`L1UncertaintySet`](@ref) restricted to the asset indices `i`.

The radius is a scalar budget shared across the universe, so it passes through unchanged; the per-asset scaling and the carried characteristic are sliced. The slice preserves the order of `i`, so the restricted set indexes the assets in the order the optimiser hands it.

# Algorithm

 1. Take `nothing_scalar_array_view(risk_ucs.sd, i)`, the per-asset scaling restricted to the selected assets, which passes a `nothing` through unchanged.
 2. Take `nothing_scalar_array_view(risk_ucs.mu, i)`, the carried characteristic restricted to the same assets, which passes a `nothing` through unchanged.
 3. Build an [`L1UncertaintySet`](@ref) from `risk_ucs.eps` and the two views. The radius is not rescaled, because it budgets the total error across whichever assets the set covers.

# Arguments

  - `risk_ucs`: ``\\ell_1`` uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::L1UncertaintySet`: The set restricted to `i`.

# Examples

```jldoctest
julia> ucs = L1UncertaintySet(; eps = 0.5, sd = [0.1, 0.2, 0.3], mu = [0.05, 0.03, 0.01]);

julia> PortfolioOptimisers.port_opt_view(ucs, [1, 3])
L1UncertaintySet
  eps ┼ Float64: 0.5
   sd ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.1, 0.3]
   mu ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.05, 0.01]
```

# Related

  - [`L1UncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::L1UncertaintySet, i, args...)::L1UncertaintySet
    return L1UncertaintySet(; eps = risk_ucs.eps,
                            sd = nothing_scalar_array_view(risk_ucs.sd, i),
                            mu = nothing_scalar_array_view(risk_ucs.mu, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Signed ``\\ell_1`` uncertainty set on the characteristic vector, with a separate error budget per sign.

The set is ``\\mathcal{A}_2 = \\{\\hat{\\boldsymbol{\\mu}} + \\boldsymbol{e} : \\boldsymbol{1}^{\\intercal} [\\boldsymbol{e} \\oslash \\boldsymbol{\\sigma}]_{+} \\leq \\epsilon_{+},\\, -\\boldsymbol{1}^{\\intercal} [\\boldsymbol{e} \\oslash \\boldsymbol{\\sigma}]_{-} \\leq \\epsilon_{-}\\}``. It bounds a *mean/characteristic* vector alone, and has no covariance analogue. [quintile](@cite) introduces it in order to *decouple* the long-short problem into two independent problems (its equations 27 and 28), which its Remark 12 then recombines only when the two legs happen to have complementary support; modelling the worst case below directly keeps the problem coupled, so that caveat does not arise.

# Mathematical definition

```math
\\underset{\\boldsymbol{\\mu} \\in \\mathcal{A}_2}{\\min}\\, \\boldsymbol{\\mu}^{\\intercal} \\boldsymbol{w} = \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w} - \\epsilon_{+} \\left[\\underset{i}{\\max}\\, (-\\sigma_i w_i)\\right]_{+} - \\epsilon_{-} \\left[\\underset{i}{\\max}\\, (\\sigma_i w_i)\\right]_{+}\\,.
```

Where:

  - ``\\epsilon_{+}``, ``\\epsilon_{-}``: Radii of the positive- and negative-error sides (`ep`, `en`).
  - ``[\\cdot]_{+}``, ``[\\cdot]_{-}``: Element-wise positive and negative parts.

Still concave and LP-representable, with one epigraph variable per sign.

Setting ``\\epsilon_{+} = \\epsilon_{-}`` does not recover ``\\mathcal{S}``: the joint set shares one budget across both signs, giving ``\\max(t_{+}, t_{-})``, whereas this one spends a budget per sign, giving ``\\epsilon_{+} t_{+} + \\epsilon_{-} t_{-}``. The two worst cases agree only when ``\\boldsymbol{w}`` is single-signed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignedL1UncertaintySet(;
        ep::Number,
        en::Number,
        sd::Option{<:VecNum} = nothing,
        mu::Option{<:VecNum} = nothing
    ) -> SignedL1UncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(ep)` and `ep >= 0`.
  - `isfinite(en)` and `en >= 0`.
  - If `sd` is provided: `!isempty(sd)` and `all(sd .> 0)`.
  - If `mu` is provided: `!isempty(mu)` and `all(isfinite, mu)`.
  - If both `sd` and `mu` are provided: `length(mu) == length(sd)`.

# Examples

```jldoctest
julia> SignedL1UncertaintySet(; ep = 0.1, en = 0.2)
SignedL1UncertaintySet
  ep ┼ Float64: 0.1
  en ┼ Float64: 0.2
  sd ┼ nothing
  mu ┴ nothing
```

# Related

  - [`L1UncertaintySet`](@ref): the joint set. Under a long-only budget the weights are single-signed, the two worst cases coincide, and the joint set is the simpler choice.
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref): the estimator that fits this set. It is mean-only, so [`sigma_ucs`](@ref) is not defined for it.
  - [`mu_ucs`](@ref): a set it builds carries in `mu` the characteristic vector its radii were calibrated on, so the consumer bounds that vector and not an unrelated one. See ADR 0050.
  - [`AbstractUncertaintySetResult`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:quintile])
"""
@concrete struct SignedL1UncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:ep_ucs])
    """
    ep
    """
    $(field_dict[:en_ucs])
    """
    en
    """
    $(field_dict[:sd_ucs])
    """
    sd
    """
    $(field_dict[:mu_l1_ucs])
    """
    mu
    function SignedL1UncertaintySet(ep::Number, en::Number, sd::Option{<:VecNum},
                                    mu::Option{<:VecNum})
        @argcheck(isfinite(ep) && ep >= zero(ep),
                  DomainError(ep, "ep must be finite and >= 0"))
        @argcheck(isfinite(en) && en >= zero(en),
                  DomainError(en, "en must be finite and >= 0"))
        if isa(sd, VecNum)
            @argcheck(!isempty(sd), IsEmptyError("sd cannot be empty"))
            @argcheck(all(x -> x > zero(x), sd),
                      DomainError(sd, "all entries of sd must be > 0"))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu),
                      IsNonFiniteError("all elements of mu must be finite"))
            if isa(sd, VecNum)
                @argcheck(length(mu) == length(sd),
                          DimensionMismatch("mu ($(length(mu))) must match sd ($(length(sd)))"))
            end
        end
        return new{typeof(ep), typeof(en), typeof(sd), typeof(mu)}(ep, en, sd, mu)
    end
end
function SignedL1UncertaintySet(ep::Number, en::Number,
                                sd::Option{<:VecNum})::SignedL1UncertaintySet
    return SignedL1UncertaintySet(ep, en, sd, nothing)
end
function SignedL1UncertaintySet(; ep::Number, en::Number, sd::Option{<:VecNum} = nothing,
                                mu::Option{<:VecNum} = nothing)::SignedL1UncertaintySet
    return SignedL1UncertaintySet(ep, en, sd, mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`SignedL1UncertaintySet`](@ref) restricted to the asset indices `i`.

Both radii are scalar budgets shared across the universe, so they pass through unchanged; the per-asset scaling and the carried characteristic are sliced. The slice preserves the order of `i`, so the restricted set indexes the assets in the order the optimiser hands it.

# Algorithm

 1. Take `nothing_scalar_array_view(risk_ucs.sd, i)`, the per-asset scaling restricted to the selected assets, which passes a `nothing` through unchanged.
 2. Take `nothing_scalar_array_view(risk_ucs.mu, i)`, the carried characteristic restricted to the same assets, which passes a `nothing` through unchanged.
 3. Build a [`SignedL1UncertaintySet`](@ref) from `risk_ucs.ep`, `risk_ucs.en` and the two views. Neither radius is rescaled, because each budgets the total error of its own sign across whichever assets the set covers.

# Arguments

  - `risk_ucs`: Signed ``\\ell_1`` uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::SignedL1UncertaintySet`: The set restricted to `i`.

# Examples

```jldoctest
julia> ucs = SignedL1UncertaintySet(; ep = 0.4, en = 0.6, sd = [0.1, 0.2, 0.3],
                                    mu = [0.05, 0.03, 0.01]);

julia> PortfolioOptimisers.port_opt_view(ucs, [1, 3])
SignedL1UncertaintySet
  ep ┼ Float64: 0.4
  en ┼ Float64: 0.6
  sd ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.1, 0.3]
  mu ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.05, 0.01]
```

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::SignedL1UncertaintySet, i, args...)::SignedL1UncertaintySet
    return SignedL1UncertaintySet(; ep = risk_ucs.ep, en = risk_ucs.en,
                                  sd = nothing_scalar_array_view(risk_ucs.sd, i),
                                  mu = nothing_scalar_array_view(risk_ucs.mu, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Shape algorithm selecting a joint ``\\ell_1`` uncertainty set.

`scaled` picks between the two sets of [quintile](@cite): `false` gives ``\\mathcal{S}`` (its equation 5) and `true` gives ``\\mathcal{A}_1`` (its equation 18). It defaults to `false` because ``\\mathcal{S}`` is the base construction of the paper, the one whose closed forms give the ``1/N`` and quintile portfolios. [`SignedL1UncertaintySetAlgorithm`](@ref) defaults the other way, because the paper defines its set only in the scaled form.

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

# Examples

```jldoctest
julia> L1UncertaintySetAlgorithm()
L1UncertaintySetAlgorithm
  method ┼ ActiveAssetsUncertaintyAlgorithm
         │   active ┴ Float64: 0.2
  scaled ┼ Bool: false
  paired ┴ Bool: false
```

# Related

  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`l1_resolve_eps`](@ref): the function that reads `method` and `paired` and returns the radius.

# References

  - $(ref_dict[:quintile])
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
    `paired`: Which closed form calibrates the radius when `method` is an [`AbstractUncertaintyEpsAlgorithm`](@ref). The number of assets a radius activates depends on the sign structure of the problem it is used in, which an uncertainty set cannot observe, so the caller must say. `false` takes the long-only ladder, for a problem with `w >= 0` and `bgt = 1` (Corollaries 4 and 11 of [quintile](@cite)). `true` takes the paired ladder, for a dollar-neutral problem with `bgt = 0` and `sbgt = 1/2`, where assets activate in long/short pairs (Corollary 7). Using the wrong one mis-calibrates the radius; it does not make the optimisation incorrect. Inert when `method` is a number.
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

The two ladders are fixed by the construction of [quintile](@cite) and need no `paired` flag: ``\\epsilon_{-}`` governs the long leg and is calibrated against the top of the ranking, ``\\epsilon_{+}`` governs the short leg and is calibrated against the bottom (its Corollary 13). `scaled` defaults to `true` because the paper defines ``\\mathcal{A}_2`` only in the volatility-scaled form, and states Corollary 13 in that form alone. [`L1UncertaintySetAlgorithm`](@ref) defaults the other way, because its base set carries no scaling.

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

# Examples

```jldoctest
julia> SignedL1UncertaintySetAlgorithm()
SignedL1UncertaintySetAlgorithm
      mp ┼ ActiveAssetsUncertaintyAlgorithm
         │   active ┴ Float64: 0.2
      mm ┼ ActiveAssetsUncertaintyAlgorithm
         │   active ┴ Float64: 0.2
  scaled ┴ Bool: true
```

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`l1_resolve_eps`](@ref): the function that reads `mp` and `mm` and returns each radius.

# References

  - $(ref_dict[:quintile])
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

Fits an ``\\ell_1`` uncertainty set on the characteristic vector, mean-only and with a calibrated radius.

`pe` supplies both the characteristic vector ``\\hat{\\boldsymbol{\\mu}}`` and, when the shape algorithm is `scaled`, the per-asset scaling ``\\hat{\\boldsymbol{\\sigma}} = \\sqrt{\\mathrm{diag}(\\hat{\\mathbf{\\Sigma}})}``. The sets are the robust best-characteristic sets of [quintile](@cite). Maximising the worst-case characteristic over one of them recovers the heuristic ``1/N``, quintile, and inverse-volatility portfolios as *exact solutions of a robust optimisation problem*, with the radius controlling how many assets are held — which is why this library ships no dedicated quintile optimiser. Compose the set with [`MeanRisk`](@ref) instead:

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
  - [`ucs`](@ref) and [`sigma_ucs`](@ref) always throw an `ArgumentError` on this estimator. The ``\\ell_1`` ball bounds a characteristic vector, [quintile](@cite) defines no covariance analogue, and there is therefore no covariance set for either to return.

# Examples

The radius is calibrated, not given. Three assets whose sample means are ``0.125``, ``0.0625`` and ``0.03125`` give the activation ladder ``[0, 0.0625, 0.125]``, and the default `active` of `0.2` targets one active asset, so the radius is the midpoint of the first rung and the second:

```jldoctest
julia> X = [0.1875 0.09375 0.0625
            0.0625 0.03125 0.0];

julia> mu_ucs(CharacteristicUncertaintySet(), X)
L1UncertaintySet
  eps ┼ Float64: 0.03125
   sd ┼ nothing
   mu ┴ Vector{Float64}: [0.125, 0.0625, 0.03125]
```

# Related

  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`mu_ucs`](@ref): the only fitting verb this estimator answers.
  - [`StandardDeviationExpectedReturns`](@ref): the characteristic need not be an expected return. A prior built on this estimator ranks on volatility, the Low Volatility factor, and reproduces Table III of [quintile](@cite).
  - [`AbstractUncertaintySetEstimator`](@ref)

# References

  - $(ref_dict[:quintile])
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

`mu` must be sorted in non-increasing order, with `sd` (when given) under the same permutation.

# Mathematical definition

```math
\\begin{align}
L_k &= \\sum_{i=1}^{k} \\frac{\\hat{\\mu}_i - \\hat{\\mu}_k}{\\sigma_i}\\,, \\quad k = 1,\\, \\ldots,\\, N\\,.
\\end{align}
```

Where:

  - ``L_k``: Activation threshold of the ``k``-th asset of the ranking.
  - $(math_dict[:mu_hat_i_rank])
  - $(math_dict[:sigma_i_ucs])
  - $(math_dict[:N])

This is the threshold of Lemma 2 (``\\sigma_i = 1``) and of Lemma 9 (scaled) of [quintile](@cite). Both lemmas state that the number of active assets is the largest ``k`` with ``L_k < \\epsilon``, so a radius in the open interval ``(L_q,\\, L_{q+1})`` activates exactly ``q`` assets, and the active weights are equal (Lemma 2) or inverse-volatility (Lemma 9). Corollaries 4 and 11 read this off at the quintile.

Two consequences bound the sequence. Every summand of ``L_1`` is zero, so ``L_1 = 0``: a strictly positive radius always activates at least one asset. And ``L_{k+1} - L_k = \\sum_{i=1}^{k+1} (\\hat{\\mu}_k - \\hat{\\mu}_{k+1}) / \\sigma_i \\geq 0`` because the ranking is non-increasing, so the sequence is non-decreasing and its rungs bracket. Equality holds exactly when ``\\hat{\\mu}_k = \\hat{\\mu}_{k+1}``, which is the tie Assumption 1 of the paper excludes.

# Algorithm

 1. Read `N = length(mu)`, the number of rungs to build.
 2. For each `k` in `1:N`, sum ``(\\hat{\\mu}_i - \\hat{\\mu}_k) / \\sigma_i`` over `i` in `1:k`, taking ``\\sigma_i = 1`` when `sd` is `nothing`, giving the `k`-th rung.

# Arguments

  - `mu`: Characteristic vector, sorted in non-increasing order.
  - `sd`: Per-asset scaling under the same permutation, or `nothing` for an unscaled ladder.

# Returns

  - `g::Vector{<:Number}`: The activation ladder, one rung per asset.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_activation_ladder([0.125, 0.0625, 0.03125], nothing)
3-element Vector{Float64}:
 0.0
 0.0625
 0.125

julia> PortfolioOptimisers.l1_activation_ladder([0.125, 0.0625, 0.03125], [0.5, 0.25, 0.125])
3-element Vector{Float64}:
 0.0
 0.125
 0.3125
```

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`l1_eps_from_ladder`](@ref)
  - [`l1_resolve_eps`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function l1_activation_ladder(mu::VecNum, sd::Option{<:VecNum})
    N = length(mu)
    return [sum(i -> (mu[i] - mu[k]) / (isnothing(sd) ? one(eltype(mu)) : sd[i]), 1:k)
            for k in 1:N]
end
"""
    l1_active_count(active::Union{<:Integer, <:AbstractFloat}, N::Int)

Convert an `active` target — a count or a fraction of the universe — into an asset count in `1:N`.

The type of `active` selects the rule, and the clamp makes a target outside the universe usable rather than an error: a count above `N` becomes `N`, and a fraction that rounds to zero becomes `1`.

# Algorithm

 1. When `active` is an `Integer`, take it as the count itself, giving `q`.
 2. Otherwise take `round(Int, active * N)`, the fraction of the universe rounded to the nearest asset, giving `q`.
 3. Return `clamp(q, 1, N)`, so the count names an asset of the ranking.

# Arguments

  - `active`: Target number of active assets, as a count (`Integer`) or a fraction of the universe (`AbstractFloat`).
  - `N`: Number of assets in the universe.

# Returns

  - `q::Int`: The target count, in `1:N`.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_active_count(3, 10)
3

julia> PortfolioOptimisers.l1_active_count(0.2, 10)
2

julia> PortfolioOptimisers.l1_active_count(20, 10)
10
```

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`l1_resolve_eps`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function l1_active_count(active::Union{<:Integer, <:AbstractFloat}, N::Int)
    q = isa(active, Integer) ? Int(active) : round(Int, active * N)
    return clamp(q, 1, N)
end
"""
    l1_eps_from_ladder(method, ladder::VecNum, N::Int)

Resolve a radius from a [`Num_UcSEps`](@ref) against an activation ladder.

The method that takes a `Number` is a passthrough: it returns the radius the caller gave, runs no procedure, and carries neither an `# Algorithm` nor a `# Validation` section. The method that takes an [`ActiveAssetsUncertaintyAlgorithm`](@ref) returns the midpoint of the open interval ``(L_q,\\, L_{q+1})`` that the closed forms of [quintile](@cite) require for exactly `q` active entries.

Above the last rung any radius activates every entry, so the top of the ladder has no interval to bisect and the method continues the final increment by a half step instead. A ladder of one rung has no increment either, and the method returns `one(eltype(ladder))`. That radius is inert rather than calibrated: a one-rung ladder is a one-asset universe, where the budget pins the single weight and every radius gives the same portfolio.

# Algorithm

 1. Read `L = length(ladder)`, the number of rungs.
 2. Clamp `q` to `1:L`, so the target names a rung.
 3. When `q < L`, read the bracketing rungs `lo, hi = ladder[q], ladder[q+1]` and return their midpoint, a radius strictly inside the open interval.
 4. Otherwise `q == L`: return `ladder[L] + (ladder[L] - ladder[L-1]) / 2`, half a final increment above the last rung, or `one(eltype(ladder))` when `L == 1` and there is no increment to continue.

# Arguments

  - `method`: A radius (`Number`) or an [`ActiveAssetsUncertaintyAlgorithm`](@ref).
  - `ladder`: Activation ladder, as [`l1_activation_ladder`](@ref) returns it.
  - `q`: Target number of active entries.
  - `args...`: Additional positional arguments (ignored by the `Number` method).

# Validation

  - `length(ladder) >= 1`, otherwise an `IsEmptyError` is thrown.
  - `ladder[q+1] > ladder[q]` when `q < L`, otherwise a `DomainError` is thrown. The rungs coincide when the characteristic has ties across the cut, the interval the closed form needs is empty, and no radius delivers the requested count. Assumption 1 of [quintile](@cite) excludes the case, so the message names the two fixes: break the tie, or pass an explicit radius.

# Returns

  - `eps::Number`: The resolved radius.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_eps_from_ladder(0.5)
0.5

julia> alg = ActiveAssetsUncertaintyAlgorithm(; active = 2);

julia> PortfolioOptimisers.l1_eps_from_ladder(alg, [0.0, 0.0625, 0.125], 2)
0.09375

julia> PortfolioOptimisers.l1_eps_from_ladder(alg, [0.0, 0.0625, 0.125], 3)
0.15625

julia> PortfolioOptimisers.l1_eps_from_ladder(alg, [0.0], 1)
1.0
```

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`l1_activation_ladder`](@ref)
  - [`l1_resolve_eps`](@ref)

# References

  - $(ref_dict[:quintile])
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
        lo, hi = ladder[q], ladder[q+1]
        @argcheck(hi > lo,
                  DomainError(method.active,
                              "cannot calibrate a radius for $(q) active entries: the characteristic has ties across the cut, so the interval ($(lo), $(hi)) required by the closed form is empty. Break the tie, or pass an explicit radius."))
        return (lo + hi) / 2
    end
    # Top of the ladder: any radius above ladder[end] activates everything.
    return L == 1 ? one(eltype(ladder)) : ladder[L] + (ladder[L] - ladder[L-1]) / 2
end
"""
    l1_resolve_eps(method, mus::VecNum, sds::Option{<:VecNum}, paired::Bool)

Resolve a radius from a [`Num_UcSEps`](@ref) against the characteristic vector.

`mus` must be sorted in non-increasing order, with `sds` under the same permutation. The method that takes a `Number` is a passthrough: it returns the radius the caller gave, runs no procedure, and carries neither an `# Algorithm` nor a `# Validation` section. The method that takes an [`ActiveAssetsUncertaintyAlgorithm`](@ref) calibrates against the long-only ladder when `paired` is `false`, and against the paired ladder when it is `true`.

# Mathematical definition

The paired ladder adds the ladder of the ranking to the ladder of the reversed, negated ranking, which is Lemma 6 of [quintile](@cite):

```math
\\begin{align}
F_m &= \\sum_{i=1}^{m} \\frac{\\hat{\\mu}_i - \\hat{\\mu}_m}{\\sigma_i}
     + \\sum_{j=1}^{m} \\frac{\\hat{\\mu}_{N-m+1} - \\hat{\\mu}_{N-j+1}}{\\sigma_{N-j+1}}\\,,
     \\quad m = 1,\\, \\ldots,\\, \\left\\lfloor N/2 \\right\\rfloor\\,.
\\end{align}
```

Where:

  - ``F_m``: Activation threshold of the ``m``-th long/short pair.
  - $(math_dict[:mu_hat_i_rank])
  - $(math_dict[:sigma_i_ucs])
  - $(math_dict[:N])

A radius in ``(F_q,\\, F_{q+1})`` makes the top ``q`` assets long and the bottom ``q`` short, so it activates ``2q`` assets (Corollary 7). Lemma 5 gives the reason the ladder stops at ``\\lfloor N/2 \\rfloor``: an optimal dollar-neutral solution satisfies ``w_i = -w_{N+1-i}``, so with an odd ``N`` the middle asset pairs with itself, its weight solves ``w = -w``, and it is never active. The largest count the paired branch can reach is therefore ``2 \\lfloor N/2 \\rfloor``, which is ``N-1`` when ``N`` is odd.

# Algorithm

 1. Read `N = length(mus)`, the size of the universe.
 2. When `paired` is `false`, build the long-only ladder with [`l1_activation_ladder`](@ref), convert `method.active` to a count with [`l1_active_count`](@ref), and return the radius [`l1_eps_from_ladder`](@ref) resolves against them.
 3. Otherwise read `half = N ÷ 2`, the number of pairs the universe admits.
 4. Add the ladder of `mus` to the ladder of the reversed, negated ranking and keep the first `half` entries, giving `ladder`, the paired ladder.
 5. Halve the target: an `Integer` `active` counts assets and gives `max(active ÷ 2, 1)` pairs; an `AbstractFloat` `active` is a fraction of the universe and gives `clamp(round(Int, active * N / 2), 1, half)` pairs. Both name pairs, so a count and the matching fraction agree whenever the count is even. An odd count truncates while the matching fraction rounds to nearest, so the two can differ by one pair: `active = 7` gives three pairs and `active = 7/N` gives four.
 6. Return the radius [`l1_eps_from_ladder`](@ref) resolves against `ladder` and the pair count.

# Arguments

  - `method`: A radius (`Number`) or an [`ActiveAssetsUncertaintyAlgorithm`](@ref).
  - `mus`: Characteristic vector, sorted in non-increasing order.
  - `sds`: Per-asset scaling under the same permutation, or `nothing`.
  - `paired`: Whether to calibrate against the paired ladder rather than the long-only one.

# Validation

  - `N ÷ 2 >= 1` when `paired` is `true`, otherwise an `ArgumentError` is thrown. A single asset admits no long/short pair, so no paired ladder exists to calibrate against.

# Returns

  - `eps::Number`: The resolved radius.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_resolve_eps(0.25)
0.25

julia> alg = ActiveAssetsUncertaintyAlgorithm(; active = 2);

julia> mus = [0.125, 0.0625, 0.03125, 0.0];

julia> PortfolioOptimisers.l1_resolve_eps(alg, mus, nothing, false)
0.09375

julia> PortfolioOptimisers.l1_resolve_eps(alg, mus, nothing, true)
0.046875
```

# Related

  - [`l1_activation_ladder`](@ref)
  - [`l1_active_count`](@ref)
  - [`l1_eps_from_ladder`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)

# References

  - $(ref_dict[:quintile])
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
    ladder = (l1_activation_ladder(mus, sds)+l1_activation_ladder(reverse(-mus),
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

The calibration runs on the ranking, and the set is returned in the universe's own order: `mu` and `sd` carry one entry per asset in the order the prior produced them, which is the order the optimiser indexes and the order [`port_opt_view`](@ref) slices. Only the ladders see the sorted vector, and a ladder needs no asset identity. The set carries ``\\hat{\\boldsymbol{\\mu}}`` in its `mu` field, so the consumer bounds the characteristic vector the radius was calibrated on. See ADR 0050.

# Algorithm

The two methods share the first four steps and differ in the fifth.

 1. Fit `ue.pe` on `X` and `F`, giving the prior `pr`.
 2. Read `alg = ue.alg`, the shape algorithm.
 3. When `alg.scaled`, take `sd = sqrt.(diag(pr.sigma))`, the per-asset scaling; otherwise take `nothing`.
 4. Take `idx = sortperm(pr.mu; rev = true)`, the ranking that sorts the characteristic non-increasing, and apply it to `pr.mu` and to `sd`.
 5. Resolve the radii with [`l1_resolve_eps`](@ref) against the sorted vectors, and build the set from them with `mu = pr.mu` and `sd` in the universe's order.

For an [`L1UncertaintySetAlgorithm`](@ref) step 5 resolves the single radius `eps` from `alg.method`, passing `alg.paired`, and builds an [`L1UncertaintySet`](@ref).

For a [`SignedL1UncertaintySetAlgorithm`](@ref) step 5 resolves two radii against two ladders and builds a [`SignedL1UncertaintySet`](@ref). `en` comes from `alg.mm` on the sorted ranking, and `ep` from `alg.mp` on the reversed, negated ranking, which is the same ladder read from the other end (Corollary 13 of [quintile](@cite)). Neither call passes `paired`, because the construction fixes both ladders. The consumer spends `en` against ``\\max_i (\\sigma_i w_i)`` and `ep` against ``\\max_i (-\\sigma_i w_i)``, so `en` prices the long leg and `ep` the short one, matching the end each was calibrated against.

# Arguments

  - `ue`: Characteristic uncertainty set estimator.
  - `X`: Data matrix (e.g. returns).
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::Union{<:L1UncertaintySet, <:SignedL1UncertaintySet}`: The uncertainty set.

# Examples

```jldoctest
julia> X = [0.1875 0.09375 0.0625
            0.0625 0.03125 0.0];

julia> alg = SignedL1UncertaintySetAlgorithm(; scaled = false);

julia> mu_ucs(CharacteristicUncertaintySet(; alg = alg), X)
SignedL1UncertaintySet
  ep ┼ Float64: 0.015625
  en ┼ Float64: 0.03125
  sd ┼ nothing
  mu ┴ Vector{Float64}: [0.125, 0.0625, 0.03125]
```

# Related

  - [`CharacteristicUncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`l1_resolve_eps`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function mu_ucs(ue::CharacteristicUncertaintySet{<:Any, <:L1UncertaintySetAlgorithm},
                X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    alg = ue.alg
    sd = alg.scaled ? sqrt.(LinearAlgebra.diag(pr.sigma)) : nothing
    idx = sortperm(pr.mu; rev = true)
    eps = l1_resolve_eps(alg.method, pr.mu[idx], alg.scaled ? sd[idx] : nothing, alg.paired)
    return L1UncertaintySet(; eps = eps, sd = sd, mu = pr.mu)
end
function mu_ucs(ue::CharacteristicUncertaintySet{<:Any, <:SignedL1UncertaintySetAlgorithm},
                X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    alg = ue.alg
    sd = alg.scaled ? sqrt.(LinearAlgebra.diag(pr.sigma)) : nothing
    idx = sortperm(pr.mu; rev = true)
    mus = pr.mu[idx]
    sds = isnothing(sd) ? nothing : sd[idx]
    # en governs the long leg, calibrated against the top of the ranking; ep governs the
    # short leg, calibrated against the bottom — i.e. the top of the reversed, negated
    # ranking, which is the same ladder read from the other end (Corollary 13).
    en = l1_resolve_eps(alg.mm, mus, sds, false)
    ep = l1_resolve_eps(alg.mp, reverse(-mus), alg.scaled ? reverse(sds) : nothing, false)
    return SignedL1UncertaintySet(; ep = ep, en = en, sd = sd, mu = pr.mu)
end
"""
    ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)
    sigma_ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)

Always throw. [`CharacteristicUncertaintySet`](@ref) is mean-only.

Both methods are refusals rather than procedures, so neither carries an `# Algorithm` section. They take the same `(X, F)` signature as the rest of the family rather than a catch-all, so that the [`ReturnsResult`](@ref) forwarders in the base reach them without ambiguity.

# Arguments

  - `ue`: Characteristic uncertainty set estimator.
  - `X`: Data matrix (e.g. returns).
  - `F`: Optional factor matrix.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - Both methods always throw an `ArgumentError`. The ``\\ell_1`` set bounds a characteristic (mean) vector, [quintile](@cite) defines no covariance analogue, and there is therefore no covariance set to return. Each message names the fix: [`mu_ucs`](@ref) for the mean set, and [`NormalUncertaintySet`](@ref), [`DeltaUncertaintySet`](@ref) or [`ARCHUncertaintySet`](@ref) for a covariance set.

# Returns

  - Neither method returns. Both throw.

# Related

  - [`CharacteristicUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
  - [`NormalUncertaintySet`](@ref)
  - [`DeltaUncertaintySet`](@ref)
  - [`ARCHUncertaintySet`](@ref)

# References

  - $(ref_dict[:quintile])
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
