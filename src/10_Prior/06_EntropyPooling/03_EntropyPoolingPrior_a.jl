"""
    ep_evar(x::VecNum, w::VecNum, alpha::Number; args::Tuple = (),
            kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)

Compute the sample entropic value at risk of a loss series and the dual variable that attains it.

`ep_evar` minimises the convex scalar objective of the sample EVaR with [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). The entropy pooling views need the value and the minimiser. The value compares a view with its prior, and the minimiser is the centre of the grid of [`GridEntropicValueatRiskView`](@ref). [EPTail](@citet) states the formula and puts the centre of the grid at the minimiser `z`. It does not state how to find that minimiser, so the bracket and the search are the library's own.

# Mathematical definition

```math
\\mathrm{EVaR}_{\\alpha}(X) = \\min_{z > 0} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
```

# Algorithm

 1. Normalise the observation probabilities in the logarithmic domain. The result is `lw`.
 2. Bracket the dual variable. The upper end `hi` is `(maximum(x) - dot(w, x)) / log(inv(alpha))`. When `hi` is not positive, `eps` of its type replaces it. The lower end is `hi * zlo_frac`.
 3. Minimise the objective over the bracket with [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). Each evaluation uses `LogExpFunctions.logsumexp`, so a small `z` does not overflow.
 4. Return the minimum as `evar`, and the minimiser as `z`.

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The observation probabilities. The function normalises them to sum to one.
  - `alpha`: The significance level.
  - $(arg_dict[:optargs]) An empty `args` gives `Optim.Brent()`, which `Optim.optimize` selects for a bracketed scalar minimisation.
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: The lower end of the bracket of the dual variable, as a fraction of the upper end. `nothing` gives `sqrt(eps(T))` for the element type `T`. The caller cannot state this value, because the type comes from the data. Above the upper end, `(maximum(x) - dot(w, x)) / log(inv(alpha))`, the objective is larger than `maximum(x)`, which is an upper bound of the EVaR. This is a proof, so the upper end is fixed and only the lower end is a parameter.

# Validation

  - `0 < zlo_frac < 1`.
  - The search converges. It is a bracketed scalar minimisation of a convex function, so it fails only when `args` or `kwargs` stop it early.

# Returns

  - `res::@NamedTuple{evar::Number, z::Number}`: The value and the dual variable that attains it.

# Related

  - [`GridEntropicValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`EntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equation 7.
"""
function ep_evar(x::VecNum, w::VecNum, alpha::Number; args::Tuple = (),
                 kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)
    lw = log.(w)
    lw .-= LogExpFunctions.logsumexp(lw)
    ila = -log(alpha)
    f = function (z)
        return z * (LogExpFunctions.logsumexp(lw .+ x ./ z) + ila)
    end
    hi = (maximum(x) - LinearAlgebra.dot(exp.(lw), x)) / ila
    ehi = eps(typeof(hi))
    hi = ifelse(hi > zero(hi), hi, ehi)
    # The default lower end is the one the element type states, which a caller that holds no
    # data cannot, so `nothing` resolves here rather than in the view that carries it.
    zlo_frac = isnothing(zlo_frac) ? sqrt(ehi) : zlo_frac
    @argcheck(zero(zlo_frac) < zlo_frac < one(zlo_frac),
              DomainError(zlo_frac, "zlo_frac must be in (0, 1)"))
    lo = hi * zlo_frac
    res = Optim.optimize(f, lo, hi, args...; kwargs...)
    @argcheck(Optim.converged(res),
              ErrorException("The search for the sample EVaR did not converge. Relax the `args` and `kwargs` of the view group, or leave them empty to take the defaults."))
    return (; evar = Optim.minimum(res), z = Optim.minimizer(res))
end
"""
    ep_evar_grid_row(x::VecNum, ebar::Number, z::Number)

Build one scaled row of the grid formulation of an entropic value-at-risk view.

`ep_evar_grid_row` returns the coefficients `exp((x - ebar) / z)` divided by their largest entry, and the reciprocal of that entry. The caller multiplies the right hand side by the reciprocal. This scale keeps each coefficient in `(0, 1]` at a small `z` too. At a small `z` the scaled bound `alpha * isc` is much smaller than one, so [`add_ep_tail_view!`](@ref) also divides an upper-bound row by its bound. A lower-bound row keeps the norm scale of [`ep_add_grid_tail_view!`](@ref).

# Algorithm

 1. Exponentiate the shifted loss series. The result is `c`, the coefficients `exp((x - ebar) / z)` before the scale.
 2. Read the largest entry of `c` into `sc`.
 3. Return `c` divided by `sc`, and `isc`, the reciprocal of `sc`.

# Arguments

  - `x`: The loss series (`-returns`).
  - `ebar`: The target entropic value at risk.
  - `z`: The dual variable of the grid point.

# Returns

  - `c::VecNum`: The scaled coefficients.
  - `isc::Number`: The scale factor of the right hand side.

# Related

  - [`GridEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskViewConstraint`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_add_grid_tail_view!`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_evar_grid_row(x::VecNum, ebar::Number, z::Number)
    c = exp.((x .- ebar) ./ z)
    sc = maximum(c)
    return c ./ sc, inv(sc)
end
"""
    ep_rlvar_tail(u::Number, z::Number, kappa::Number)

Compute the smallest tail penalty that the pair of power cones of one observation allows.

The primal programme of the relativistic value at risk has two power cones and two non-negative variables per observation. This function minimises the sum of the two variables in closed form. With this closed form, a point of the primal programme gives a row that is linear in the posterior probabilities. [EPRLVaR](@citet) states the cones but not the closed form, which is the library's own.

# Mathematical definition

```math
\\begin{align}
\\varphi_{\\kappa}(u, z) &= \\dfrac{\\kappa}{1+\\kappa} \\left(\\dfrac{2\\kappa}{(1+\\kappa) z}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma - u}{2}\\right)^{\\frac{1+\\kappa}{\\kappa}} + \\kappa (1-\\kappa)^{\\frac{1-\\kappa}{\\kappa}} \\left(\\dfrac{z}{2\\kappa}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma + u}{2}\\right)^{-\\frac{1-\\kappa}{\\kappa}}\\,,\\\\
\\sigma &= \\sqrt{u^{2} + \\dfrac{(1 - \\kappa^{2}) z^{2}}{\\kappa^{2}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_u])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:rlvar_sigma])

# Arguments

  - `u`: The shifted loss of the observation, `t - x` for a loss `x`.
  - `z`: The dual variable of the primal programme.
  - `kappa`: The deformation parameter, in `(0, 1)`.

# Returns

  - `phi::Number`: The smallest sum of the two tail variables of the observation.

# Related

  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_grid_row`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
function ep_rlvar_tail(u::Number, z::Number, kappa::Number)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    sigma = sqrt(u^2 + opk * omk * (z * ik)^2)
    psi = kappa / opk * (2 * kappa / (opk * z))^ik * ((sigma - u) / 2)^(opk * ik)
    theta = kappa * omk^(omk * ik) * (z / (2 * kappa))^ik * ((sigma + u) / 2)^(-omk * ik)
    return psi + theta
end
"""
    ep_rlvar_shift(x::VecNum, w::VecNum, kappa::Number, lnk::Number, z::Number;
                   args::Tuple = (), kwargs::NamedTuple = (;),
                   bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Minimise the primal objective of the relativistic value at risk over its shift variable, at a fixed dual variable.

The objective is the primal programme of [EPRLVaR](@cite). [EPRLVaR](@citet) does not state how to search for the minimum, so the bracket of the shift is the library's own.

# Mathematical definition

```math
\\begin{align}
\\underset{t}{\\min} &\\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:kappa_rm])

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The observation probabilities, which sum to one.
  - `kappa`: The deformation parameter, in `(0, 1)`.
  - `lnk`: The Kaniadakis logarithm of `inv(alpha * T)`, from [`kappa_log`](@ref).
  - `z`: The dual variable of the primal programme.
  - $(arg_dict[:optargs]) An empty `args` gives `Optim.Brent()`, which `Optim.optimize` selects for a bracketed scalar minimisation.
  - $(arg_dict[:optkwargs])
  - `bracket`: The spans of the searches of a relativistic value-at-risk view, or `nothing` for the default spans of [`RelativisticValueatRiskViewBracket`](@ref). This function reads only `tspan`. The span is a margin and not a proof, so make it wider if the minimiser is at an end of the bracket. [`ep_rlvar`](@ref) reads the other two fields.

# Validation

  - The search converges. It is a bracketed scalar minimisation of a convex function, so it fails only when `args` or `kwargs` stop it early. A minimiser at an end of the bracket does not make the search fail, because `Optim` reports that end as converged.

# Returns

  - `res::@NamedTuple{risk::Number, t::Number}`: The smallest value of the objective, and the shift that attains it.

# Algorithm

 1. Bracket the shift by the loss range, widened on each side by `tspan` times the span of the losses. The minimiser is near the largest loss, so the bracket holds it with a wide margin.
 2. Minimise the objective over the bracket with [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/).

# Related

  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_tail`](@ref)
  - [`RelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
function ep_rlvar_shift(x::VecNum, w::VecNum, kappa::Number, lnk::Number, z::Number;
                        args::Tuple = (), kwargs::NamedTuple = (;),
                        bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    # `nothing` resolves to the default bracket rather than to bare numbers, so the spans
    # are written once, in the constructor that validates them.
    tspan = something(bracket, RelativisticValueatRiskViewBracket()).tspan
    T = length(x)
    xmin, xmax = extrema(x)
    span = xmax - xmin
    span = ifelse(span > zero(span), span, max(abs(xmax), one(xmax)))
    f = function (t)
        acc = zero(t * one(eltype(w)))
        for j in eachindex(x, w)
            acc += w[j] * ep_rlvar_tail(t - x[j], z, kappa)
        end
        return t + z * lnk + T * acc
    end
    res = Optim.optimize(f, xmin - tspan * span, xmax + tspan * span, args...; kwargs...)
    @argcheck(Optim.converged(res),
              ErrorException("The search for the shift of the sample RLVaR did not converge. Relax the `args` and `kwargs` of the view group, or leave them empty to take the defaults."))
    return (; risk = Optim.minimum(res), t = Optim.minimizer(res))
end
"""
    ep_rlvar(x::VecNum, w::VecNum, alpha::Number, kappa::Number; args::Tuple = (),
             kwargs::NamedTuple = (;), bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Compute the sample relativistic value at risk of a loss series and the primal point that attains it.

`ep_rlvar` minimises the primal objective of the sample RLVaR over its two variables. [`ep_rlvar_tail`](@ref) already minimises out the power cones of each observation. The entropy pooling views need the value and the minimiser. The value compares a view with its prior, and the minimiser is the centre of the grid of [`GridRelativisticValueatRiskView`](@ref). The objective is the primal programme of [EPRLVaR](@cite). [EPRLVaR](@citet) does not state how to search for the minimum, so the two searches and their brackets are the library's own.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) &= \\underset{t,\\, z > 0}{\\min} \\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_stat])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:kappa_rm])

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The observation probabilities. The function normalises them to sum to one.
  - `alpha`: The significance level.
  - `kappa`: The deformation parameter, in `(0, 1)`.
  - $(arg_dict[:optargs]) They go to both searches. An empty `args` gives `Optim.Brent()`, which `Optim.optimize` selects for a bracketed scalar minimisation.
  - $(arg_dict[:optkwargs]) They go to both searches.
  - `bracket`: The spans of the searches, or `nothing` for the default spans of [`RelativisticValueatRiskViewBracket`](@ref). This function reads `log_zlo` and `log_zhi`. They are the ends of the bracket of the logarithm of the dual variable, as offsets from the logarithm of the loss range. They are a margin and not a proof, so widen one if the minimiser is at an end of the bracket. [`ep_rlvar_shift`](@ref) reads `tspan`.

# Validation

  - Both searches converge. Each is a bracketed scalar minimisation of a unimodal function, so a search fails only when `args` or `kwargs` stop it early. A minimiser at an end of a bracket does not make a search fail, because `Optim` reports that end as converged.

# Returns

  - `res::@NamedTuple{rlvar::Number, t::Number, z::Number}`: The value and the primal pair that attains it.

# Algorithm

 1. Minimise over the logarithm of the dual variable with [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). The bracket goes from `exp(log_zlo)` to `exp(log_zhi)` times the loss range, which is about `2e-9` to about `2e4` for the default bracket. The objective is convex in the pair, so the partial minimum over the shift is convex in the dual variable. The logarithm is increasing, so the function of the outer minimisation is unimodal, but it is not convex in general.
 2. Minimise over the shift at each candidate dual variable with [`ep_rlvar_shift`](@ref).
 3. Do the inner minimisation again at the dual variable of the minimum, so that the returned shift is the shift that attains the value.

# Related

  - [`ep_rlvar_tail`](@ref)
  - [`ep_rlvar_shift`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`RelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
function ep_rlvar(x::VecNum, w::VecNum, alpha::Number, kappa::Number; args::Tuple = (),
                  kwargs::NamedTuple = (;),
                  bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    # `nothing` resolves to the default bracket rather than to bare numbers, so the spans
    # are written once, in the constructor that validates them.
    bkt = something(bracket, RelativisticValueatRiskViewBracket())
    log_zlo, log_zhi = bkt.log_zlo, bkt.log_zhi
    T = length(x)
    wi = w ./ sum(w)
    lnk = kappa_log(inv(alpha * T), kappa)
    xmin, xmax = extrema(x)
    span = xmax - xmin
    span = ifelse(span > zero(span), span, max(abs(xmax), one(xmax)))
    lspan = log(span)
    res = Optim.optimize(u -> ep_rlvar_shift(x, wi, kappa, lnk, exp(u); args = args,
                                             kwargs = kwargs, bracket = bracket).risk,
                         lspan + log_zlo, lspan + log_zhi, args...; kwargs...)
    @argcheck(Optim.converged(res),
              ErrorException("The search for the dual variable of the sample RLVaR did not converge. Relax the `args` and `kwargs` of the view group, or leave them empty to take the defaults."))
    z = exp(Optim.minimizer(res))
    shift = ep_rlvar_shift(x, wi, kappa, lnk, z; args = args, kwargs = kwargs,
                           bracket = bracket)
    return (; rlvar = shift.risk, t = shift.t, z = z)
end
"""
    ep_rlvar_grid_row(x::VecNum, vbar::Number, t::Number, z::Number, alpha::Number,
                      kappa::Number)

Build one scaled row of the grid formulation of a relativistic value-at-risk view.

`ep_rlvar_grid_row` returns the coefficients `T * phi(t - x, z)` divided by their largest entry, and the target of the row divided by the same entry. This scale keeps each finite coefficient in `(0, 1]`, at a small `z` too. At a small `z` the scaled target is much smaller than one, so [`add_ep_tail_view!`](@ref) also divides an upper-bound row by its target. A lower-bound row keeps the norm scale of [`ep_add_grid_tail_view!`](@ref). [EPRLVaR](@citet) states the primal programme and no grid of its points. The row is the library's own, and it uses the primal programme as the grid rows of the entropic value at risk in [EPTail](@cite) use its primal.

# Arguments

  - `x`: The loss series (`-returns`).
  - `vbar`: The target relativistic value at risk.
  - `t`: The shift variable of the grid point.
  - `z`: The dual variable of the grid point.
  - `alpha`: The significance level.
  - `kappa`: The deformation parameter, in `(0, 1)`.

# Returns

  - `c::VecNum`: The scaled coefficients.
  - `b::Number`: The scaled target of the row.

# Related

  - [`GridRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskViewConstraint`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_add_grid_tail_view!`](@ref)
  - [`ep_tail_views!`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equations 10 and 11.
  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
function ep_rlvar_grid_row(x::VecNum, vbar::Number, t::Number, z::Number, alpha::Number,
                           kappa::Number)
    T = length(x)
    lnk = kappa_log(inv(alpha * T), kappa)
    c = T .* ep_rlvar_tail.(t .- x, z, kappa)
    sc = maximum(c)
    return c ./ sc, (vbar - t - z * lnk) / sc
end
"""
    ep_row_tilt(w::VecNum, c::VecNum, b::Number; iters::Integer = 200)

Tilt a probability vector so that one linear row holds with equality, at the smallest relative entropy.

The row of a grid point is linear in the posterior probabilities. So the posterior that makes the row tight, at the smallest relative entropy from the prior, is an exponential tilt of the prior along the coefficients of the row. This tilt is the entropy pooling posterior of that one row, and it needs no solver. The tilt is the closed form of [meucci2008](@cite) for one equality constraint. Its use as the anchor of a grid is the library's own.

# Mathematical definition

```math
\\begin{align}
q_{j}(\\theta) &= \\dfrac{w_{j} e^{-\\theta c_{j}}}{\\sum_{i=1}^{T} w_{i} e^{-\\theta c_{i}}}\\,,\\\\
\\sum_{j=1}^{T} q_{j}(\\theta) c_{j} &= b\\,.
\\end{align}
```

The value of the row under the tilt decreases strictly as ``\\theta`` increases, from ``\\max_{j} c_{j}`` to ``\\min_{j} c_{j}`` over the observations with ``w_{j} > 0``. So the tilt exists if and only if ``b`` is strictly inside that range. A tilt cannot put mass on an observation of zero prior probability, so the coefficient of such an observation sets no bound.

# Arguments

  - `w`: The prior probabilities, which sum to one.
  - `c`: The coefficients of the row.
  - `b`: The value that the row must take.
  - `iters::Integer = 200`: The largest number of bisection steps. The bisection also stops when the midpoint does not change. For `Float64` this happens near step 64, so the limit applies only to a number type of higher precision.

# Validation

  - `iters >= 1`.

# Returns

  - `q::Option{VecNum}`: The tilted probabilities. The function returns `nothing` when no tilt attains `b`, because `b` is outside the range of `c` over the support of `w`. It also returns `nothing` when the tilted row is not finite.

# Algorithm

 1. Return `nothing` when `b` is outside the open range of `c` over the observations with `w > 0`.
 2. Bracket the root. Double the tilt away from zero until the value of the row crosses `b`.
 3. Bisect the bracket until the midpoint does not change, or for `iters` steps if that comes first.
 4. Return `nothing` when the sum of the tilted weights is zero or not a number. Otherwise return the normalised weights.

# Related

  - [`ep_rlvar_anchor`](@ref)
  - [`ep_rlvar_grid_row`](@ref)

# References

  - $(ref_dict[:meucci2008]) Appendix A.3, Equation 90.
"""
function ep_row_tilt(w::VecNum, c::VecNum, b::Number; iters::Integer = 200)
    @argcheck(iters >= one(iters), DomainError(iters, "iters must be >= 1"))
    # The range is read over the support of `w`: a tilt leaves a zero prior probability at
    # zero, so a target past the support's extreme sends the doubling below to underflow
    # every `exp`, and the row to `0/0`. Issue #1260.
    lo, hi = extrema(ci for (ci, wi) in zip(c, w) if wi > zero(wi))
    if !(lo < b < hi)
        return nothing
    end
    q = Vector{promote_type(eltype(w), eltype(c), typeof(b))}(undef, length(w))
    # Each call leaves the unnormalised tilt in `q`, so the last call is the answer.
    row = function (th)
        q .= (-th) .* c
        q .= w .* exp.(q .- maximum(q))
        return LinearAlgebra.dot(q, c) / sum(q)
    end
    # The row's value falls as the tilt rises, so one side of zero holds the root. Doubling
    # reaches it, because the value tends to an end of the range of `c` and `b` is inside.
    sgn = ifelse(row(zero(b)) >= b, one(b), -one(b))
    thb = sgn
    while (row(thb) - b) * sgn > zero(b)
        thb *= 2
    end
    tha = zero(b)
    for _ in 1:iters
        thm = (tha + thb) / 2
        if (thm == tha || thm == thb)
            break
        end
        ((row(thm) - b) * sgn > zero(b)) ? (tha = thm) : (thb = thm)
    end
    row(thb)
    # A sum that is zero or not a number means the tilt underflowed.
    return sum(q) > zero(eltype(q)) ? q ./ sum(q) : nothing
end
"""
    ep_evar_anchor(x::VecNum, w::VecNum, alpha::Number, rhs::Number, z::Number;
                   iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                   args::Tuple = (), kwargs::NamedTuple = (;),
                   zlo_frac::Option{<:Number} = nothing)

Find the dual variable of the entropic value at risk that a posterior attains when it meets an upper-bound view.

A grid point states the view as one row. A posterior that makes the row tight reaches the target only if that grid point is the point that the posterior attains. `ep_evar_anchor` finds the dual variable that meets both conditions, and the grid of [`GridEntropicValueatRiskView`](@ref) has this dual variable as its centre. The function calls no solver. [EPTail](@citet) puts the centre of the grid at the dual variable of the prior. The anchor at the dual variable of the posterior is the library's own.

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The prior probabilities, which sum to one.
  - `alpha`: The significance level.
  - `rhs`: The target entropic value at risk.
  - `z`: The dual variable at the start of the iteration.
  - `iters::Integer = 50`: The largest number of steps of the iteration.
  - `tol::Number = 1e-10`: The relative distance from the target at which the iteration stops.
  - `tilt_iters::Integer = 200`: The largest number of bisection steps of the tilt of one row, in [`ep_row_tilt`](@ref).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: The lower end of the bracket of the dual variable, as a fraction of the upper end, which goes to [`ep_evar`](@ref).

# Returns

  - `res::Option{@NamedTuple{z::Number, w::VecNum}}`: The dual variable and the posterior that attains it, or `nothing` when the iteration does not reach the target.

# Algorithm

 1. Build the row of the current dual variable with [`ep_evar_grid_row`](@ref). Return `nothing` when the row is not finite.
 2. Tilt the prior with [`ep_row_tilt`](@ref), so that the row is tight. Return `nothing` when no probability vector makes the row tight.
 3. Compute the dual variable again with [`ep_evar`](@ref), as the minimiser at the tilted probabilities.
 4. Stop when the entropic value at risk of the tilted probabilities is within `tol` of the target. Return the dual variable and those probabilities.
 5. Return `nothing` if `iters` steps do not reach the target.

# Related

  - [`ep_row_tilt`](@ref)
  - [`ep_evar`](@ref)
  - [`ep_evar_grid`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equations 10 and 11, and footnote 9.
"""
function ep_evar_anchor(x::VecNum, w::VecNum, alpha::Number, rhs::Number, z::Number;
                        iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                        args::Tuple = (), kwargs::NamedTuple = (;),
                        zlo_frac::Option{<:Number} = nothing)
    for _ in 1:iters
        c, isc = ep_evar_grid_row(x, rhs, z)
        b = alpha * isc
        if !(all(isfinite, c) && isfinite(b))
            return nothing
        end
        q = ep_row_tilt(w, c, b; iters = tilt_iters)
        if isnothing(q)
            return nothing
        end
        res = ep_evar(x, q, alpha; args = args, kwargs = kwargs, zlo_frac = zlo_frac)
        if abs(res.evar - rhs) <= tol * abs(rhs)
            return (; z = res.z, w = q)
        end
        z = res.z
    end
    return nothing
end
"""
    ep_evar_grid(x::VecNum, w::VecNum, alpha::Number, op::Symbol, rhs::Number,
                 zstar::Number, pct::Number, K::Integer; iters::Integer = 50,
                 tol::Number = 1e-10, tilt_iters::Integer = 200, args::Tuple = (),
                 kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)

Build the grid of dual variables on which an entropic value-at-risk view is written.

[EPTail](@citet) builds a grid of an odd number `K` of equidistant points. It moves the dual variable of the prior up and down by a percentage, and `pct` is that percentage. The anchor is the library's own. For a view with an upper-bound half, the centre of the grid is the dual variable that [`ep_evar_anchor`](@ref) finds. For a lower-bound view, and for a view whose anchor does not converge, the centre is the dual variable of the prior. The grid of [`GridRelativisticValueatRiskView`](@ref) also translates its shift, but an EVaR grid does not need to. The shift of the primal programme of EVaR has a closed form in the target and the dual variable, and the row `exp((x - rhs) / z)` of [`ep_evar_grid_row`](@ref) already contains it.

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The prior probabilities, which sum to one.
  - `alpha`: The significance level.
  - `op`: The comparison operator of the view.
  - `rhs`: The target entropic value at risk.
  - `zstar`: The dual variable that attains the prior EVaR of the asset.
  - `pct`: The half-width of the grid, as a fraction of the dual variable at its centre.
  - `K`: The number of grid points.
  - `iters::Integer = 50`: The largest number of steps of the anchor.
  - `tol::Number = 1e-10`: The relative distance from the target at which the anchor stops.
  - `tilt_iters::Integer = 200`: The largest number of bisection steps of the tilt of one row, in [`ep_row_tilt`](@ref).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: The lower end of the bracket of the dual variable, as a fraction of the upper end, which goes to [`ep_evar`](@ref).

# Returns

  - `z::VecNum`: The dual variable of each grid point.

# Algorithm

 1. Use the dual variable of the prior as the centre.
 2. If the view has an upper-bound half, use the dual variable of [`ep_evar_anchor`](@ref) instead. Keep the dual variable of the prior if the anchor does not converge.
 3. Span the dual variable from `zc * (1 - pct)` to `zc * (1 + pct)` in `K` points. `K` is odd, so the centre is a point of the grid, and a grid of one point is the centre alone.

# Related

  - [`ep_evar_anchor`](@ref)
  - [`ep_evar_grid_row`](@ref)
  - [`ep_add_evar_view!`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equations 9 to 11, and footnote 9.
"""
function ep_evar_grid(x::VecNum, w::VecNum, alpha::Number, op::Symbol, rhs::Number,
                      zstar::Number, pct::Number, K::Integer; iters::Integer = 50,
                      tol::Number = 1e-10, tilt_iters::Integer = 200, args::Tuple = (),
                      kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)
    # EVaR is translation-equivariant and its dual variable is translation-invariant, so
    # the row of a grid point already carries the whole of the translation to the target.
    # The upper-bound half needs more: it reaches the target only where the grid holds the
    # dual variable the posterior itself attains, which is not the prior's. The anchor puts
    # the centre of the grid on it.
    zc = zstar
    anc = if op == :geq
        nothing
    else
        ep_evar_anchor(x, w, alpha, rhs, zstar; iters = iters, tol = tol,
                       tilt_iters = tilt_iters, args = args, kwargs = kwargs,
                       zlo_frac = zlo_frac)
    end
    if !isnothing(anc)
        zc = anc.z
    end
    # `K` is odd, so the centre is a grid point, and a grid of one point is the centre
    # alone. `range` refuses a single point between two ends that differ, so that case is
    # written out rather than left to raise from `Base`.
    return if isone(K)
        [zc]
    else
        collect(range(zc * (one(pct) - pct), zc * (one(pct) + pct); length = K))
    end
end
"""
    ep_rlvar_anchor(x::VecNum, w::VecNum, alpha::Number, kappa::Number, rhs::Number,
                    t::Number, z::Number; iters::Integer = 50, tol::Number = 1e-10,
                    tilt_iters::Integer = 200, args::Tuple = (),
                    kwargs::NamedTuple = (;),
                    bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Find the primal point of the relativistic value at risk that a posterior attains when it meets an upper-bound view.

A grid point states the view as one row. A posterior that makes the row tight reaches the target only if that grid point is the point that the posterior attains. `ep_rlvar_anchor` finds the pair that meets both conditions, and the grid of [`GridRelativisticValueatRiskView`](@ref) has this pair as its centre. The function calls no solver. [EPRLVaR](@citet) states no grid and no anchor, so both are the library's own, and both use the primal programme.

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The prior probabilities, which sum to one.
  - `alpha`: The significance level.
  - `kappa`: The deformation parameter, in `(0, 1)`.
  - `rhs`: The target relativistic value at risk.
  - `t`: The shift variable at the start of the iteration.
  - `z`: The dual variable at the start of the iteration.
  - `iters::Integer = 50`: The largest number of steps of the iteration.
  - `tol::Number = 1e-10`: The relative distance from the target at which the iteration stops.
  - `tilt_iters::Integer = 200`: The largest number of bisection steps of the tilt of one row, in [`ep_row_tilt`](@ref).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: The spans of the searches, which go to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Returns

  - `res::Option{@NamedTuple{t::Number, z::Number, w::VecNum}}`: The pair and the posterior that attains it, or `nothing` when the iteration does not reach the target.

# Algorithm

 1. Build the row of the current pair with [`ep_rlvar_grid_row`](@ref). Return `nothing` when the row is not finite.
 2. Tilt the prior with [`ep_row_tilt`](@ref), so that the row is tight. Return `nothing` when no probability vector makes the row tight.
 3. Compute the pair again with [`ep_rlvar`](@ref), as the minimiser at the tilted probabilities.
 4. Stop when the relativistic value at risk of the tilted probabilities is within `tol` of the target. Return the pair and those probabilities.
 5. Return `nothing` if `iters` steps do not reach the target.

# Related

  - [`ep_row_tilt`](@ref)
  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_grid`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
function ep_rlvar_anchor(x::VecNum, w::VecNum, alpha::Number, kappa::Number, rhs::Number,
                         t::Number, z::Number; iters::Integer = 50, tol::Number = 1e-10,
                         tilt_iters::Integer = 200, args::Tuple = (),
                         kwargs::NamedTuple = (;),
                         bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    for _ in 1:iters
        c, b = ep_rlvar_grid_row(x, rhs, t, z, alpha, kappa)
        if !(all(isfinite, c) && isfinite(b))
            return nothing
        end
        q = ep_row_tilt(w, c, b; iters = tilt_iters)
        if isnothing(q)
            return nothing
        end
        res = ep_rlvar(x, q, alpha, kappa; args = args, kwargs = kwargs, bracket = bracket)
        if abs(res.rlvar - rhs) <= tol * abs(rhs)
            return (; t = res.t, z = res.z, w = q)
        end
        t, z = res.t, res.z
    end
    return nothing
end
"""
    ep_rlvar_grid(x::VecNum, w::VecNum, alpha::Number, kappa::Number, op::Symbol,
                  rhs::Number, zstar::Number, pv::Number, pct::Number, K::Integer;
                  iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                  args::Tuple = (), kwargs::NamedTuple = (;),
                  bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Build the grid of primal points a relativistic value-at-risk view is written on.

[EPRLVaR](@citet) states the primal programme and no grid of its points. The grid, its translation and its anchor are the library's own, and they use the primal programme.

For a view with an upper-bound half, the centre of the grid is the pair that [`ep_rlvar_anchor`](@ref) finds. Each shift then minimises the objective at the posterior of that pair. For a lower-bound view, and for a view whose anchor does not converge, the centre is the dual variable of the prior. Each shift then minimises the objective under the prior probabilities, minus the distance from the prior value to the target. The relativistic value at risk and the shift that attains it are both translation-equivariant, so a posterior that moves the value to the target acts, to first order, like a translation of each loss by that distance.

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The prior probabilities, which sum to one.
  - `alpha`: The significance level.
  - `kappa`: The deformation parameter, in `(0, 1)`.
  - `op`: The comparison operator of the view.
  - `rhs`: The target relativistic value at risk.
  - `zstar`: The dual variable that attains the prior RLVaR of the asset.
  - `pv`: The prior RLVaR of the asset.
  - `pct`: The half-width of the grid, as a fraction of the dual variable at its centre.
  - `K`: The number of grid points.
  - `iters::Integer = 50`: The largest number of steps of the anchor.
  - `tol::Number = 1e-10`: The relative distance from the target at which the anchor stops.
  - `tilt_iters::Integer = 200`: The largest number of bisection steps of the tilt of one row, in [`ep_row_tilt`](@ref).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: The spans of the searches, which go to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Returns

  - `t::VecNum`: The shift variable of each grid point.
  - `z::VecNum`: The dual variable of each grid point.

# Algorithm

 1. Use the pair of the prior as the centre. Use the distance from the prior value to the target as the translation of each shift.
 2. If the view has an upper-bound half, use the pair and the posterior of [`ep_rlvar_anchor`](@ref) instead, with no translation. Keep the pair of the prior if the anchor does not converge.
 3. Span the dual variable from `zc * (1 - pct)` to `zc * (1 + pct)` in `K` points. `K` is odd, so the centre is a point of the grid, and a grid of one point is the centre alone.
 4. Minimise the objective over the shift at each point with [`ep_rlvar_shift`](@ref), and subtract the translation.

# Related

  - [`ep_rlvar_anchor`](@ref)
  - [`ep_rlvar_shift`](@ref)
  - [`ep_add_rlvar_view!`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
function ep_rlvar_grid(x::VecNum, w::VecNum, alpha::Number, kappa::Number, op::Symbol,
                       rhs::Number, zstar::Number, pv::Number, pct::Number, K::Integer;
                       iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                       args::Tuple = (), kwargs::NamedTuple = (;),
                       bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    lnk = kappa_log(inv(alpha * length(x)), kappa)
    # RLVaR is translation-equivariant, and so is the shift that attains it: subtracting
    # `delta` from every loss subtracts `delta` from both. A posterior that moves the RLVaR
    # to the target behaves, to first order, like that translation.
    zc, wc, delta = zstar, w, pv - rhs
    # Issue #530. To first order is not enough for the upper-bound half, which reaches the
    # target only where the grid holds the pair the posterior itself attains. That pair is
    # not the prior's, and over the fixture it reaches 8.4e5 times the prior's dual
    # variable, which no `pct` spans. The anchor puts the centre of the grid on it.
    anc = if op == :geq
        nothing
    else
        ep_rlvar_anchor(x, w, alpha, kappa, rhs,
                        ep_rlvar_shift(x, w, kappa, lnk, zstar; args = args,
                                       kwargs = kwargs, bracket = bracket).t - delta, zstar;
                        iters = iters, tol = tol, tilt_iters = tilt_iters, args = args,
                        kwargs = kwargs, bracket = bracket)
    end
    if !isnothing(anc)
        zc, wc, delta = anc.z, anc.w, zero(delta)
    end
    # `K` is odd, so the centre is a grid point, and a grid of one point is the centre
    # alone. `range` refuses a single point between two ends that differ, so that case is
    # written out rather than left to raise from `Base`.
    z = if isone(K)
        [zc]
    else
        collect(range(zc * (one(pct) - pct), zc * (one(pct) + pct); length = K))
    end
    t = [ep_rlvar_shift(x, wc, kappa, lnk, zk; args = args, kwargs = kwargs,
                        bracket = bracket).t - delta for zk in z]
    return t, z
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series and coefficient of every asset, the significance level and the target of a linear conditional value-at-risk view.

[`ep_add_cvar_view!`](@ref) makes one of these for each view that uses the linear formulation. From it, [`add_ep_tail_view!`](@ref) then writes the dual representation of CVaR into the model, one block for each asset. Every coefficient is positive. A view whose coefficients have both signs uses another formulation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearConditionalValueatRiskViewConstraint(x, coef, alpha, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equation 6.
"""
@concrete struct LinearConditionalValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_losses])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the ordered tail window of every asset that an integer conditional value-at-risk view names.

[`ep_add_cvar_view!`](@ref) makes one of these for each view that uses the integer formulation. Each entry of `ord` and of `x` belongs to one asset of the view, and `coef` holds the coefficient of that asset. A view of one asset has one entry, and a view of several assets has one entry for each asset. The window is in ascending order, so the largest loss is last, and the tail that the binary vector marks is at the end of the window. [EPTail](@citet) sorts the window in descending order, and its Equation 3 sets the probability of a marked observation to its posterior probability. The library lets the lowest marked observation hold a part of its probability, so the tail mass can equal `alpha`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IntegerConditionalValueatRiskViewConstraint(ord, x, coef, alpha, op, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equations 3 to 5.
"""
@concrete struct IntegerConditionalValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_ord])
    """
    ord
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_op])
    """
    op
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series and coefficient of every asset, the significance level and the target of a conic entropic value-at-risk view.

[`ep_add_evar_view!`](@ref) makes one of these for each view that uses the conic formulation. From it, [`add_ep_tail_view!`](@ref) then writes the relative entropy cone of the dual representation of EVaR under probabilities that are not uniform, one cone for each asset. Every coefficient is positive. A view whose coefficients have both signs uses another formulation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConicEntropicValueatRiskViewConstraint(x, coef, alpha, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 1, and Section 3.1. $(ref_dict[:no_original_source])
"""
@concrete struct ConicEntropicValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_losses])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the grid of dual variables from which an upper-bound or equality entropic value-at-risk view selects one point.

A lower-bound grid view is a set of rows on the posterior probabilities alone. These rows go into the constraint dictionary, and not into this tail view constraint. An equality view makes both parts. Its rows go into the dictionary, and its selector block goes into this constraint.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GridEntropicValueatRiskViewConstraint(x, z, alpha, rhs, M)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equations 10 and 11.
"""
@concrete struct GridEntropicValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_zgrid])
    """
    z
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:bigM])
    """
    M
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series and coefficient of every asset, the significance level, the deformation parameter and the target of a conic relativistic value-at-risk view.

[`ep_add_rlvar_view!`](@ref) makes one of these for each view that uses the conic formulation. From it, [`add_ep_tail_view!`](@ref) then writes the power cones of the dual representation of RLVaR, one pair for each observation of each asset. Every coefficient is positive. A view whose coefficients have both signs uses another formulation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConicRelativisticValueatRiskViewConstraint(x, coef, alpha, kappa, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 3.2, Equation 7.
"""
@concrete struct ConicRelativisticValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_losses])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_kappa])
    """
    kappa
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the grid of primal points from which an upper-bound or equality relativistic value-at-risk view selects one point.

A lower-bound grid view is a set of rows on the posterior probabilities alone. These rows go into the constraint dictionary, and not into this tail view constraint. An equality view makes both parts. Its rows go into the dictionary, and its selector block goes into this constraint. [EPRLVaR](@citet) states no upper-bound view of the relativistic value at risk. The grid and its selector are the library's own, and they use the primal programme as the entropic grid of [`GridEntropicValueatRiskViewConstraint`](@ref) does.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GridRelativisticValueatRiskViewConstraint(x, t, z, alpha, kappa, rhs, M)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
@concrete struct GridRelativisticValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_rlvar_tgrid])
    """
    t
    """
    $(field_dict[:ep_rlvar_zgrid])
    """
    z
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_kappa])
    """
    kappa
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:rlvar_bigM])
    """
    M
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the tail view constraints of the sequential convex formulations.

A sequential tail view constraint holds a view that reads as a lower bound, and it splits the assets of the view into two sides. The risk measures of the dual side are concave in the posterior probabilities, and each takes its exact dual block. The risk measures of the primal side have a negative coefficient, and each takes a linear upper bound read at a fixed posterior. The three subtypes differ only in the risk measure. [`add_ep_tail_view!`](@ref) and [`ep_refine_tail_view`](@ref) each have one method for the whole family. Each subtype has a method of [`ep_tail_dual_block!`](@ref) and of [`ep_tail_surrogate_row`](@ref) for its risk measure.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`SequentialConditionalValueatRiskViewConstraint`](@ref)
  - [`SequentialEntropicValueatRiskViewConstraint`](@ref)
  - [`SequentialRelativisticValueatRiskViewConstraint`](@ref)
  - [`ep_refine_tail_view`](@ref)
"""
abstract type AbstractSequentialTailViewConstraint <: AbstractEntropyPoolingTailView end
"""
$(DocStringExtensions.TYPEDEF)

Carries the two sides of a sequential conditional value-at-risk view, its surrogate row and its stopping rule.

[`ep_add_cvar_view!`](@ref) makes one of these for each view that uses [`SequentialConditionalValueatRiskView`](@ref), and reads its first surrogate row with [`ep_sequential_start`](@ref). [`ep_refine_tail_view`](@ref) then reads the row again at each posterior that [`entropy_pooling`](@ref) finds. [EPTail](@citet) states the dual formulation of the dual side. The sequential formulation, the surrogate row and the later reads are the library's own.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialConditionalValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, rhs, iters, tol)
    SequentialConditionalValueatRiskViewConstraint(; xd, cd, xp, cp, c, b, alpha, rhs, iters, tol)

Arguments and keywords correspond to the fields above.

# Related

  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_refine_tail_view`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equation 6.
"""
@concrete struct SequentialConditionalValueatRiskViewConstraint <:
                 AbstractSequentialTailViewConstraint
    """
    $(field_dict[:ep_seq_xd])
    """
    xd
    """
    $(field_dict[:ep_seq_cd])
    """
    cd
    """
    $(field_dict[:ep_seq_xp])
    """
    xp
    """
    $(field_dict[:ep_seq_cp])
    """
    cp
    """
    $(field_dict[:ep_seq_row])
    """
    c
    """
    $(field_dict[:ep_seq_b])
    """
    b
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the two sides of a sequential entropic value-at-risk view, its surrogate row, its stopping rule, and the settings of the search for the dual variable.

[`ep_add_evar_view!`](@ref) makes one of these for each view that uses [`SequentialEntropicValueatRiskView`](@ref), and reads its first surrogate row with [`ep_sequential_start`](@ref). [`ep_refine_tail_view`](@ref) then reads the row again at each posterior that [`entropy_pooling`](@ref) finds. Each read runs [`ep_evar`](@ref) once for each asset of the primal side, with `args`, `kwargs` and `zlo_frac`. [EPRLVaR](@citet) states the dual formulation of the dual side, and [EPTail](@citet) states the formula of the entropic value at risk that the primal side reads. The sequential formulation, the surrogate row and the later reads are the library's own.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialEntropicValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, rhs, iters, tol,
                                                args, kwargs, zlo_frac)
    SequentialEntropicValueatRiskViewConstraint(; xd, cd, xp, cp, c, b, alpha, rhs, iters, tol,
                                                args, kwargs, zlo_frac)

Arguments and keywords correspond to the fields above.

# Related

  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_refine_tail_view`](@ref)
  - [`ep_evar`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equation 7.
  - $(ref_dict[:EPRLVaR]) Section 2, Equation 1.
"""
@concrete struct SequentialEntropicValueatRiskViewConstraint <:
                 AbstractSequentialTailViewConstraint
    """
    $(field_dict[:ep_seq_xd])
    """
    xd
    """
    $(field_dict[:ep_seq_cd])
    """
    cd
    """
    $(field_dict[:ep_seq_xp])
    """
    xp
    """
    $(field_dict[:ep_seq_cp])
    """
    cp
    """
    $(field_dict[:ep_seq_row])
    """
    c
    """
    $(field_dict[:ep_seq_b])
    """
    b
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
    """
    $(field_dict[:optargs]) They go to the search of [`ep_evar`](@ref) that each read runs.
    """
    args
    """
    $(field_dict[:optkwargs]) They go to the same search as `args`.
    """
    kwargs
    """
    $(field_dict[:ep_tv_evar_zlo_frac])
    """
    zlo_frac
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the two sides of a sequential relativistic value-at-risk view, its surrogate row, its stopping rule, and the settings of the search for the primal pair.

[`ep_add_rlvar_view!`](@ref) makes one of these for each view that uses [`SequentialRelativisticValueatRiskView`](@ref), and reads its first surrogate row with [`ep_sequential_start`](@ref). [`ep_refine_tail_view`](@ref) then reads the row again at each posterior that [`entropy_pooling`](@ref) finds. Each read runs [`ep_rlvar`](@ref) once for each asset of the primal side, with `args`, `kwargs` and `bracket`. [EPRLVaR](@citet) states the dual and the primal formulations. The sequential formulation, the surrogate row and the later reads are the library's own.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialRelativisticValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, kappa, rhs,
                                                    iters, tol, args, kwargs, bracket)
    SequentialRelativisticValueatRiskViewConstraint(; xd, cd, xp, cp, c, b, alpha, kappa, rhs,
                                                    iters, tol, args, kwargs, bracket)

Arguments and keywords correspond to the fields above.

# Related

  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_refine_tail_view`](@ref)
  - [`ep_rlvar`](@ref)

# References

  - $(ref_dict[:EPRLVaR]) Section 2, Equations 4 and 5.
"""
@concrete struct SequentialRelativisticValueatRiskViewConstraint <:
                 AbstractSequentialTailViewConstraint
    """
    $(field_dict[:ep_seq_xd])
    """
    xd
    """
    $(field_dict[:ep_seq_cd])
    """
    cd
    """
    $(field_dict[:ep_seq_xp])
    """
    xp
    """
    $(field_dict[:ep_seq_cp])
    """
    cp
    """
    $(field_dict[:ep_seq_row])
    """
    c
    """
    $(field_dict[:ep_seq_b])
    """
    b
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_kappa])
    """
    kappa
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
    """
    $(field_dict[:optargs]) They go to the searches of [`ep_rlvar`](@ref) that each read runs.
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
end
"""
    ep_tail_dual_block!(model::JuMP.Model, pw, tv::AbstractEntropyPoolingTailView, x::VecNum,
                        sc1::Number)

Add the dual block of one asset's risk measure to an entropy pooling JuMP model, and return the expression that attains the measure.

The dual representation of each measure is the maximum of ``\\boldsymbol{\\nu}^{\\intercal} \\boldsymbol{x}`` over a set of weights that depends on the posterior probabilities. `ep_tail_dual_block!` adds that set to the model and returns the linear expression. The caller bounds the expression alone, or in a coefficient-weighted sum over several assets. The tail view constraint `tv` names the measure and holds its level. The fixed tail view constraint of a measure and its sequential tail view constraint use the same method.

The conditional value-at-risk block is Equation 6 of [EPTail](@cite). The entropic block is Equation 1 of [EPRLVaR](@cite), and the relativistic block is Equations 4 and 7 of [EPRLVaR](@cite). [EPRLVaR](@citet) prints the bound of its Equation 7 as `0 <= nu_j <= 0`. The library reads it as the bound of Equation 4, `0 <= nu_j <= 1`.

# JuMP formulation

## Variables

  - `pw`: $(math_dict[:ep_post_probs]) The caller gives it, and every entry below reads it.
  - `nu`: $(math_dict[:ep_tail_nu]) The conditional value-at-risk method gives it a lower bound of zero, and the two other methods bound it to ``[0, 1]``.
  - `tau`, `varsigma`: ``\\boldsymbol{\\tau}`` and ``\\boldsymbol{\\varsigma}``, two ``T \\times 1`` vectors with a lower bound of zero. The relativistic value-at-risk method makes them. The cones and the budget already imply both bounds. With the bounds stated, the solver reports `OPTIMAL` where it otherwise reports `SLOW_PROGRESS`.

## Expressions

  - The method returns ``\\sum_{j=1}^{T} \\nu_{j} x_{j}``, which has no name.

## Constraints

No row has a name. The conditional value-at-risk method, for [`LinearConditionalValueatRiskViewConstraint`](@ref) and [`SequentialConditionalValueatRiskViewConstraint`](@ref), registers two:

  - ``s_{c1} \\left(\\nu_{j} - \\dfrac{p_{j}}{\\alpha}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,T``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.

The entropic value-at-risk method, for [`ConicEntropicValueatRiskViewConstraint`](@ref) and [`SequentialEntropicValueatRiskViewConstraint`](@ref), registers two:

  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.
  - ``\\left(s_{c1} \\ln\\left(\\dfrac{1}{\\alpha}\\right),\\, s_{c1} \\boldsymbol{p},\\, s_{c1} \\boldsymbol{\\nu}\\right) \\in \\mathcal{K}_{\\mathrm{re}}(2T+1)``.

The relativistic value-at-risk method, for [`ConicRelativisticValueatRiskViewConstraint`](@ref) and [`SequentialRelativisticValueatRiskViewConstraint`](@ref), registers four:

  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\dfrac{\\tau_{j} - \\varsigma_{j}}{2\\kappa} - \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\right) \\leq 0``.
  - ``\\left(s_{c1} \\tau_{j},\\, s_{c1} T p_{j},\\, s_{c1} \\nu_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\left(\\dfrac{1}{1+\\kappa}\\right)``, ``\\forall\\, j = 1,\\ldots,T``.
  - ``\\left(s_{c1} \\nu_{j},\\, s_{c1} T p_{j},\\, s_{c1} \\varsigma_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)``, ``\\forall\\, j = 1,\\ldots,T``.

Where:

  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_tail_nu])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - ``\\boldsymbol{\\tau}``, ``\\boldsymbol{\\varsigma}``: ``T \\times 1`` vectors that hold the Kaniadakis entropy budget of ``\\boldsymbol{\\nu}``.
  - ``\\mathcal{K}_{\\mathrm{re}}(1 + 2T) = \\{(u,\\, \\boldsymbol{v},\\, \\boldsymbol{s}) : u \\geq \\sum_{j=1}^{T} s_{j} \\ln(s_{j} / v_{j})\\}``: The relative entropy cone.
  - ``\\mathcal{K}_{\\mathrm{pow}}(\\pi) = \\{(a, b, c) : a^{\\pi} b^{1-\\pi} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}``: The power cone.

## Relaxation

$(val_dict[:relax])

The block is exact. The set that it adds is the dual description of the measure, so the largest value of the returned expression is the measure of the asset under `pw`.

# Arguments

  - `model`: The entropy pooling JuMP model.
  - `pw`: The vector of posterior probability variables.
  - `tv`: The tail view constraint. It names the measure and holds its level. For the relativistic method, it also holds the deformation parameter.
  - `x`: The loss series of the asset.
  - `sc1`: The constraint scale factor.

# Returns

  - `expr::JuMP.AffExpr`: The expression ``\\sum_{j=1}^{T} \\nu_{j} x_{j}``.

# Related

  - [`add_ep_tail_view!`](@ref)
  - [`LinearConditionalValueatRiskViewConstraint`](@ref)
  - [`ConicEntropicValueatRiskViewConstraint`](@ref)
  - [`ConicRelativisticValueatRiskViewConstraint`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equation 6.
  - $(ref_dict[:EPRLVaR]) Section 2, Equations 1 and 4, and Section 3.2, Equation 7.
"""
function ep_tail_dual_block!(model::JuMP.Model, pw,
                             tv::Union{LinearConditionalValueatRiskViewConstraint,
                                       SequentialConditionalValueatRiskViewConstraint},
                             x::VecNum, sc1::Number)
    alpha = tv.alpha
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0)
    JuMP.@constraints(model, begin
                          [j = 1:T], sc1 * (nu[j] - pw[j] / alpha) <= 0
                          sc1 * (sum(nu) - one(alpha)) == 0
                      end)
    return LinearAlgebra.dot(nu, x)
end
function ep_tail_dual_block!(model::JuMP.Model, pw,
                             tv::Union{ConicEntropicValueatRiskViewConstraint,
                                       SequentialEntropicValueatRiskViewConstraint},
                             x::VecNum, sc1::Number)
    alpha = tv.alpha
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0, upper_bound = 1)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(nu) - one(alpha)) == 0
                          [sc1 * log(inv(alpha)); sc1 * pw; sc1 * nu] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
    return LinearAlgebra.dot(nu, x)
end
function ep_tail_dual_block!(model::JuMP.Model, pw,
                             tv::Union{ConicRelativisticValueatRiskViewConstraint,
                                       SequentialRelativisticValueatRiskViewConstraint},
                             x::VecNum, sc1::Number)
    (; alpha, kappa) = tv
    T = length(x)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik2 = inv(2 * kappa)
    lnk = kappa_log(inv(alpha * T), kappa)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0, upper_bound = 1)
    # Both bounds are implied by the cones and the budget: the first slot of a power cone
    # is non-negative, and the budget is loosest at the largest `varsigma` the second cone
    # allows, which is non-negative. Stating them is what turns a `SLOW_PROGRESS` report
    # into an `OPTIMAL` one.
    tau = JuMP.@variable(model, [1:T], lower_bound = 0)
    varsigma = JuMP.@variable(model, [1:T], lower_bound = 0)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(nu) - one(alpha)) == 0
                          sc1 * (sum(tau - varsigma) * ik2 - lnk) <= 0
                          [j = 1:T],
                          [sc1 * tau[j], sc1 * T * pw[j], sc1 * nu[j]] in
                          JuMP.MOI.PowerCone(inv(opk))
                          [j = 1:T],
                          [sc1 * nu[j], sc1 * T * pw[j], sc1 * varsigma[j]] in
                          JuMP.MOI.PowerCone(omk)
                      end)
    return LinearAlgebra.dot(nu, x)
end
"""
    ep_var_multiplier(x::VecNum, w::VecNum, alpha::Number)

Find the largest minimiser of the primal of the conditional value at risk of a loss series under observation probabilities.

# Mathematical definition

The conditional value at risk is the minimum over ``\\eta`` of ``\\eta + \\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} (x_{j} - \\eta)^{+}``. This function of ``\\eta`` is convex and piecewise linear, and its kinks are at the losses. Its largest minimiser is the loss at which the tail mass first reaches ``\\alpha``:

```math
\\eta^{\\star} = x_{(s)}\\,, \\quad s = \\min\\left\\{k : \\sum_{i=1}^{k} w_{(i)} \\geq \\alpha\\right\\}\\,,
```

with the losses sorted in descending order, ``x_{(1)} \\geq x_{(2)} \\geq \\ldots``.

When the tail mass of a loss is exactly ``\\alpha``, the minimisers form an interval. [`ValueatRisk`](@ref) takes the smallest point of the interval, so in this case the two values are adjacent losses. Every minimiser gives the same conditional value at risk, and a bound read at any of them is tight at `w`.

Where:

  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

# Arguments

  - `x`: The loss series (`-returns`).
  - `w`: The observation probabilities. The function normalises them to sum to one.
  - `alpha`: The significance level.

# Returns

  - `eta::Number`: The largest minimiser, one of the losses of `x`.

# Related

  - [`ep_tail_surrogate_row`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`ValueatRisk`](@ref)
"""
function ep_var_multiplier(x::VecNum, w::VecNum, alpha::Number)
    o = sortperm(x; rev = true)
    sw = sum(w)
    cw = zero(promote_type(eltype(w), typeof(alpha)))
    for j in o
        cw += w[j] / sw
        if cw >= alpha
            return x[j]
        end
    end
    return x[o[end]]
end
"""
    ep_tail_surrogate_row(tv::AbstractSequentialTailViewConstraint, x::VecNum, w::VecNum)

Read the linear upper bound of one asset's risk measure at a posterior.

Each sequential formulation bounds the measure of an asset on its primal side by an affine function of the posterior probabilities, ``r_{0} + \\boldsymbol{r}^{\\intercal} \\boldsymbol{p}``. The bound is tight at the probabilities `w` at which the function reads it. So the value ``r_{0} + \\boldsymbol{r}^{\\intercal} \\boldsymbol{w}`` is the measure of the asset under `w`.

Neither [EPTail](@cite) nor [EPRLVaR](@cite) states the row. The library builds it as the tangent of the primal programme of the measure. This programme is Equation 7 of [EPTail](@cite) for the entropic value at risk, and Equation 5 of [EPRLVaR](@cite) for the relativistic value at risk. The conditional value-at-risk row reads the minimisation of the conditional value at risk over its threshold, which neither work states.

# Mathematical definition

The conditional value-at-risk method reads the primal at the value at risk ``\\eta`` of [`ep_var_multiplier`](@ref):

```math
r_{0} = \\eta\\,, \\quad r_{j} = \\dfrac{(x_{j} - \\eta)^{+}}{\\alpha}\\,.
```

The entropic value-at-risk method reads the dual variable ``z`` and the value of [`ep_evar`](@ref), and takes the tangent of the concave primal at `w`:

```math
r_{0} = \\mathrm{EVaR}_{\\alpha}(X) - z\\,, \\quad r_{j} = \\dfrac{z e^{x_{j}/z}}{\\sum_{k=1}^{T} w_{k} e^{x_{k}/z}}\\,.
```

The relativistic value-at-risk method reads the pair ``(t, z)`` of [`ep_rlvar`](@ref), at which the primal is linear in the probabilities:

```math
r_{0} = t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\,, \\quad r_{j} = T \\varphi_{\\kappa}(t - x_{j},\\, z)\\,.
```

Where:

  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:evar_stat])
  - ``\\eta``: The value at risk of the loss series under ``\\boldsymbol{w}``.
  - ``z``: The dual variable that attains the measure under ``\\boldsymbol{w}``.
  - ``t``: The shift that attains the relativistic value at risk under ``\\boldsymbol{w}``.

# Arguments

  - `tv`: The sequential tail view constraint. It names the measure, and holds its level and the settings of its search.
  - `x`: The loss series of the asset (`-returns`).
  - `w`: The observation probabilities at which the function reads the bound, which sum to one.

# Returns

  - `r::VecNum`: The coefficients of the bound, one for each observation.
  - `r0::Number`: The constant of the bound.

# Related

  - [`ep_refine_tail_view`](@ref)
  - [`ep_var_multiplier`](@ref)
  - [`ep_evar`](@ref)
  - [`ep_rlvar`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.3, Equation 7.
  - $(ref_dict[:EPRLVaR]) Section 2, Equation 5.
"""
function ep_tail_surrogate_row(tv::SequentialConditionalValueatRiskViewConstraint,
                               x::VecNum, w::VecNum)
    alpha = tv.alpha
    eta = ep_var_multiplier(x, w, alpha)
    return max.(x .- eta, zero(eta)) ./ alpha, eta
end
function ep_tail_surrogate_row(tv::SequentialEntropicValueatRiskViewConstraint, x::VecNum,
                               w::VecNum)
    (; alpha, args, kwargs, zlo_frac) = tv
    res = ep_evar(x, w, alpha; args = args, kwargs = kwargs, zlo_frac = zlo_frac)
    z = res.z
    # The exponentials are shifted by the largest loss, so the ratio does not overflow at
    # a small dual variable.
    r = exp.((x .- maximum(x)) ./ z)
    r .*= z / LinearAlgebra.dot(w, r)
    return r, res.evar - z
end
function ep_tail_surrogate_row(tv::SequentialRelativisticValueatRiskViewConstraint,
                               x::VecNum, w::VecNum)
    (; alpha, kappa, args, kwargs, bracket) = tv
    res = ep_rlvar(x, w, alpha, kappa; args = args, kwargs = kwargs, bracket = bracket)
    T = length(x)
    lnk = kappa_log(inv(alpha * T), kappa)
    return T .* ep_rlvar_tail.(res.t .- x, res.z, kappa), res.t + res.z * lnk
end
function ep_refine_iters(tv::AbstractSequentialTailViewConstraint)
    return tv.iters
end
"""
    ep_refine_tail_view(tv::AbstractSequentialTailViewConstraint, w::VecNum)

Read the surrogate row of a sequential tail view again at a posterior, and tell whether the old row was already tight there.

# Algorithm

 1. Return `tv` and `true` when the primal side is empty. The view is then convex, its dual blocks are exact, and no row needs a new read.
 2. Clamp `w` at zero from below, and normalise it to sum to one. A conic solver returns a posterior whose smallest entries are a rounding error below zero, and the searches behind the rows refuse a negative probability.
 3. For each asset of the primal side, read its bound at `w` with [`ep_tail_surrogate_row`](@ref). Add the coefficient-weighted bounds into a new row `c`, `b`. The value of the new row at `w` is the coefficient-weighted sum of the measures at `w`.
 4. Calculate the gap between the old row of `tv` and the new row at `w`. The old row bounds the same sum from the same side, so the gap is the slack that the last solve left.
 5. Return `tv` with the new row. Also return `true` if the gap is at most `tol` times the larger of the absolute target and the largest absolute loss of the primal side.

# Arguments

  - `tv`: The sequential tail view constraint.
  - `w`: The posterior probabilities of the last solve.

# Returns

  - `tv::AbstractSequentialTailViewConstraint`: The tail view constraint with its row read again at `w`.
  - `tight::Bool`: `true` if the row that `tv` held before the call was tight at `w`.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_tail_surrogate_row`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)
"""
function ep_refine_tail_view(tv::AbstractSequentialTailViewConstraint, w::VecNum)
    (; xp, cp, c, b, rhs, tol) = tv
    if isempty(xp)
        return tv, true
    end
    # A conic solver returns a posterior with entries a rounding error below zero, which
    # the searches behind the rows refuse, so the posterior is clamped before it is read.
    wi = max.(w, zero(eltype(w)))
    wi ./= sum(wi)
    cn = zero(c)
    bn = zero(b)
    scale = abs(rhs)
    for (x, ci) in zip(xp, cp)
        r, r0 = ep_tail_surrogate_row(tv, x, wi)
        cn .+= ci .* r
        bn += ci * r0
        scale = max(scale, maximum(abs, x))
    end
    gap = abs((LinearAlgebra.dot(cn, wi) + bn) - (LinearAlgebra.dot(c, wi) + b))
    return rebuild_estimator(tv, (; c = cn, b = bn)), gap <= tol * scale
end
"""
    ep_sequential_start(tv::AbstractSequentialTailViewConstraint, w::VecNum)

Read the first surrogate row of a sequential tail view at probabilities under which the row can meet the view.

A linear upper bound is tight where the function reads it. But the value of the row cannot fall below the smallest value of the row over the simplex. If the target is further below the prior than that, the first solve has no feasible point.

This happens with the relativistic value at risk. At the pair of the prior, its bound has a floor at the shift plus the deformed logarithm, and only the tail term above the floor can change. `ep_sequential_start` moves the multipliers toward the target with a chain of exponential tilts before a solver runs. The anchors of [`GridEntropicValueatRiskView`](@ref) and [`GridRelativisticValueatRiskView`](@ref) use the same method to find the centres of their grids. The function calls no solver.

# Algorithm

 1. Normalise `w` to sum to one, and read the row at `w` with [`ep_refine_tail_view`](@ref). Return `tv` when its primal side is empty, because the view is then convex and the row is empty.
 2. Calculate `need`, the value that the row must reach. It is the target, minus the constant of the row, minus the coefficient-weighted sum of the measures of the dual side under the current probabilities. [`ep_tail_surrogate_row`](@ref) gives each measure.
 3. Return `tv` when the largest coefficient of the row is at least `need`. A probability vector then meets the row, and the solve can start.
 4. Otherwise, tilt the probabilities with [`ep_row_tilt`](@ref), so that the row goes nine tenths of the way from its current value to its largest coefficient. Read the row again at the tilted probabilities, and go back to step 2.
 5. Do at most `iters` steps. When the tilt does not exist, return `tv` with its last row.

# Arguments

  - `tv`: The sequential tail view constraint, with any row.
  - `w`: The prior probability weights.

# Returns

  - `tv::AbstractSequentialTailViewConstraint`: The tail view constraint with its first row.

# Related

  - [`ep_refine_tail_view`](@ref)
  - [`ep_tail_surrogate_row`](@ref)
  - [`ep_row_tilt`](@ref)
  - [`ep_evar_anchor`](@ref)
  - [`ep_rlvar_anchor`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)
"""
function ep_sequential_start(tv::AbstractSequentialTailViewConstraint, w::VecNum)
    (; xd, cd, xp, iters) = tv
    wi = w ./ sum(w)
    tv, _ = ep_refine_tail_view(tv, wi)
    if isempty(xp)
        return tv
    end
    for _ in 1:iters
        (; c, b, rhs) = tv
        need = rhs - b
        for (x, ci) in zip(xd, cd)
            r, r0 = ep_tail_surrogate_row(tv, x, wi)
            need -= ci * (r0 + LinearAlgebra.dot(r, wi))
        end
        cmax = maximum(c)
        if need <= cmax
            return tv
        end
        v = LinearAlgebra.dot(c, wi)
        q = ep_row_tilt(wi, c, v + 0.9 * (cmax - v))
        if isnothing(q)
            return tv
        end
        wi = q
        tv, _ = ep_refine_tail_view(tv, wi)
    end
    return tv
end
"""
    add_ep_tail_view!(model::JuMP.Model, pw, tv::AbstractEntropyPoolingTailView,
                      sc1::Number)

Add the variables and constraints of one tail view to an entropy pooling JuMP model.

Every conditional, entropic and relativistic value-at-risk view goes into the model through `add_ep_tail_view!`. Each formulation has its own method, which dispatches on the tail view constraint that [`ep_tail_views!`](@ref) made. The three dual tail view constraints share one method, and the three sequential tail view constraints share another. Both methods write one dual block for each asset with [`ep_tail_dual_block!`](@ref), and they differ in the row that bounds the sum. [EPTail](@citet) and [EPRLVaR](@citet) state the dual, the integer and the entropic grid formulations. The sequential formulation and the grid formulation of the relativistic value at risk are the library's own.

# JuMP formulation

This section covers the five methods and every entry that each method registers. No entry has a name. One model holds one block for each view, so a name collides at the second view of a family. Nothing reads these entries back by name.

## Variables

  - `pw`: $(math_dict[:ep_post_probs]) The caller gives it, and every entry below reads it.
  - `nu`, `tau`, `varsigma`: [`ep_tail_dual_block!`](@ref) makes them once for each asset of a dual tail view constraint, and once for each asset of the dual side of a sequential tail view constraint. Its `# JuMP formulation` section names them.
  - `y`, `q`: ``\\boldsymbol{y}`` and ``\\boldsymbol{q}``, two ``\\bar{s} \\times 1`` vectors that the [`IntegerConditionalValueatRiskViewConstraint`](@ref) method makes once for each asset. `y` is binary, and `q` has a lower bound of zero.
  - `y`: ``\\boldsymbol{y}``, a binary ``K \\times 1`` vector that the [`GridEntropicValueatRiskViewConstraint`](@ref) and [`GridRelativisticValueatRiskViewConstraint`](@ref) methods make. It selects the grid point at which the posterior meets the view.

## Expressions

  - ``\\varepsilon``: The left hand side of the view, which each method builds once and which has no name. For the dual tail view constraints, it is the coefficient-weighted sum of the expressions that [`ep_tail_dual_block!`](@ref) returns for each asset. For the integer tail view constraint, it is the coefficient-weighted sum of the tail sums of each asset. For the sequential tail view constraints, it is the coefficient-weighted sum of the expressions of the dual side, plus the surrogate row ``b + \\boldsymbol{c}^{\\intercal} \\boldsymbol{p}``.

## Constraints

The method of the three dual tail view constraints, [`LinearConditionalValueatRiskViewConstraint`](@ref), [`ConicEntropicValueatRiskViewConstraint`](@ref) and [`ConicRelativisticValueatRiskViewConstraint`](@ref), registers one block of [`ep_tail_dual_block!`](@ref) for each asset that the view names, and one row:

  - ``s_{c1} \\left(\\bar{c} - \\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j}\\right) \\leq 0``.

The method of the three sequential tail view constraints, [`SequentialConditionalValueatRiskViewConstraint`](@ref), [`SequentialEntropicValueatRiskViewConstraint`](@ref) and [`SequentialRelativisticValueatRiskViewConstraint`](@ref), registers one block of [`ep_tail_dual_block!`](@ref) for each asset of the dual side, and one row over both sides. When the largest absolute coefficient of the surrogate row is larger than one, the method divides the row by it. So the coefficients of the surrogate row stay in ``[-1, 1]``, at a small dual variable of a relativistic value at risk too:

  - ``s_{c1} \\left(\\bar{c} - \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} - b - \\sum_{j=1}^{T} c_{j} p_{j}\\right) \\Big/ \\max\\left(1, \\lVert \\boldsymbol{c} \\rVert_{\\infty}\\right) \\leq 0``.

The [`IntegerConditionalValueatRiskViewConstraint`](@ref) method registers five rows for each asset that the view names, over the window of the ``\\bar{s}`` largest losses of that asset:

  - ``s_{c1} \\left(q_{j} - y_{j}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(q_{j} - p_{[j]}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(p_{[j]} - (1 - y_{j-1}) - q_{j}\\right) \\leq 0``, ``\\forall\\, j = 2,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(y_{j} - y_{j+1}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}-1``.
  - ``s_{c1} \\left(\\sum_{j=1}^{\\bar{s}} q_{j} - \\alpha\\right) = 0``.

The method also registers one row on ``\\varepsilon = \\sum_{i} \\dfrac{\\gamma_{i}}{\\alpha} \\sum_{j=1}^{\\bar{s}} q_{i,\\,j} x_{i,\\,[j]}``. The operator of the view selects one of these three rows:

  - ``s_{c1} \\left(\\varepsilon - \\bar{c}\\right) = 0`` under `:eq`.
  - ``s_{c1} \\left(\\bar{c} - \\varepsilon\\right) \\leq 0`` under `:geq`.
  - ``s_{c1} \\left(\\varepsilon - \\bar{c}\\right) \\leq 0`` under `:leq`.

The [`GridEntropicValueatRiskViewConstraint`](@ref) and [`GridRelativisticValueatRiskViewConstraint`](@ref) methods each register the rows below. Each method divides the row of each grid point by its bound:

  - ``s_{c1} \\left(\\sum_{k=1}^{K} y_{k} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\dfrac{c_{k,\\,j}}{b_{k}} p_{j} - 1 - M M_{k} (1 - y_{k})\\right) \\leq 0``, ``\\forall\\, k = 1,\\ldots,K``, with ``M_{k} = \\max_{j} c_{k,\\,j} / b_{k} - 1``.

Where:

  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:cvar_target])
  - ``x_{i,\\,j}``: The loss of asset ``i`` at observation ``j``, which is the negated return.
  - ``\\boldsymbol{\\nu}_{i}``: The ``T \\times 1`` vector of weights that attains the measure of asset ``i``, from its block of [`ep_tail_dual_block!`](@ref).
  - ``\\bar{c}``: The target of the view, for each of the three measures. A sequential tail view constraint holds the view as a lower bound, so its target has the sign of that orientation.
  - ``\\bar{s}``: The length of the window of the largest losses of one asset, from [`ep_sbar`](@ref).
  - ``x_{[j]}``, ``p_{[j]}``: The loss and the posterior probability of the observation at position ``j`` of that window, which is in ascending order.
  - ``\\gamma_{i}``: The coefficient that the view gives asset ``i``.
  - ``\\mathcal{P}``: The assets on the dual side of a sequential tail view constraint.
  - ``b``, ``\\boldsymbol{c}``: The surrogate row that a sequential tail view constraint holds, from [`ep_tail_surrogate_row`](@ref).
  - ``\\boldsymbol{y}``: A binary vector. In the integer conditional value-at-risk method, it marks the tail of the window of one asset. In the two grid methods, it selects one grid point.
  - ``\\boldsymbol{q}``: The ``\\bar{s} \\times 1`` vector that holds the tail mass of each observation of the window. The tail mass is ``p_{[j]}`` above the lowest marked observation, a part of ``p_{[j]}`` at that observation, and zero below it.
  - ``\\varepsilon``: The left hand side of an integer conditional value-at-risk view, which is the coefficient-weighted sum of the posterior CVaRs of the assets.
  - ``K``: The number of grid points that the tail view constraint holds.
  - ``c_{k,\\,j}``: The scaled coefficient of observation ``j`` at grid point ``k``, from [`ep_evar_grid_row`](@ref) or [`ep_rlvar_grid_row`](@ref).
  - ``b_{k}``: The scaled bound of grid point ``k``, from the same two functions. For the entropic value at risk, it is ``\\alpha`` times the reciprocal that [`ep_evar_grid_row`](@ref) returns. For the relativistic value at risk, it is the target that [`ep_rlvar_grid_row`](@ref) returns.
  - ``M_{k}``: The smallest big-M constant that releases the row of grid point ``k``.
  - ``M``: The big-M multiplier that the grid tail view constraint holds.

## Relaxation

$(val_dict[:relax])

The two grid methods and the sequential method bound the statistic. The dual method is exact. The integer method is exact for each posterior whose tail of mass ``\\alpha`` is inside the window of the ``\\bar{s}`` largest losses. When ``\\bar{s} = T``, this holds for every posterior.

The sequential method is a restriction of the view, and each re-solve makes it tighter.

 1. **Direction.** The surrogate row bounds each measure of the primal side from above, and each of these measures has a negative coefficient. So the value of the row is at or below the left hand side of the view. A posterior that meets the row meets the view.

 2. **Quantity.** The coefficient-weighted sum of the posterior measures that the view names, which is a statistic of `pw`.

 3. **Tightness.** The row is tight at the posterior at which the function read it. [`entropy_pooling`](@ref) reads the row again at each posterior with [`ep_refine_tail_view`](@ref), and solves again. It stops when the slack is within the `tol` of the tail view constraint, or after `iters` re-solves. At the fixed point the posterior meets the view to that tolerance. Before the fixed point, the posterior meets the view with slack.

The two grid methods are also a restriction of the view.

 1. **Direction.** Every grid point is a feasible point of the primal programme of the statistic, so its row bounds the statistic from above. The block asks one grid point to hold, so the posterior statistic is at or below the target. The encoding can only be tighter than the view, so a posterior that meets the block meets the view.

 2. **Quantity.** The posterior entropic value at risk under the [`GridEntropicValueatRiskViewConstraint`](@ref) method, and the posterior relativistic value at risk under the [`GridRelativisticValueatRiskViewConstraint`](@ref) method. Both are statistics of `pw`.

 3. **Tightness.** The bound is tight when the grid holds the point that the posterior attains. [`ep_evar_anchor`](@ref) and [`ep_rlvar_anchor`](@ref) put the centre of the grid at that point. When the anchor does not converge, the grid uses the point of the prior as its centre, and the posterior statistic can be strictly below the target. In that case, increase `pct` or `K`.

Each method divides each grid row by its bound, so that the right hand side is one. At a small dual variable, the bound ``b_{k}`` is far below the feasibility tolerance of a solver. At that scale the whole row is inside the tolerance, and the solver accepts any posterior on it. The selector can then select that grid point when the posterior does not meet the view.

``M M_{k}`` releases the rows of the grid points that the selector does not select. ``\\boldsymbol{p}`` sums to one, so the left hand side of row ``k`` is never larger than ``\\max_{j} c_{k,\\,j} / b_{k}``, and ``M_{k}`` is the smallest constant that releases the row. An ``M`` below one can remove a posterior that the view accepts, so ``M \\geq 1``. An ``M`` above one gives the released rows a margin for a posterior whose sum is one only to the tolerance of the solver. But it also increases the slack that the integrality tolerance opens on the selected row, so the default is ``M = 1``.

The lower-bound half of a grid view is a relaxation, and this function does not receive it. The view needs the primal objective to be at least the target at every dual variable, and the rows ask for this only at the ``K`` grid points. So the posterior statistic, which is the minimum over all dual variables, can be below the target. [`ep_add_evar_view!`](@ref) and [`ep_add_rlvar_view!`](@ref) add those rows to the entropy pooling constraint dictionary, not to the model.

# Arguments

  - `model`: The entropy pooling JuMP model.
  - `pw`: The vector of posterior probability variables.
  - `tv`: The tail view constraint.
  - `sc1`: The constraint scale factor.

# Returns

  - `nothing`: The function mutates `model` in-place.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`ep_tail_dual_block!`](@ref)
  - [`LinearConditionalValueatRiskViewConstraint`](@ref)
  - [`IntegerConditionalValueatRiskViewConstraint`](@ref)
  - [`ConicEntropicValueatRiskViewConstraint`](@ref)
  - [`GridEntropicValueatRiskViewConstraint`](@ref)
  - [`ConicRelativisticValueatRiskViewConstraint`](@ref)
  - [`GridRelativisticValueatRiskViewConstraint`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`entropy_pooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equations 3 to 6, and Section 2.3, Equations 10 and 11.
  - $(ref_dict[:EPRLVaR]) Section 3.2, Equation 7.
"""
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::Union{LinearConditionalValueatRiskViewConstraint,
                                     ConicEntropicValueatRiskViewConstraint,
                                     ConicRelativisticValueatRiskViewConstraint},
                           sc1::Number)
    (; x, coef, rhs) = tv
    expr = JuMP.AffExpr()
    for (xi, ci) in zip(x, coef)
        JuMP.add_to_expression!(expr, ci, ep_tail_dual_block!(model, pw, tv, xi, sc1))
    end
    JuMP.@constraint(model, sc1 * (rhs - expr) <= 0)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw, tv::AbstractSequentialTailViewConstraint,
                           sc1::Number)
    (; xd, cd, c, b, rhs) = tv
    expr = JuMP.AffExpr(b)
    for (xi, ci) in zip(xd, cd)
        JuMP.add_to_expression!(expr, ci, ep_tail_dual_block!(model, pw, tv, xi, sc1))
    end
    JuMP.add_to_expression!(expr, LinearAlgebra.dot(c, pw))
    # The tail function of the relativistic measure can reach the thousands at a small
    # dual variable, so the row is divided by its largest coefficient, as the grid rows are.
    # Below one the row is left alone, so a CVaR row keeps its natural units.
    s = max(one(eltype(c)), maximum(abs, c; init = zero(eltype(c))))
    JuMP.@constraint(model, sc1 * (rhs - expr) / s <= 0)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::IntegerConditionalValueatRiskViewConstraint, sc1::Number)
    (; ord, x, coef, alpha, op, rhs) = tv
    expr = JuMP.AffExpr()
    for (ordi, xi, ci) in zip(ord, x, coef)
        sb = length(ordi)
        y = JuMP.@variable(model, [1:sb], binary = true)
        q = JuMP.@variable(model, [1:sb], lower_bound = 0)
        JuMP.@constraints(model,
                          begin
                              [j = 1:sb], sc1 * (q[j] - y[j]) <= 0
                              [j = 1:sb], sc1 * (q[j] - pw[ordi[j]]) <= 0
                              [j = 2:sb],
                              sc1 * (pw[ordi[j]] - (one(alpha) - y[j - 1]) - q[j]) <= 0
                              [j = 1:(sb - 1)], sc1 * (y[j] - y[j + 1]) <= 0
                              sc1 * (sum(q) - alpha) == 0
                          end)
        JuMP.add_to_expression!(expr, ci / alpha, LinearAlgebra.dot(q, xi))
    end
    if op == :eq
        JuMP.@constraint(model, sc1 * (expr - rhs) == 0)
    elseif op == :geq
        JuMP.@constraint(model, sc1 * (rhs - expr) <= 0)
    else
        JuMP.@constraint(model, sc1 * (expr - rhs) <= 0)
    end
    return nothing
end
