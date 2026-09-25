"""
    set_initial_w!(args...)
    set_initial_w!(w::VecNum, wi::VecNum)

Set initial (warm-start) values for portfolio weight variables in the JuMP model.

The no-op fallback does nothing when `wi` is not provided. The two-argument method sets JuMP start values for each weight variable.

# Arguments

  - `w::VecNum`: Vector of JuMP weight variables.
  - `wi::VecNum`: Vector of initial weight values.

# Returns

  - `nothing`.

# Related

  - [`set_w!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_initial_w!(args...)
    return nothing
end
function set_initial_w!(w::VecNum, wi::VecNum)
    @argcheck(length(wi) == length(w),
              DimensionMismatch("wi ($(length(wi))) must match w ($(length(w)))"))
    JuMP.set_start_value.(w, wi)
    return nothing
end
"""
    set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})

Create portfolio weight variables in the JuMP model and optionally set initial values.

Registers a vector of weight variables `w` of length `size(X, 2)` in the model. If `wi` is provided, sets the initial values via [`set_initial_w!`](@ref).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix (shape: observations × assets).
  - `wi`: Optional initial weight values.

# Returns

  - `nothing`.

# Related

  - [`set_initial_w!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})
    JuMP.@variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
"""
    process_model(model, retcode)

Extract the solution from an optimised JuMP model based on the return code.

On success, extracts the optimised weights from the model. On failure, returns an empty solution.

# Arguments

  - `model`: Optimised JuMP model.
  - `retcode`: Optimisation return code ([`OptimisationSuccess`](@ref) or [`OptimisationFailure`](@ref)).

# Returns

  - Solution object.

# Related

  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function process_model(model::JuMP.Model, ::OptimisationSuccess)
    k = JuMP.value(model[:k])
    ik = !iszero(k) ? inv(k) : 1
    w = JuMP.value.(model[:w]) * ik
    return JuMPOptimisationSolution(; w = w)
end
function process_model(model::JuMP.Model, ::OptimisationFailure)
    return JuMPOptimisationSolution(;
                                    w = fill(convert(JuMP.value_type(typeof(model)), NaN),
                                             length(model[:w])))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Attempt to solve the JuMP model using each solver in `opt.opt.slv` in order.

Tries each solver sequentially, checking feasibility and finite non-zero weights. Returns a `(retcode, solution)` tuple where `retcode` is [`OptimisationSuccess`](@ref) or [`OptimisationFailure`](@ref) and `solution` is a [`JuMPOptimisationSolution`](@ref).

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`JuMPOptimisationSolution`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function optimise_JuMP_model!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                              datatype::DataType = Float64)
    trials = Dict()
    success = false
    for solver in opt.opt.slv
        try
            JuMP.set_optimizer(model, solver.solver; add_bridges = solver.add_bridges)
        catch err
            trials[solver.name] = Dict(:set_optimizer => err)
            continue
        end
        set_solver_attributes(model, solver.settings)
        try
            JuMP.optimize!(model)
        catch err
            trials[solver.name] = Dict(:optimize! => err)
            continue
        end
        trial = Dict{Symbol, Any}(:settings => solver.settings)
        try
            JuMP.assert_is_solved_and_feasible(model; solver.check_sol...)
            all_finite_weights = all(isfinite, JuMP.value.(model[:w]))
            all_non_zero_weights = !all(x -> isapprox(x, zero(datatype)),
                                        abs.(JuMP.value.(model[:w])))
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            trial[:assert_is_solved_and_feasible] = err
        end
        trial[:err] = JuMP.solution_summary(model)
        trials[solver.name] = trial
    end
    retcode = if success
        OptimisationSuccess(; res = trials)
    else
        @warn("Failed to solve optimisation problem.\nCheck `result.retcode.res` on the returned result for per-solver diagnostics.")
        OptimisationFailure(; res = trials)
    end
    return retcode, process_model(model, retcode)
end
"""
    set_portfolio_returns!(model::JuMP.Model, X::MatNum)

Compute and register portfolio returns expression `X * w` in the JuMP model.

If the expression already exists in the model, returns it directly (idempotent).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.

# Returns

  - The portfolio returns expression.

# Related

  - [`set_net_portfolio_returns!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_portfolio_returns!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :X))
        return model[Symbol(prefix, :X)]
    end
    w = get_w(model, prefix)
    return state_set!(model, prefix, :X, JuMP.@expression(model, X * w))
end
"""
    set_net_portfolio_returns!(model::JuMP.Model, X::MatNum)

Compute and register net portfolio returns (after fees) in the JuMP model.

Calls [`set_portfolio_returns!`](@ref) and subtracts the fees if any are registered. The model's
`:fees` expression holds the per period terms `l`, `s` and `tn`, which are rates per period, so it
is subtracted from every observation. The model's `:one_time_fees` expression holds the two fixed
terms, which are charged one time for the whole holding period, and `:fee_fa` names the clock they
fall on: a `nothing` or [`FirstObservationFees`](@ref) clock subtracts them from the first
observation alone, and an [`AmortisedFees`](@ref) spreads them over the observation count of the
fit. That is the rule [`charge_fees`](@ref) states at the value level, and
[`charge_one_time_fees`](@ref) applies it here.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.

# Returns

  - The net portfolio returns expression.

# Related

  - [`set_portfolio_returns!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_net_portfolio_returns!(model::JuMP.Model, X::MatNum;
                                    prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :net_X))
        return model[Symbol(prefix, :net_X)]
    end
    Xe = set_portfolio_returns!(model, X; prefix = prefix)
    # `:fees` and `:one_time_fees` are shared and not recreated by a nested build, so both
    # are read bare.
    if !haskey(model, :fees)
        return state_set!(model, prefix, :net_X, JuMP.@expression(model, Xe))
    end
    fees = model[:fees]
    net = JuMP.@expression(model, Xe .- fees)
    if haskey(model, :one_time_fees) && !isempty(net)
        # The clock the fee states decides where the two fixed terms land, exactly as
        # `charge_fees` decides it at the value level.
        net = charge_one_time_fees(model, net, model[:one_time_fees], get_T(model),
                                   model[:fee_fa])
    end
    return state_set!(model, prefix, :net_X, net)
end
"""
    set_asset_returns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register portfolio asset gross returns `X .+ 1` in the JuMP model.

Used in drawdown and logarithmic return computations.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns expression.

# Returns

  - The gross asset returns expression `X .+ 1`.

# Related

  - [`set_portfolio_drawdowns_plus_one!`](@ref)
  - [`set_portfolio_returns!`](@ref)
"""
function set_asset_returns_plus_one!(model::JuMP.Model, X::MatNum;
                                     prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :Xap1))
        return model[Symbol(prefix, :Xap1)]
    end
    return state_set!(model, prefix, :Xap1, JuMP.@expression(model, X .+ one(eltype(X))))
end
"""
    set_asset_neg_returns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register negative asset gross returns `-X .+ 1` in the JuMP model.

Used in drawdown and logarithmic return computations.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns expression.

# Returns

  - The negative gross asset returns expression `-X .+ 1`.

# Related

  - [`set_portfolio_drawdowns_plus_one!`](@ref)
  - [`set_portfolio_returns!`](@ref)
"""
function set_asset_neg_returns_plus_one!(model::JuMP.Model, X::MatNum;
                                         prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :nXap1))
        return model[Symbol(prefix, :nXap1)]
    end
    return state_set!(model, prefix, :nXap1, JuMP.@expression(model, -X .+ one(eltype(X))))
end
"""
    set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register absolute drawdowns plus one in the JuMP model.

Computes `absolute_drawdown_arr(X) .+ 1` and registers it in the model.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Portfolio returns expression.

# Returns

  - The drawdowns-plus-one expression.

# Related

  - [`set_asset_returns_plus_one!`](@ref)
"""
function set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum;
                                           prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :ddap1))
        return model[Symbol(prefix, :ddap1)]
    end
    _ddap1 = absolute_drawdown_arr(X) .+ one(eltype(X))
    return state_set!(model, prefix, :ddap1, JuMP.@expression(model, _ddap1))
end
"""
    scalarise_risk_expression!(model, r, X, T, ...) -> nothing

Scalarise a risk expression and add it to the JuMP model objective.

Generic function stub; concrete methods are defined in constraint and risk measure files. Each method adds the appropriate risk objective term for a given risk measure type `r`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
function scalarise_risk_expression! end
"""
    set_risk_constraints!(model, r, X, T, ...) -> nothing

Set risk constraints in the JuMP model for a given risk measure.

Generic function stub; concrete methods are defined in constraint and risk measure files. Each method configures the appropriate risk constraint expressions for a given risk measure type `r`.

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
function set_risk_constraints! end
