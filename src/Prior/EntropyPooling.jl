function entropy_pooling(p::AbstractVector;
                         A_eq::Union{<:AbstractMatrix, Nothing} = nothing,
                         b_eq::Union{<:AbstractVector, Nothing} = nothing,
                         A_ineq::Union{<:AbstractMatrix, Nothing} = nothing,
                         b_ineq::Union{<:AbstractVector, Nothing} = nothing,
                         scale::Real = 1, optim_args::Tuple = (),
                         optim_kwargs::NamedTuple = (;))
    lhs, rhs, lb, ub = if !isnothing(A_eq) && !isnothing(b_eq)
        @smart_assert(!isempty(A_eq))
        @smart_assert(!isempty(b_eq))
        A_eq = vcat(ones(eltype(A_eq), 1, size(A_eq, 2)), A_eq)
        b_eq = vcat(eltype(b_eq)[1], b_eq)
        A_eq, b_eq, fill(-Inf, length(b_eq)), fill(Inf, length(b_eq))
    else
        ones(eltype(p), 1, length(p)), eltype(p)[1], [-Inf], [Inf]
    end

    lhs_ineq, rhs_ineq, lb_ineq, ub_ineq = if !isnothing(A_ineq) && !isnothing(b_ineq)
        @smart_assert(!isempty(A_ineq))
        @smart_assert(!isempty(b_ineq))
        A_ineq, b_ineq, fill(-Inf, length(b_ineq)), fill(Inf, length(b_ineq))
    else
        nothing, nothing, nothing, nothing
    end

    if !isnothing(lhs_ineq)
        lhs = vcat(lhs, lhs_ineq)
        rhs = vcat(rhs, rhs_ineq)
        lb = vcat(lb, lb_ineq)
        ub = vcat(ub, ub_ineq)
    end

    log_p = log.(p)
    x0 = zeros(size(lhs, 1))
    G = similar(x0)
    last_x = similar(x0)

    grad = similar(G)
    log_x = similar(log_p)
    y = similar(log_p)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            log_x .= log_p .- one(eltype(log_p)) .- lhs' * x
            y .= exp.(log_x)
            grad .= rhs - lhs * y
        end
    end
    function f(x)
        common_op(x)
        return scale * (dot(x, grad) - dot(y, log_x - log_p))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return scale * G
    end
    result = Optim.optimize(f, g!, lb, ub, x0, optim_args...; optim_kwargs...)

    # Compute posterior probabilities
    x = Optim.minimizer(result)
    q = exp.(log_p .- one(eltype(log_p)) .- lhs' * x)

    return q
end

function entropy_pooling(p::AbstractVector,
                         solvers::Union{<:Solver, <:AbstractVector{<:Solver}};
                         A_eq::Union{<:AbstractMatrix, Nothing} = nothing,
                         b_eq::Union{<:AbstractVector, Nothing} = nothing,
                         A_ineq::Union{<:AbstractMatrix, Nothing} = nothing,
                         b_ineq::Union{<:AbstractVector, Nothing} = nothing)
    model = Model()
    S = length(p)
    log_p = log.(p)
    # Decision variables (posterior probabilities)
    @variables(model, begin
                   q[1:S]
                   t
               end)
    # Equality constraints from A_ineq and b_ineq if provided
    if !isnothing(A_eq) && !isnothing(b_eq)
        @smart_assert(!isempty(A_eq))
        @smart_assert(!isempty(b_eq))
        @constraint(model, constr_eq, A_eq * q == b_eq)
    end
    # Inequality constraints from A_ineq and b_ineq if provided
    if !isnothing(A_ineq) && !isnothing(b_ineq)
        @smart_assert(!isempty(A_ineq))
        @smart_assert(!isempty(b_ineq))
        @constraint(model, constr_ineq, A_ineq * q <= b_ineq)
    end
    # Equality constraints from A_eq and b_eq and probabilities equal to 1
    @constraints(model, begin
                     sum(q) == 1
                     [t; ones(S); q] in MOI.RelativeEntropyCone(2 * S + 1)
                 end)

    @objective(model, Min, t - dot(q, log_p))
    # Solve the optimization problem
    success, solvers_tried = optimise_JuMP_model(model, solvers)
    return if success
        value.(q)
    else
        @warn("model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        fill(NaN, length(q))
    end
end

export entropy_pooling
