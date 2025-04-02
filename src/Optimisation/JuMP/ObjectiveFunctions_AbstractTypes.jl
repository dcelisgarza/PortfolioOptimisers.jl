abstract type OptimisationType end
abstract type ClusteringOptimisationType <: OptimisationType end
abstract type HierarchicalClusteringOptimisationType <: ClusteringOptimisationType end
abstract type JuMPOptimisationType <: OptimisationType end
abstract type ObjectiveFunction end
abstract type CustomObjective end
function add_custom_objective_term!(obj_func::ObjectiveFunction,
                                    ret_type::PortfolioReturnType, custom_obj::Nothing,
                                    obj_expr)
    return nothing
end
function set_objective_penalty!(model::JuMP.Model)
    @expression(model, op, zero(AffExpr))
    return nothing
end
function set_model_scales!(model::JuMP.Model, so::Real, sc::Real, ss::Real)
    @expressions(model, begin
                     so, so
                     sc, sc
                     ss, ss
                 end)
    return nothing
end
function set_initial_w!(::AbstractVector, ::Nothing)
    return nothing
end
function set_initial_w!(w::AbstractVector, wi::AbstractVector)
    @smart_assert(length(wi) == length(w))
    set_start_value.(w, wi)
    return nothing
end
function set_w!(model::JuMP.Model, X::AbstractMatrix, wi::Union{Nothing, <:AbstractVector})
    @variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
