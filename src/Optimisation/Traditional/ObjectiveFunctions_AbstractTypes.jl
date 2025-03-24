abstract type OptimisationType end
abstract type ClusteringOptimisationType <: OptimisationType end
abstract type HierarchicalClusteringOptimisationType <: ClusteringOptimisationType end
abstract type TraditionalOptimisationType <: OptimisationType end
abstract type ObjectiveFunction end
abstract type CustomObjective end
struct NoCustomObjective <: CustomObjective end
function add_custom_objective_term!(port, obj_func::ObjectiveFunction,
                                    ret_type::PortfolioReturnType,
                                    custom_obj::NoCustomObjective, obj_expr)
    return nothing
end
function set_objective_penalty!(model::JuMP.Model)
    @expression(model, obj_pen, zero(AffExpr))
    return nothing
end
function set_model_scales!(model::JuMP.Model, os::Real, cs::Real)
    @expressions(model, begin
                     os, os
                     cs, cs
                 end)
    return nothing
end
function set_initial_w!(w::AbstractVector, wi::AbstractVector)
    if isempty(wi)
        return nothing
    end
    @smart_assert(length(wi) == length(w))
    set_start_value.(w, wi)
    return nothing
end
function set_w!(model::JuMP.Model, X::AbstractMatrix, wi::AbstractVector)
    @variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
function optimise! end

export NoCustomObjective, optimise!
