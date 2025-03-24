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
function create_objective_penalty!(model::JuMP.Model)
    @expression(model, obj_pen, zero(AffExpr))
    return nothing
end

export NoCustomObjective
