abstract type OptimisationType end
abstract type ClusteringOptimisationType <: OptimisationType end
abstract type HierarchicalClusteringOptimisationType <: ClusteringOptimisationType end
abstract type TraditionalOptimisationType <: OptimisationType end
abstract type ObjectiveFunction end
abstract type CustomObjective end
struct NoCustomObjective <: CustomObjective end
abstract type CustomConstraint end
struct NoCustomConstraint <: CustomConstraint end
function add_objective_function_penalty!(port, obj_expr, c)
    model = port.model
    if haskey(model, :l1)
        l1 = model[:l1]
        add_to_expression!(obj_expr, c, l1)
    end
    if haskey(model, :l2)
        l2 = model[:l2]
        add_to_expression!(obj_expr, c, l2)
    end
    if haskey(model, :np)
        np = model[:np]
        add_to_expression!(obj_expr, c, np)
    end
    if haskey(model, :cp)
        cp = model[:cp]
        add_to_expression!(obj_expr, c, cp)
    end
    return nothing
end
function add_custom_objective_term!(port, obj_func::ObjectiveFunction,
                                    ret_type::PortfolioReturnType,
                                    custom_obj::NoCustomObjective, obj_expr)
    return nothing
end

export NoCustomObjective
