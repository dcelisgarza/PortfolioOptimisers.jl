abstract type OptimisationType end
abstract type ClusteringOptimisationType <: OptimisationType end
abstract type HierarchicalClusteringOptimisationType <: ClusteringOptimisationType end
abstract type TraditionalOptimisationType <: OptimisationType end
abstract type ObjectiveFunction end
abstract type CustomObjective end
struct NoCustomObjective <: CustomObjective end
function add_objective_function_penalty!(port, obj_expr, c)
    model = port.model
    if haskey(model, :l1_reg)
        l1_reg = model[:l1_reg]
        add_to_expression!(obj_expr, c, l1_reg)
    end
    if haskey(model, :l2_reg)
        l2_reg = model[:l2_reg]
        add_to_expression!(obj_expr, c, l2_reg)
    end
    if haskey(model, :network_penalty)
        network_penalty = model[:network_penalty]
        add_to_expression!(obj_expr, c, network_penalty)
    end
    if haskey(model, :cluster_penalty)
        cluster_penalty = model[:cluster_penalty]
        add_to_expression!(obj_expr, c, cluster_penalty)
    end
    return nothing
end
function add_custom_objective_term!(port, obj_func::ObjectiveFunction,
                                    ret_type::PortfolioReturnType,
                                    custom_obj::NoCustomObjective, obj_expr)
    return nothing
end

export NoCustomObjective
