struct JuMP_ConstraintModel
    lcm::LinearConstraintModel
    budget::Union{<:Real, <:BudgetConstraint}
    lsb::LongShortSum
    bit::BuyInThreshold
    card::Integer
    gcard::PartialLinearConstraintModel
    cadj::AdjacencyConstraintModel
    nadj::AdjacencyConstraintModel
end
