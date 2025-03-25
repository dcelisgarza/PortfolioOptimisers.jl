struct JuMP_ConstraintModel
    lcm::LinearConstraintModel
    budget::Union{<:Real, <:BudgetConstraint}
    lsb::LongShortBounds
    bit::BuyInThreshold
    card::Integer
    gcard::PartialLinearConstraintModel
    cadj::AdjacencyConstraint
    nadj::AdjacencyConstraint
end