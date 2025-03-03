abstract type AbstractLinearConstraintSide end
abstract type AbstractLinearConstraint end
abstract type ComparisonOperators end
struct EQ <: ComparisonOperators end
struct LEQ <: ComparisonOperators end
struct GEQ <: ComparisonOperators end
function comparison_sign_ineq_flag(::EQ)
    return 1, false
end
function comparison_sign_ineq_flag(::LEQ)
    return 1, true
end
function comparison_sign_ineq_flag(::GEQ)
    return -1, true
end

function linear_constraints end

export linear_constraints, EQ, LEQ, GEQ
