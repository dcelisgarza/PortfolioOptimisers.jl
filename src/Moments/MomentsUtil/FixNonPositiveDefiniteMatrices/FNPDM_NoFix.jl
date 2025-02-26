struct FNPDM_NoFix <: FixNonPositiveDefiniteMatrix end
function fix_non_positive_definite_matrix(::FNPDM_NoFix, X::AbstractMatrix)
    return X
end
function fix_non_positive_definite_matrix!(::FNPDM_NoFix, ::AbstractMatrix)
    return nothing
end
