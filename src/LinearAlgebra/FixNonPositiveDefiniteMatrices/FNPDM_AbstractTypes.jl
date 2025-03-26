abstract type FixNonPositiveDefiniteMatrix end
function fix_non_positive_definite_matrix! end
function fix_non_positive_definite_matrix!(::Nothing, ::AbstractMatrix)
    return nothing
end

export fix_non_positive_definite_matrix!
