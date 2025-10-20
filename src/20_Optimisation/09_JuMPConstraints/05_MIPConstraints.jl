function get_mip_ss(ss::Real, args...)
    return ss
end
function get_mip_ss(::Nothing, wb::WeightBounds)
    return max(maximum(abs.(wb.lb)), maximum(abs.(wb.ub))) * 1000
end
function mip_wb(::Any, ::Nothing, args...)
    return nothing
end
function mip_wb(model::JuMP.Model, wb::WeightBounds, il::AbstractVector, is::AbstractVector)
    sc = model[:sc]
    w = model[:w]
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        @constraint(model, w_mip_lb, sc * (w - is ⊙ lb) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        @constraint(model, w_mip_ub, sc * (w - il ⊙ ub) <= 0)
    end
    return nothing
end
function short_mip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                         lt::Union{Nothing, <:BuyInThreshold},
                                         st::Union{Nothing, <:BuyInThreshold},
                                         ffl::Union{Nothing, <:Real, <:AbstractVector},
                                         ffs::Union{Nothing, <:Real, <:AbstractVector},
                                         ss::Union{Nothing, <:Real}, lt_flag::Bool,
                                         st_flag::Bool, ffl_flag::Bool, ffs_flag::Bool)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    ss = get_mip_ss(ss, wb)
    N = length(w)
    @variables(model, begin
                   ilb[1:N], (binary = true)
                   isb[1:N], (binary = true)
               end)
    @expression(model, i_mip, ilb + isb)
    if isa(k, Real)
        @expressions(model, begin
                         il, ilb
                         is, isb
                     end)
    else
        @variables(model, begin
                       ilf[1:N] >= 0
                       isf[1:N] >= 0
                   end)
        @constraints(model, begin
                         ilf_ub, sc * (ilf .- k) <= 0
                         isf_ub, sc * (isf .- k) <= 0
                         ilfd_ub, sc * (ilf - ss * ilb) <= 0
                         isfd_ub, sc * (isf - ss * isb) <= 0
                         ilfd_lb, sc * ((ilf + ss * (1 .- ilb)) .- k) >= 0
                         isfd_lb, sc * ((isf + ss * (1 .- isb)) .- k) >= 0
                     end)
        @expressions(model, begin
                         il, ilf
                         is, isf
                     end)
    end
    @constraint(model, i_mip_ub, sc * (i_mip .- 1) <= 0)
    mip_wb(model, wb, il, is)
    if lt_flag
        @constraint(model, w_mip_lt, sc * (w - il ⊙ lt.val + ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        @constraint(model, w_mip_st, sc * (w + is ⊙ st.val - ss * (1 .- isb)) <= 0)
    end
    if ffl_flag
        @expression(model, ffl, dot_scalar(ffl, ilb))
        add_to_fees!(model, ffl)
    end
    if ffs_flag
        @expression(model, ffs, dot_scalar(ffs, isb))
        add_to_fees!(model, ffs)
    end
    return i_mip
end
function mip_constraints(model::JuMP.Model, wb::WeightBounds,
                         ffl::Union{Nothing, <:Real, <:AbstractVector},
                         lt::Union{Nothing, <:BuyInThreshold}, ss::Union{Nothing, <:Real},
                         lt_flag::Bool, ffl_flag::Bool)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    @variable(model, ib[1:N], binary = true)
    if isa(k, Real)
        @expression(model, i_mip, ib)
    else
        ss = get_mip_ss(ss, wb)
        @variable(model, ibf[1:N] >= 0)
        @constraints(model, begin
                         ibf_ub, sc * (ibf .- k) <= 0
                         ibfd_ub, sc * (ibf - ss * ib) <= 0
                         ibfd_lb, sc * ((ibf + ss * (1 .- ib)) .- k) >= 0
                     end)
        @expression(model, i_mip, ibf)
    end
    mip_wb(model, wb, i_mip, i_mip)
    if lt_flag
        @constraint(model, w_mip_lt, sc * (w - i_mip ⊙ lt.val) >= 0)
    end
    if ffl_flag
        @expression(model, ffl, dot_scalar(ffl, ib))
        add_to_fees!(model, ffl)
    end
    return ib
end
function set_iplg_constraints!(model::JuMP.Model,
                               plgs::Union{<:AbstractPhylogenyConstraintResult,
                                           <:AbstractVector{<:AbstractPhylogenyConstraintResult}})
    ib = model[:ib]
    sc = model[:sc]
    for (i, plg) in enumerate(plgs)
        if !isa(plg, IntegerPhylogeny)
            continue
        end
        A = plg.A
        B = plg.B
        model[Symbol(:card_plg_, i)] = @constraint(model, sc * (A * ib ⊖ B) <= 0)
    end
    return nothing
end
function set_mip_constraints!(model::JuMP.Model, wb::WeightBounds,
                              card::Union{Nothing, <:Integer},
                              gcard::Union{Nothing, <:LinearConstraint},
                              plg::Union{Nothing, <:AbstractPhylogenyConstraintResult,
                                         <:AbstractVector{<:AbstractPhylogenyConstraintResult}},
                              lt::Union{Nothing, <:BuyInThreshold},
                              st::Union{Nothing, <:BuyInThreshold},
                              fees::Union{Nothing, <:Fees}, ss::Union{Nothing, <:Real})
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    iplg_flag = isa(plg, IntegerPhylogeny) ||
                isa(plg, AbstractVector) && any(x -> isa(x, IntegerPhylogeny), plg)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    ffl_flag, ffs_flag, ffl, ffs = if !isnothing(fees)
        non_zero_real_or_vec(fees.fl), non_zero_real_or_vec(fees.fs), fees.fl, fees.fs
    else
        false, false, nothing, nothing
    end
    if !(card_flag || gcard_flag || iplg_flag || lt_flag || st_flag || ffl_flag || ffs_flag)
        return nothing
    end
    ib = if (st_flag || ffl_flag || ffs_flag) && haskey(model, :sw)
        short_mip_threshold_constraints(model, wb, lt, st, ffl, ffs, ss, lt_flag, st_flag,
                                        ffl_flag, ffs_flag)
    else
        mip_constraints(model, wb, ffl, lt, ss, lt_flag, ffl_flag)
    end
    sc = model[:sc]
    if card_flag
        @constraint(model, card, sc * (sum(ib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            @constraint(model, gcard_ineq, sc * (A * ib ⊖ B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            @constraint(model, gcard_eq, sc * (A * ib ⊖ B) == 0)
        end
    end
    if iplg_flag
        set_iplg_constraints!(model, plg)
    end
    return nothing
end
function smip_wb(::Any, ::Nothing, args...)
    return nothing
end
function smip_wb(model::JuMP.Model, wb::WeightBounds, smtx::AbstractMatrix,
                 smtx_expr::AbstractVector{<:AbstractJuMPScalar}, il::AbstractVector,
                 is::AbstractVector, key::Symbol = :set_w_mip_, i::Integer = 1)
    sc = model[:sc]
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        lb = [sum(lb[view(smtx, j, :)]) for j in axes(smtx, 1)]
        model[Symbol(key, :lb_, i)] = @constraint(model, sc * (smtx_expr - lb ⊙ is) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        ub = [sum(ub[view(smtx, j, :)]) for j in axes(smtx, 1)]
        model[Symbol(key, :ub_, i)] = @constraint(model, sc * (smtx_expr - ub ⊙ il) <= 0)
    end
    return nothing
end
function short_smip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                          smtx::Union{Nothing, <:AbstractMatrix},
                                          lt::Union{Nothing, <:BuyInThreshold},
                                          st::Union{Nothing, <:BuyInThreshold},
                                          ss::Union{Nothing, <:Real}, lt_flag::Bool,
                                          st_flag::Bool, key1::Symbol = :si,
                                          key7::Symbol = :smtx_expr_,
                                          key8::Symbol = :set_w_mip_, i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    ss = get_mip_ss(ss, wb)
    N = size(smtx, 1)
    ilb, isb = model[Symbol(key1, :lb_, i)], model[Symbol(key1, :ub_, i)] = @variables(model,
                                                                                       begin
                                                                                           [1:N],
                                                                                           (binary = true)
                                                                                           [1:N],
                                                                                           (binary = true)
                                                                                       end)
    key2 = Symbol(key1, :_mip_)
    i_mip = model[Symbol(key2, i)] = @expression(model, ilb + isb)
    il, is = if isa(k, Real)
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = @expressions(model,
                                                                                begin
                                                                                    ilb
                                                                                    isb
                                                                                end)
    else
        key3 = Symbol(key1, :lf_)
        key4 = Symbol(key1, :sf_)
        ilf, isf = model[Symbol(key3, i)], model[Symbol(key4, i)] = @variables(model,
                                                                               begin
                                                                                   [1:N],
                                                                                   (lower_bound = 0)
                                                                                   [1:N],
                                                                                   (lower_bound = 0)
                                                                               end)
        key5 = Symbol(key1, :lfd_)
        key6 = Symbol(key1, :sfd_)
        model[Symbol(key3, :ub_, i)], model[Symbol(key4, :ub_, i)], model[Symbol(key5, :ub_, i)], model[Symbol(key6, :ub_, i)], model[Symbol(key5, :lb_, i)], model[Symbol(key6, :lb_, i)] = @constraints(model,
                                                                                                                                                                                                          begin
                                                                                                                                                                                                              sc *
                                                                                                                                                                                                              (ilf .-
                                                                                                                                                                                                               k) <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              sc *
                                                                                                                                                                                                              (isf .-
                                                                                                                                                                                                               k) <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              sc *
                                                                                                                                                                                                              (ilf -
                                                                                                                                                                                                               ss *
                                                                                                                                                                                                               ilb) <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              sc *
                                                                                                                                                                                                              (isf -
                                                                                                                                                                                                               ss *
                                                                                                                                                                                                               isb) <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              sc *
                                                                                                                                                                                                              ((ilf +
                                                                                                                                                                                                                ss *
                                                                                                                                                                                                                (1 .-
                                                                                                                                                                                                                 ilb)) .-
                                                                                                                                                                                                               k) >=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              sc *
                                                                                                                                                                                                              ((isf +
                                                                                                                                                                                                                ss *
                                                                                                                                                                                                                (1 .-
                                                                                                                                                                                                                 isb)) .-
                                                                                                                                                                                                               k) >=
                                                                                                                                                                                                              0
                                                                                                                                                                                                          end)
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = @expressions(model,
                                                                                begin
                                                                                    ilf
                                                                                    isf
                                                                                end)
    end
    model[Symbol(key2, :ub, i)] = @constraint(model, sc * (i_mip .- 1) <= 0)
    smtx_expr = model[Symbol(key7, i)] = @expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, il, is, key8, i)
    if lt_flag
        model[Symbol(key8, :lt, i)] = @constraint(model,
                                                  sc * (smtx_expr - il ⊙ lt.val +
                                                        ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        model[Symbol(key8, :st, i)] = @constraint(model,
                                                  sc * (smtx_expr + is ⊙ st.val -
                                                        ss * (1 .- isb)) <= 0)
    end
    return i_mip
end
function smip_constraints(model::JuMP.Model, wb::WeightBounds,
                          smtx::Union{Nothing, <:AbstractMatrix},
                          lt::Union{Nothing, <:BuyInThreshold}, ss::Union{Nothing, <:Real},
                          lt_flag::Bool, key1::Symbol = :sib_, key2::Symbol = :i_smip_,
                          key3::Symbol = :isbf_, key4::Symbol = :smtx_expr_,
                          key5::Symbol = :set_w_mip_, key6::Symbol = :w_smip_lt_,
                          i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = size(smtx, 1)
    sib = model[Symbol(key1, i)] = @variable(model, [1:N], binary = true)
    i_smip = if isa(k, Real)
        model[Symbol(key2, i)] = @expression(model, sib)
    else
        ss = get_mip_ss(ss, wb)
        isbf = model[Symbol(key3, i)] = @variable(model, [1:N], lower_bound = 0)
        model[Symbol(key3, :_ub_, i)], model[Symbol(key3, :d_ub_, i)], model[Symbol(key3, :d_lb_, i)] = @constraints(model,
                                                                                                                     begin
                                                                                                                         sc *
                                                                                                                         (isbf .-
                                                                                                                          k) <=
                                                                                                                         0
                                                                                                                         sc *
                                                                                                                         (isbf -
                                                                                                                          ss *
                                                                                                                          sib) <=
                                                                                                                         0
                                                                                                                         sc *
                                                                                                                         ((isbf +
                                                                                                                           ss *
                                                                                                                           (1 .-
                                                                                                                            sib)) .-
                                                                                                                          k) >=
                                                                                                                         0
                                                                                                                     end)
        model[Symbol(key2, i)] = @expression(model, isbf)
    end
    smtx_expr = model[Symbol(key4, i)] = @expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, i_smip, i_smip, key5, i)
    if lt_flag
        model[Symbol(key6, i)] = @constraint(model, sc * (smtx_expr - i_smip ⊙ lt.val) >= 0)
    end
    return sib
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Union{Nothing, <:Integer},
                                   gcard::Union{Nothing, <:LinearConstraint},
                                   smtx::Union{Nothing, <:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold},
                                   st::Union{Nothing, <:BuyInThreshold},
                                   ss::Union{Nothing, <:Real}, i::Integer = 1)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = model[:sc]
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag, :si,
                                         :smtx_expr_, :set_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sib_, :i_smip_, :isbf_,
                         :smtx_expr_, :set_w_mip_, :w_smip_lt_, i)
    end
    if card_flag
        model[Symbol(:scard_, i)] = @constraint(model, sc * (sum(sib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            model[Symbol(:sgcard_ineq_, i)] = @constraint(model,
                                                          sc *
                                                          (A * transpose(smtx) * sib ⊖ B) <=
                                                          0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            model[Symbol(:sgcard_eq_, i)] = @constraint(model,
                                                        sc *
                                                        (A * transpose(smtx) * sib ⊖ B) ==
                                                        0)
        end
    end
    return nothing
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::AbstractVector{<:Integer},
                                   gcard::AbstractVector{<:LinearConstraint},
                                   smtx::AbstractVector{<:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   st::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   ss::Union{Nothing, <:Real})
    for (i, (c, g, s)) in enumerate(zip(card, gcard, smtx))
        lti = isa(lt, Union{Nothing, <:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Union{Nothing, <:BuyInThreshold}) ? st : st[i]
        set_all_smip_constraints!(model, wb, c, g, s, lti, sti, ss, i)
    end
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Union{Nothing, <:Integer},
                                   smtx::Union{Nothing, <:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold},
                                   st::Union{Nothing, <:BuyInThreshold},
                                   ss::Union{Nothing, <:Real}, i::Integer = 1)
    card_flag = !isnothing(card)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || lt_flag || st_flag)
        return nothing
    end
    sc = model[:sc]
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag, :si,
                                         :smtx_expr_, :set_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sib_, :i_smip_, :isbf_,
                         :smtx_expr_, :set_w_mip_, :w_smip_lt_, i)
    end
    model[Symbol(:scard_, i)] = @constraint(model, sc * (sum(sib) - card) <= 0)
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::AbstractVector{<:Integer},
                                   smtx::AbstractVector{<:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   st::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   ss::Union{Nothing, <:Real})
    for (i, (c, s)) in enumerate(zip(card, smtx))
        lti = isa(lt, Union{Nothing, <:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Union{Nothing, <:BuyInThreshold}) ? st : st[i]
        set_scardmip_constraints!(model, wb, c, s, lti, sti, ss, i)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::Union{Nothing, <:LinearConstraint},
                                    smtx::Union{Nothing, <:AbstractMatrix},
                                    lt::Union{Nothing, <:BuyInThreshold},
                                    st::Union{Nothing, <:BuyInThreshold},
                                    ss::Union{Nothing, <:Real}, i::Integer = 1)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = model[:sc]
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag,
                                         :sgi, :sgmtx_expr_, :setg_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sgib_, :i_sgmip_, :isgbf_,
                         :sgmtx_expr_, :setg_w_mip_, :w_sgmip_lt_, i)
    end
    if !isnothing(gcard.ineq)
        A = gcard.ineq.A
        B = gcard.ineq.B
        model[Symbol(:sgcard_ineq_, i)] = @constraint(model,
                                                      sc *
                                                      (A * transpose(smtx) * sib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        model[Symbol(:sgcard_eq_, i)] = @constraint(model,
                                                    sc * (A * transpose(smtx) * sib ⊖ B) ==
                                                    0)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::AbstractVector{<:LinearConstraint},
                                    smtx::AbstractVector{<:AbstractMatrix},
                                    lt::Union{Nothing, <:BuyInThreshold,
                                              <:AbstractVector{<:BuyInThreshold},
                                              <:AbstractVector{<:Union{Nothing,
                                                                       <:BuyInThreshold}}},
                                    st::Union{Nothing, <:BuyInThreshold,
                                              <:AbstractVector{<:BuyInThreshold},
                                              <:AbstractVector{<:Union{Nothing,
                                                                       <:BuyInThreshold}}},
                                    ss::Union{Nothing, <:Real})
    for (i, (gc, s)) in enumerate(zip(gcard, smtx))
        lti = isa(lt, Union{Nothing, <:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Union{Nothing, <:BuyInThreshold}) ? st : st[i]
        set_sgcardmip_constraints!(model, wb, gc, s, lti, sti, ss, i)
    end
    return nothing
end
function set_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                               card::Union{Nothing, <:Integer, <:AbstractVector{<:Integer}},
                               gcard::Union{Nothing, <:LinearConstraint,
                                            <:AbstractVector{<:LinearConstraint}},
                               smtx::Union{Nothing, <:AbstractMatrix,
                                           <:AbstractVector{<:AbstractMatrix}},
                               sgmtx::Union{Nothing, <:AbstractMatrix,
                                            <:AbstractVector{<:AbstractMatrix}},
                               lt::Union{Nothing, <:BuyInThreshold},
                               st::Union{Nothing, <:BuyInThreshold},
                               glt::Union{Nothing, <:BuyInThreshold},
                               gst::Union{Nothing, <:BuyInThreshold},
                               ss::Union{Nothing, <:Real})
    if smtx === sgmtx
        set_all_smip_constraints!(model, wb, card, gcard, smtx, lt, st, ss)
    else
        set_scardmip_constraints!(model, wb, card, smtx, lt, st, ss)
        set_sgcardmip_constraints!(model, wb, gcard, sgmtx, glt, gst, ss)
    end
    return nothing
end
