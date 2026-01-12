function get_mip_ss(ss::Number, args...)
    return ss
end
function get_mip_ss(::Nothing, wb::WeightBounds)
    lb = wb.lb
    ub = wb.ub
    lb_mag = if isnothing(lb)
        0.0
    else
        idx = isfinite.(lb)
        lbv = view(lb, idx)
        isempty(lbv) ? 0.0 : maximum(abs.(lbv))
    end
    ub_mag = if isnothing(ub)
        0.0
    else
        idx = isfinite.(ub)
        ubv = view(ub, idx)
        isempty(ubv) ? 0.0 : maximum(abs.(ubv))
    end
    return (iszero(lb_mag) && iszero(ub_mag)) ? 1000.0 : max(lb_mag, ub_mag) * 1000.0
end
function mip_wb(::Any, ::Nothing, args...)
    return nothing
end
function mip_wb(model::JuMP.Model, wb::WeightBounds, il::VecNum, is::VecNum)
    sc = model[:sc]
    w = model[:w]
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        JuMP.@constraint(model, w_mip_lb, sc * (w - is ⊙ lb) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        JuMP.@constraint(model, w_mip_ub, sc * (w - il ⊙ ub) <= 0)
    end
    return nothing
end
function short_mip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                         lt::Option{<:BuyInThreshold},
                                         st::Option{<:BuyInThreshold},
                                         ffl::Option{<:Num_VecNum},
                                         ffs::Option{<:Num_VecNum}, ss::Option{<:Number},
                                         lt_flag::Bool, st_flag::Bool, ffl_flag::Bool,
                                         ffs_flag::Bool)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    ss = get_mip_ss(ss, wb)
    N = length(w)
    JuMP.@variables(model, begin
                        ilb[1:N], (binary = true)
                        isb[1:N], (binary = true)
                    end)
    JuMP.@expression(model, i_mip, ilb + isb)
    if isa(k, Number)
        JuMP.@expressions(model, begin
                              il, ilb
                              is, isb
                          end)
    else
        JuMP.@variables(model, begin
                            ilf[1:N] >= 0
                            isf[1:N] >= 0
                        end)
        JuMP.@constraints(model, begin
                              ilf_ub, sc * (ilf .- k) <= 0
                              isf_ub, sc * (isf .- k) <= 0
                              ilfd_ub, sc * (ilf - ss * ilb) <= 0
                              isfd_ub, sc * (isf - ss * isb) <= 0
                              ilfd_lb, sc * ((ilf + ss * (1 .- ilb)) .- k) >= 0
                              isfd_lb, sc * ((isf + ss * (1 .- isb)) .- k) >= 0
                          end)
        JuMP.@expressions(model, begin
                              il, ilf
                              is, isf
                          end)
    end
    JuMP.@constraint(model, i_mip_ub, sc * (i_mip .- 1) <= 0)
    mip_wb(model, wb, il, is)
    if lt_flag
        JuMP.@constraint(model, w_mip_lt, sc * (w - il ⊙ lt.val + ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        JuMP.@constraint(model, w_mip_st, sc * (w + is ⊙ st.val - ss * (1 .- isb)) <= 0)
    end
    if ffl_flag
        JuMP.@expression(model, ffl, dot_scalar(ffl, ilb))
        add_to_fees!(model, ffl)
    end
    if ffs_flag
        JuMP.@expression(model, ffs, dot_scalar(ffs, isb))
        add_to_fees!(model, ffs)
    end
    return i_mip
end
function mip_constraints(model::JuMP.Model, wb::WeightBounds, ffl::Option{<:Num_VecNum},
                         lt::Option{<:BuyInThreshold}, ss::Option{<:Number}, lt_flag::Bool,
                         ffl_flag::Bool)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    JuMP.@variable(model, ib[1:N], binary = true)
    if isa(k, Number)
        JuMP.@expression(model, i_mip, ib)
    else
        ss = get_mip_ss(ss, wb)
        JuMP.@variable(model, ibf[1:N] >= 0)
        JuMP.@constraints(model, begin
                              ibf_ub, sc * (ibf .- k) <= 0
                              ibfd_ub, sc * (ibf - ss * ib) <= 0
                              ibfd_lb, sc * ((ibf + ss * (1 .- ib)) .- k) >= 0
                          end)
        JuMP.@expression(model, i_mip, ibf)
    end
    mip_wb(model, wb, i_mip, i_mip)
    if lt_flag
        JuMP.@constraint(model, w_mip_lt, sc * (w - i_mip ⊙ lt.val) >= 0)
    end
    if ffl_flag
        JuMP.@expression(model, ffl, dot_scalar(ffl, ib))
        add_to_fees!(model, ffl)
    end
    return ib
end
function set_iplg_constraints!(model::JuMP.Model, plgs::PhC_VecPhC)
    ib = model[:ib]
    sc = model[:sc]
    for (i, plg) in enumerate(plgs)
        if !isa(plg, IntegerPhylogeny)
            continue
        end
        A = plg.A
        B = plg.B
        model[Symbol(:card_plg_, i)] = JuMP.@constraint(model, sc * (A * ib ⊖ B) <= 0)
    end
    return nothing
end
function set_mip_constraints!(model::JuMP.Model, wb::WeightBounds, card::Option{<:Integer},
                              gcard::Option{<:LinearConstraint}, plg::Option{<:PhC_VecPhC},
                              lt::Option{<:BuyInThreshold}, st::Option{<:BuyInThreshold},
                              fees::Option{<:Fees}, ss::Option{<:Number})
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
        JuMP.@constraint(model, card, sc * (sum(ib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            JuMP.@constraint(model, gcard_ineq, sc * (A * ib ⊖ B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            JuMP.@constraint(model, gcard_eq, sc * (A * ib ⊖ B) == 0)
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
function smip_wb(model::JuMP.Model, wb::WeightBounds, smtx::MatNum,
                 smtx_expr::VecJuMPScalar, il::VecNum, is::VecNum,
                 key::Symbol = :set_w_mip_, i::Integer = 1)
    sc = model[:sc]
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        lb = smtx * lb
        model[Symbol(key, :lb_, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr - lb ⊙ is) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        ub = smtx * ub
        model[Symbol(key, :ub_, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr - ub ⊙ il) <= 0)
    end
    return nothing
end
function short_smip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                          smtx::Option{<:MatNum},
                                          lt::Option{<:BuyInThreshold},
                                          st::Option{<:BuyInThreshold},
                                          ss::Option{<:Number}, lt_flag::Bool,
                                          st_flag::Bool, key1::Symbol = :si,
                                          key7::Symbol = :smtx_expr_,
                                          key8::Symbol = :set_w_mip_, i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    ss = get_mip_ss(ss, wb)
    N = size(smtx, 1)
    ilb, isb = model[Symbol(key1, :lb_, i)], model[Symbol(key1, :ub_, i)] = JuMP.@variables(model,
                                                                                            begin
                                                                                                [1:N],
                                                                                                (binary = true)
                                                                                                [1:N],
                                                                                                (binary = true)
                                                                                            end)
    key2 = Symbol(key1, :_mip_)
    i_mip = model[Symbol(key2, i)] = JuMP.@expression(model, ilb + isb)
    il, is = if isa(k, Number)
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = JuMP.@expressions(model,
                                                                                     begin
                                                                                         ilb
                                                                                         isb
                                                                                     end)
    else
        key3 = Symbol(key1, :lf_)
        key4 = Symbol(key1, :sf_)
        ilf, isf = model[Symbol(key3, i)], model[Symbol(key4, i)] = JuMP.@variables(model,
                                                                                    begin
                                                                                        [1:N],
                                                                                        (lower_bound = 0)
                                                                                        [1:N],
                                                                                        (lower_bound = 0)
                                                                                    end)
        key5 = Symbol(key1, :lfd_)
        key6 = Symbol(key1, :sfd_)
        model[Symbol(key3, :ub_, i)], model[Symbol(key4, :ub_, i)], model[Symbol(key5, :ub_, i)], model[Symbol(key6, :ub_, i)], model[Symbol(key5, :lb_, i)], model[Symbol(key6, :lb_, i)] = JuMP.@constraints(model,
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
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = JuMP.@expressions(model,
                                                                                     begin
                                                                                         ilf
                                                                                         isf
                                                                                     end)
    end
    model[Symbol(key2, :ub, i)] = JuMP.@constraint(model, sc * (i_mip .- 1) <= 0)
    smtx_expr = model[Symbol(key7, i)] = JuMP.@expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, il, is, key8, i)
    if lt_flag
        model[Symbol(key8, :lt, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr - il ⊙ lt.val +
                                                             ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        model[Symbol(key8, :st, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr + is ⊙ st.val -
                                                             ss * (1 .- isb)) <= 0)
    end
    return i_mip
end
function smip_constraints(model::JuMP.Model, wb::WeightBounds, smtx::Option{<:MatNum},
                          lt::Option{<:BuyInThreshold}, ss::Option{<:Number}, lt_flag::Bool,
                          key1::Symbol = :sib_, key2::Symbol = :i_smip_,
                          key3::Symbol = :isbf_, key4::Symbol = :smtx_expr_,
                          key5::Symbol = :set_w_mip_, key6::Symbol = :w_smip_lt_,
                          i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = size(smtx, 1)
    sib = model[Symbol(key1, i)] = JuMP.@variable(model, [1:N], binary = true)
    i_smip = if isa(k, Number)
        model[Symbol(key2, i)] = JuMP.@expression(model, sib)
    else
        ss = get_mip_ss(ss, wb)
        isbf = model[Symbol(key3, i)] = JuMP.@variable(model, [1:N], lower_bound = 0)
        model[Symbol(key3, :_ub_, i)], model[Symbol(key3, :d_ub_, i)], model[Symbol(key3, :d_lb_, i)] = JuMP.@constraints(model,
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
        model[Symbol(key2, i)] = JuMP.@expression(model, isbf)
    end
    smtx_expr = model[Symbol(key4, i)] = JuMP.@expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, i_smip, i_smip, key5, i)
    if lt_flag
        model[Symbol(key6, i)] = JuMP.@constraint(model,
                                                  sc * (smtx_expr - i_smip ⊙ lt.val) >= 0)
    end
    return sib
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer},
                                   gcard::Option{<:LinearConstraint},
                                   smtx::Option{<:MatNum}, lt::Option{<:BuyInThreshold},
                                   st::Option{<:BuyInThreshold}, ss::Option{<:Number},
                                   i::Integer = 1)
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
        model[Symbol(:scard_, i)] = JuMP.@constraint(model, sc * (sum(sib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            model[Symbol(:sgcard_ineq_, i)] = JuMP.@constraint(model,
                                                               sc * (A * sib ⊖ B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            model[Symbol(:sgcard_eq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) == 0)
        end
    end
    return nothing
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds, card::VecInt,
                                   gcard::VecLc, smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                   st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (c, g, s)) in enumerate(zip(card, gcard, smtx))
        lti = isa(lt, Option{<:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Option{<:BuyInThreshold}) ? st : st[i]
        set_all_smip_constraints!(model, wb, c, g, s, lti, sti, ss, i)
    end
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer}, smtx::Option{<:MatNum},
                                   lt::Option{<:BuyInThreshold},
                                   st::Option{<:BuyInThreshold}, ss::Option{<:Number},
                                   i::Integer = 1)
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
    model[Symbol(:scard_, i)] = JuMP.@constraint(model, sc * (sum(sib) - card) <= 0)
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds, card::VecInt,
                                   smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                   st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (c, s)) in enumerate(zip(card, smtx))
        lti = isa(lt, Option{<:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Option{<:BuyInThreshold}) ? st : st[i]
        set_scardmip_constraints!(model, wb, c, s, lti, sti, ss, i)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::Option{<:LinearConstraint},
                                    smtx::Option{<:MatNum}, lt::Option{<:BuyInThreshold},
                                    st::Option{<:BuyInThreshold}, ss::Option{<:Number},
                                    i::Integer = 1)
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
        model[Symbol(:sgcard_ineq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        model[Symbol(:sgcard_eq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) == 0)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds, gcard::VecLc,
                                    smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                    st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (gc, s)) in enumerate(zip(gcard, smtx))
        lti = isa(lt, Option{<:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Option{<:BuyInThreshold}) ? st : st[i]
        set_sgcardmip_constraints!(model, wb, gc, s, lti, sti, ss, i)
    end
    return nothing
end
function set_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                               card::Option{<:Int_VecInt}, gcard::Option{<:Lc_VecLc},
                               smtx::Option{<:MatNum_VecMatNum},
                               sgmtx::Option{<:MatNum_VecMatNum}, lt::Option{<:Bt_VecOptBt},
                               st::Option{<:Bt_VecOptBt}, glt::Option{<:Bt_VecOptBt},
                               gst::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    if smtx === sgmtx
        set_all_smip_constraints!(model, wb, card, gcard, smtx, lt, st, ss)
    else
        set_scardmip_constraints!(model, wb, card, smtx, lt, st, ss)
        set_sgcardmip_constraints!(model, wb, gcard, sgmtx, glt, gst, ss)
    end
    return nothing
end
