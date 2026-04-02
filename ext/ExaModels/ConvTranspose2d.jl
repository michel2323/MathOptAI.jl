# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ConvTranspose2d,
    x,
)
    (Hin, Win, Cin) = p.input_size
    kH, kW, Cout, Cin_w = size(p.weight)
    @assert Cin == Cin_w
    (pH, pW), (sH, sW) = p.padding, p.stride
    (opH, opW) = p.output_padding
    Hout = (Hin - 1) * sH - 2 * pH + kH + opH
    Wout = (Win - 1) * sW - 2 * pW + kW + opW
    n_out = Hout * Wout * Cout
    n_spatial = Hout * Wout
    # Output variables
    y = ExaModels.variable(core, n_out)
    bias_expanded = [p.bias[c] for h in 1:Hout, w in 1:Wout, c in 1:Cout] |> vec
    b_param = ExaModels.parameter(core, bias_expanded)
    c1 = ExaModels.constraint(
        core,
        y[i] - b_param[i] for i in 1:n_out;
        lcon = 0.0,
        ucon = 0.0,
    )
    # Index maps (created once, reused)
    cout_map = ExaModels.parameter(
        core,
        Float64[fld1(i, n_spatial) for i in 1:n_out],
    )
    sp_map = ExaModels.parameter(
        core,
        Float64[mod1(i, n_spatial) for i in 1:n_out],
    )
    in_map = Vector{Float64}(undef, n_spatial)
    for kh in 1:kH, kw in 1:kW, cin in 1:Cin
        w_vec = [p.weight[kh, kw, cout, cin] for cout in 1:Cout]
        all(iszero, w_vec) && continue
        # Build input-index map for valid output positions
        fill!(in_map, 1.0)
        in_ch_base = (cin - 1) * Hin * Win
        for wo in 1:Wout
            w_num = wo - kw + pW
            (w_num < 0 || w_num % sW != 0) && continue
            wi = w_num ÷ sW + 1
            (1 <= wi <= Win) || continue
            in_col_base = in_ch_base + (wi - 1) * Hin
            @inbounds for ho in 1:Hout
                h_num = ho - kh + pH
                (h_num < 0 || h_num % sH != 0) && continue
                hi = h_num ÷ sH + 1
                (1 <= hi <= Hin) || continue
                in_map[(wo - 1) * Hout + ho] = Float64(in_col_base + hi)
            end
        end
        w_p = ExaModels.parameter(core, w_vec)
        in_p = ExaModels.parameter(core, in_map)
        ExaModels.constraint!(
            core,
            c1,
            i => -w_p[cout_map[i]] * x[in_p[sp_map[i]]] for i in 1:n_out
        )
    end
    return y, MathOptAI.Formulation(p, Any[y], Any[c1])
end
