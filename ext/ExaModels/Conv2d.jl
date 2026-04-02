# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Conv2d,
    x,
)
    (Hin, Win, Cin) = p.input_size
    kH, kW, Cin_w, Cout = size(p.weight)
    @assert Cin == Cin_w
    (pH, pW), (sH, sW) = p.padding, p.stride
    Hout = fld(Hin + 2 * pH - kH, sH) + 1
    Wout = fld(Win + 2 * pW - kW, sW) + 1
    n_out = Hout * Wout * Cout
    n_spatial = Hout * Wout
    # Output variables
    y = ExaModels.variable(core, n_out)
    # Base constraint: y[i] = bias[channel]
    bias_expanded = [p.bias[c] for h in 1:Hout, w in 1:Wout, c in 1:Cout] |> vec
    b_param = ExaModels.parameter(core, bias_expanded)
    c1 = ExaModels.constraint(
        core,
        y[i] - b_param[i] for i in 1:n_out;
        lcon = 0.0,
        ucon = 0.0,
    )
    # ── Sparse augmentation ──────────────────────────────────────────────
    #
    # For each kernel position (m, n, cin), ONE constraint! call augments
    # ALL n_out constraints at once using a flat 1:n_out range iterator.
    #
    # Two "index map" parameters (created ONCE, reused across all calls)
    # decompose the flat index i into its spatial and channel components:
    #   cout_map[i] = which output channel  (for w_p indexing)
    #   sp_map[i]   = spatial position      (for in_p indexing)
    #
    # Per-(m,n,cin) parameters:
    #   w_p   – Cout-length:     kernel weight per output channel
    #   in_p  – n_spatial-length: input variable index per spatial position
    #
    # Expression:  i => -w_p[cout_map[i]] * x[in_p[sp_map[i]]]
    #
    # All closures capture Parameter{S,O} objects with identical types,
    # so Julia compiles the SIMDFunction ONCE and reuses it.
    # The iterator 1:n_out is an AbstractRange → _adapt_gen skips collect().
    #
    # constraint! calls: kH*kW*Cin (e.g. 36)
    # Parameter storage:  2*n_out + kH*kW*Cin*(Cout + n_spatial)
    cout_map_vec = Float64[fld1(i, n_spatial) for i in 1:n_out]
    sp_map_vec = Float64[mod1(i, n_spatial) for i in 1:n_out]
    cout_map = ExaModels.parameter(core, cout_map_vec)
    sp_map = ExaModels.parameter(core, sp_map_vec)
    in_map = Vector{Float64}(undef, n_spatial)
    for m in 1:kH, n in 1:kW, cin in 1:Cin
        # Valid output spatial range
        ho_min = max(1, cld(1 - m + pH, sH) + 1)
        ho_max = min(Hout, fld(Hin - m + pH, sH) + 1)
        wo_min = max(1, cld(1 - n + pW, sW) + 1)
        wo_max = min(Wout, fld(Win - n + pW, sW) + 1)
        (ho_min > ho_max || wo_min > wo_max) && continue
        # Weight vector (Cout elements)
        w_vec = [p.weight[kH - m + 1, kW - n + 1, cin, cout] for cout in 1:Cout]
        all(iszero, w_vec) && continue
        # Input-index map (n_spatial elements, shared across cout)
        fill!(in_map, 1.0)  # dummy for boundary
        in_ch_base = (cin - 1) * Hin * Win
        for wo in wo_min:wo_max
            wi = sW * (wo - 1) + n - pW
            in_col_base = in_ch_base + (wi - 1) * Hin
            @inbounds for ho in ho_min:ho_max
                in_map[(wo - 1) * Hout + ho] =
                    Float64(in_col_base + sH * (ho - 1) + m - pH)
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
