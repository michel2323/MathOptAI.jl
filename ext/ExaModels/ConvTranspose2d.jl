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
    n_in = Hin * Win * Cin
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
    # Augment column-by-column (same Affine pattern as Conv2d)
    for j in 1:n_in
        hi = mod1(j, Hin)
        wi = fld1(mod1(j, Hin * Win), Hin)
        ci = fld1(j, Hin * Win)
        # Input pixel (hi, wi, ci) "stamps" the kernel onto output
        w_col = zeros(Float64, n_out)
        for kh in 1:kH, kw in 1:kW
            ho = (hi - 1) * sH + kh - pH
            wo = (wi - 1) * sW + kw - pW
            if 1 <= ho <= Hout && 1 <= wo <= Wout
                for co in 1:Cout
                    out_idx = (co - 1) * Hout * Wout + (wo - 1) * Hout + ho
                    w_col[out_idx] += p.weight[kh, kw, co, ci]
                end
            end
        end
        if all(iszero, w_col)
            continue
        end
        w_param = ExaModels.parameter(core, w_col)
        xj = x[j]
        ExaModels.constraint!(core, c1, i => -w_param[i] * xj for i in 1:n_out)
    end
    return y, MathOptAI.Formulation(p, Any[y], Any[c1])
end
