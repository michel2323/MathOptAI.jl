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
    n_in = Hin * Win * Cin
    # Output variables
    y = ExaModels.variable(core, n_out)
    # Base: y[i] - bias[channel] = 0
    bias_expanded = [p.bias[c] for h in 1:Hout, w in 1:Wout, c in 1:Cout] |> vec
    b_param = ExaModels.parameter(core, bias_expanded)
    c1 = ExaModels.constraint(
        core,
        y[i] - b_param[i] for i in 1:n_out;
        lcon = 0.0,
        ucon = 0.0,
    )
    # Augment column-by-column: for each input pixel j, compute the weight
    # vector w_j of length n_out where w_j[i] is the weight connecting input j
    # to output i (zero if not connected). Then: y[i] -= w_j[i] * x[j]
    # This follows the exact same pattern as Affine.jl.
    idx_map = MathOptAI.PaddedArrayView(
        reshape(collect(1:n_in), Hin, Win, Cin),
        p.padding,
    )
    for j in 1:n_in
        # Find which (k, wi, hi) this linear index corresponds to
        hi = mod1(j, Hin)
        wi = fld1(mod1(j, Hin * Win), Hin)
        k = fld1(j, Hin * Win)
        # Build weight column: which output pixels does input j contribute to?
        w_col = zeros(Float64, n_out)
        for m in 1:kH, n in 1:kW
            # Output pixel that uses input (hi,wi) at kernel position (m,n):
            # hi = sH*(ho-1) + m - pH  =>  ho = (hi + pH - m)/sH + 1
            h_num = hi + pH - m
            w_num = wi + pW - n
            if h_num >= 0 && h_num % sH == 0 && w_num >= 0 && w_num % sW == 0
                ho = h_num ÷ sH + 1
                wo = w_num ÷ sW + 1
                if 1 <= ho <= Hout && 1 <= wo <= Wout
                    for c in 1:Cout
                        out_idx = (c - 1) * Hout * Wout + (wo - 1) * Hout + ho
                        w_col[out_idx] += p.weight[kH - m + 1, kW - n + 1, k, c]
                    end
                end
            end
        end
        # Skip if this input pixel contributes nothing (e.g., all zero weights)
        if all(iszero, w_col)
            continue
        end
        w_param = ExaModels.parameter(core, w_col)
        xj = x[j]
        ExaModels.constraint!(core, c1, i => -w_param[i] * xj for i in 1:n_out)
    end
    return y, MathOptAI.Formulation(p, Any[y], Any[c1])
end
