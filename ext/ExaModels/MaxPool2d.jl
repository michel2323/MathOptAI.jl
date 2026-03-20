# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.MaxPool2d,
    x,
)
    (Hin, Win, C) = p.input_size
    (kH, kW), (pH, pW), (sH, sW) = p.kernel_size, p.padding, p.stride
    Hout = fld(Hin + 2 * pH - kH, sH) + 1
    Wout = fld(Win + 2 * pW - kW, sW) + 1
    n_out = Hout * Wout * C
    idx_map = MathOptAI.PaddedArrayView(
        reshape(collect(1:(Hin * Win * C)), Hin, Win, C),
        p.padding,
    )
    y = ExaModels.variable(core, n_out)
    # Build max expression for each output pixel and constrain y[i] == max(...)
    # ExaModels supports `max` in constraint generators.
    for c in 1:C, w in 1:Wout, h in 1:Hout
        out_idx = (c - 1) * Hout * Wout + (w - 1) * Hout + h
        # Collect input indices in this pooling window
        x_indices = Int[]
        for n_k in 1:kW, m_k in 1:kH
            xi = idx_map[sH*(h-1)+m_k, sW*(w-1)+n_k, c]
            if !iszero(xi)
                push!(x_indices, xi)
            end
        end
        # Build nested max
        expr = x[x_indices[1]]
        for i in 2:length(x_indices)
            expr = max(expr, x[x_indices[i]])
        end
        ExaModels.constraint(core, y[out_idx] - expr for j in 1:1; lcon = 0.0, ucon = 0.0)
    end
    return y, MathOptAI.Formulation(p, Any[y], Any[])
end
