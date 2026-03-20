# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.BatchNorm2d,
    x,
)
    H, W, C = p.input_size
    n = H * W * C
    # Expand per-channel scale/bias to full spatial extent
    scale_expanded = [p.scale[div(i - 1, H * W) + 1] for i in 1:n]
    bias_expanded = [p.bias[div(i - 1, H * W) + 1] for i in 1:n]
    s_param = ExaModels.parameter(core, scale_expanded)
    b_param = ExaModels.parameter(core, bias_expanded)
    y = ExaModels.variable(core, n)
    c1 = ExaModels.constraint(
        core,
        y[i] - s_param[i] * x[i] - b_param[i] for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.BatchNorm2d},
    x,
)
    inner = p.predictor
    H, W, C = inner.input_size
    n = H * W * C
    y = Vector{Any}(undef, n)
    for c in 1:C
        s, b = inner.scale[c], inner.bias[c]
        for i in ((c - 1) * H * W + 1):(c * H * W)
            y[i] = s * x[i] + b
        end
    end
    return y, MathOptAI.Formulation(p)
end
