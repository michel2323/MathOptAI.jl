# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SkipConnection,
    x::ExaModels.AbstractVariable,
)
    z, inner_form = MathOptAI.add_predictor(core, p.inner, x)
    n = _length(x)
    y = ExaModels.variable(core, n)
    cons = ExaModels.constraint(
        core,
        y[i] - z[i] - x[i] for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y,
    MathOptAI.Formulation(
        p,
        [inner_form.variables; y],
        [inner_form.constraints; cons],
    )
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SkipConnection,
    x::AbstractVector,
)
    z, inner_form = MathOptAI.add_predictor(core, p.inner, x)
    n = length(x)
    y = ExaModels.variable(core, n)
    cons = [
        ExaModels.constraint(
            core,
            y[i] - z[i] - x[i];
            lcon = 0.0,
            ucon = 0.0,
        ) for i in 1:n
    ]
    return y,
    MathOptAI.Formulation(
        p,
        [inner_form.variables; y],
        [inner_form.constraints; cons...],
    )
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.SkipConnection},
    x,
)
    inner = MathOptAI.ReducedSpace(p.predictor.inner)
    z, _ = MathOptAI.add_predictor(core, inner, x)
    y = [z[i] + x[i] for i in 1:_length(x)]
    return y, MathOptAI.Formulation(p)
end
