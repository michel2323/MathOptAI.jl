# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    BatchNorm2d(
        scale::Vector{T},
        bias::Vector{T};
        input_size::Tuple{Int,Int,Int},
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents a batch normalization layer in
evaluation mode. At eval time, batch normalization is a per-channel affine
transform applied identically at every spatial position:

```math
y_{h,w,c} = \\text{scale}_c \\cdot x_{h,w,c} + \\text{bias}_c
```

where `scale` and `bias` are precomputed from the learned weight/bias and the
running mean/variance:

```math
\\text{scale}_c = \\frac{\\text{weight}_c}{\\sqrt{\\text{running\\_var}_c + \\varepsilon}}
\\qquad
\\text{bias}_c = \\text{bias}_c - \\text{running\\_mean}_c \\cdot \\text{scale}_c
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:12]);

julia> predictor = MathOptAI.BatchNorm2d([2.0, 3.0], [0.1, 0.2]; input_size = (2, 3, 2))
BatchNorm2d((2, 3, 2), 2 channels)

julia> y, formulation = MathOptAI.add_predictor(model, predictor, x);

julia> length(y)
12
```
"""
struct BatchNorm2d{T} <: AbstractPredictor
    input_size::Tuple{Int,Int,Int}  # (H, W, C)
    scale::Vector{T}                # per-channel scale, length C
    bias::Vector{T}                 # per-channel bias, length C

    function BatchNorm2d(
        scale::Vector{T},
        bias::Vector{T};
        input_size::Tuple{Int,Int,Int},
    ) where {T}
        return new{T}(input_size, scale, bias)
    end
end

function Base.show(io::IO, p::BatchNorm2d)
    return print(io, "BatchNorm2d($(p.input_size), $(length(p.scale)) channels)")
end

output_size(p::BatchNorm2d, ::Any) = p.input_size

function (predictor::BatchNorm2d)(x::Vector)
    H, W, C = predictor.input_size
    @assert length(x) == H * W * C
    y = similar(x, length(x))
    for c in 1:C
        s, b = predictor.scale[c], predictor.bias[c]
        for i in ((c - 1) * H * W + 1):(c * H * W)
            y[i] = s * x[i] + b
        end
    end
    return y
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::BatchNorm2d,
    x::Vector,
)
    H, W, C = predictor.input_size
    m = H * W * C
    y = add_variables(model, x, m, "moai_BatchNorm2d")
    cons = JuMP.@constraint(
        model,
        [i in 1:m],
        y[i] == begin
            c = div(i - 1, H * W) + 1
            predictor.scale[c] * x[i] + predictor.bias[c]
        end,
    )
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:BatchNorm2d},
    x::Vector,
)
    inner = predictor.predictor
    H, W, C = inner.input_size
    m = H * W * C
    y = JuMP.@expression(
        model,
        [i in 1:m],
        begin
            c = div(i - 1, H * W) + 1
            inner.scale[c] * x[i] + inner.bias[c]
        end,
    )
    return y, Formulation(predictor)
end
