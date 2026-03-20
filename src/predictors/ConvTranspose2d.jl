# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ConvTranspose2d(
        weight::Array{T,4},
        bias::Vector{T};
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = (1, 1),
        padding::Tuple{Int,Int} = (0, 0),
        output_padding::Tuple{Int,Int} = (0, 0),
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents a 2-dimensional transposed
convolutional layer (sometimes called "deconvolution").

The transposed convolution is the gradient of a convolution with respect to its
input. It is commonly used for learned upsampling in encoder-decoder networks.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2])
2-element Vector{VariableRef}:
 x[1]
 x[2]

julia> weight = reshape([1.0, 2.0, 3.0, 4.0], 2, 1, 1, 1);

julia> bias = [0.5];

julia> predictor = MathOptAI.ConvTranspose2d(weight, bias; input_size = (1, 2, 1), stride = (1, 1))
ConvTranspose2d{Float64}((1, 2, 1), [1.0 3.0; 2.0 4.0;;;; ], [0.5], (1, 1), (0, 0), (0, 0))

julia> y, formulation = MathOptAI.add_predictor(model, predictor, vec(x));

julia> y
3-element Vector{VariableRef}:
 moai_ConvTranspose2d[1]
 moai_ConvTranspose2d[2]
 moai_ConvTranspose2d[3]
```
"""
struct ConvTranspose2d{T} <: AbstractPredictor
    input_size::Tuple{Int,Int,Int}  # (height, width, channels)
    weight::Array{T,4}             # (kH, kW, Cout, Cin)
    bias::Vector{T}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}
    output_padding::Tuple{Int,Int}

    function ConvTranspose2d(
        weight::Array{T,4},
        bias::Vector{T};
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = (1, 1),
        padding::Tuple{Int,Int} = (0, 0),
        output_padding::Tuple{Int,Int} = (0, 0),
    ) where {T}
        return new{T}(input_size, weight, bias, stride, padding, output_padding)
    end
end

function _convtranspose2d_output_size(f::ConvTranspose2d)
    (Hin, Win, _) = f.input_size
    kH, kW = size(f.weight, 1), size(f.weight, 2)
    Cout = size(f.weight, 3)
    (pH, pW), (sH, sW) = f.padding, f.stride
    (opH, opW) = f.output_padding
    Hout = (Hin - 1) * sH - 2 * pH + kH + opH
    Wout = (Win - 1) * sW - 2 * pW + kW + opW
    return Hout, Wout, Cout
end

function (f::ConvTranspose2d)(model::JuMP.AbstractModel, x::Vector)
    (Hin, Win, Cin) = f.input_size
    kH, kW, Cout, Cin_w = size(f.weight)
    @assert Cin == Cin_w
    (pH, pW), (sH, sW) = f.padding, f.stride
    Hout, Wout, _ = _convtranspose2d_output_size(f)
    X = reshape(x, Hin, Win, Cin)
    # For each output pixel, sum contributions from input pixels through the
    # transposed kernel. An input pixel (hi, wi) contributes to output pixel
    # (ho, wo) when: ho = (hi-1)*sH + kh - pH  and  wo = (wi-1)*sW + kw - pW
    # Inverting: kh = ho - (hi-1)*sH + pH  and  kw = wo - (wi-1)*sW + pW
    return JuMP.@expression(
        model,
        [ho in 1:Hout, wo in 1:Wout, co in 1:Cout],
        f.bias[co] + sum(
            f.weight[ho-(hi-1)*sH+pH, wo-(wi-1)*sW+pW, co, ci] * X[hi, wi, ci]
            for ci in 1:Cin,
                hi in 1:Hin,
                wi in 1:Win
            if (
                1 <= ho - (hi - 1) * sH + pH <= kH &&
                1 <= wo - (wi - 1) * sW + pW <= kW
            )
        ),
    )
end

function output_size(f::ConvTranspose2d, ::NTuple{2,Int})
    return _convtranspose2d_output_size(f)
end

function output_size(f::ConvTranspose2d, ::NTuple{3,Int})
    return _convtranspose2d_output_size(f)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ConvTranspose2d,
    x::Vector,
)
    Y = predictor(model, x)
    y = add_variables(model, x, length(Y), "moai_ConvTranspose2d")
    cons = JuMP.@constraint(model, [i in 1:length(Y)], y[i] == Y[i])
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:ConvTranspose2d},
    x::Vector,
)
    return vec(predictor.predictor(model, x)), Formulation(predictor)
end
