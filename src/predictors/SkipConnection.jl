# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SkipConnection(inner::AbstractPredictor) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = f(x) + x
```
where \$f\$ is an [`AbstractPredictor`](@ref) and \$x\$ is the input.

The inner predictor must produce output of the same dimension as its input.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.SkipConnection(
           MathOptAI.Pipeline(
               MathOptAI.Affine([1.0 0.0; 0.0 1.0], [0.1, 0.2]),
               MathOptAI.ReLU(),
           ),
       )
SkipConnection(Pipeline with layers:
 * Affine(A, b) [input: 2, output: 2]
 * ReLU())

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SkipConnection[1]
 moai_SkipConnection[2]
```
"""
struct SkipConnection <: AbstractPredictor
    inner::AbstractPredictor
end

function Base.show(io::IO, p::SkipConnection)
    return print(io, "SkipConnection(", p.inner, ")")
end

output_size(p::SkipConnection, input_size) = output_size(p.inner, input_size)

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::SkipConnection,
    x::Vector,
)
    z, inner_form = add_predictor(model, predictor.inner, x)
    if length(z) != length(x)
        error(
            "SkipConnection requires inner predictor output dimension " *
            "($(length(z))) to match input dimension ($(length(x)))",
        )
    end
    y = add_variables(model, x, length(x), "moai_SkipConnection")
    cons = JuMP.@constraint(model, y .== z .+ x)
    return y, Formulation(predictor, Any[inner_form; y], Any[cons])
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:SkipConnection},
    x::Vector,
)
    inner = predictor.predictor
    z, _ = add_predictor(model, ReducedSpace(inner.inner), x)
    y = JuMP.@expression(model, z .+ x)
    return y, Formulation(predictor)
end
