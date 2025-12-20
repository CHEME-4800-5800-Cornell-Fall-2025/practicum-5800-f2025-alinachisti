# src/Types.jl

export MyClassicalHopfieldNetworkModel

"""
    MyClassicalHopfieldNetworkModel

Container for a classical Hopfield network.

Fields
------
- `memories::Array{Int32,2}`   : N×P matrix of stored bipolar patterns (-1, +1),
                                 each column is one encoded image.
- `W::Array{Float32,2}`        : N×N Hopfield weight matrix.
- `b::Vector{Float32}`         : N-length bias vector (all zeros for classical Hopfield).
- `energy::Vector{Float32}`    : length-P vector; energy of each stored pattern.
- `number_of_pixels::Int`      : N, number of neurons / pixels.
- `number_of_images::Int`      : P, number of stored images.
"""
mutable struct MyClassicalHopfieldNetworkModel
    memories::Array{Int32,2}
    W::Array{Float32,2}
    b::Vector{Float32}
    energy::Vector{Float32}
    number_of_pixels::Int
    number_of_images::Int
end

# Optional generic alias (not required by the notebook, but harmless)
const ClassicalHopfieldNetwork = MyClassicalHopfieldNetworkModel
