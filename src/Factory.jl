# src/Factory.jl

export build

"""
    build(::Type{MyClassicalHopfieldNetworkModel}, config)

Construct a Hopfield network from a configuration NamedTuple.

Expected usage in the notebook:

    model = build(MyClassicalHopfieldNetworkModel, (
        memories = linearimagecollection,
    ))

where `linearimagecollection` is an Array{Int32,2} of size
(number_of_pixels, number_of_images_to_learn) with entries -1 or +1.
"""
function build(::Type{MyClassicalHopfieldNetworkModel}, config::NamedTuple)
    haskey(config, :memories) ||
        error("build(MyClassicalHopfieldNetworkModel, config): config must have a `memories` field.")

    # Ensure the memories array has the expected concrete type
    memories = Array{Int32,2}(config.memories)
    N, P = size(memories)

    # Hebbian learning rule:
    # X is N×P with columns as patterns (values -1/+1)
    X = Float32.(memories)
    W = (X * X') / N

    # Remove self-connections
    @inbounds for i in 1:N
        W[i, i] = 0.0f0
    end

    # Classical Hopfield: zero bias
    b = zeros(Float32, N)

    # Precompute energy of each stored pattern
    energy_vec = Vector{Float32}(undef, P)
    @inbounds for μ in 1:P
        s = view(X, :, μ)
        # E = -1/2 sᵀ W s - bᵀ s, but b = 0
        energy_vec[μ] = -0.5f0 * dot(s, W * s)
    end

    return MyClassicalHopfieldNetworkModel(memories, W, b, energy_vec, N, P)
end
