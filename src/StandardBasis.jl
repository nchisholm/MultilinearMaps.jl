import LinearAlgebra: dot

export StandardBasis, StandardUnitVector

# Needed if we want to allow the recursive parent-child pattern relationship
# between StandardBasis and StandardUnitVector
# abstract type AbstractBasis{V<:AbstractVector} <: AbstractVector{V} end

"""
    StandardUnitVector(i, n)

Represents a standard unit vector of a vector space of dimension `n`.
"""
struct StandardUnitVector <: AbstractVector{Bool}
    i::Int
    length::Int
    function StandardUnitVector(i::Int, length::Int)
        # FIXME: check below causes (expected) performance overhead...
        # 1 ≤ i ≤ length || throw(ArgumentError("$i out of bounds with $length"))
        # This is OK as long as we use dynamic
        @boundscheck 1 ≤ i ≤ length || throw(DomainError(i, "invalid direction"))
        # TODO: make this error more like the BoundsError for an array
        new(i, length)
    end
end

StandardUnitVector(i, length) =
    StandardUnitVector(convert(Int, i), convert(Int, length))

@noinline _domain_error(i) = throw(DomainError(i, "out of bounds"))

const StdUnitVec = StandardUnitVector

Base.IndexStyle(::Type{<:StdUnitVec}) = IndexLinear()
Base.size(e::StdUnitVec) = (e.length,)

@inline function Base.getindex(e::StdUnitVec, i::Int)
    @boundscheck checkbounds(e, i)
    e.i == i
end

Base.show(io::IO, e::StdUnitVec) = print(io, typeof(e), "(", e.i, ")")
Base.show(io::IO, ::MIME"text/plain", e::StdUnitVec) =
    print(io, e, "\n  ", "𝐞̂", e.i,
          " ", "(standard unit vector of a $(length(e))-dimensional vector space")

Base.:(==)(e1::StdUnitVec, e2::StdUnitVec) = e1 === e2

@inline Base.@assume_effects :foldable function dot(e1::StdUnitVec, e2::StdUnitVec)
    length(e1) == length(e2) ||
        throw(DimensionMismatch("length of the first vector ($(length(e1))) does not match the length of the second ($(length(e2)))"))
    e1 == e2
    # ifelse(length(e1) == 0, false, e1 == e2)  # XXX: slower, but why?
end
# dot(e1::StdUnitVec, e2::StdUnitVec) => throw an error?

@inline function dot(e::StdUnitVec, v::AbstractVector)
    length(e) == length(v) ||
        throw(DimensionMismatch("standard unit vector has length $(length(e)), which does not match the length of the array $(length(v))"))
    v[e.i]
end

@inline dot(v::AbstractVector, e::StdUnitVec) = dot(e, v)  # commute

# TODO: matrix multiplication and `'` (hermetian) operations


"""
    StandardBasis(N)

Represents the standard or canoncical basis of a vector space with dimension
`N`, and serves as a collection of the standard unit vectors.
"""
struct StandardBasis
    ndims::Int
    function StandardBasis(n::Int)
        n ≥ 0 || throw(DomainError("Dimension of vector space must be nonnegative"))
        new(n)
    end
end

StandardBasis(n) = StandardBasis(convert(Int, n))
StandardBasis(n, dim) = StandardBasis(n)[dim]

Base.ndims(::Type{StandardBasis}) = 1
Base.IteratorSize(T::Type{StandardBasis}) = Base.HasShape{ndims(T)}()
Base.IteratorEltype(::Type{StandardBasis}) = StandardUnitVector
Base.IndexStyle(::Type{StandardBasis}) = IndexLinear()

Base.length(b::StandardBasis) = b.ndims
Base.size(b::StandardBasis) = (length(b),)

@inline Base.getindex(b::StandardBasis, i) =
    StandardUnitVector(i, b.ndims)
