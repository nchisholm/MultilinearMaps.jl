import Base
import LinearAlgebra: dot

export StandardBasis

# Needed if we want to allow the recursive parent-child pattern relationship
# between StandardBasis and StandardUnitVector
# abstract type AbstractBasis{V<:AbstractVector} <: AbstractVector{V} end

"""
    StandardUnitVector(i...)

Represents a standard unit vector of a vector space formed by the
tensor product of `N` vector spaces.
"""
struct StandardUnitVector <: AbstractVector{Bool}
    length::Int
    i::Int
    function StandardUnitVector(length::Integer, i::Integer)
        0 < i ≤ length || throw(DomainError("No dimension $i in vector space with dimension $length"))
        new(length, i)
    end
end

const StdUnitVec = StandardUnitVector

Base.size(e::StdUnitVec) = (e.length,)

Base.IndexStyle(::Type{<:StdUnitVec}) = IndexLinear()

function Base.getindex(e::StdUnitVec, i::Int)
    @boundscheck checkbounds(e, i)
    e.i == i
end

Base.show(io::IO, e::StdUnitVec) = print(io, typeof(e), "(", e.i, ")")
Base.show(io::IO, ::MIME"text/plain", e::StdUnitVec) =
    print(io, e, "\n  ", "𝐞̂", e.i,
          " ", "(standard unit vector of a vector space of dimension $(length(e)))")

function dot(e1::StdUnitVec, e2::StdUnitVec)
    length(e1) == length(e2) ||
        throw(DimensionMismatch("length of the first vector ($(length(e1))) does not match the length of the second ($(length(e2)))"))
    length(e1) == 0 ? false : e1.i == e2.i
end
# dot(e1::StdUnitVec, e2::StdUnitVec) => throw an error?

function dot(e::StdUnitVec, v::AbstractVector)
    length(e) == length(v) ||
        throw(DimensionMismatch("standard unit vector has length $(length(e)), which does not match the length of the array $(length(v))"))
    length(e) == 0 ? zero(eltype(v)) : v[e.i]
end

dot(v::AbstractVector, e::StdUnitVec) = dot(e, v)  # commute


"""
    StandardBasis(n)

Represents the standard or canoncical basis of a vector space with dimension
`n`, and serves as a collection of the standard unit vectors.
"""
struct StandardBasis <: AbstractVector{StandardUnitVector}
    ndims::Int
    function StandardBasis(n::Integer)
        n ≥ 0 || throw(DomainError("Dimension of vector space must be nonnegative"))
        new(n)
    end
end

Base.size(basis::StandardBasis) = (basis.ndims,)

Base.IndexStyle(::Type{<:StandardBasis}) = IndexLinear()

Base.getindex(basis::StandardBasis, i::Integer) =
    StandardUnitVector(basis.ndims, i)
