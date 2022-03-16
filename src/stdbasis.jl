using LinearAlgebra
import LinearAlgebra: dot
import StaticArrays: StaticVector

export StdUnitVector, standardbasis


# TODO: build these up
#
# abstract type Basis{D} end
# struct StandardBasis{D} end
#
# Could define `iterate` and `getindex` on singleton types <: `StandardBasis{N}
# where N`, to produce the `N` standard unit vectors.  We would also have
# dualbasis(StandardBasis{3}) === StandardBasis{3}()
# Then, we could work with to arbitrary sets of basis vectors ...

struct StdUnitVector{D} <: StaticVector{D,Bool}
    direction::Int
    @inline function StdUnitVector{D}(::Unsafe, d::Int) where D
        D isa Int && D > 0 || _throw_dimensionality_error(D)
        new(d)
    end
end
# Could also define the notion of a scaled unit vector

@inline function StdUnitVector{D}(::Safe, d::Int) where D
    1 ≤ d ≤ D || _throw_dims_error(D, d)
    StdUnitVector{D}(UNSAFE, d::Int)
end

@inline StdUnitVector{D}(d::Int) where D = StdUnitVector{D}(SAFE, d::Int)


# @inline _check_dimensionality(::Val{D}) where D =
#     D isa Int && D > 0 || _throw_dimensionality_error(D)
# @inline _check_direction(::Val{D}, d::Int) where D =
#     1 ≤ d ≤ D || _throw_dims_error(D, d)

@noinline _throw_dimensionality_error(D) =
    throw(DomainError(D, "Number of dimensions `D` must be a positive `Int`"))
@noinline _throw_dims_error(D, d) =
    throw(DomainError(d, "No vector in $(d)th dimension of a basis spanning ℝ^$D"))

"""
Return the `N` standard unit vectors of an `N`-dimensional standard
basis.
"""
@inline standardbasis(N::Int) = ntuple(i -> StdUnitVector{N}(i), Val(N))
@inline standardbasis(N::Integer) = standardbasis(convert(Int, N))

# TODO: implement multidimensional basis sets
# (like Cartesian indices, but over basis vectors)
@inline standardbasis(Ns::TupleN{<:Integer}) = map(standardbasis, Ns)
# @inline standardbasis(::TupleN{<:Integer}) =
#     error("Not implemented: multidimensional standard basis not yet supported.")

@inline standardbasis(Ns::Vararg{<:Integer}) = standardbasis(Ns)

@inline standardbasis(iter, args...) =
    standardbasis(Arr.size(iter, args...))

"""
    direction(e::StdUnitVector)::Int

Returns an `Int` indicatring the direction in which `e` points.
"""
@inline direction(e::StdUnitVector) = e.direction

# @inline Base.length(::StdUnitVector{D}) where D = D
# @inline Base.size(e::StdUnitVector) = (length(e),)
# Base.IndexStyle(::StdUnitVector) = IndexLinear()

Base.:(==)(es::StdUnitVector...) = ===(es...)

@inline function Base.getindex(e::StdUnitVector, i::Int)
    @boundscheck checkbounds(e, i)  # NOTE: uses `size(e)`
    direction(e) == i
end

Base.show(io::IO, e::StdUnitVector{D}) where D = print(io, "𝐞̂{$D}_$(direction(e))")

# The dot product

@inline dot(e1::StdUnitVector, e2::StdUnitVector) =
    (_SA.same_size(e1, e2); e1 === e2)

# @inline dot(e::StdUnitVector{D}, v::StaticVector{D}) where D =
#     (@boundscheck _check_dot(e, v); @inbounds v[direction(e)])

@inline dot(e::StdUnitVector, v::StaticVector) =
    _SA._vecdot(_SA.same_size(e, v), e, v, dot)
@inline dot(v::StaticVector, e::StdUnitVector) = dot(e, v)

@inline function _SA._vecdot(sz::_SA.Size, a::StdUnitVector, b::StaticArray, ::typeof(dot))
    # eltype(StdUnitVector) == `Bool` so...
    @assert promote_type(eltype(a), eltype(b)) === eltype(b)
    if _SA.Length(sz) == 0  # No elements!
        # should be unreachable because there is no zero-dimensional unit vector
        zero(eltype(b))
    else
        @inbounds b[direction(a)]
    end
end

@inline dot(e::StdUnitVector, v::AbstractArray) =
    dot(e, _SA.SizedArray{_SA.size_tuple(_SA.Size(e))}(v))
@inline dot(v::AbstractArray, e::StdUnitVector) = dot(e, v)

# TODO: add specialized arithmetic operations +, -,
# scalar and matrix multiplication (*), etc.

# For example, (+) should spit out a `StaticVector` since the size is known.
# Right now, usual `Array`s are emitted.
