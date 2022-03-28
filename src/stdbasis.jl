using LinearAlgebra
import LinearAlgebra: dot
# using StaticArrays: StaticVector, StaticMatrix

export StdUnitVector, StdBasis, ⊗

"""
    StdUnitVector{D,T}([::Safety=Safe,] d::Int)

Represents a standard unit vector of a vector space over `T`.

The unit vector points along the `d`th dimension of the vector space. The `d`th
element of a `StdUnitVector` is equal to `one(T)` and all others are equal to
`zero(T)`.  Note that, upon construction, `T` is automatically narrowed to the
narrowest element type that fully represents elements of the basis vectors using
`suv_eltype`.  For example, if the vector space is over the `Real` numbers,

    StdUnitVector{D}(Real, 1) === StdUnitVector{D,Bool}(1) == [1, 0]

Setting `safety` to `Unsafe()` skips checking if `1 ≤ d ≤ D`, which may improve
performace.

Values of type `T` are assumed to be members of an algebraic field.   If `T` is
not supplied, it is defaulted to `Real`.
"""
struct StdUnitVector{D,T} <: AbstractVector{T}
    direction::Int

    # For internal use; does not narrow T.  Use outer constructors instead.
    @inline StdUnitVector{D,T}(s::Safety, d::Int) where {D,T} =
        new{valid_space_ndims(D), suv_eltype(T)}(valid_space_dim(s,D,d))
end
# TODO scaled unit vectors?
# TODO unit vectors are also (multi)linear maps... (Uh oh, Ouroborio!)

# Safety first!
@inline StdUnitVector{D,T}(d::Int) where {D,T} =
    StdUnitVector{D,T}(SAFE, d::Int)

# Assume vector space is isomporphic to ℝᴰ by default.
@inline StdUnitVector{D}(s::Safety, d::Int) where D =
    StdUnitVector{D,Real}(s, d)

@inline StdUnitVector{D}(d::Int) where D =
    StdUnitVector{D,Real}(SAFE, d)

"""
    suv_eltype(T::Type)::Type

Compute the minimal eltype of a standard unit vector belonging to a vector
space over a given type `T`.  For example `suv_eltype(Real) === Bool`
because a standard basis over the real numbers is minimally expressed as

    (𝐞̂₁, 𝐞̂₂) == (Bool[1, 0], Bool[0, 1])

An error is thrown if an appropriate element type is unknown because no
corresponding method for `suv_eltype(T)` has been implemented.
"""
@inline suv_eltype(::Type{T}) where {T<:Real} = Bool

# @inline suv_eltype(::Type{<:Complex}) = Complex{Bool}

@noinline suv_eltype(::Type{T}) where {T<:Number} =
    error("Narrowest element type of a standard unit vector of a vector space " *
          "over the field of $T numbers is unknown.")
@noinline suv_eltype(::Type{T}) where T =
    error("Narrowest element type of vector space over the field of $T values is unknown.")

# Validate that the number of spatial dimensions of a vector space is nonegative
@inline valid_space_ndims(D) =
    (D isa Int && D > 0 || _space_ndims_err(D); return D)
@noinline _space_ndims_err(D) =
    throw(DomainError(D, "Dimension of vector space must be a positive `Int`"))

# Validate that the spatial dimension in which a unit vector points is
# appropriate for a vector space with `D` dimensions.
@inline valid_space_dim(::Safe, D::Int, d::Int) =
    (1 ≤ d ≤ D || _space_dim_err(D, d); return d)
# ... Or don't validate for performance reasons.
@inline valid_space_dim(::Unsafe, D::Int, d::Int) = d

@noinline _space_dim_err(D, d) =
    throw(DomainError(d, "No vector in $(d)th dimension of a $D-dimensional vector space."))

"""
    direction(e::StdUnitVector)::Int

Returns an `Int` indicatring the direction in which `e` points.
"""
@inline direction(e::StdUnitVector) = e.direction

# Array Interface

Base.IndexStyle(::Type{<:StdUnitVector}) = IndexLinear()

@inline Arr.axes_types(::Type{<:StdUnitVector{D}}) where {D} = Tuple{Arr.SOneTo{D}}
@inline Arr.axes(e::StdUnitVector) = tuple(Arr.axes_types(e, static(1))())

@inline Base.axes(e::StdUnitVector) = Arr.axes(e::StdUnitVector)

@inline Base.length(::StdUnitVector{D}) where {D} = D
@inline Base.size(e::StdUnitVector) = dynamic(Arr.size(e))

@inline Base.getindex(e::StdUnitVector, ::Colon) = e

@inline function Base.getindex(e::StdUnitVector, i::Int)
    @boundscheck checkbounds(e, i)
    direction(e) == i
end

@inline Base.getindex(::StdUnitVector, ::Any) = error("Not implemented")
# @inline Base.getindex(e::StdUnitVector, i...) = SubArray(e, i)

# Avoid indirection of the SubArray wrapper
@inline Base.view(e::StdUnitVector, ::Colon) = e

Base.show(io::IO, e::StdUnitVector) =
    print(io, typeof(e), "(", direction(e), ")")
Base.show(io::IO, ::MIME"text/plain", e::StdUnitVector) =
    print(io,
          "𝐞̂_", direction(e), " ",
          "(standard unit vector of a vector space {>:Bool}^$(length(e)))")

# The dot product

@inline dot(e1::StdUnitVector, e2::StdUnitVector) =
    (samesize(e1, e2); e1 === e2)

@inline dot(e::StdUnitVector, v::AbstractVector) =
    _dot(samesize(e,v), promote_eltype(e,v), e, v)

@inline dot(v::AbstractVector, e::StdUnitVector) = dot(e, v)  # commute

@inline function _dot(::Size{0}, T::Type, e::StdUnitVector, v::AbstractVector)
    @assert false   # Presently, zero dimensional unit vectors not defined
    zero(T)
end

@inline function _dot(::Size, ::Type{T}, e::StdUnitVector, v::AbstractVector) where {T<:Real}
    @inbounds v[direction(e)]::T
end

# TODO: add specialized arithmetic operations +, -, scalar and matrix
# multiplication (*), etc.  For example, (+) should spit out a `StaticVector`
# since the size is known.  Right now, usual `Array`s are emitted.


"""
    Basis{D,T}

Represent a `D`-dimensional basis for a vector space over a field of type `T`.
"""
abstract type Basis{D,T} end

@inline Base.length(::Basis{D}) where D = static(D)
@inline field(::Basis{<:Any, T}) where T = T
@inline Base.firstindex(::Basis) = 1
@inline Base.lastindex(sb::Basis) = length(sb)

# Made of three vectors that span the vector space for which
#     det([ 𝐞₁  𝐞₂  ⋯  𝐞ₙ ]) ≠ 0
# but are otherwise arbitrary.
struct GeneralBasis{D,T,L #= D*D =#} <: Basis{D,T}
    data::NTuple{L,T}  # Some representation of a statically-sized matrix
    GeneralBasis(::Unsafe, ::Tuple) = error("Not implemented.")
    # UNSAFE because basis vectors should be checked if they span
    # whatever field (e.g. ℝᴰ) that the vector space is over.
end

# Dual basis, so we can enforce invariance under changes of basis / coordinate
# system.
struct Dual{D,T,B<:Basis,L #= D*D =#} <: Basis{D,T}
    parent::B
    data::NTuple{L,T}
    Dual(::Basis) = error("Not implemented")
    # Also needs some method to compute the dual basis, possibly at the same as
    # constructuion so they can be linked.  The rows of the inverse transpose of
    # the column-wise matrix of vectors formed by the original basis is the dual
    # basis.
    #     Dual(b::GeneralBasis) = new{...}(b, inv(matrix(::GeneralBasis))')
end

# Some algorithms can probably be made more efficient if we know that a given
# basis is orthonormal (ON).
struct ONBasis{D,T,L #= D*D =#}
    data::NTuple{L,T}
    GeneralBasis(::Unsafe, ::Tuple) = error("Not implemented.")
    # Unsafe in that the caller must guarantee that the basis is actually
    # orthonormal
end

"""
    StdBasis{D,T}()
    StdBasis{D}(::Type{T})

The set of "standard basis vectors"; the ordered collection of each possible
instance of `StdUnitVector{D,T}`.

See also `StdUnitVector` and `Basis`.
"""
struct StdBasis{D,T} <: Basis{D,T}
    function StdBasis{D,T}() where {D,T}
        new{valid_space_ndims(D), T}()
    end
end

# Because writing StdBasis{D}(T::Type) might be easier to remember than
# StdBasis{D,T}() --- it's easy to accidentally omit the trailing `()`.
@inline StdBasis{D}(::Type{T}) where {D,T} = StdBasis{D,T}()

@inline StdBasis{D}() where D = StdBasis{D}(Real)  # basis in ℝᴰ by default

@inline Base.eltype(::Type{StdBasis{D,T}}) where {D,T} = StdUnitVector{D,T}

# @inline Base.IteratorSize(::Type{<:StdBasis}) = Base.HasShape{1}()

# @inline Base.ndims(::Type{<:StdBasis}) = 1
# @inline Base.ndims(::SB) where {SB<:StdBasis} = Base.ndims(SB)

@inline Base.iterate(b::StdBasis{D}, i::Int=1) where D =
    i ≤ D ? (eltype(b)(UNSAFE, i), i+1) : nothing

# Array Interface

# @inline Arr.axes_types(::Type{T}) where {T<:StdBasis} =
#     Arr.axes_types(eltype(T))
# @inline Arr.axes(sb::StdBasis) = Arr.axes(first(sb))
#
# @inline Base.axes(sb::StdBasis) = Arr.axes(sb::StdBasis)

# @inline Base.size(sb::StdBasis, dim...) = dynamic(Arr.size(sb, dim...))

@inline _getindex(✓::Safety, sb::StdBasis, i::Int) = eltype(sb)(✓, i)

Base.@propagate_inbounds Base.getindex(sb::StdBasis, i::Int) =
    _getindex(inbounds_safety(), sb, i)

Base.Tuple(sb::StdBasis{D}) where D = NTuple{D}(sb)

# @inline Base.getindex(sb::StdBasis, ::Colon, ::Colon) = sb
#
# @inline function Base.getindex(sb::StdBasis{D,T}, ::Colon, j::Int) where {D,T}
#     @boundscheck checkbounds(sb, :, j)
#     StdUnitVector{D,T}(UNSAFE, j)
# end
# # Symmetry of the standard basis vector matrix.
# @inline Base.getindex(sb::StdBasis, i::Int, ::Colon) = sb[:,i]
#
# @inline function Base.getindex(sb::StdBasis, i::Int, j::Int)
#     @boundscheck checkbounds(sb, i, j)
#     @inbounds sb[:,j][i]
# end
#
# @noinline Base.getindex(::StdBasis, i::Any, j::Any) = error("Not implemented")
#
# @inline Base.view(sb::StdBasis, I::Vararg{Union{Colon,Int}}) =
#     Base.getindex(sb, I...)

Base.show(io::IO, sb::StdBasis) =
    print(io, Union{typeof(sb), StdBasis}, "{", length(sb), "}",
          "(", field(sb), ")")
function Base.show(io::IO, ::MIME"text/plain", sb::StdBasis)
    D = length(sb)
    T = eltype(eltype(sb))
    print(io,
          "Standard basis of a vector space {<:", T, "}^", D, ":\n",
          "  {", join(ntuple(i -> "𝐞̂$(subscripts(i))", Val(D)), ", "), "}")
end


"""
Lazy representation of a tensor product of vectors.
"""
struct TensorProduct{T, N, VV<:NTuple{N,AbstractVector}} <: AbstractArray{T,N}
    operands::VV

    function TensorProduct(vs::Vararg{Union{AbstractVector,TensorProduct}})
        # sz = map(Arr.static_length, vs)
        vs♭ = tuplejoin_deep(map(operands(TensorProduct), vs))
        new{promote_eltype(vs♭...), length(vs♭), typeof(vs♭)}(vs♭)
    end
end

⊗(vs::Union{AbstractVector, TensorProduct}...) = TensorProduct(vs...)

# Array Interface

Base.IndexStyle(::Type{<:TensorProduct}) = IndexCartesian()

@generated Arr.axes_types(::Type{<:TensorProduct{<:Any,N,VV}}) where {N,VV} =
    Tuple{ntuple(i -> Arr.axes_types(fieldtype(VV, i), static(1)), Val(N))...}

@inline Arr.axes(vv::TensorProduct) =
    map(operand -> Arr.axes(operand, static(1)), vv.operands)

@inline Base.axes(vv::TensorProduct) = Arr.axes(vv::TensorProduct)

@inline Base.size(vv::TensorProduct) = dynamic(Arr.size(vv))

@inline function Base.getindex(vv::TensorProduct, I::Vararg{Int,N}) where N
    @boundscheck checkbounds(vv, I...)
    ops = vv.operands
    prod(ntuple(i -> ops[i][I[i]], Val(ndims(vv))))
end

Base.show(io::IO, vv::TensorProduct) =
    print(io, "TensorProduct{", eltype(vv), ", ", ndims(vv), "}",
          "(", join(vv.operands, ", ") , ")")

Base.show(io::IO, ::MIME"text/plain", vv::TensorProduct) =
    print(io, join(size(vv), "×"), " ",
          "TensorProduct{", eltype(vv), ", ", ndims(vv) ,"}:\n  ",
          join(vv.operands, " ⊗ "))


struct TensorProductBasis{D, T, BB<:TupleN{Basis}} <: Basis{D, T}
    bases::BB
    function TensorProductBasis(bases::Vararg{Basis})
        D = mapreduce(length, *, bases)
        T = promote_type(map(field, bases)...)
        new{D, T, typeof(bases)}(bases)
    end
end

function Base.show(io::IO, bb::TensorProductBasis)
    print(io, Union{typeof(bb), TensorProductBasis},
          "{", length(bb), ",", field(bb), "}", bb.bases)
end
