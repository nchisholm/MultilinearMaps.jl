module MultilinearForms

using Base: @propagate_inbounds
using StaticArrays

export AbstractMultilinearForm, MultilinearForm, ContractedMultilinearForm,
    dimension, order

include("util.jl")
include("stdbasis.jl")


abstract type AbstractMultilinearForm{K,D} end

# For now, everything is in Cartesian space
basis(::AbstractMultilinearForm{<:Any,D}) where {D} = basis(StdUnitVector{D})

# Convenient type aliases for passing sets of vectors
const FormArgs{K,D} = NTuple{K,StaticVector{D}}

# Helper method so we aren't forced to wrap arguments in a tuple. Note that the
# `::Vararg{K}` is required or things will infinitely recurse.
(mf::AbstractMultilinearForm{K})(vs::Vararg{Any,K}) where {K} = mf(vs)

struct MultilinearForm{K,D,F} <: AbstractMultilinearForm{K,D}
    f::F
    @inline MultilinearForm{K,D}(f::F) where {K,D,F} = new{K,D,F}(f)
end

# Call that procudes a scalar value
(mf::MultilinearForm{K,D})(vs::FormArgs{K,D}) where {K,D} =
    mf.f(vs...)

# Dimension of the tensor product of vector spaces that the form works on, i.e.,
# the tensorial order
order(::Type{<:AbstractMultilinearForm{K}}) where {K} = K
order(::M) where {M<:AbstractMultilinearForm} = order(M)

# Dimension of the vector space for each individual argument
dimension(::Type{<:AbstractMultilinearForm{<:Any,D}}) where {D} = D
dimension(::M) where {M<:AbstractMultilinearForm} = dimension(M)
# Call that produces a "contracted" form

const ContractionArgs{K,D} = NTuple{K,Union{StaticVector{D},Colon}}

# Represent a partially contracted form
struct ContractedMultilinearForm{K,D,K′,M<:MultilinearForm{K′,D},
    T<:ContractionArgs{K′,D}} <: AbstractMultilinearForm{K,D}
    parent::M
    args::T
    function ContractedMultilinearForm{K,D,K′}(parent::M, args::T) where {K,D,K′,M,T}
        @assert K < K′  # parent must take fewer arguments than contracted form
        new{K,D,K′,M,T}(parent, args)
    end
end

function _contractargs(T::Type{<:ContractedMultilinearForm{K}}) where {K}
    # We need to intercalate the "concrete" parent arguments with the "free
    # arguments" of the contracted form.
    #
    # Here, we let "vs" be the free arguments and "us" be the parent arguments.
    parent_argTs = fieldtypes(fieldtype(T, :args))
    j = 0
    [parent_argTs[i] === Colon ? :(vs[$(j += 1)]) : :(cmf.args[$i])
     for i ∈ eachindex(parent_argTs)]
end

# Use @_inline_meta?
@generated function (cmf::ContractedMultilinearForm)(vs::FormArgs)
    :(cmf.parent.f($(_contractargs(cmf)...)))
end


@generated function (mf::MultilinearForm{K,D})(args::ContractionArgs{K,D}) where {K,D}
    # "Dispatch" is controled by where `:`s appear in `args`.
    K′ = count(arg -> arg === Colon, fieldtypes(args))
    if K′ == K  # All arguments are (:), so this is an identity operation
        :(mf)
    elseif K′ < K
        :(ContractedMultilinearForm{$K′,D,K}(mf, args))
    else # should never happen
        :(@assert false)
    end
end

# function (cmf::ContractedMultilinearForm{K})(args::ContractionArgs{K}) where K
#     # Just modify the args passed to the parent appropriately
#     j = 0
#     args′ = map(eachindex ) do i
#         args[i] isa Colon || cmf.args[i] isa Colon ? Colon :
#     end
# end

# IDEA: define a macro @IndexLabels i, j, k ... or use Symbolics variables
struct IndexLabel{S} end
# use like i = IndexLabel{:i}()

# ----------
# Interfaces
# ----------


# Iteration

# NOTE inlining is important to performance here
@inline function Base.iterate(mf::AbstractMultilinearForm)
    # Piggy-back off of iterate(::CartesianIndices)
    (I, state) = iterate(CartesianIndices(mf))
    return (unsafe_getindex(mf, I), state)
    #   -> (mf[I], state)
    # Should be safe to elide the unit vector validity check
end

@inline function Base.iterate(mf::AbstractMultilinearForm{K}, state) where K
    maybe(iterate(CartesianIndices(mf), state)) do (I′, state′)
        (unsafe_getindex(mf, I′), state′)
    end
end

Base.IteratorSize(::Type{<:AbstractMultilinearForm{K}}) where {K} = Base.HasShape{K}()

Base.IndexStyle(::Type{<:AbstractMultilinearForm}) = IndexCartesian()
Base.IndexStyle(mf::AbstractMultilinearForm) = Base.IndexStyle(typeof(mf))

@inline Base.eltype(mf::AbstractMultilinearForm) = eltype(first(mf))

@inline Base.size(mf::AbstractMultilinearForm) = Tuple(Size(mf))
@inline Base.size(::AbstractMultilinearForm{K,D}, dim::Int) where {K,D} =
    dim ∈ 1:K ? D : 1

@inline Base.length(mf::AbstractMultilinearForm) = Int(Length(mf))

# `StaticArrays` triats

@inline StaticArrays.Size(::Type{<:AbstractMultilinearForm{K,D}}) where {K,D} =
    Size(ntuple(_ -> D, Val(K)))
@inline StaticArrays.Size(mf::AbstractMultilinearForm) = Size(typeof(mf))

@inline StaticArrays.Length(MF::Type{<:AbstractMultilinearForm}) = Length(Size(MF))
@inline StaticArrays.Length(mf::AbstractMultilinearForm) = Length(Size(mf))

# Indexing

Base.CartesianIndices(::AbstractMultilinearForm{K,D}) where {K,D} =
    CartesianIndices(ntuple(_ -> SOneTo(D), Val(K)))

@inline Base.getindex(mf::AbstractMultilinearForm{K,D}, I::Vararg{Int,K}) where {K,D} =
    mf(map(StdUnitVector{D}, I))

@inline Base.getindex(mf::AbstractMultilinearForm{K}, I::CartesianIndex{K}) where K =
    Base.getindex(mf, Tuple(I)...)

@inline Base.firstindex(mf::AbstractMultilinearForm) = Base.first(CartesianIndices(mf))

@inline Base.lastindex(mf::AbstractMultilinearForm) = Base.last(CartesianIndices(mf))

# UNSAFE indexing; do not check for validitiy of the StdUnitVectors in each direction

@inline unsafe_getindex(mf::AbstractMultilinearForm{K,D}, I::Vararg{Int,K}) where {K,D} =
    mf(map(i -> StdUnitVector{D}(UNSAFE, i), I))

@inline unsafe_getindex(mf::AbstractMultilinearForm{K}, I::CartesianIndex{K}) where K =
    unsafe_getindex(mf, Tuple(I)...)

# `similar` and `StaticArrays.similar_type`

StaticArrays.similar_type(MF::Type{<:AbstractMultilinearForm}, ElType::Type, S::Size = Size(MF)) =
    similar_type(StaticArray, ElType, S)

StaticArrays.similar_type(mf::AbstractMultilinearForm, ElType::Type = eltype(mf), S::Size = Size(mf)) =
    similar_type(StaticArray, ElType, S)

StaticArrays.similar_type(mf::AbstractMultilinearForm, S::Size) =
    similar_type(StaticArray, eltype(mf), S)

Base.similar(MF::Type{<:AbstractMultilinearForm}, ElType::Type, S::Size = Size(MF)) =
    similar_type(MArray, ElType, S)(undef)

Base.similar(mf::AbstractMultilinearForm, ElType::Type = eltype(mf), S::Size = Size(mf)) =
    similar_type(MArray, ElType, S)(undef)

Base.similar(mf::AbstractMultilinearForm, S::Size) =
    similar_type(MArray, eltype(mf), S)(undef)

Base.similar(::AbstractMultilinearForm{K}, ::Type{T}, s::Dims) where {K,T} =
    Array{T,K}(undef, s)

Base.similar(mf::AbstractMultilinearForm{K}, s::Dims) where {K} =
    Array{eltype(mf),K}(undef, s)

# AHOY MATIES! We be type pirates here.
# Also, this appears to have a small runtime cost, perhaps to compute `eltype(sized_gen)`
@inline function StaticArrays.sacollect(::Type{SA}, sized_gen) where {SA<:StaticArray}
    SA′ = similar_type(SA, eltype(sized_gen), Size(sized_gen))
    return StaticArrays.sacollect(SA′, sized_gen)
end

# Allow to collect MultilinearForms as `SArray(mf)`, `MArray(mf)`, etc.
@inline (::Type{SA})(mf::AbstractMultilinearForm) where {SA<:StaticArray} =
    StaticArrays.sacollect(SA, mf)

# NOTE it would be nice if ~sacollect~ had a generic method that could
# handle iterators that possessed a ~Size~ trait without having to specify the
# size in the type ~SA~.  We have hacked that together above, but maybe
# something like this should be considered for inclusion in ~StaticArrays~
# itself.

end
