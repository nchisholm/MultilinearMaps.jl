export MultilinearMap, MultilinearForm, AtomicMultilinearMap


abstract type MultilinearMap{Sz,T} end

# TODO docstring
@inline function MultilinearMap{Sz}(f) where {Sz<:Size}
    AtomicMultilinearMap{Sz}(f)
end

# TODO docstring
@inline MultilinearMap{Sz #= Tuple =#}(f) where Sz =
    MultilinearMap{typeof(static(Sz::Size))}(f)


const MultilinearMapN{N} = MultilinearMap{<:Size{N}}

const MultilinearForm{N,D} = MultilinearMap{CubeSize{N,D}}


# Default Evaluation

const VecOrColon = Union{AbstractVector,Colon}

# Identity operation
@inline (f::MultilinearMap{<:Size{N}})(::Vararg{Colon,N}) where {N} = f

# Partial evaluation / contraction
@inline (f::MultilinearMap{<:Size{N}})(args::Vararg{VecOrColon,N}) where {N} =
    PartialMap(f, args...)

function Base.show(io::IO, ::MIME"text/plain", f::MultilinearMap)
    print(io, f, ":\n  ",
          join(["V$(subscripts(i))" for i ∈ 1:ndims(f)], " × "),
          " ↦ ", eltype(f))
end


"""
Wraps a function `impl` that is assumed to have a method
`impl(::Vararg{AbstractVector})`, which is itself assumed to be
multilinear in its arguments, i.e., linear when all of the arguments but one
are held fixed.
"""
struct AtomicMultilinearMap{Sz<:Size,T,F} <: MultilinearMap{Sz,T}
    impl::F
    function AtomicMultilinearMap{Sz}(impl::F) where {Sz<:SizeS,F}
        sz::Sz = static(known(Sz))            # convert Type{<:Size} -> sz::Size
        args1 = map(dim -> StdUnitVector{known(dim)}(1), sz)
        T = typeof(impl(args1...))            # determine output type
        new{Sz, T, F}(impl)
    end
end

@inline (f::AtomicMultilinearMap{<:Size{0}})() = f.impl()  # disambiguate
@inline (f::AtomicMultilinearMap{<:Size{N}})(vs::Vararg{AbstractVector,N}) where {N} =
    f.impl(vs...)

function Base.show(io::IO, f::AtomicMultilinearMap)
    print(io, "MultilinearMap{", size(f), "}(", f.impl, ")")
end
