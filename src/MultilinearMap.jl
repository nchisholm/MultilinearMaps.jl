export MultilinearMap, MultilinearForm, AtomicMultilinearMap


abstract type MultilinearMap{N, Sz<:SizeS{N}, T} end

# TODO docstring
@inline MultilinearMap{N,Sz}(f) where {N,Sz} =
    AtomicMultilinearMap{N,Sz}(f)

# TODO docstring
@inline MultilinearMap(::Sz, f) where {N, Sz<:SizeS{N} #= Size =#} =
    AtomicMultilinearMap{N, Sz}(f)


const MultilinearForm{N,D} = MultilinearMap{N, CubeSize{N,D}}

# Default Evaluation

const VecOrColon = Union{AbstractVector,Colon}

# Identity operation
@inline (f::MultilinearMap{N})(::Vararg{Colon,N}) where {N} = f

# Partial evaluation / contraction
@inline (f::MultilinearMap{N})(args::Vararg{VecOrColon,N}) where {N} =
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
struct AtomicMultilinearMap{N, Sz<:Size{N}, T, F} <: MultilinearMap{N,Sz,T}
    impl::F
    # NOTE: only allow fully static sizes for now
    function AtomicMultilinearMap{N, Sz}(impl::F) where {N, Sz<:SizeS{N}, F}
        sz::Sz = static(known(Sz))            # convert Type{<:Size} -> sz::Size
        args1 = map(dim -> StdUnitVector{known(dim)}(1), sz)
        T = typeof(impl(args1...))            # determine output type
        new{N, Sz, T, F}(impl)
    end
end

@inline (f::AtomicMultilinearMap{0})() = f.impl()  # disambiguate
@inline (f::AtomicMultilinearMap{N})(vs::Vararg{AbstractVector,N}) where {N} =
    f.impl(vs...)

function Base.show(io::IO, f::AtomicMultilinearMap)
    print(io, "@MultilinearMap{", size(f), "}(", f.impl, ")")
end
