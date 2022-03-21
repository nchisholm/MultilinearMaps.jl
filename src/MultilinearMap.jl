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

@inline (f::MultilinearMap{<:Size{N}})(::Vararg{Colon,N}) where {N} = f
@inline (f::MultilinearMap{<:Size{N}})(args::Vararg{VecOrColon,N}) where {N} =
    PartialMap(f, args...)


# Wraps a function of vectors
struct AtomicMultilinearMap{Sz<:Size,T,F} <: MultilinearMap{Sz,T}
    impl::F
    function AtomicMultilinearMap{Sz}(impl::F) where {Sz<:SizeS,F}
        sz::Sz = static(known(Sz))
        args1 = map(dim -> StdUnitVector{known(dim)}(1), sz)
        T = typeof(impl(args1...))      # determine output type
        new{Sz, T, F}(impl)
    end
end

@inline (f::AtomicMultilinearMap{<:Size{0}})() = f.impl()  # disambiguate
@inline (f::AtomicMultilinearMap{<:Size{N}})(vs::Vararg{AbstractVector,N}) where {N} =
    f.impl(vs...)

function Base.show(io::IO, f::AtomicMultilinearMap)
    fn_name = Symbol(f.impl)
    print(io, "MultilinearMap{", size(f), "}(", fn_name, ")")
end

function Base.show(io::IO, ::MIME"text/plain", f::AtomicMultilinearMap)
    print(io, "MultilinearMap{(", join(size(f), ","), ")}(", f.impl, "):\n  ",
          join(["V_$i" for i ∈ 1:ndims(f)], " × "), " → ", eltype(f))
end
