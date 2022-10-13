export MultilinearMap, MultilinearForm, AtomicMultilinearMap


abstract type MultilinearMap{N, Sz<:SizeS{N}, T} end

# TODO docstring
@inline MultilinearMap{N,Sz}(f) where {N,Sz} =
    AtomicMultilinearMap{N,Sz}(f)

# TODO docstring
@inline MultilinearMap(f, ::Sz) where {N, Sz<:SizeS{N} #= Size =#} =
    AtomicMultilinearMap{N, Sz}(f)

const MultilinearForm{N,D} = MultilinearMap{N, CubeSize{N,D}}

@inline nargs(::MultilinearMap{N}) where {N} = N

# Argument wrapper/trait
abstract type ApplyMode end
struct FullApply <: ApplyMode end
struct PartialApply <: ApplyMode end

@inline ApplyMode() = FullApply()
@inline ApplyMode(::Colon, ::Vararg) = PartialApply()
@inline ApplyMode(::Any, args...) = ApplyMode(args...)

# Choose full versus partial evaluation
@inline (f::MultilinearMap{N})(args::Vararg{Any,N}) where N =
    f(ApplyMode(args...), args...)

# Fallback if wrong argument count
@noinline (f::MultilinearMap)(args...) =
    throw(ArgumentError("expected $(ndims(f)) arguments but got $(length(args)) of them"))

# Identity operation
@inline (f::MultilinearMap)(::PartialApply, ::Vararg{Colon}) = f

# Partial evaluation / contraction
@inline (f::MultilinearMap)(::PartialApply, args...) = PartialMap(f, args...)

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
        # T = Base.promote_op(impl, map(dim -> StdUnitVector{known(dim)}, sz)...)
        args1 = map(dim -> StdUnitVector{known(dim)}(1), sz)
        T = typeof(impl(args1...))            # determine output type
        new{N, Sz, T, F}(impl)
    end
end

@inline (f::AtomicMultilinearMap)(::FullApply, args...) = f.impl(args...)

function Base.show(io::IO, f::AtomicMultilinearMap)
    print(io, "@MultilinearMap{", size(f), "}(", f.impl, ")")
end
