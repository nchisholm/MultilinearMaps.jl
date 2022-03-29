# Represent a partially contracted form by wrapping another MultilinearMap of a
# "larger" size.
struct PartialMap{N, Sz<:Size{N}, T,
                  MM<:MultilinearMap,
                  TT<:Tuple} <: MultilinearMap{N, Sz, T}
    parent::MM
    args::TT
    function PartialMap(parent::MultilinearMap, args0::Vararg{VecOrColon})
        sz = _appliedsize(parent, args0)
        args1 = map(dim -> StdUnitVector{known(dim)}(1), sz)
        T = typeof(parent(_parentargs(args0, args1)...))
        new{length(sz), typeof(sz), T, typeof(parent), typeof(args0)}(parent, args0)
    end
end

# Returns the size of `parent(args...)`
@inline _appliedsize(parent::MultilinearMap, args::Tuple) =
    _appliedsize(Arr.size(parent), args)
@inline _appliedsize(::Tuple{}, ::Tuple{}) = ()
@inline _appliedsize((dim, sz...)::Size{N}, (arg, args...)::NTuple{N,Any}) where N =
    arg isa Colon ? (dim, _appliedsize(sz, args)...) : _appliedsize(sz, args)

# Returns the the full set of arguments to be passed to the *parent* of a
# `PartialMap` when the `PartialMap` is applied to `args`.
@inline _parentargs(::Tuple{}, ::Tuple{}) = ()
@noinline _parentargs(::Tuple{}, args::Tuple) = error("Internal error. Please report this result as a bug.")
@inline _parentargs((_, fixedargs...)::Tuple{Colon, Vararg}, (arg, args...)::Tuple) =
    (arg, _parentargs(fixedargs, args)...)
@inline _parentargs((arg, fixedargs...)::Tuple, args::Tuple) =
    (arg, _parentargs(fixedargs, args)...)

@inline (f::PartialMap{N})(vs::Vararg{VecOrColon,N}) where N =
    f.parent(_parentargs(f.args, vs)...)

function Base.show(io::IO, f::PartialMap)
    pp_args = "(" * join(map(arg -> arg isa Colon ? ":" : arg, f.args), ", ") * ")"
    print(io, f.parent, pp_args)
end
