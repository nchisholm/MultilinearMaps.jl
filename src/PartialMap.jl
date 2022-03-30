# Represent a partially contracted form by wrapping another MultilinearMap of a
# "larger" size.
struct PartialMap{N, Sz<:Size{N}, T,
                  MM<:MultilinearMap,
                  TT<:Tuple} <: MultilinearMap{N, Sz, T}
    parent::MM
    args::TT
    function PartialMap(parent::MultilinearMap{M}, args0::Vararg{Any,M}) where M
        sz = _appliedsize(parent, args0)
        args1 = map(dim -> StdUnitVector{known(dim)}(1), sz)
        T = typeof(_reapplyargs(parent, args0, args1))
        # T = promote_eltype(Bool, parent, _removetype(Colon, args0)...)
        new{length(sz), typeof(sz), T, typeof(parent), typeof(args0)}(parent, args0)
    end
end

# Returns the size of `parent(args...)`
@inline _appliedsize(parent::MultilinearMap, args::Tuple) =
    _appliedsize(Arr.size(parent), args)
@inline _appliedsize(::Tuple{}, ::Tuple{}) = ()
@inline _appliedsize((dim, sz...)::Size{N}, (arg, args...)::NTuple{N,Any}) where N =
    arg isa Colon ? (dim, _appliedsize(sz, args)...) : _appliedsize(sz, args)

# Returns the set of arguments to be reapplied to the parent of a `PartialMap`
# when the `PartialMap` is applied to some arguments.  The first arguments is a
# tuple of the original arguments given when the `PartialMap` was created, and
# the second argument is a tuple of the "new" arguments to be applied.
@inline _reapplyargs(::Tuple{}, ::Tuple{}) = ()
@inline _reapplyargs(::Tuple{}, args::Tuple) = args  # Gave too many args, but will be handled
@inline _reapplyargs((_, fixedargs...)::Tuple{Colon, Vararg}, (arg, args...)::Tuple) =
    (arg, _reapplyargs(fixedargs, args)...)
@inline _reapplyargs((arg, fixedargs...)::Tuple, args::Tuple) =
    (arg, _reapplyargs(fixedargs, args)...)

@inline _reapplyargs(f::MultilinearMap, args0, args1) = f(_reapplyargs(args0, args1)...)

@inline (f::PartialMap)(vs...) = f.parent(_reapplyargs(f.args, vs)...)

function Base.show(io::IO, f::PartialMap)
    pp_args = "(" * join(map(arg -> arg isa Colon ? ":" : arg, f.args), ", ") * ")"
    print(io, f.parent, pp_args)
end
