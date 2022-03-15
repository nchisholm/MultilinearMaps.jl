@inline Base.IteratorSize(F::Type{<:MultilinearMap}) = Base.HasShape{ndims(F)}()

# NOTE inlining is important to performance here
@inline function Base.iterate(f::MultilinearMap)
    # Piggy-back off of iterate(::CartesianIndices)
    (I, state) = iterate(CartesianIndices(f))
    return (_getindex(UNSAFE, f, I), state)
    #   -> (f[I], state)
    # Should be safe to elide the unit vector validity check
end

@inline function Base.iterate(f::MultilinearMap{K}, state) where {K}
    maybe(iterate(CartesianIndices(f), state)) do (I′, state′)
        (_getindex(UNSAFE, f, I′), state′)
    end
end
