# Represent a partially contracted form
struct SlicedMultilinearMap{Sz<:StaticSize,
                            M<:AtomicMultilinearMap,
                            TT<:Tuple} <: MultilinearMap{Sz}
    parent::M
    args::TT
    function SlicedMultilinearMap(parent::AtomicMultilinearMap{<:StaticSize{N}},
                                  args::Vararg{Union{Colon,AbstractVector}, N}
                                  ) where N
        MM = typeof(parent)
        TT = typeof(args)
        Sz′ = _contracted_size(MM, TT)
        new{Sz′, MM, TT}(parent, args)
    end
end

@generated function _contracted_size(::Type{MM}, ::Type{ArgsType}) where {MM<:MultilinearMap, ArgsType<:Tuple}
    Sz = ArrayInterface.known_size(MM)
    ArgTs = fieldtypes(ArgsType)
    @assert length(Sz) == length(ArgTs)
    Tuple{[StaticInt{dimlen} for (dimlen, ArgT) ∈ zip(Sz, ArgTs) if ArgT === Colon]...}
end

# Use @_inline_meta?
@generated function (cmf::SlicedMultilinearMap{<:StaticSize{N}})(vs::Vararg{AbstractVector, N}) where N
    # We need to intercalate the "concrete" parent arguments with the "free
    # arguments" of the contracted form.
    #
    # Here, we let "vs" be the free arguments and "us" be the parent arguments.
    parent_argTs = fieldtypes(fieldtype(cmf, :args))
    j = 0
    vs′ = [parent_argTs[i] === Colon ? :(vs[$(j += 1)]) : :(cmf.args[$i])
           for i ∈ eachindex(parent_argTs)]

    return :(cmf.parent.impl($(vs′...)))
end

@inline (f::MultilinearMap{<:StaticSize{N}})(::Vararg{Colon, N}) where N = f
@inline (f::MultilinearMap{<:StaticSize{N}})(args::Vararg{Union{Colon,AbstractVector}}) where N =
    SlicedMultilinearMap(f, args...)
