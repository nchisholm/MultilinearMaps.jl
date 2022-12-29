#
# Algebra
#
# Multilinear maps form a vector space.  That is, we can take linear
# combinations of multilinear maps and generally produce another multilinear
# map.  Here, we define the necessary operations `ScalarMultiple` and `Sum`.
#
# TODO: tensor products, contractions?

# Comparison
Base.:(==)(f1::MultilinearMap, f2::MultilinearMap) =
    _sizes_match(f1, f2) && all(Iterators.map(==, f1, f2))
