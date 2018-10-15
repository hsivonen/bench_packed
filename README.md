This crate contains two benchmarks extracted from encoding_rs and
encoding_bench. These benchmarks demonstrate a regression form the `simd`
crate to `packed_simd` transition at `opt_level=2`.

```
RUST_FLAGS='-C opt_level=2' cargo bench
```