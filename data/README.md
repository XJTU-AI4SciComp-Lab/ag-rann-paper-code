# Data

## Allen-Cahn reference solution

`allen_cahn/AC_cosine.mat` contains the Allen-Cahn reference solution used by
the numerical experiment.

- Size: 559,196 bytes
- SHA-256: `266e97eedd9ef543d7f6d370fa5f06388761aa7da2b97e476562dcf1d644a175`
- Variables used by the code: `tt`, `x`, and `uu`

## Large generated ensemble output

The 100-seed ensemble output is 55,899,416 bytes and is therefore generated
locally instead of being tracked in the repository. Its reference SHA-256 is:

`b62d2b294ddf8ef35f05a7815775b89b8860b90aa4aadb06a389a45c5f180280`

Regenerate it with:

```matlab
run_experiment("example4_ensemble", 100)
```

The generated file is written to
`results/example4_allen_cahn/result.mat`.
