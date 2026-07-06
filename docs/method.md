# Method Overview

AG-RaNN is a residual-driven randomized neural-network framework. Hidden-layer
parameters remain fixed after construction, while output coefficients are
computed from a linear least-squares system.

## Frequency-based initialization

The initial random-feature scale is selected by comparing residual frequency
content with candidate basis functions. This replaces a manually tuned random
range with a constructive, problem-informed choice.

## Neuron growth

The current residual is evaluated at collocation points. New neurons are placed
using high-residual locations and residual frequency information, increasing
width while retaining a linear output solve.

## Layer growth

When width alone is insufficient, localized basis functions are constructed
from residual indicators and added as a new hidden layer. The implementation
keeps the paper's original feature formulas and update order.

## Domain splitting

For discontinuous or piecewise-smooth solutions, the approximation range is
used to split the domain. Separate randomized subnetworks then specialize on
the resulting subdomains.

## PDE benchmarks

- Example 1: five Poisson cases (1-D, 2-D, 3-D, oscillatory and sharp targets);
- Example 2: discontinuous advection-reaction;
- Example 3: four Burgers cases;
- Example 4: Allen-Cahn with 100 random seeds and ensemble averaging.
