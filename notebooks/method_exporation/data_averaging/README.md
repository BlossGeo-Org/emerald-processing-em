# Figuring out data averaging

## Using sums of squares
### Means
The population mean is calculated as expected, simply by averaging the means.  Let

$$\mathbf{x} = \left[\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n\right]^T$$

be the individual data sets with means and standard deviations given by:

```math
\mathbf{\mu} = \left[\mu_1, \mu_2, ..., \mu_n\right]^T
```
```math
\mathbf{\sigma} = \left[\sigma_1, \sigma_2, ..., \sigma_n\right]^T .
```
Let $$\mathbf{m}$$ be a vector containing the number of observations in each dataset $$\mathbf{x}_i$$:

```math
\mathbf{m} = \left[n_1, n_2, ..., n_n\right]^T .
```
The mean of the entire population is the mean of the means:
```math
\mu_{total} = \frac{1}{\sum{n_i}}\sum_i{\mu_i} .
```
### Standard deviations/variances
If the means are identical, the total variance is the geometric mean of the individual variances.  Otherwise, the Sum of Squares formulae are used.  Let the sums of squares error, SSE, be given by:

```math
SSE = \sum_i{n_i\sigma_i^2}
```
and the sum of squares of groups, SSG, be

```math
SSG = \sum_i{n_i\left(\mu_i - \mu_{total}\right)^2} .
```
The sum of squares total, SST, would normally be given by
```math
SST = \sum_{i,j}{\left(x_{ij} - \mu_{total}\right)^2}
```
but since we don't have the individual $$x_{ij}$$ values we use SSE and SSG:
```math
SST = SSE + SSG .
```
The total variance, $$\sigma_{total}^2$$, is finally given by
```math
\sigma_{total}^2 = \frac{SSG + SSE}{\sum{n_i}} ,
```
with the square root giving the standard deviation.

Note that if the datasets represent a sample rather than the entire population, $$n_i$$ is replaced with $$n_i - 1$$.

In the case where the sample numbers are equal within each stack, the actual values of $$n$$ cancel out.  Beginning from the expression for variance derived above:
```math
\sigma_{total}^2 = \frac{\sum{n_i\left(\mu_i - \mu_{total}\right)^2} + \sum{n_i\sigma_i^2}}{\sum{n_i}} ,
```
we let $$n_i$$ equal a scalar $$a$$, since $$n_i = n_j\forall i,j$$.  The variance then becomes
```math
\sigma_{total}^2 = \frac{a\sum{\left(\mu_i - \mu_{total}\right)^2} + a\sum{\sigma_i^2}}{a\sum{1}}
```
and the $a$ values cancel.

## References
https://www.reddit.com/r/statistics/comments/18armdi/q_if_i_know_the_mean_and_variance_of_two/

https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation

