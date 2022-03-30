# MLDS: Multilinear Dynamical System

**Unofficial** Python implementation of MLDS:  
"Multilinear dynamical systems for tensor time series.",
Rogers, Mark, Lei Li, and Stuart J. Russell.
Advances in Neural Information Processing Systems 26 (2013): 2634-2642.  
The original implementation is found at the author's homepage
\[[link](http://www.cs.cmu.edu/~leili/mlds/index.html)\].


## Model

$$\mathcal{Z}_1\sim\mathcal{N}(\mathcal{U}_0,\mathcal{Q}_0)$$
$$\mathcal{Z}_{n+1}|\mathcal{Z}_{n}\sim\mathcal{N}(\mathcal{A}\otimes\mathcal{Z}_n,\mathcal{Q})$$

Then, original tensors are represented by

$$\mathcal{X}_n|\mathcal{Z}_n\sim\mathcal{N}(\mathcal{C}\otimes\mathcal{Z}_n,\mathcal{R})$$

where, 
- the initial state covariance, $\mathcal{Q}_0$
- the transition covariance, $\mathcal{Q}$
- the observation covariace, $\mathcal{R}$

The shapes of these covariaces can be specified in
'full', 'diag', and 'isotropic', independently.

## Reference

```bibtex
@article{rogers2013multilinear,
  title={Multilinear dynamical systems for tensor time series},
  author={Rogers, Mark and Li, Lei and Russell, Stuart J},
  journal={Advances in Neural Information Processing Systems},
  volume={26},
  pages={2634--2642},
  year={2013},
  publisher={Citeseer}
}
```