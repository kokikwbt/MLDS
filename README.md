# MLDS
Multi-linear dynamical system
\[[link](http://www.cs.cmu.edu/~leili/mlds/index.html)\]

#### Initialize mode

`class` mlds. **MLDS**
#### Notes


#### Methods

`__init__($self, X, ranks)`
Initialize self.
###### Parameters:

  * X: nd-array
    * tensor of shape T x N1 x ... x NM
  * ranks: int list
    * size of latent tensor Z

`em($self, max_iter=10, tol=1.e-5, cov_types)`
Estimate mlds parameters
###### Parameters:

  * max_iter: int
  * tol: float, optional
  * cov_types: list
    * a

`save_params($self, outdir="./out/")`
save the current parameters of mlds model.
###### Parameters:

  * outdir: string
