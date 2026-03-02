[Lil'Log](https://lilianweng.github.io/) | [Posts](https://lilianweng.github.io/) [Archive](https://lilianweng.github.io/archives) [Search](https://lilianweng.github.io/search/) [Tags](https://lilianweng.github.io/tags/) [FAQ](https://lilianweng.github.io/faq)
# Some Math behind Neural Tangent Kernel
Date: September 8, 2022 | Estimated Reading Time: 17 min | Author: Lilian Weng Table of Contents [Basics](#basics) [Vector-to-vector Derivative](#vector-to-vector-derivative) [Differential Equations](#differential-equations) [Central Limit Theorem](#central-limit-theorem) [Taylor Expansion](#taylor-expansion) [Kernel & Kernel Methods](#kernel--kernel-methods) [Gaussian Processes](#gaussian-processes) [Notation](#notation) [Neural Tangent Kernel](#neural-tangent-kernel) [Infinite Width Networks](#infinite-width-networks) [Connection with Gaussian Processes](#connection-with-gaussian-processes) [Deterministic Neural Tangent Kernel](#deterministic-neural-tangent-kernel) [Linearized Models](#linearized-models) [Lazy Training](#lazy-training) [Citation](#citation) [References](#references)
Neural networks are [well known](https://lilianweng.github.io/posts/2019-03-14-overfit/) to be over-parameterized and can often easily fit data with near-zero training loss with decent generalization performance on test dataset. Although all these parameters are initialized at random, the optimization process can consistently lead to similarly good outcomes. And this is true even when the number of model parameters exceeds the number of training data points.
Neural tangent kernel (NTK) ( [Jacot et al. 2018](https://arxiv.org/abs/1806.07572) ) is a kernel to explain the evolution of neural networks during training via gradient descent. It leads to great insights into why neural networks with enough width can consistently converge to a global minimum when trained to minimize an empirical loss. In the post, we will do a deep dive into the motivation and definition of NTK, as well as the proof of a deterministic convergence at different initializations of neural networks with infinite width by characterizing NTK in such a setting.
🤓 Different from my previous posts, this one mainly focuses on a small number of core papers, less on the breadth of the literature review in the field. There are many interesting works after NTK, with modification or expansion of the theory for understanding the learning dynamics of NNs, but they won’t be covered here. The goal is to show all the math behind NTK in a clear and easy-to-follow format, so the post is quite math-intensive. If you notice any mistakes, please let me know and I will be happy to correct them quickly. Thanks in advance!
# Basics [#](#basics)
This section contains reviews of several very basic concepts which are core to understanding of neural tangent kernel. Feel free to skip.
## Vector-to-vector Derivative [#](#vector-to-vector-derivative)
Given an input vector x ∈ R n (as a column vector) and a function f : R n → R m , the derivative of f with respective to x is a m × n matrix, also known as [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) :
J = ∂ f ∂ x = [ ∂ f 1 ∂ x 1 … ∂ f 1 ∂ x n ⋮ ∂ f m ∂ x 1 … ∂ f m ∂ x n ] ∈ R m × n
Throughout the post, I use integer subscript(s) to refer to a single entry out of a vector or matrix value; i.e. x i indicates the i -th value in the vector x and f i ( . ) is the i -th entry in the output of the function.
The gradient of a vector with respect to a vector is defined as ∇ x f = J ⊤ ∈ R n × m and this formation is also valid when m = 1 (i.e., scalar output).
## Differential Equations [#](#differential-equations)
Differential equations describe the relationship between one or multiple functions and their derivatives. There are two main types of differential equations.
(1) ODE (Ordinary differential equation) contains only an unknown function of one random variable. ODEs are the main form of differential equations used in this post. A general form of ODE looks like ( x , y , d y d x , … , d n y d x n ) = 0 . (2) PDE (Partial differential equation) contains unknown multivariable functions and their partial derivatives.
Let’s review the simplest case of differential equations and its solution. Separation of variables (Fourier method) can be used when all the terms containing one variable can be moved to one side, while the other terms are all moved to the other side. For example,
Given a is a constant scalar: d y d x = a y Move same variables to the same side: d y y = a d x Put integral on both sides: ∫ d y y = ∫ a d x ln ⁡ ( y ) = a x + C ′ Finally y = e a x + C ′ = C e a x
## Central Limit Theorem [#](#central-limit-theorem)
Given a collection of i.i.d. random variables, x 1 , … , x N with mean μ and variance σ 2 , the Central Limit Theorem (CTL) states that the expectation would be Gaussian distributed when N becomes really large.
x ¯ = 1 N ∑ i = 1 N x i ∼ N ( μ , σ 2 n ) when N → ∞
CTL can also apply to multidimensional vectors, and then instead of a single scale σ 2 we need to compute the covariance matrix of random variable Σ .
## Taylor Expansion [#](#taylor-expansion)
The [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series) is to express a function as an infinite sum of components, each represented in terms of this function’s derivatives. The Tayler expansion of a function f ( x ) at x = a can be written as: f ( x ) = f ( a ) + ∑ k = 1 ∞ 1 k ! ( x − a ) k ∇ x k f ( x ) | x = a where ∇ k denotes the k -th derivative.
The first-order Taylor expansion is often used as a linear approximation of the function value:
f ( x ) ≈ f ( a ) + ( x − a ) ∇ x f ( x ) | x = a
## Kernel & Kernel Methods [#](#kernel--kernel-methods)
A [kernel](https://en.wikipedia.org/wiki/Kernel_method) is essentially a similarity function between two data points, K : X × X → R . It describes how sensitive the prediction for one data sample is to the prediction for the other; or in other words, how similar two data points are. The kernel should be symmetric, K ( x , x ′ ) = K ( x ′ , x ) .
Depending on the problem structure, some kernels can be decomposed into two feature maps, one corresponding to one data point, and the kernel value is an inner product of these two features: K ( x , x ′ ) = ⟨ φ ( x ) , φ ( x ′ ) ⟩ .
Kernel methods are a type of non-parametric, instance-based machine learning algorithms. Assuming we have known all the labels of training samples { x ( i ) , y ( i ) } , the label for a new input x is predicted by a weighted sum ∑ i K ( x ( i ) , x ) y ( i ) .
## Gaussian Processes [#](#gaussian-processes)
Gaussian process (GP) is a non-parametric method by modeling a multivariate Gaussian probability distribution over a collection of random variables. GP assumes a prior over functions and then updates the posterior over functions based on what data points are observed.
Given a collection of data points { x ( 1 ) , … , x ( N ) } , GP assumes that they follow a jointly multivariate Gaussian distribution, defined by a mean μ ( x ) and a covariance matrix Σ ( x ) . Each entry at location ( i , j ) in the covariance matrix Σ ( x ) is defined by a kernel Σ i , j = K ( x ( i ) , x ( j ) ) , also known as a covariance function . The core idea is – if two data points are deemed similar by the kernel, the function outputs should be close, too. Making predictions with GP for unknown data points is equivalent to drawing samples from this distribution, via a conditional distribution of unknown data points given observed ones.
Check [this post](https://distill.pub/2019/visual-exploration-gaussian-processes/) for a high-quality and highly visualization tutorial on what Gaussian Processes are.
# Notation [#](#notation)
Let us consider a fully-connected neural networks with parameter θ , f ( . ; θ ) : R n 0 → R n L . Layers are indexed from 0 (input) to L (output), each containing n 0 , … , n L neurons, including the input of size n 0 and the output of size n L . There are P = ∑ l = 0 L − 1 ( n l + 1 ) n l + 1 parameters in total and thus we have θ ∈ R P .
The training dataset contains N data points, D = { x ( i ) , y ( i ) } i = 1 N . All the inputs are denoted as X = { x ( i ) } i = 1 N and all the labels are denoted as Y = { y ( i ) } i = 1 N .
Now let’s look into the forward pass computation in every layer in detail. For l = 0 , … , L − 1 , each layer l defines an affine transformation A ( l ) with a weight matrix w ( l ) ∈ R n l × n l + 1 and a bias term b ( l ) ∈ R n l + 1 , as well as a pointwise nonlinearity function σ ( . ) which is [Lipschitz continuous](https://en.wikipedia.org/wiki/Lipschitz_continuity) .
A ( 0 ) = x A ~ ( l + 1 ) ( x ) = 1 n l w ( l ) ⊤ A ( l ) + β b ( l ) ∈ R n l + 1 ; pre-activations A ( l + 1 ) ( x ) = σ ( A ~ ( l + 1 ) ( x ) ) ∈ R n l + 1 ; post-activations
Note that the NTK parameterization applies a rescale weight 1 / n l on the transformation to avoid divergence with infinite-width networks. The constant scalar β ≥ 0 controls how much effort the bias terms have.
All the network parameters are initialized as an i.i.d Gaussian N ( 0 , 1 ) in the following analysis.
# Neural Tangent Kernel [#](#neural-tangent-kernel)
Neural tangent kernel (NTK) ( [Jacot et al. 2018](https://arxiv.org/abs/1806.07572) ) is an important concept for understanding neural network training via gradient descent. At its core, it explains how updating the model parameters on one data sample affects the predictions for other samples.
Let’s start with the intuition behind NTK, step by step.
The empirical loss function L : R P → R + to minimize during training is defined as follows, using a per-sample cost function ℓ : R n 0 × R n L → R + :
L ( θ ) = 1 N ∑ i = 1 N ℓ ( f ( x ( i ) ; θ ) , y ( i ) )
and according to the chain rule. the gradient of the loss is:
∇ θ L ( θ ) = 1 N ∑ i = 1 N ∇ θ f ( x ( i ) ; θ ) ⏟ size P × n L ∇ f ℓ ( f , y ( i ) ) ⏟ size n L × 1
When tracking how the network parameter θ evolves in time, each gradient descent update introduces a small incremental change of an infinitesimal step size. Because of the update step is small enough, it can be approximately viewed as a derivative on the time dimension:
d θ d t = − ∇ θ L ( θ ) = − 1 N ∑ i = 1 N ∇ θ f ( x ( i ) ; θ ) ∇ f ℓ ( f , y ( i ) )
Again, by the chain rule, the network output evolves according to the derivative:
d f ( x ; θ ) d t = d f ( x ; θ ) d θ d θ d t = − 1 N ∑ i = 1 N ∇ θ f ( x ; θ ) ⊤ ∇ θ f ( x ( i ) ; θ ) ⏟ Neural tangent kernel ∇ f ℓ ( f , y ( i ) )
Here we find the Neural Tangent Kernel (NTK) , as defined in the blue part in the above formula, K : R n 0 × R n 0 → R n L × n L :
K ( x , x ′ ; θ ) = ∇ θ f ( x ; θ ) ⊤ ∇ θ f ( x ′ ; θ )
where each entry in the output matrix at location ( m , n ) , 1 ≤ m , n ≤ n L is:
K m , n ( x , x ′ ; θ ) = ∑ p = 1 P ∂ f m ( x ; θ ) ∂ θ p ∂ f n ( x ′ ; θ ) ∂ θ p
The “feature map” form of one input x is φ ( x ) = ∇ θ f ( x ; θ ) .
# Infinite Width Networks [#](#infinite-width-networks)
To understand why the effect of one gradient descent is so similar for different initializations of network parameters, several pioneering theoretical work starts with infinite width networks. We will look into detailed proof using NTK of how it guarantees that infinite width networks can converge to a global minimum when trained to minimize an empirical loss.
## Connection with Gaussian Processes [#](#connection-with-gaussian-processes)
Deep neural networks have deep connection with gaussian processes ( [Neal 1994](https://www.cs.toronto.edu/~radford/ftp/pin.pdf) ). The output functions of a L -layer network, f i ( x ; θ ) for i = 1 , … , n L , are i.i.d. centered Gaussian process of covariance Σ ( L ) , defined recursively as:
Σ ( 1 ) ( x , x ′ ) = 1 n 0 x ⊤ x ′ + β 2 λ ( l + 1 ) ( x , x ′ ) = [ Σ ( l ) ( x , x ) Σ ( l ) ( x , x ′ ) Σ ( l ) ( x ′ , x ) Σ ( l ) ( x ′ , x ′ ) ] Σ ( l + 1 ) ( x , x ′ ) = E f ∼ N ( 0 , λ ( l ) ) [ σ ( f ( x ) ) σ ( f ( x ′ ) ) ] + β 2
[Lee & Bahri et al. (2018)](https://arxiv.org/abs/1711.00165) showed a proof by mathematical induction:
(1) Let’s start with L = 1 , when there is no nonlinearity function and the input is only processed by a simple affine transformation:
f ( x ; θ ) = A ~ ( 1 ) ( x ) = 1 n 0 w ( 0 ) ⊤ x + β b ( 0 ) where A ~ m ( 1 ) ( x ) = 1 n 0 ∑ i = 1 n 0 w i m ( 0 ) x i + β b m ( 0 ) for 1 ≤ m ≤ n 1
Since the weights and biases are initialized i.i.d., all the output dimensions of this network A ~ 1 ( 1 ) ( x ) , … , A ~ n 1 ( 1 ) ( x ) are also i.i.d. Given different inputs, the m -th network outputs A ~ m ( 1 ) ( . ) have a joint multivariate Gaussian distribution, equivalent to a Gaussian process with covariance function (We know that mean μ w = μ b = 0 and variance σ w 2 = σ b 2 = 1 )
Σ ( 1 ) ( x , x ′ ) = E [ A ~ m ( 1 ) ( x ) A ~ m ( 1 ) ( x ′ ) ] = E [ ( 1 n 0 ∑ i = 1 n 0 w i , m ( 0 ) x i + β b m ( 0 ) ) ( 1 n 0 ∑ i = 1 n 0 w i , m ( 0 ) x i ′ + β b m ( 0 ) ) ] = 1 n 0 σ w 2 ∑ i = 1 n 0 ∑ j = 1 n 0 x i x ′ j + β μ b n 0 ∑ i = 1 n 0 w i m ( x i + x i ′ ) + σ b 2 β 2 = 1 n 0 x ⊤ x ′ + β 2
(2) Using induction, we first assume the proposition is true for L = l , a l -layer network, and thus A ~ m ( l ) ( . ) is a Gaussian process with covariance Σ ( l ) and { A ~ i ( l ) } i = 1 n l are i.i.d.
Then we need to prove the proposition is also true for L = l + 1 . We compute the outputs by:
f ( x ; θ ) = A ~ ( l + 1 ) ( x ) = 1 n l w ( l ) ⊤ σ ( A ~ ( l ) ( x ) ) + β b ( l ) where A ~ m ( l + 1 ) ( x ) = 1 n l ∑ i = 1 n l w i m ( l ) σ ( A ~ i ( l ) ( x ) ) + β b m ( l ) for 1 ≤ m ≤ n l + 1
We can infer that the expectation of the sum of contributions of the previous hidden layers is zero:
E [ w i m ( l ) σ ( A ~ i ( l ) ( x ) ) ] = E [ w i m ( l ) ] E [ σ ( A ~ i ( l ) ( x ) ) ] = μ w E [ σ ( A ~ i ( l ) ( x ) ) ] = 0 E [ ( w i m ( l ) σ ( A ~ i ( l ) ( x ) ) ) 2 ] = E [ w i m ( l ) 2 ] E [ σ ( A ~ i ( l ) ( x ) ) 2 ] = σ w 2 Σ ( l ) ( x , x ) = Σ ( l ) ( x , x )
Since { A ~ i ( l ) ( x ) } i = 1 n l are i.i.d., according to central limit theorem, when the hidden layer gets infinitely wide n l → ∞ , A ~ m ( l + 1 ) ( x ) is Gaussian distributed with variance β 2 + Var ( A ~ i ( l ) ( x ) ) . Note that A ~ 1 ( l + 1 ) ( x ) , … , A ~ n l + 1 ( l + 1 ) ( x ) are still i.i.d.
A ~ m ( l + 1 ) ( . ) is equivalent to a Gaussian process with covariance function:
Σ ( l + 1 ) ( x , x ′ ) = E [ A ~ m ( l + 1 ) ( x ) A ~ m ( l + 1 ) ( x ′ ) ] = 1 n l σ ( A ~ i ( l ) ( x ) ) ⊤ σ ( A ~ i ( l ) ( x ′ ) ) + β 2 ;similar to how we get Σ ( 1 )
When n l → ∞ , according to central limit theorem,
Σ ( l + 1 ) ( x , x ′ ) → E f ∼ N ( 0 , Λ ( l ) ) [ σ ( f ( x ) ) ⊤ σ ( f ( x ′ ) ) ] + β 2
The form of Gaussian processes in the above process is referred to as the Neural Network Gaussian Process (NNGP) ( [Lee & Bahri et al. (2018)](https://arxiv.org/abs/1711.00165) ).
## Deterministic Neural Tangent Kernel [#](#deterministic-neural-tangent-kernel)
Finally we are now prepared enough to look into the most critical proposition from the NTK paper:
When n 1 , … , n L → ∞ (network with infinite width), the NTK converges to be:
(1) deterministic at initialization, meaning that the kernel is irrelevant to the initialization values and only determined by the model architecture; and (2) stays constant during training.
The proof depends on mathematical induction as well:
(1) First of all, we always have K ( 0 ) = 0 . When L = 1 , we can get the representation of NTK directly. It is deterministic and does not depend on the network initialization. There is no hidden layer, so there is nothing to take on infinite width.
f ( x ; θ ) = A ~ ( 1 ) ( x ) = 1 n 0 w ( 0 ) ⊤ x + β b ( 0 ) K ( 1 ) ( x , x ′ ; θ ) = ( ∂ f ( x ′ ; θ ) ∂ w ( 0 ) ) ⊤ ∂ f ( x ; θ ) ∂ w ( 0 ) + ( ∂ f ( x ′ ; θ ) ∂ b ( 0 ) ) ⊤ ∂ f ( x ; θ ) ∂ b ( 0 ) = 1 n 0 x ⊤ x ′ + β 2 = Σ ( 1 ) ( x , x ′ )
(2) Now when L = l , we assume that a l -layer network with P ~ parameters in total, θ ~ = ( w ( 0 ) , … , w ( l − 1 ) , b ( 0 ) , … , b ( l − 1 ) ) ∈ R P ~ , has a NTK converging to a deterministic limit when n 1 , … , n l − 1 → ∞ .
K ( l ) ( x , x ′ ; θ ~ ) = ∇ θ ~ A ~ ( l ) ( x ) ⊤ ∇ θ ~ A ~ ( l ) ( x ′ ) → K ∞ ( l ) ( x , x ′ )
Note that K ∞ ( l ) has no dependency on θ .
Next let’s check the case L = l + 1 . Compared to a l -layer network, a ( l + 1 ) -layer network has additional weight matrix w ( l ) and bias b ( l ) and thus the total parameters contain θ = ( θ ~ , w ( l ) , b ( l ) ) .
The output function of this ( l + 1 ) -layer network is:
f ( x ; θ ) = A ~ ( l + 1 ) ( x ; θ ) = 1 n l w ( l ) ⊤ σ ( A ~ ( l ) ( x ) ) + β b ( l )
And we know its derivative with respect to different sets of parameters; let denote A ~ ( l ) = A ~ ( l ) ( x ) for brevity in the following equation:
∇ w ( l ) f ( x ; θ ) = 1 n l σ ( A ~ ( l ) ) ⊤ ∈ R 1 × n l ∇ b ( l ) f ( x ; θ ) = β ∇ θ ~ f ( x ; θ ) = 1 n l ∇ θ ~ σ ( A ~ ( l ) ) w ( l ) = 1 n l [ σ ˙ ( A ~ 1 ( l ) ) ∂ A ~ 1 ( l ) ∂ θ ~ 1 … σ ˙ ( A ~ n l ( l ) ) ∂ A ~ n l ( l ) ∂ θ ~ 1 ⋮ σ ˙ ( A ~ 1 ( l ) ) ∂ A ~ 1 ( l ) ∂ θ ~ P ~ … σ ˙ ( A ~ n l ( l ) ) ∂ A ~ n l ( l ) ∂ θ ~ P ~ ] w ( l ) ∈ R P ~ × n l + 1
where σ ˙ is the derivative of σ and each entry at location ( p , m ) , 1 ≤ p ≤ P ~ , 1 ≤ m ≤ n l + 1 in the matrix ∇ θ ~ f ( x ; θ ) can be written as
∂ f m ( x ; θ ) ∂ θ ~ p = ∑ i = 1 n l w i m ( l ) σ ˙ ( A ~ i ( l ) ) ∇ θ ~ p A ~ i ( l )
The NTK for this ( l + 1 ) -layer network can be defined accordingly:
K ( l + 1 ) ( x , x ′ ; θ ) = ∇ θ f ( x ; θ ) ⊤ ∇ θ f ( x ; θ ) = ∇ w ( l ) f ( x ; θ ) ⊤ ∇ w ( l ) f ( x ; θ ) + ∇ b ( l ) f ( x ; θ ) ⊤ ∇ b ( l ) f ( x ; θ ) + ∇ θ ~ f ( x ; θ ) ⊤ ∇ θ ~ f ( x ; θ ) = 1 n l [ σ ( A ~ ( l ) ) σ ( A ~ ( l ) ) ⊤ + β 2 + w ( l ) ⊤ [ σ ˙ ( A ~ 1 ( l ) ) σ ˙ ( A ~ 1 ( l ) ) ∑ p = 1 P ~ ∂ A ~ 1 ( l ) ∂ θ ~ p ∂ A ~ 1 ( l ) ∂ θ ~ p … σ ˙ ( A ~ 1 ( l ) ) σ ˙ ( A ~ n l ( l ) ) ∑ p = 1 P ~ ∂ A ~ 1 ( l ) ∂ θ ~ p ∂ A ~ n l ( l ) ∂ θ ~ p ⋮ σ ˙ ( A ~ n l ( l ) ) σ ˙ ( A ~ 1 ( l ) ) ∑ p = 1 P ~ ∂ A ~ n l ( l ) ∂ θ ~ p ∂ A ~ 1 ( l ) ∂ θ ~ p … σ ˙ ( A ~ n l ( l ) ) σ ˙ ( A ~ n l ( l ) ) ∑ p = 1 P ~ ∂ A ~ n l ( l ) ∂ θ ~ p ∂ A ~ n l ( l ) ∂ θ ~ p ] w ( l ) ] = 1 n l [ σ ( A ~ ( l ) ) σ ( A ~ ( l ) ) ⊤ + β 2 + w ( l ) ⊤ [ σ ˙ ( A ~ 1 ( l ) ) σ ˙ ( A ~ 1 ( l ) ) K 11 ( l ) … σ ˙ ( A ~ 1 ( l ) ) σ ˙ ( A ~ n l ( l ) ) K 1 n l ( l ) ⋮ σ ˙ ( A ~ n l ( l ) ) σ ˙ ( A ~ 1 ( l ) ) K n l 1 ( l ) … σ ˙ ( A ~ n l ( l ) ) σ ˙ ( A ~ n l ( l ) ) K n l n l ( l ) ] w ( l ) ]
where each individual entry at location ( m , n ) , 1 ≤ m , n ≤ n l + 1 of the matrix K ( l + 1 ) can be written as:
K m n ( l + 1 ) = 1 n l [ σ ( A ~ m ( l ) ) σ ( A ~ n ( l ) ) + β 2 + ∑ i = 1 n l ∑ j = 1 n l w i m ( l ) w i n ( l ) σ ˙ ( A ~ i ( l ) ) σ ˙ ( A ~ j ( l ) ) K i j ( l ) ]
When n l → ∞ , the section in blue and green has the limit (See the proof in the [previous section](#connection-with-gaussian-processes) ):
1 n l σ ( A ~ ( l ) ) σ ( A ~ ( l ) ) + β 2 → Σ ( l + 1 )
and the red section has the limit:
∑ i = 1 n l ∑ j = 1 n l w i m ( l ) w i n ( l ) σ ˙ ( A ~ i ( l ) ) σ ˙ ( A ~ j ( l ) ) K i j ( l ) → ∑ i = 1 n l ∑ j = 1 n l w i m ( l ) w i n ( l ) σ ˙ ( A ~ i ( l ) ) σ ˙ ( A ~ j ( l ) ) K ∞ , i j ( l )
Later, [Arora et al. (2019)](https://arxiv.org/abs/1904.11955) provided a proof with a weaker limit, that does not require all the hidden layers to be infinitely wide, but only requires the minimum width to be sufficiently large.
## Linearized Models [#](#linearized-models)
From the [previous section](#neural-tangent-kernel) , according to the derivative chain rule, we have known that the gradient update on the output of an infinite width network is as follows; For brevity, we omit the inputs in the following analysis:
d f ( θ ) d t = − η ∇ θ f ( θ ) ⊤ ∇ θ f ( θ ) ∇ f L = − η ∇ θ f ( θ ) ⊤ ∇ θ f ( θ ) ∇ f L = − η K ( θ ) ∇ f L = − η K ∞ ∇ f L ; for infinite width network
To track the evolution of θ in time, let’s consider it as a function of time step t . With Taylor expansion, the network learning dynamics can be simplified as:
f ( θ ( t ) ) ≈ f lin ( θ ( t ) ) = f ( θ ( 0 ) ) + ∇ θ f ( θ ( 0 ) ) ⏟ formally ∇ θ f ( x ; θ ) | θ = θ ( 0 ) ( θ ( t ) − θ ( 0 ) )
Such formation is commonly referred to as the linearized model, given θ ( 0 ) , f ( θ ( 0 ) ) , and ∇ θ f ( θ ( 0 ) ) are all constants. Assuming that the incremental time step t is extremely small and the parameter is updated by gradient descent:
θ ( t ) − θ ( 0 ) = − η ∇ θ L ( θ ) = − η ∇ θ f ( θ ) ⊤ ∇ f L f lin ( θ ( t ) ) − f ( θ ( 0 ) ) = − η ∇ θ f ( θ ( 0 ) ) ⊤ ∇ θ f ( X ; θ ( 0 ) ) ∇ f L d f ( θ ( t ) ) d t = − η K ( θ ( 0 ) ) ∇ f L d f ( θ ( t ) ) d t = − η K ∞ ∇ f L ; for infinite width network
Eventually we get the same learning dynamics, which implies that a neural network with infinite width can be considerably simplified as governed by the above linearized model ( [Lee & Xiao, et al. 2019](https://arxiv.org/abs/1902.06720) ).
In a simple case when the empirical loss is an MSE loss, ∇ θ L ( θ ) = f ( X ; θ ) − Y , the dynamics of the network becomes a simple linear ODE and it can be solved in a closed form:
d f ( θ ) d t = − η K ∞ ( f ( θ ) − Y ) d g ( θ ) d t = − η K ∞ g ( θ ) ; let g ( θ ) = f ( θ ) − Y ∫ d g ( θ ) g ( θ ) = − η ∫ K ∞ d t g ( θ ) = C e − η K ∞ t
When t = 0 , we have C = f ( θ ( 0 ) ) − Y and therefore,
f ( θ ) = ( f ( θ ( 0 ) ) − Y ) e − η K ∞ t + Y = f ( θ ( 0 ) ) e − K ∞ t + ( I − e − η K ∞ t ) Y
## Lazy Training [#](#lazy-training)
People observe that when a neural network is heavily over-parameterized, the model is able to learn with the training loss quickly converging to zero but the network parameters hardly change. Lazy training refers to the phenomenon. In other words, when the loss L has a decent amount of reduction, the change in the differential of the network f (aka the Jacobian matrix) is still very small.
Let θ ( 0 ) be the initial network parameters and θ ( T ) be the final network parameters when the loss has been minimized to zero. The delta change in parameter space can be approximated with first-order Taylor expansion:
y ^ = f ( θ ( T ) ) ≈ f ( θ ( 0 ) ) + ∇ θ f ( θ ( 0 ) ) ( θ ( T ) − θ ( 0 ) ) Thus Δ θ = θ ( T ) − θ ( 0 ) ≈ ‖ y ^ − f ( θ ( 0 ) ) ‖ ‖ ∇ θ f ( θ ( 0 ) ) ‖
Still following the first-order Taylor expansion, we can track the change in the differential of f :
∇ θ f ( θ ( T ) ) ≈ ∇ θ f ( θ ( 0 ) ) + ∇ θ 2 f ( θ ( 0 ) ) Δ θ = ∇ θ f ( θ ( 0 ) ) + ∇ θ 2 f ( θ ( 0 ) ) ‖ y ^ − f ( x ; θ ( 0 ) ) ‖ ‖ ∇ θ f ( θ ( 0 ) ) ‖ Thus Δ ( ∇ θ f ) = ∇ θ f ( θ ( T ) ) − ∇ θ f ( θ ( 0 ) ) = ‖ y ^ − f ( x ; θ ( 0 ) ) ‖ ∇ θ 2 f ( θ ( 0 ) ) ‖ ∇ θ f ( θ ( 0 ) ) ‖
Let κ ( θ ) be the relative change of the differential of f to the change in the parameter space:
κ ( θ = Δ ( ∇ θ f ) ‖ ∇ θ f ( θ ( 0 ) ) ‖ = ‖ y ^ − f ( θ ( 0 ) ) ‖ ∇ θ 2 f ( θ ( 0 ) ) ‖ ∇ θ f ( θ ( 0 ) ) ‖ 2
[Chizat et al. (2019)](https://arxiv.org/abs/1812.07956) showed the proof for a two-layer neural network that E [ κ ( θ 0 ) ] → 0 (getting into the lazy regime) when the number of hidden neurons → ∞ . Also, recommend [this post](https://rajatvd.github.io/NTK/) for more discussion on linearized models and lazy training.
# Citation [#](#citation)
Cited as:
Weng, Lilian. (Sep 2022). Some math behind neural tangent kernel. Lil’Log. https://lilianweng.github.io/posts/2022-09-08-ntk/.
Or
@article{weng2022ntk,
title = "Some Math behind Neural Tangent Kernel" ,
author = "Weng, Lilian" ,
journal = "Lil'Log" ,
year = "2022" ,
month = "Sep" ,
url = "https://lilianweng.github.io/posts/2022-09-08-ntk/" } copy
# References [#](#references)
[1] Jacot et al. [“Neural Tangent Kernel: Convergence and Generalization in Neural Networks.”](https://arxiv.org/abs/1806.07572) NeuriPS 2018.
[2]Radford M. Neal. “Priors for Infinite Networks.” Bayesian Learning for Neural Networks. Springer, New York, NY, 1996. 29-53.
[3] Lee & Bahri et al. [“Deep Neural Networks as Gaussian Processes.”](https://arxiv.org/abs/1711.00165) ICLR 2018.
[4] Chizat et al. [“On Lazy Training in Differentiable Programming”](https://arxiv.org/abs/1812.07956) NeuriPS 2019.
[5] Lee & Xiao, et al. [“Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent.”](https://arxiv.org/abs/1902.06720) NeuriPS 2019.
[6] Arora, et al. [“On Exact Computation with an Infinitely Wide Neural Net.”](https://arxiv.org/abs/1904.11955) NeurIPS 2019.
[7] (YouTube video) [“Neural Tangent Kernel: Convergence and Generalization in Neural Networks”](https://www.youtube.com/watch?v=raT2ECrvbag) by Arthur Jacot, Nov 2018.
[8] (YouTube video) [“Lecture 7 - Deep Learning Foundations: Neural Tangent Kernels”](https://www.youtube.com/watch?v=DObobAnELkU) by Soheil Feizi, Sep 2020.
[9] [“Understanding the Neural Tangent Kernel.”](https://rajatvd.github.io/NTK/) Rajat’s Blog.
[10] [“Neural Tangent Kernel.”](https://appliedprobability.blog/2021/03/10/neural-tangent-kernel/) Applied Probability Notes, Mar 2021.
[11] [“Some Intuition on the Neural Tangent Kernel.”](https://www.inference.vc/neural-tangent-kernels-some-intuition-for-kernel-gradient-descent/) inFERENCe, Nov 2020.
[Foundation](https://lilianweng.github.io/tags/foundation/) [Neural-Tangent-Kernel](https://lilianweng.github.io/tags/neural-tangent-kernel/) [Learning-Dynamics](https://lilianweng.github.io/tags/learning-dynamics/) [« Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) [» Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/) © 2025 [Lil'Log](https://lilianweng.github.io/) Powered by [Hugo](https://gohugo.io/) & [PaperMod](https://git.io/hugopapermod)