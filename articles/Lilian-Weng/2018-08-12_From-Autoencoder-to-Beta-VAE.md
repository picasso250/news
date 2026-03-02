[Lil'Log](https://lilianweng.github.io/) | [Posts](https://lilianweng.github.io/) [Archive](https://lilianweng.github.io/archives) [Search](https://lilianweng.github.io/search/) [Tags](https://lilianweng.github.io/tags/) [FAQ](https://lilianweng.github.io/faq)
# From Autoencoder to Beta-VAE
Date: August 12, 2018 | Estimated Reading Time: 21 min | Author: Lilian Weng Table of Contents [Notation](#notation) [Autoencoder](#autoencoder) [Denoising Autoencoder](#denoising-autoencoder) [Sparse Autoencoder](#sparse-autoencoder) [Contractive Autoencoder](#contractive-autoencoder) [VAE: Variational Autoencoder](#vae-variational-autoencoder) [Loss Function: ELBO](#loss-function-elbo) [Reparameterization Trick](#reparameterization-trick) [Beta-VAE](#beta-vae) [VQ-VAE and VQ-VAE-2](#vq-vae-and-vq-vae-2) [TD-VAE](#td-vae) [References](#references) Autocoders are a family of neural network models aiming to learn compressed latent variables of high-dimensional data. Starting from the basic autocoder model, this post reviews several variations, including denoising, sparse, and contractive autoencoders, and then Variational Autoencoder (VAE) and its modification beta-VAE.
[Updated on 2019-07-18: add a section on [VQ-VAE & VQ-VAE-2](#vq-vae-and-vq-vae-2) .] [Updated on 2019-07-26: add a section on [TD-VAE](#td-vae) .]
Autocoder is invented to reconstruct high-dimensional data using a neural network model with a narrow bottleneck layer in the middle (oops, this is probably not true for [Variational Autoencoder](#vae-variational-autoencoder) , and we will investigate it in details in later sections). A nice byproduct is dimension reduction: the bottleneck layer captures a compressed latent encoding. Such a low-dimensional representation can be used as en embedding vector in various applications (i.e. search), help data compression, or reveal the underlying data generative factors.
# Notation [#](#notation)
| Symbol | Mean |
|---|---|
| D | The dataset, D = { x ( 1 ) , x ( 2 ) , … , x ( n ) } , contains n data samples; | D | = n . |
| x ( i ) | Each data point is a vector of d dimensions, x ( i ) = [ x 1 ( i ) , x 2 ( i ) , … , x d ( i ) ] . |
| x | One data sample from the dataset, x ∈ D . |
| x ′ | The reconstructed version of x . |
| x ~ | The corrupted version of x . |
| z | The compressed code learned in the bottleneck layer. |
| a j ( l ) | The activation function for the j -th neuron in the l -th hidden layer. |
| g ϕ ( . ) | The encoding function parameterized by ϕ . |
| f θ ( . ) | The decoding function parameterized by θ . |
| q ϕ ( z | x ) | Estimated posterior probability function, also known as probabilistic encoder . |
| p θ ( x | z ) | Likelihood of generating true data sample given the latent code, also known as probabilistic decoder . |
# Autoencoder [#](#autoencoder)
Autoencoder is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation. The idea was originated in [the 1980s](https://en.wikipedia.org/wiki/Autoencoder) , and later promoted by the seminal paper by [Hinton & Salakhutdinov, 2006](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.3788&rep=rep1&type=pdf) .
It consists of two networks:
Encoder network: It translates the original high-dimension input into the latent low-dimensional code. The input size is larger than the output size. Decoder network: The decoder network recovers the data from the code, likely with larger and larger output layers. Illustration of autoencoder model architecture.
The encoder network essentially accomplishes the [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) , just like how we would use Principal Component Analysis (PCA) or Matrix Factorization (MF) for. In addition, the autoencoder is explicitly optimized for the data reconstruction from the code. A good intermediate representation not only can capture latent variables, but also benefits a full [decompression](https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html) process.
The model contains an encoder function g ( . ) parameterized by ϕ and a decoder function f ( . ) parameterized by θ . The low-dimensional code learned for input x in the bottleneck layer is z = g ϕ ( x ) and the reconstructed input is x ′ = f θ ( g ϕ ( x ) ) .
The parameters ( θ , ϕ ) are learned together to output a reconstructed data sample same as the original input, x ≈ f θ ( g ϕ ( x ) ) , or in other words, to learn an identity function. There are various metrics to quantify the difference between two vectors, such as cross entropy when the activation function is sigmoid, or as simple as MSE loss:
L AE ( θ , ϕ ) = 1 n ∑ i = 1 n ( x ( i ) − f θ ( g ϕ ( x ( i ) ) ) ) 2
# Denoising Autoencoder [#](#denoising-autoencoder)
Since the autoencoder learns the identity function, we are facing the risk of “overfitting” when there are more network parameters than the number of data points.
To avoid overfitting and improve the robustness, Denoising Autoencoder (Vincent et al. 2008) proposed a modification to the basic autoencoder. The input is partially corrupted by adding noises to or masking some values of the input vector in a stochastic manner, x ~ ∼ M D ( x ~ | x ) . Then the model is trained to recover the original input (note: not the corrupt one).
x ~ ( i ) ∼ M D ( x ~ ( i ) | x ( i ) ) L DAE ( θ , ϕ ) = 1 n ∑ i = 1 n ( x ( i ) − f θ ( g ϕ ( x ~ ( i ) ) ) ) 2
where M D defines the mapping from the true data samples to the noisy or corrupted ones.
Illustration of denoising autoencoder model architecture.
This design is motivated by the fact that humans can easily recognize an object or a scene even the view is partially occluded or corrupted. To “repair” the partially destroyed input, the denoising autoencoder has to discover and capture relationship between dimensions of input in order to infer missing pieces.
For high dimensional input with high redundancy, like images, the model is likely to depend on evidence gathered from a combination of many input dimensions to recover the denoised version rather than to overfit one dimension. This builds up a good foundation for learning robust latent representation.
The noise is controlled by a stochastic mapping M D ( x ~ | x ) , and it is not specific to a particular type of corruption process (i.e. masking noise, Gaussian noise, salt-and-pepper noise, etc.). Naturally the corruption process can be equipped with prior knowledge
In the experiment of the original DAE paper, the noise is applied in this way: a fixed proportion of input dimensions are selected at random and their values are forced to 0. Sounds a lot like dropout, right? Well, the denoising autoencoder was proposed in 2008, 4 years before the dropout paper ( [Hinton, et al. 2012](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) ) ;)
**Stacked Denoising Autoencoder**: In the old days when it was still hard to train deep neural networks, stacking denoising autoencoders was a way to build deep models ([Vincent et al., 2010](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)). The denoising autoencoders are trained layer by layer. Once one layer has been trained, it is fed with clean, uncorrupted inputs to learn the encoding in the next layer.
<figure>
<img src="stacking-dae.png" style="width: 100%;" />
<figcaption>Stacking denoising autoencoders. (Image source: <a href="http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf" target="_blank">Vincent et al., 2010</a>)</figcaption>
</figure>
# Sparse Autoencoder [#](#sparse-autoencoder)
Sparse Autoencoder applies a “sparse” constraint on the hidden unit activation to avoid overfitting and improve robustness. It forces the model to only have a small number of hidden units being activated at the same time, or in other words, one hidden neuron should be inactivate most of time.
Recall that common [activation functions](http://cs231n.github.io/neural-networks-1/#actfun) include sigmoid, tanh, relu, leaky relu, etc. A neuron is activated when the value is close to 1 and inactivate with a value close to 0.
Let’s say there are s l neurons in the l -th hidden layer and the activation function for the j -th neuron in this layer is labelled as a j ( l ) ( . ) , j = 1 , … , s l . The fraction of activation of this neuron ρ ^ j is expected to be a small number ρ , known as sparsity parameter ; a common config is ρ = 0.05 .
ρ ^ j ( l ) = 1 n ∑ i = 1 n [ a j ( l ) ( x ( i ) ) ] ≈ ρ
This constraint is achieved by adding a penalty term into the loss function. The KL-divergence D KL measures the difference between two Bernoulli distributions, one with mean ρ and the other with mean ρ ^ j ( l ) . The hyperparameter β controls how strong the penalty we want to apply on the sparsity loss.
L SAE ( θ ) = L ( θ ) + β ∑ l = 1 L ∑ j = 1 s l D KL ( ρ ‖ ρ ^ j ( l ) ) = L ( θ ) + β ∑ l = 1 L ∑ j = 1 s l ρ log ⁡ ρ ρ ^ j ( l ) + ( 1 − ρ ) log ⁡ 1 − ρ 1 − ρ ^ j ( l ) The KL divergence between a Bernoulli distribution with mean ρ = 0.25 and a Bernoulli distribution with mean 0 ≤ ρ ^ ≤ 1 .
k -Sparse Autoencoder
In k -Sparse Autoencoder ( [Makhzani and Frey, 2013](https://arxiv.org/abs/1312.5663) ), the sparsity is enforced by only keeping the top k highest activations in the bottleneck layer with linear activation function.
First we run feedforward through the encoder network to get the compressed code: z = g ( x ) .
Sort the values in the code vector z . Only the k largest values are kept while other neurons are set to 0. This can be done in a ReLU layer with an adjustable threshold too. Now we have a sparsified code: z ′ = Sparsify ( z ) .
Compute the output and the loss from the sparsified code, L = | x − f ( z ′ ) | 2 2 .
And, the back-propagation only goes through the top k activated hidden units!
Filters of the k-sparse autoencoder for different sparsity levels k, learnt from MNIST with 1000 hidden units.. (Image source: [Makhzani and Frey, 2013](https://arxiv.org/abs/1312.5663) )
# Contractive Autoencoder [#](#contractive-autoencoder)
Similar to sparse autoencoder, Contractive Autoencoder ( [Rifai, et al, 2011](http://www.icml-2011.org/papers/455_icmlpaper.pdf) ) encourages the learned representation to stay in a contractive space for better robustness.
It adds a term in the loss function to penalize the representation being too sensitive to the input, and thus improve the robustness to small perturbations around the training data points. The sensitivity is measured by the Frobenius norm of the Jacobian matrix of the encoder activations with respect to the input:
‖ J f ( x ) ‖ F 2 = ∑ i j ( ∂ h j ( x ) ∂ x i ) 2
where h j is one unit output in the compressed code z = f ( x ) .
This penalty term is the sum of squares of all partial derivatives of the learned encoding with respect to input dimensions. The authors claimed that empirically this penalty was found to carve a representation that corresponds to a lower-dimensional non-linear manifold, while staying more invariant to majority directions orthogonal to the manifold.
# VAE: Variational Autoencoder [#](#vae-variational-autoencoder)
The idea of Variational Autoencoder ( [Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114) ), short for VAE , is actually less similar to all the autoencoder models above, but deeply rooted in the methods of variational bayesian and graphical model.
Instead of mapping the input into a fixed vector, we want to map it into a distribution. Let’s label this distribution as p θ , parameterized by θ . The relationship between the data input x and the latent encoding vector z can be fully defined by:
Prior p θ ( z ) Likelihood p θ ( x | z ) Posterior p θ ( z | x )
Assuming that we know the real parameter θ ∗ for this distribution. In order to generate a sample that looks like a real data point x ( i ) , we follow these steps:
First, sample a z ( i ) from a prior distribution p θ ∗ ( z ) . Then a value x ( i ) is generated from a conditional distribution p θ ∗ ( x | z = z ( i ) ) .
The optimal parameter θ ∗ is the one that maximizes the probability of generating real data samples:
θ ∗ = arg ⁡ max θ ∏ i = 1 n p θ ( x ( i ) )
Commonly we use the log probabilities to convert the product on RHS to a sum:
θ ∗ = arg ⁡ max θ ∑ i = 1 n log ⁡ p θ ( x ( i ) )
Now let’s update the equation to better demonstrate the data generation process so as to involve the encoding vector:
p θ ( x ( i ) ) = ∫ p θ ( x ( i ) | z ) p θ ( z ) d z
Unfortunately it is not easy to compute p θ ( x ( i ) ) in this way, as it is very expensive to check all the possible values of z and sum them up. To narrow down the value space to facilitate faster search, we would like to introduce a new approximation function to output what is a likely code given an input x , q ϕ ( z | x ) , parameterized by ϕ .
The graphical model involved in Variational Autoencoder. Solid lines denote the generative distribution p _ θ ( . ) and dashed lines denote the distribution q _ ϕ ( z | x ) to approximate the intractable posterior p _ θ ( z | x ) .
Now the structure looks a lot like an autoencoder:
The conditional probability p θ ( x | z ) defines a generative model, similar to the decoder f θ ( x | z ) introduced above. p θ ( x | z ) is also known as probabilistic decoder . The approximation function q ϕ ( z | x ) is the probabilistic encoder , playing a similar role as g ϕ ( z | x ) above.
## Loss Function: ELBO [#](#loss-function-elbo)
The estimated posterior q ϕ ( z | x ) should be very close to the real one p θ ( z | x ) . We can use [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) to quantify the distance between these two distributions. KL divergence D KL ( X | Y ) measures how much information is lost if the distribution Y is used to represent X.
In our case we want to minimize D KL ( q ϕ ( z | x ) | p θ ( z | x ) ) with respect to ϕ .
But why use D KL ( q ϕ | p θ ) (reversed KL) instead of D KL ( p θ | q ϕ ) (forward KL)? Eric Jang has a great explanation in his [post](https://blog.evjang.com/2016/08/variational-bayes.html) on Bayesian Variational methods. As a quick recap:
Forward and reversed KL divergence have different demands on how to match two distributions. (Image source: [blog.evjang.com/2016/08/variational-bayes.html](https://blog.evjang.com/2016/08/variational-bayes.html) ) Forward KL divergence: D KL ( P | Q ) = E z ∼ P ( z ) log ⁡ P ( z ) Q ( z ) ; we have to ensure that Q(z)>0 wherever P(z)>0. The optimized variational distribution q ( z ) has to cover over the entire p ( z ) . Reversed KL divergence: D KL ( Q | P ) = E z ∼ Q ( z ) log ⁡ Q ( z ) P ( z ) ; minimizing the reversed KL divergence squeezes the Q ( z ) under P ( z ) .
Let’s now expand the equation:
D KL ( q ϕ ( z | x ) ‖ p θ ( z | x ) ) = ∫ q ϕ ( z | x ) log ⁡ q ϕ ( z | x ) p θ ( z | x ) d z = ∫ q ϕ ( z | x ) log ⁡ q ϕ ( z | x ) p θ ( x ) p θ ( z , x ) d z ; Because p ( z | x ) = p ( z , x ) / p ( x ) = ∫ q ϕ ( z | x ) ( log ⁡ p θ ( x ) + log ⁡ q ϕ ( z | x ) p θ ( z , x ) ) d z = log ⁡ p θ ( x ) + ∫ q ϕ ( z | x ) log ⁡ q ϕ ( z | x ) p θ ( z , x ) d z ; Because ∫ q ( z | x ) d z = 1 = log ⁡ p θ ( x ) + ∫ q ϕ ( z | x ) log ⁡ q ϕ ( z | x ) p θ ( x | z ) p θ ( z ) d z ; Because p ( z , x ) = p ( x | z ) p ( z ) = log ⁡ p θ ( x ) + E z ∼ q ϕ ( z | x ) [ log ⁡ q ϕ ( z | x ) p θ ( z ) − log ⁡ p θ ( x | z ) ] = log ⁡ p θ ( x ) + D KL ( q ϕ ( z | x ) ‖ p θ ( z ) ) − E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z )
So we have:
D KL ( q ϕ ( z | x ) ‖ p θ ( z | x ) ) = log ⁡ p θ ( x ) + D KL ( q ϕ ( z | x ) ‖ p θ ( z ) ) − E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z )
Once rearrange the left and right hand side of the equation,
log ⁡ p θ ( x ) − D KL ( q ϕ ( z | x ) ‖ p θ ( z | x ) ) = E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z ) − D KL ( q ϕ ( z | x ) ‖ p θ ( z ) )
The LHS of the equation is exactly what we want to maximize when learning the true distributions: we want to maximize the (log-)likelihood of generating real data (that is log ⁡ p θ ( x ) ) and also minimize the difference between the real and estimated posterior distributions (the term D KL works like a regularizer). Note that p θ ( x ) is fixed with respect to q ϕ .
The negation of the above defines our loss function:
L VAE ( θ , ϕ ) = − log ⁡ p θ ( x ) + D KL ( q ϕ ( z | x ) ‖ p θ ( z | x ) ) = − E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z ) + D KL ( q ϕ ( z | x ) ‖ p θ ( z ) ) θ ∗ , ϕ ∗ = arg ⁡ min θ , ϕ L VAE
In Variational Bayesian methods, this loss function is known as the variational lower bound , or evidence lower bound . The “lower bound” part in the name comes from the fact that KL divergence is always non-negative and thus − L VAE is the lower bound of log ⁡ p θ ( x ) .
− L VAE = log ⁡ p θ ( x ) − D KL ( q ϕ ( z | x ) ‖ p θ ( z | x ) ) ≤ log ⁡ p θ ( x )
Therefore by minimizing the loss, we are maximizing the lower bound of the probability of generating real data samples.
## Reparameterization Trick [#](#reparameterization-trick)
The expectation term in the loss function invokes generating samples from z ∼ q ϕ ( z | x ) . Sampling is a stochastic process and therefore we cannot backpropagate the gradient. To make it trainable, the reparameterization trick is introduced: It is often possible to express the random variable z as a deterministic variable z = T ϕ ( x , ϵ ) , where ϵ is an auxiliary independent random variable, and the transformation function T ϕ parameterized by ϕ converts ϵ to z .
For example, a common choice of the form of q ϕ ( z | x ) is a multivariate Gaussian with a diagonal covariance structure:
z ∼ q ϕ ( z | x ( i ) ) = N ( z ; μ ( i ) , σ 2 ( i ) I ) z = μ + σ ⊙ ϵ , where ϵ ∼ N ( 0 , I ) ; Reparameterization trick.
where ⊙ refers to element-wise product.
Illustration of how the reparameterization trick makes the z sampling process trainable.(Image source: Slide 12 in Kingma’s NIPS 2015 workshop [talk](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf) )
The reparameterization trick works for other types of distributions too, not only Gaussian.
In the multivariate Gaussian case, we make the model trainable by learning the mean and variance of the distribution, μ and σ , explicitly using the reparameterization trick, while the stochasticity remains in the random variable ϵ ∼ N ( 0 , I ) .
Illustration of variational autoencoder model with the multivariate Gaussian assumption.
# Beta-VAE [#](#beta-vae)
If each variable in the inferred latent representation z is only sensitive to one single generative factor and relatively invariant to other factors, we will say this representation is disentangled or factorized. One benefit that often comes with disentangled representation is good interpretability and easy generalization to a variety of tasks.
For example, a model trained on photos of human faces might capture the gentle, skin color, hair color, hair length, emotion, whether wearing a pair of glasses and many other relatively independent factors in separate dimensions. Such a disentangled representation is very beneficial to facial image generation.
β-VAE ( [Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl) ) is a modification of Variational Autoencoder with a special emphasis to discover disentangled latent factors. Following the same incentive in VAE, we want to maximize the probability of generating real data, while keeping the distance between the real and estimated posterior distributions small (say, under a small constant δ ):
max ϕ , θ E x ∼ D [ E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z ) ] subject to D KL ( q ϕ ( z | x ) ‖ p θ ( z ) ) < δ
We can rewrite it as a Lagrangian with a Lagrangian multiplier β under the [KKT condition](https://www.cs.cmu.edu/~ggordon/10725-F12/slides/16-kkt.pdf) . The above optimization problem with only one inequality constraint is equivalent to maximizing the following equation F ( θ , ϕ , β ) :
F ( θ , ϕ , β ) = E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z ) − β ( D KL ( q ϕ ( z | x ) ‖ p θ ( z ) ) − δ ) = E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z ) − β D KL ( q ϕ ( z | x ) ‖ p θ ( z ) ) + β δ ≥ E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z ) − β D KL ( q ϕ ( z | x ) ‖ p θ ( z ) ) ; Because β , δ ≥ 0
The loss function of β -VAE is defined as:
L BETA ( ϕ , β ) = − E z ∼ q ϕ ( z | x ) log ⁡ p θ ( x | z ) + β D KL ( q ϕ ( z | x ) ‖ p θ ( z ) )
where the Lagrangian multiplier β is considered as a hyperparameter.
Since the negation of L BETA ( ϕ , β ) is the lower bound of the Lagrangian F ( θ , ϕ , β ) . Minimizing the loss is equivalent to maximizing the Lagrangian and thus works for our initial optimization problem.
When β = 1 , it is same as VAE. When β > 1 , it applies a stronger constraint on the latent bottleneck and limits the representation capacity of z . For some conditionally independent generative factors, keeping them disentangled is the most efficient representation. Therefore a higher β encourages more efficient latent encoding and further encourages the disentanglement. Meanwhile, a higher β may create a trade-off between reconstruction quality and the extent of disentanglement.
[Burgess, et al. (2017)](https://arxiv.org/pdf/1804.03599.pdf) discussed the distentangling in β -VAE in depth with an inspiration by the [information bottleneck theory](https://lilianweng.github.io/posts/2017-09-28-information-bottleneck/) and further proposed a modification to β -VAE to better control the encoding representation capacity.
# VQ-VAE and VQ-VAE-2 [#](#vq-vae-and-vq-vae-2)
The VQ-VAE (“Vector Quantised-Variational AutoEncoder”; [van den Oord, et al. 2017](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf) ) model learns a discrete latent variable by the encoder, since discrete representations may be a more natural fit for problems like language, speech, reasoning, etc.
Vector quantisation (VQ) is a method to map K -dimensional vectors into a finite set of “code” vectors. The process is very much similar to [KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) algorithm. The optimal centroid code vector that a sample should be mapped to is the one with minimum euclidean distance.
Let e ∈ R K × D , i = 1 , … , K be the latent embedding space (also known as “codebook”) in VQ-VAE, where K is the number of latent variable categories and D is the embedding size. An individual embedding vector is e i ∈ R D , i = 1 , … , K .
The encoder output E ( x ) = z e goes through a nearest-neighbor lookup to match to one of K embedding vectors and then this matched code vector becomes the input for the decoder D ( . ) :
z q ( x ) = Quantize ( E ( x ) ) = e k where k = arg ⁡ min i ‖ E ( x ) − e i ‖ 2
Note that the discrete latent variables can have different shapes in differnet applications; for example, 1D for speech, 2D for image and 3D for video.
The architecture of VQ-VAE (Image source: [van den Oord, et al. 2017](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf) )
Because argmin() is non-differentiable on a discrete space, the gradients ∇ z L from decoder input z q is copied to the encoder output z e . Other than reconstruction loss, VQ-VAE also optimizes:
VQ loss : The L2 error between the embedding space and the encoder outputs. Commitment loss : A measure to encourage the encoder output to stay close to the embedding space and to prevent it from fluctuating too frequently from one code vector to another. L = ‖ x − D ( e k ) ‖ 2 2 ⏟ reconstruction loss + ‖ sg [ E ( x ) ] − e k ‖ 2 2 ⏟ VQ loss + β ‖ E ( x ) − sg [ e k ] ‖ 2 2 ⏟ commitment loss
where sg [ . ] is the stop_gradient operator.
The embedding vectors in the codebook is updated through EMA (exponential moving average). Given a code vector e i , say we have n i encoder output vectors, { z i , j } j = 1 n i , that are quantized to e i :
N i ( t ) = γ N i ( t − 1 ) + ( 1 − γ ) n i ( t ) m i ( t ) = γ m i ( t − 1 ) + ( 1 − γ ) ∑ j = 1 n i ( t ) z i , j ( t ) e i ( t ) = m i ( t ) / N i ( t )
where ( t ) refers to batch sequence in time. N i and m i are accumulated vector count and volume, respectively.
VQ-VAE-2 ( [Ali Razavi, et al. 2019](https://arxiv.org/abs/1906.00446) ) is a two-level hierarchical VQ-VAE combined with self-attention autoregressive model.
Stage 1 is to train a hierarchical VQ-VAE : The design of hierarchical latent variables intends to separate local patterns (i.e., texture) from global information (i.e., object shapes). The training of the larger bottom level codebook is conditioned on the smaller top level code too, so that it does not have to learn everything from scratch. Stage 2 is to learn a prior over the latent discrete codebook so that we sample from it and generate images. In this way, the decoder can receive input vectors sampled from a similar distribution as the one in training. A powerful autoregressive model enhanced with multi-headed self-attention layers is used to capture the prior distribution (like [PixelSNAIL; Chen et al 2017](https://arxiv.org/abs/1712.09763) ).
Considering that VQ-VAE-2 depends on discrete latent variables configured in a simple hierarchical setting, the quality of its generated images are pretty amazing.
Architecture of hierarchical VQ-VAE and multi-stage image generation. (Image source: [Ali Razavi, et al. 2019](https://arxiv.org/abs/1906.00446) ) The VQ-VAE-2 algorithm. (Image source: [Ali Razavi, et al. 2019](https://arxiv.org/abs/1906.00446) )
# TD-VAE [#](#td-vae)
TD-VAE (“Temporal Difference VAE”; [Gregor et al., 2019](https://arxiv.org/abs/1806.03107) ) works with sequential data. It relies on three main ideas, described below.
State-space model as a Markov Chain model.
1. State-Space Models In (latent) state-space models, a sequence of unobserved hidden states z = ( z 1 , … , z T ) determine the observation states x = ( x 1 , … , x T ) . Each time step in the Markov chain model in Fig. 13 can be trained in a similar manner as in Fig. 6, where the intractable posterior p ( z | x ) is approximated by a function q ( z | x ) .
2. Belief State An agent should learn to encode all the past states to reason about the future, named as belief state , b t = b e l i e f ( x 1 , … , x t ) = b e l i e f ( b t − 1 , x t ) . Given this, the distribution of future states conditioned on the past can be written as p ( x t + 1 , … , x T | x 1 , … , x t ) ≈ p ( x t + 1 , … , x T | b t ) . The hidden states in a recurrent policy are used as the agent’s belief state in TD-VAE. Thus we have b t = RNN ( b t − 1 , x t ) .
3. Jumpy Prediction Further, an agent is expected to imagine distant futures based on all the information gathered so far, suggesting the capability of making jumpy predictions, that is, predicting states several steps further into the future.
Recall what we have learned from the variance lower bound [above](#loss-function-elbo) :
log ⁡ p ( x ) ≥ log ⁡ p ( x ) − D KL ( q ( z | x ) ‖ p ( z | x ) ) = E z ∼ q log ⁡ p ( x | z ) − D KL ( q ( z | x ) ‖ p ( z ) ) = E z ∼ q log ⁡ p ( x | z ) − E z ∼ q log ⁡ q ( z | x ) p ( z ) = E z ∼ q [ log ⁡ p ( x | z ) − log ⁡ q ( z | x ) + log ⁡ p ( z ) ] = E z ∼ q [ log ⁡ p ( x , z ) − log ⁡ q ( z | x ) ] log ⁡ p ( x ) ≥ E z ∼ q [ log ⁡ p ( x , z ) − log ⁡ q ( z | x ) ]
Now let’s model the distribution of the state x t as a probability function conditioned on all the past states x < t and two latent variables, z t and z t − 1 , at current time step and one step back:
log ⁡ p ( x t | x < t ) ≥ E ( z t − 1 , z t ) ∼ q [ log ⁡ p ( x t , z t − 1 , z t | x < t ) − log ⁡ q ( z t − 1 , z t | x ≤ t ) ]
Continue expanding the equation:
log ⁡ p ( x t | x < t ) ≥ E ( z t − 1 , z t ) ∼ q [ log ⁡ p ( x t , z t − 1 , z t | x < t ) − log ⁡ q ( z t − 1 , z t | x ≤ t ) ] ≥ E ( z t − 1 , z t ) ∼ q [ log ⁡ p ( x t | z t − 1 , z t , x < t ) + log ⁡ p ( z t − 1 , z t | x < t ) − log ⁡ q ( z t − 1 , z t | x ≤ t ) ] ≥ E ( z t − 1 , z t ) ∼ q [ log ⁡ p ( x t | z t ) + log ⁡ p ( z t − 1 | x < t ) + log ⁡ p ( z t | z t − 1 ) − log ⁡ q ( z t − 1 , z t | x ≤ t ) ] ≥ E ( z t − 1 , z t ) ∼ q [ log ⁡ p ( x t | z t ) + log ⁡ p ( z t − 1 | x < t ) + log ⁡ p ( z t | z t − 1 ) − log ⁡ q ( z t | x ≤ t ) − log ⁡ q ( z t − 1 | z t , x ≤ t ) ]
Notice two things:
The red terms can be ignored according to Markov assumptions. The blue term is expanded according to Markov assumptions. The green term is expanded to include an one-step prediction back to the past as a smoothing distribution.
Precisely, there are four types of distributions to learn:
p D ( . ) is the decoder distribution: p ( x t ∣ z t ) is the encoder by the common definition; p ( x t ∣ z t ) → p D ( x t ∣ z t ) ; p T ( . ) is the transition distribution: p ( z t ∣ z t − 1 ) captures the sequential dependency between latent variables; p ( z t ∣ z t − 1 ) → p T ( z t ∣ z t − 1 ) ; p B ( . ) is the belief distribution: Both p ( z t − 1 ∣ x < t ) and q ( z t ∣ x ≤ t ) can use the belief states to predict the latent variables; p ( z t − 1 ∣ x < t ) → p B ( z t − 1 ∣ b t − 1 ) ; q ( z t ∣ x ≤ t ) → p B ( z t ∣ b t ) ; p S ( . ) is the smoothing distribution: The back-to-past smoothing term q ( z t − 1 ∣ z t , x ≤ t ) can be rewritten to be dependent of belief states too; q ( z t − 1 ∣ z t , x ≤ t ) → p S ( z t − 1 ∣ z t , b t − 1 , b t ) ;
To incorporate the idea of jumpy prediction, the sequential ELBO has to not only work on t , t + 1 , but also two distant timestamp t 1 < t 2 . Here is the final TD-VAE objective function to maximize:
J t 1 , t 2 = E [ log ⁡ p D ( x t 2 | z t 2 ) + log ⁡ p B ( z t 1 | b t 1 ) + log ⁡ p T ( z t 2 | z t 1 ) − log ⁡ p B ( z t 2 | b t 2 ) − log ⁡ p S ( z t 1 | z t 2 , b t 1 , b t 2 ) ] A detailed overview of TD-VAE architecture, very nicely done. (Image source: [TD-VAE paper](https://arxiv.org/abs/1806.03107) )
Cited as:
@article{weng2018VAE,
title = "From Autoencoder to Beta-VAE" ,
author = "Weng, Lilian" ,
journal = "lilianweng.github.io" ,
year = "2018" ,
url = "https://lilianweng.github.io/posts/2018-08-12-vae/" } copy
# References [#](#references)
[1] Geoffrey E. Hinton, and Ruslan R. Salakhutdinov. [“Reducing the dimensionality of data with neural networks.”](https://pdfs.semanticscholar.org/c50d/ca78e97e335d362d6b991ae0e1448914e9a3.pdf) Science 313.5786 (2006): 504-507.
[2] Pascal Vincent, et al. [“Extracting and composing robust features with denoising autoencoders.”](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) ICML, 2008.
[3] Pascal Vincent, et al. [“Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion.”](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) . Journal of machine learning research 11.Dec (2010): 3371-3408.
[4] Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov. “Improving neural networks by preventing co-adaptation of feature detectors.” arXiv preprint arXiv:1207.0580 (2012).
[5] [Sparse Autoencoder](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) by Andrew Ng.
[6] Alireza Makhzani, Brendan Frey (2013). [“k-sparse autoencoder”](https://arxiv.org/abs/1312.5663) . ICLR 2014.
[7] Salah Rifai, et al. [“Contractive auto-encoders: Explicit invariance during feature extraction.”](http://www.icml-2011.org/papers/455_icmlpaper.pdf) ICML, 2011.
[8] Diederik P. Kingma, and Max Welling. [“Auto-encoding variational bayes.”](https://arxiv.org/abs/1312.6114) ICLR 2014.
[9] [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) on jaan.io
[10] Youtube tutorial: [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) by Arxiv Insights
[11] [“A Beginner’s Guide to Variational Methods: Mean-Field Approximation”](https://blog.evjang.com/2016/08/variational-bayes.html) by Eric Jang.
[12] Carl Doersch. [“Tutorial on variational autoencoders.”](https://arxiv.org/abs/1606.05908) arXiv:1606.05908, 2016.
[13] Irina Higgins, et al. [" β -VAE: Learning basic visual concepts with a constrained variational framework."](https://openreview.net/forum?id=Sy2fzU9gl) ICLR 2017.
[14] Christopher P. Burgess, et al. [“Understanding disentangling in beta-VAE.”](https://arxiv.org/abs/1804.03599) NIPS 2017.
[15] Aaron van den Oord, et al. [“Neural Discrete Representation Learning”](https://arxiv.org/abs/1711.00937) NIPS 2017.
[16] Ali Razavi, et al. [“Generating Diverse High-Fidelity Images with VQ-VAE-2”](https://arxiv.org/abs/1906.00446) . arXiv preprint arXiv:1906.00446 (2019).
[17] Xi Chen, et al. [“PixelSNAIL: An Improved Autoregressive Generative Model.”](https://arxiv.org/abs/1712.09763) arXiv preprint arXiv:1712.09763 (2017).
[18] Karol Gregor, et al. [“Temporal Difference Variational Auto-Encoder.”](https://arxiv.org/abs/1806.03107) ICLR 2019.
[Autoencoder](https://lilianweng.github.io/tags/autoencoder/) [Generative-Model](https://lilianweng.github.io/tags/generative-model/) [Image-Generation](https://lilianweng.github.io/tags/image-generation/) [« Flow-based Deep Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/) [» Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) © 2025 [Lil'Log](https://lilianweng.github.io/) Powered by [Hugo](https://gohugo.io/) & [PaperMod](https://git.io/hugopapermod)