[Lil'Log](https://lilianweng.github.io/) | [Posts](https://lilianweng.github.io/) [Archive](https://lilianweng.github.io/archives) [Search](https://lilianweng.github.io/search/) [Tags](https://lilianweng.github.io/tags/) [FAQ](https://lilianweng.github.io/faq)
# Flow-based Deep Generative Models
Date: October 13, 2018 | Estimated Reading Time: 21 min | Author: Lilian Weng Table of Contents [Types of Generative Models](#types-of-generative-models) [Linear Algebra Basics Recap](#linear-algebra-basics-recap) [Jacobian Matrix and Determinant](#jacobian-matrix-and-determinant) [Change of Variable Theorem](#change-of-variable-theorem) [What is Normalizing Flows?](#what-is-normalizing-flows) [Models with Normalizing Flows](#models-with-normalizing-flows) [RealNVP](#realnvp) [NICE](#nice) [Glow](#glow) [Models with Autoregressive Flows](#models-with-autoregressive-flows) [MADE](#made) [PixelRNN](#pixelrnn) [WaveNet](#wavenet) [Masked Autoregressive Flow](#masked-autoregressive-flow) [Inverse Autoregressive Flow](#inverse-autoregressive-flow) [VAE + Flows](#vae--flows) [Reference](#reference) In this post, we are looking into the third type of generative models: flow-based generative models. Different from GAN and VAE, they explicitly learn the probability density function of the input data.
So far, I’ve written about two types of generative models, [GAN](https://lilianweng.github.io/posts/2017-08-20-gan/) and [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) . Neither of them explicitly learns the probability density function of real data, p ( x ) (where x ∈ D ) — because it is really hard! Taking the generative model with latent variables as an example, p ( x ) = ∫ p ( x | z ) p ( z ) d z can hardly be calculated as it is intractable to go through all possible values of the latent code z .
Flow-based deep generative models conquer this hard problem with the help of [normalizing flows](https://arxiv.org/abs/1505.05770) , a powerful statistics tool for density estimation. A good estimation of p ( x ) makes it possible to efficiently complete many downstream tasks: sample unobserved but realistic new data points (data generation), predict the rareness of future events (density estimation), infer latent variables, fill in incomplete data samples, etc.
# Types of Generative Models [#](#types-of-generative-models)
Here is a quick summary of the difference between GAN, VAE, and flow-based generative models:
Generative adversarial networks: GAN provides a smart solution to model the data generation, an unsupervised learning problem, as a supervised one. The discriminator model learns to distinguish the real data from the fake samples that are produced by the generator model. Two models are trained as they are playing a [minimax](https://en.wikipedia.org/wiki/Minimax) game. Variational autoencoders: VAE inexplicitly optimizes the log-likelihood of the data by maximizing the evidence lower bound (ELBO). Flow-based generative models: A flow-based generative model is constructed by a sequence of invertible transformations. Unlike other two, the model explicitly learns the data distribution p ( x ) and therefore the loss function is simply the negative log-likelihood. Comparison of three categories of generative models.
# Linear Algebra Basics Recap [#](#linear-algebra-basics-recap)
We should understand two key concepts before getting into the flow-based generative model: the Jacobian determinant and the change of variable rule. Pretty basic, so feel free to skip.
## Jacobian Matrix and Determinant [#](#jacobian-matrix-and-determinant)
Given a function of mapping a n -dimensional input vector x to a m -dimensional output vector, f : R n ↦ R m , the matrix of all first-order partial derivatives of this function is called the Jacobian matrix , J where one entry on the i-th row and j-th column is J i j = ∂ f i ∂ x j .
J = [ ∂ f 1 ∂ x 1 … ∂ f 1 ∂ x n ⋮ ⋱ ⋮ ∂ f m ∂ x 1 … ∂ f m ∂ x n ]
The determinant is one real number computed as a function of all the elements in a squared matrix. Note that the determinant only exists for square matrices . The absolute value of the determinant can be thought of as a measure of “how much multiplication by the matrix expands or contracts space”.
The determinant of a nxn matrix M is:
det M = det [ a 11 a 12 … a 1 n a 21 a 22 … a 2 n ⋮ ⋮ ⋮ a n 1 a n 2 … a n n ] = ∑ j 1 j 2 … j n ( − 1 ) τ ( j 1 j 2 … j n ) a 1 j 1 a 2 j 2 … a n j n
where the subscript under the summation j 1 j 2 … j n are all permutations of the set {1, 2, …, n}, so there are n ! items in total; τ ( . ) indicates the [signature](https://en.wikipedia.org/wiki/Parity_of_a_permutation) of a permutation.
The determinant of a square matrix M detects whether it is invertible: If det ( M ) = 0 then M is not invertible (a singular matrix with linearly dependent rows or columns; or any row or column is all 0); otherwise, if det ( M ) ≠ 0 , M is invertible.
The determinant of the product is equivalent to the product of the determinants: det ( A B ) = det ( A ) det ( B ) . ( [proof](https://proofwiki.org/wiki/Determinant_of_Matrix_Product) )
## Change of Variable Theorem [#](#change-of-variable-theorem)
Let’s review the change of variable theorem specifically in the context of probability density estimation, starting with a single variable case.
Given a random variable z and its known probability density function z ∼ π ( z ) , we would like to construct a new random variable using a 1-1 mapping function x = f ( z ) . The function f is invertible, so z = f − 1 ( x ) . Now the question is how to infer the unknown probability density function of the new variable , p ( x ) ?
∫ p ( x ) d x = ∫ π ( z ) d z = 1 ; Definition of probability distribution. p ( x ) = π ( z ) | d z d x | = π ( f − 1 ( x ) ) | d f − 1 d x | = π ( f − 1 ( x ) ) | ( f − 1 ) ′ ( x ) |
By definition, the integral ∫ π ( z ) d z is the sum of an infinite number of rectangles of infinitesimal width Δ z . The height of such a rectangle at position z is the value of the density function π ( z ) . When we substitute the variable, z = f − 1 ( x ) yields Δ z Δ x = ( f − 1 ( x ) ) ′ and Δ z = ( f − 1 ( x ) ) ′ Δ x . Here | ( f − 1 ( x ) ) ′ | indicates the ratio between the area of rectangles defined in two different coordinate of variables z and x respectively.
The multivariable version has a similar format:
z ∼ π ( z ) , x = f ( z ) , z = f − 1 ( x ) p ( x ) = π ( z ) | det d z d x | = π ( f − 1 ( x ) ) | det d f − 1 d x |
where det ∂ f ∂ z is the Jacobian determinant of the function f . The full proof of the multivariate version is out of the scope of this post; ask Google if interested ;)
# What is Normalizing Flows? [#](#what-is-normalizing-flows)
Being able to do good density estimation has direct applications in many machine learning problems, but it is very hard. For example, since we need to run backward propagation in deep learning models, the embedded probability distribution (i.e. posterior p ( z | x ) ) is expected to be simple enough to calculate the derivative easily and efficiently. That is why Gaussian distribution is often used in latent variable generative models, even though most of real world distributions are much more complicated than Gaussian.
Here comes a Normalizing Flow (NF) model for better and more powerful distribution approximation. A normalizing flow transforms a simple distribution into a complex one by applying a sequence of invertible transformation functions. Flowing through a chain of transformations, we repeatedly substitute the variable for the new one according to the change of variables theorem and eventually obtain a probability distribution of the final target variable.
Illustration of a normalizing flow model, transforming a simple distribution p _ 0 ( z _ 0 ) to a complex one p _ K ( z _ K ) step by step.
As defined in Fig. 2,
z i − 1 ∼ p i − 1 ( z i − 1 ) z i = f i ( z i − 1 ) , thus z i − 1 = f i − 1 ( z i ) p i ( z i ) = p i − 1 ( f i − 1 ( z i ) ) | det d f i − 1 d z i |
Then let’s convert the equation to be a function of z i so that we can do inference with the base distribution.
p i ( z i ) = p i − 1 ( f i − 1 ( z i ) ) | det d f i − 1 d z i | = p i − 1 ( z i − 1 ) | det ( d f i d z i − 1 ) − 1 | ; According to the inverse func theorem. = p i − 1 ( z i − 1 ) | det d f i d z i − 1 | − 1 ; According to a property of Jacobians of invertible func. log ⁡ p i ( z i ) = log ⁡ p i − 1 ( z i − 1 ) − log ⁡ | det d f i d z i − 1 |
(*) A note on the “inverse function theorem” : If y = f ( x ) and x = f − 1 ( y ) , we have:
d f − 1 ( y ) d y = d x d y = ( d y d x ) − 1 = ( d f ( x ) d x ) − 1
(*) A note on “Jacobians of invertible function” : The determinant of the inverse of an invertible matrix is the inverse of the determinant: det ( M − 1 ) = ( det ( M ) ) − 1 , [because](#jacobian-matrix-and-determinant) det ( M ) det ( M − 1 ) = det ( M ⋅ M − 1 ) = det ( I ) = 1 .
Given such a chain of probability density functions, we know the relationship between each pair of consecutive variables. We can expand the equation of the output x step by step until tracing back to the initial distribution z 0 .
x = z K = f K ∘ f K − 1 ∘ ⋯ ∘ f 1 ( z 0 ) log ⁡ p ( x ) = log ⁡ π K ( z K ) = log ⁡ π K − 1 ( z K − 1 ) − log ⁡ | det d f K d z K − 1 | = log ⁡ π K − 2 ( z K − 2 ) − log ⁡ | det d f K − 1 d z K − 2 | − log ⁡ | det d f K d z K − 1 | = … = log ⁡ π 0 ( z 0 ) − ∑ i = 1 K log ⁡ | det d f i d z i − 1 |
The path traversed by the random variables z i = f i ( z i − 1 ) is the flow and the full chain formed by the successive distributions π i is called a normalizing flow . Required by the computation in the equation, a transformation function f i should satisfy two properties:
It is easily invertible. Its Jacobian determinant is easy to compute.
# Models with Normalizing Flows [#](#models-with-normalizing-flows)
With normalizing flows in our toolbox, the exact log-likelihood of input data log ⁡ p ( x ) becomes tractable. As a result, the training criterion of flow-based generative model is simply the negative log-likelihood (NLL) over the training dataset D :
L ( D ) = − 1 | D | ∑ x ∈ D log ⁡ p ( x )
## RealNVP [#](#realnvp)
The RealNVP (Real-valued Non-Volume Preserving; [Dinh et al., 2017](https://arxiv.org/abs/1605.08803) ) model implements a normalizing flow by stacking a sequence of invertible bijective transformation functions. In each bijection f : x ↦ y , known as affine coupling layer , the input dimensions are split into two parts:
The first d dimensions stay same; The second part, d + 1 to D dimensions, undergo an affine transformation (“scale-and-shift”) and both the scale and shift parameters are functions of the first d dimensions. y 1 : d = x 1 : d y d + 1 : D = x d + 1 : D ⊙ exp ⁡ ( s ( x 1 : d ) ) + t ( x 1 : d )
where s ( . ) and t ( . ) are scale and translation functions and both map R d ↦ R D − d . The ⊙ operation is the element-wise product.
Now let’s check whether this transformation satisfy two basic properties for a flow transformation.
Condition 1 : “It is easily invertible.”
Yes and it is fairly straightforward.
{ y 1 : d = x 1 : d y d + 1 : D = x d + 1 : D ⊙ exp ⁡ ( s ( x 1 : d ) ) + t ( x 1 : d ) ⇔ { x 1 : d = y 1 : d x d + 1 : D = ( y d + 1 : D − t ( y 1 : d ) ) ⊙ exp ⁡ ( − s ( y 1 : d ) )
Condition 2 : “Its Jacobian determinant is easy to compute.”
Yes. It is not hard to get the Jacobian matrix and determinant of this transformation. The Jacobian is a lower triangular matrix.
J = [ I d 0 d × ( D − d ) ∂ y d + 1 : D ∂ x 1 : d diag ( exp ⁡ ( s ( x 1 : d ) ) ) ]
Hence the determinant is simply the product of terms on the diagonal.
det ( J ) = ∏ j = 1 D − d exp ⁡ ( s ( x 1 : d ) ) j = exp ⁡ ( ∑ j = 1 D − d s ( x 1 : d ) j )
So far, the affine coupling layer looks perfect for constructing a normalizing flow :)
Even better, since (i) computing f − 1 does not require computing the inverse of s or t and (ii) computing the Jacobian determinant does not involve computing the Jacobian of s or t , those functions can be arbitrarily complex ; i.e. both s and t can be modeled by deep neural networks.
In one affine coupling layer, some dimensions (channels) remain unchanged. To make sure all the inputs have a chance to be altered, the model reverses the ordering in each layer so that different components are left unchanged. Following such an alternating pattern, the set of units which remain identical in one transformation layer are always modified in the next. Batch normalization is found to help training models with a very deep stack of coupling layers.
Furthermore, RealNVP can work in a multi-scale architecture to build a more efficient model for large inputs. The multi-scale architecture applies several “sampling” operations to normal affine layers, including spatial checkerboard pattern masking, squeezing operation, and channel-wise masking. Read the [paper](https://arxiv.org/abs/1605.08803) for more details on the multi-scale architecture.
## NICE [#](#nice)
The NICE (Non-linear Independent Component Estimation; [Dinh, et al. 2015](https://arxiv.org/abs/1410.8516) ) model is a predecessor of [RealNVP](#realnvp) . The transformation in NICE is the affine coupling layer without the scale term, known as additive coupling layer .
{ y 1 : d = x 1 : d y d + 1 : D = x d + 1 : D + m ( x 1 : d ) ⇔ { x 1 : d = y 1 : d x d + 1 : D = y d + 1 : D − m ( y 1 : d )
## Glow [#](#glow)
The Glow ( [Kingma and Dhariwal, 2018](https://arxiv.org/abs/1807.03039) ) model extends the previous reversible generative models, NICE and RealNVP, and simplifies the architecture by replacing the reverse permutation operation on the channel ordering with invertible 1x1 convolutions.
One step of flow in the Glow model. (Image source: [Kingma and Dhariwal, 2018](https://arxiv.org/abs/1807.03039) )
There are three substeps in one step of flow in Glow.
Substep 1: Activation normalization (short for “actnorm”)
It performs an affine transformation using a scale and bias parameter per channel, similar to batch normalization, but works for mini-batch size 1. The parameters are trainable but initialized so that the first minibatch of data have mean 0 and standard deviation 1 after actnorm.
Substep 2: Invertible 1x1 conv
Between layers of the RealNVP flow, the ordering of channels is reversed so that all the data dimensions have a chance to be altered. A 1×1 convolution with equal number of input and output channels is a generalization of any permutation of the channel ordering.
Say, we have an invertible 1x1 convolution of an input h × w × c tensor h with a weight matrix W of size c × c . The output is a h × w × c tensor, labeled as f = conv2d ( h ; W ) . In order to apply the change of variable rule, we need to compute the Jacobian determinant | det ∂ f / ∂ h | .
Both the input and output of 1x1 convolution here can be viewed as a matrix of size h × w . Each entry x i j ( i = 1 , … , h , j = 1 , … , w ) in h is a vector of c channels and each entry is multiplied by the weight matrix W to obtain the corresponding entry y i j in the output matrix respectively. The derivative of each entry is ∂ x i j W / ∂ x i j = W and there are h × w such entries in total:
log ⁡ | det ∂ conv2d ( h ; W ) ∂ h | = log ⁡ ( | det W | h ⋅ w | ) = h ⋅ w ⋅ log ⁡ | det W |
The inverse 1x1 convolution depends on the inverse matrix W − 1 . Since the weight matrix is relatively small, the amount of computation for the matrix determinant ( [tf.linalg.det](https://www.tensorflow.org/api_docs/python/tf/linalg/det) ) and inversion ( [tf.linalg.inv](https://www.tensorflow.org/api_docs/python/tf/linalg/inv) ) is still under control.
Substep 3: Affine coupling layer
The design is same as in RealNVP.
Three substeps in one step of flow in Glow. (Image source: [Kingma and Dhariwal, 2018](https://arxiv.org/abs/1807.03039) )
# Models with Autoregressive Flows [#](#models-with-autoregressive-flows)
The autoregressive constraint is a way to model sequential data, x = [ x 1 , … , x D ] : each output only depends on the data observed in the past, but not on the future ones. In other words, the probability of observing x i is conditioned on x 1 , … , x i − 1 and the product of these conditional probabilities gives us the probability of observing the full sequence:
p ( x ) = ∏ i = 1 D p ( x i | x 1 , … , x i − 1 ) = ∏ i = 1 D p ( x i | x 1 : i − 1 )
How to model the conditional density is of your choice. It can be a univariate Gaussian with mean and standard deviation computed as a function of x 1 : i − 1 , or a multilayer neural network with x 1 : i − 1 as the input.
If a flow transformation in a normalizing flow is framed as an autoregressive model — each dimension in a vector variable is conditioned on the previous dimensions — this is an autoregressive flow .
This section starts with several classic autoregressive models (MADE, PixelRNN, WaveNet) and then we dive into autoregressive flow models (MAF and IAF).
## MADE [#](#made)
MADE (Masked Autoencoder for Distribution Estimation; [Germain et al., 2015](https://arxiv.org/abs/1502.03509) ) is a specially designed architecture to enforce the autoregressive property in the autoencoder efficiently . When using an autoencoder to predict the conditional probabilities, rather than feeding the autoencoder with input of different observation windows D times, MADE removes the contribution from certain hidden units by multiplying binary mask matrices so that each input dimension is reconstructed only from previous dimensions in a given ordering in a single pass .
In a multilayer fully-connected neural network, say, we have L hidden layers with weight matrices W 1 , … , W L and an output layer with weight matrix V . The output x ^ has each dimension x ^ i = p ( x i | x 1 : i − 1 ) .
Without any mask, the computation through layers looks like the following:
h 0 = x h l = activation l ( W l h l − 1 + b l ) x ^ = σ ( V h L + c ) Demonstration of how MADE works in a three-layer feed-forward neural network. (Image source: [Germain et al., 2015](https://arxiv.org/abs/1502.03509) )
To zero out some connections between layers, we can simply element-wise multiply every weight matrix by a binary mask matrix. Each hidden node is assigned with a random “connectivity integer” between 1 and D − 1 ; the assigned value for the k -th unit in the l -th layer is denoted by m k l . The binary mask matrix is determined by element-wise comparing values of two nodes in two layers.
h l = activation l ( ( W l ⊙ M W l ) h l − 1 + b l ) x ^ = σ ( ( V ⊙ M V ) h L + c ) M k ′ , k W l = 1 m k ′ l ≥ m k l − 1 = { 1 , if m k ′ l ≥ m k l − 1 0 , otherwise M d , k V = 1 d ≥ m k L = { 1 , if d > m k L 0 , otherwise
A unit in the current layer can only be connected to other units with equal or smaller numbers in the previous layer and this type of dependency easily propagates through the network up to the output layer. Once the numbers are assigned to all the units and layers, the ordering of input dimensions is fixed and the conditional probability is produced with respect to it. See a great illustration in To make sure all the hidden units are connected to the input and output layers through some paths, the m k l is sampled to be equal or greater than the minimal connectivity integer in the previous layer, min k ′ m k ′ l − 1 .
MADE training can be further facilitated by:
Order-agnostic training : shuffle the input dimensions, so that MADE is able to model any arbitrary ordering; can create an ensemble of autoregressive models at the runtime. Connectivity-agnostic training : to avoid a model being tied up to a specific connectivity pattern constraints, resample m k l for each training minibatch.
## PixelRNN [#](#pixelrnn)
PixelRNN ( [Oord et al, 2016](https://arxiv.org/abs/1601.06759) ) is a deep generative model for images. The image is generated one pixel at a time and each new pixel is sampled conditional on the pixels that have been seen before.
Let’s consider an image of size n × n , x = { x 1 , … , x n 2 } , the model starts generating pixels from the top left corner, from left to right and top to bottom (See Fig. 6).
The context for generating one pixel in PixelRNN. (Image source: [Oord et al, 2016](https://arxiv.org/abs/1601.06759) )
Every pixel x i is sampled from a probability distribution conditional over the the past context: pixels above it or on the left of it when in the same row. The definition of such context looks pretty arbitrary, because how visual [attention](https://lilianweng.github.io/posts/2018-06-24-attention/) is attended to an image is more flexible. Somehow magically a generative model with such a strong assumption works.
One implementation that could capture the entire context is the Diagonal BiLSTM . First, apply the skewing operation by offsetting each row of the input feature map by one position with respect to the previous row, so that computation for each row can be parallelized. Then the LSTM states are computed with respect to the current pixel and the pixels on the left.
(a) PixelRNN with diagonal BiLSTM. (b) Skewing operation that offsets each row in the feature map by one with regards to the row above. (Image source: [Oord et al, 2016](https://arxiv.org/abs/1601.06759) ) [ o i , f i , i i , g i ] = σ ( K s s ⊛ h i − 1 + K i s ⊛ x i ) ; σ is tanh for g, but otherwise sigmoid; ⊛ is convolution operation. c i = f i ⊙ c i − 1 + i i ⊙ g i ; ⊙ is elementwise product. h i = o i ⊙ tanh ⁡ ( c i )
where ⊛ denotes the convolution operation and ⊙ is the element-wise multiplication. The input-to-state component K i s is a 1x1 convolution, while the state-to-state recurrent component is computed with a column-wise convolution K s s with a kernel of size 2x1.
The diagonal BiLSTM layers are capable of processing an unbounded context field, but expensive to compute due to the sequential dependency between states. A faster implementation uses multiple convolutional layers without pooling to define a bounded context box. The convolution kernel is masked so that the future context is not seen, similar to [MADE](#MADE) . This convolution version is called PixelCNN .
PixelCNN with masked convolution constructed by an elementwise product of a mask tensor and the convolution kernel before applying it. (Image source: http://slazebni.cs.illinois.edu/spring17/lec13_advanced.pdf)
## WaveNet [#](#wavenet)
WaveNet ( [Van Den Oord, et al. 2016](https://arxiv.org/abs/1609.03499) ) is very similar to PixelCNN but applied to 1-D audio signals. WaveNet consists of a stack of causal convolution which is a convolution operation designed to respect the ordering: the prediction at a certain timestamp can only consume the data observed in the past, no dependency on the future. In PixelCNN, the causal convolution is implemented by masked convolution kernel. The causal convolution in WaveNet is simply to shift the output by a number of timestamps to the future so that the output is aligned with the last input element.
One big drawback of convolution layer is a very limited size of receptive field. The output can hardly depend on the input hundreds or thousands of timesteps ago, which can be a crucial requirement for modeling long sequences. WaveNet therefore adopts dilated convolution ( [animation](https://github.com/vdumoulin/conv_arithmetic#dilated-convolution-animations) ), where the kernel is applied to an evenly-distributed subset of samples in a much larger receptive field of the input.
Visualization of WaveNet models with a stack of (top) causal convolution layers and (bottom) dilated convolution layers. (Image source: [Van Den Oord, et al. 2016](https://arxiv.org/abs/1609.03499) )
WaveNet uses the gated activation unit as the non-linear layer, as it is found to work significantly better than ReLU for modeling 1-D audio data. The residual connection is applied after the gated activation.
z = tanh ⁡ ( W f , k ⊛ x ) ⊙ σ ( W g , k ⊛ x )
where W f , k and W g , k are convolution filter and gate weight matrix of the k -th layer, respectively; both are learnable.
## Masked Autoregressive Flow [#](#masked-autoregressive-flow)
Masked Autoregressive Flow ( MAF ; [Papamakarios et al., 2017](https://arxiv.org/abs/1705.07057) ) is a type of normalizing flows, where the transformation layer is built as an autoregressive neural network. MAF is very similar to Inverse Autoregressive Flow (IAF) introduced later. See more discussion on the relationship between MAF and IAF in the next section.
Given two random variables, z ∼ π ( z ) and x ∼ p ( x ) and the probability density function π ( z ) is known, MAF aims to learn p ( x ) . MAF generates each x i conditioned on the past dimensions x 1 : i − 1 .
Precisely the conditional probability is an affine transformation of z , where the scale and shift terms are functions of the observed part of x .
Data generation, producing a new x :
x i ∼ p ( x i | x 1 : i − 1 ) = z i ⊙ σ i ( x 1 : i − 1 ) + μ i ( x 1 : i − 1 ) , where z ∼ π ( z )
Density estimation, given a known x :
p ( x ) = ∏ i = 1 D p ( x i | x 1 : i − 1 )
The generation procedure is sequential, so it is slow by design. While density estimation only needs one pass the network using architecture like [MADE](#MADE) . The transformation function is trivial to inverse and the Jacobian determinant is easy to compute too.
## Inverse Autoregressive Flow [#](#inverse-autoregressive-flow)
Similar to MAF, Inverse autoregressive flow ( IAF ; [Kingma et al., 2016](https://arxiv.org/abs/1606.04934) ) models the conditional probability of the target variable as an autoregressive model too, but with a reversed flow, thus achieving a much efficient sampling process.
First, let’s reverse the affine transformation in MAF:
z i = x i − μ i ( x 1 : i − 1 ) σ i ( x 1 : i − 1 ) = − μ i ( x 1 : i − 1 ) σ i ( x 1 : i − 1 ) + x i ⊙ 1 σ i ( x 1 : i − 1 )
If let:
x ~ = z , p ~ ( . ) = π ( . ) , x ~ ∼ p ~ ( x ~ ) z ~ = x , π ~ ( . ) = p ( . ) , z ~ ∼ π ~ ( z ~ ) μ ~ i ( z ~ 1 : i − 1 ) = μ ~ i ( x 1 : i − 1 ) = − μ i ( x 1 : i − 1 ) σ i ( x 1 : i − 1 ) σ ~ ( z ~ 1 : i − 1 ) = σ ~ ( x 1 : i − 1 ) = 1 σ i ( x 1 : i − 1 )
Then we would have,
x ~ i ∼ p ( x ~ i | z ~ 1 : i ) = z ~ i ⊙ σ ~ i ( z ~ 1 : i − 1 ) + μ ~ i ( z ~ 1 : i − 1 ) , where z ~ ∼ π ~ ( z ~ )
IAF intends to estimate the probability density function of x ~ given that π ~ ( z ~ ) is already known. The inverse flow is an autoregressive affine transformation too, same as in MAF, but the scale and shift terms are autoregressive functions of observed variables from the known distribution π ~ ( z ~ ) . See the comparison between MAF and IAF in
Comparison of MAF and IAF. The variable with known density is in green while the unknown one is in red.
Computations of the individual elements x ~ i do not depend on each other, so they are easily parallelizable (only one pass using MADE). The density estimation for a known x ~ is not efficient, because we have to recover the value of z ~ i in a sequential order, z ~ i = ( x ~ i − μ ~ i ( z ~ 1 : i − 1 ) ) / σ ~ i ( z ~ 1 : i − 1 ) , thus D times in total.
| | Base distribution | Target distribution | Model | Data generation | Density estimation |
|---|---|---|---|---|---|
| MAF | z ∼ π ( z ) | x ∼ p ( x ) | x i = z i ⊙ σ i ( x 1 : i − 1 ) + μ i ( x 1 : i − 1 ) | Sequential; slow | One pass; fast |
| IAF | z ~ ∼ π ~ ( z ~ ) | x ~ ∼ p ~ ( x ~ ) | x ~ i = z ~ i ⊙ σ ~ i ( z ~ 1 : i − 1 ) + μ ~ i ( z ~ 1 : i − 1 ) | One pass; fast | Sequential; slow |
| ———- | ———- | ———- | ———- | ———- | ———- |
# VAE + Flows [#](#vae--flows)
In [Variational Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder) , if we want to model the posterior p ( z | x ) as a more complicated distribution rather than simple Gaussian. Intuitively we can use normalizing flow to transform the base Gaussian for better density approximation. The encoder then would predict a set of scale and shift terms ( μ i , σ i ) which are all functions of input x . Read the [paper](https://arxiv.org/abs/1809.05861) for more details if interested.
If you notice mistakes and errors in this post, don’t hesitate to contact me at [lilian dot wengweng at gmail dot com] and I would be very happy to correct them right away!
See you in the next post :D
Cited as:
@article{weng2018flow,
title = "Flow-based Deep Generative Models" ,
author = "Weng, Lilian" ,
journal = "lilianweng.github.io" ,
year = "2018" ,
url = "https://lilianweng.github.io/posts/2018-10-13-flow-models/" } copy
# Reference [#](#reference)
[1] Danilo Jimenez Rezende, and Shakir Mohamed. [“Variational inference with normalizing flows.”](https://arxiv.org/abs/1505.05770) ICML 2015.
[2] [Normalizing Flows Tutorial, Part 1: Distributions and Determinants](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang.
[3] [Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows](https://blog.evjang.com/2018/01/nf2.html) by Eric Jang.
[4] [Normalizing Flows](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html) by Adam Kosiorek.
[5] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. [“Density estimation using Real NVP.”](https://arxiv.org/abs/1605.08803) ICLR 2017.
[6] Laurent Dinh, David Krueger, and Yoshua Bengio. [“NICE: Non-linear independent components estimation.”](https://arxiv.org/abs/1410.8516) ICLR 2015 Workshop track.
[7] Diederik P. Kingma, and Prafulla Dhariwal. [“Glow: Generative flow with invertible 1x1 convolutions.”](https://arxiv.org/abs/1807.03039) arXiv:1807.03039 (2018).
[8] Germain, Mathieu, Karol Gregor, Iain Murray, and Hugo Larochelle. [“Made: Masked autoencoder for distribution estimation.”](https://arxiv.org/abs/1502.03509) ICML 2015.
[9] Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. [“Pixel recurrent neural networks.”](https://arxiv.org/abs/1601.06759) ICML 2016.
[10] Diederik P. Kingma, et al. [“Improved variational inference with inverse autoregressive flow.”](https://arxiv.org/abs/1606.04934) NIPS. 2016.
[11] George Papamakarios, Iain Murray, and Theo Pavlakou. [“Masked autoregressive flow for density estimation.”](https://arxiv.org/abs/1705.07057) NIPS 2017.
[12] Jianlin Su, and Guang Wu. [“f-VAEs: Improve VAEs with Conditional Flows.”](https://arxiv.org/abs/1809.05861) arXiv:1809.05861 (2018).
[13] Van Den Oord, Aaron, et al. [“WaveNet: A generative model for raw audio.”](https://arxiv.org/abs/1609.03499) SSW. 2016.
[Architecture](https://lilianweng.github.io/tags/architecture/) [Generative-Model](https://lilianweng.github.io/tags/generative-model/) [Image-Generation](https://lilianweng.github.io/tags/image-generation/) [Math-Heavy](https://lilianweng.github.io/tags/math-heavy/) [« Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/posts/2018-11-30-meta-learning/) [» From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) © 2025 [Lil'Log](https://lilianweng.github.io/) Powered by [Hugo](https://gohugo.io/) & [PaperMod](https://git.io/hugopapermod)