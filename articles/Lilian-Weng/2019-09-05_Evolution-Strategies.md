[Lil'Log](https://lilianweng.github.io/) | [Posts](https://lilianweng.github.io/) [Archive](https://lilianweng.github.io/archives) [Search](https://lilianweng.github.io/search/) [Tags](https://lilianweng.github.io/tags/) [FAQ](https://lilianweng.github.io/faq)
# Evolution Strategies
Date: September 5, 2019 | Estimated Reading Time: 22 min | Author: Lilian Weng Table of Contents [What are Evolution Strategies?](#what-are-evolution-strategies) [Simple Gaussian Evolution Strategies](#simple-gaussian-evolution-strategies) [Covariance Matrix Adaptation Evolution Strategies (CMA-ES)](#covariance-matrix-adaptation-evolution-strategies-cma-es) [Updating the Mean](#updating-the-mean) [Controlling the Step Size](#controlling-the-step-size) [Adapting the Covariance Matrix](#adapting-the-covariance-matrix) [Natural Evolution Strategies](#natural-evolution-strategies) [Natural Gradients](#natural-gradients) [Estimation using Fisher Information Matrix](#estimation-using-fisher-information-matrix) [NES Algorithm](#nes-algorithm) [Applications: ES in Deep Reinforcement Learning](#applications-es-in-deep-reinforcement-learning) [OpenAI ES for RL](#openai-es-for-rl) [Exploration with ES](#exploration-with-es) [CEM-RL](#cem-rl) [Extension: EA in Deep Learning](#extension-ea-in-deep-learning) [Hyperparameter Tuning: PBT](#hyperparameter-tuning-pbt) [Network Topology Optimization: WANN](#network-topology-optimization-wann) [References](#references) Gradient descent is not the only option when learning optimal model parameters. Evolution Strategies (ES) works out well in the cases where we don't know the precise analytic form of an objective function or cannot compute the gradients directly. This post dives into several classic ES methods, as well as how ES can be used in deep reinforcement learning.
Stochastic gradient descent is a universal choice for optimizing deep learning models. However, it is not the only option. With black-box optimization algorithms, you can evaluate a target function f ( x ) : R n → R , even when you don’t know the precise analytic form of f ( x ) and thus cannot compute gradients or the Hessian matrix. Examples of black-box optimization methods include [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) , [Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing) and [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) .
Evolution Strategies (ES) is one type of black-box optimization algorithms, born in the family of Evolutionary Algorithms (EA) . In this post, I would dive into a couple of classic ES methods and introduce a few applications of how ES can play a role in deep reinforcement learning.
# What are Evolution Strategies? [#](#what-are-evolution-strategies)
Evolution strategies (ES) belong to the big family of evolutionary algorithms. The optimization targets of ES are vectors of real numbers, x ∈ R n .
Evolutionary algorithms refer to a division of population-based optimization algorithms inspired by natural selection . Natural selection believes that individuals with traits beneficial to their survival can live through generations and pass down the good characteristics to the next generation. Evolution happens by the selection process gradually and the population grows better adapted to the environment.
How natural selection works. (Image source: Khan Academy: [Darwin, evolution, & natural selection](https://www.khanacademy.org/science/biology/her/evolution-and-natural-selection/a/darwin-evolution-natural-selection) )
Evolutionary algorithms can be summarized in the following [format](https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/06-blackBoxOpt.pdf) as a general optimization solution:
Let’s say we want to optimize a function f ( x ) and we are not able to compute gradients directly. But we still can evaluate f ( x ) given any x and the result is deterministic. Our belief in the probability distribution over x as a good solution to f ( x ) optimization is p θ ( x ) , parameterized by θ . The goal is to find an optimal configuration of θ .
Here given a fixed format of distribution (i.e. Gaussian), the parameter θ carries the knowledge about the best solutions and is being iteratively updated across generations.
Starting with an initial value of θ , we can continuously update θ by looping three steps as follows:
Generate a population of samples D = { ( x i , f ( x i ) } where x i ∼ p θ ( x ) . Evaluate the “fitness” of samples in D . Select the best subset of individuals and use them to update θ , generally based on fitness or rank.
In Genetic Algorithms (GA) , another popular subcategory of EA, x is a sequence of binary codes, x ∈ { 0 , 1 } n . While in ES, x is just a vector of real numbers, x ∈ R n .
# Simple Gaussian Evolution Strategies [#](#simple-gaussian-evolution-strategies)
[This](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) is the most basic and canonical version of evolution strategies. It models p θ ( x ) as a n -dimensional isotropic Gaussian distribution, in which θ only tracks the mean μ and standard deviation σ .
θ = ( μ , σ ) , p θ ( x ) ∼ N ( μ , σ 2 I ) = μ + σ N ( 0 , I )
The process of Simple-Gaussian-ES, given x ∈ R n :
Initialize θ = θ ( 0 ) and the generation counter t = 0 Generate the offspring population of size Λ by sampling from the Gaussian distribution: D ( t + 1 ) = { x i ( t + 1 ) ∣ x i ( t + 1 ) = μ ( t ) + σ ( t ) y i ( t + 1 ) where y i ( t + 1 ) ∼ N ( x | 0 , I ) , ; i = 1 , … , Λ } . Select a top subset of λ samples with optimal f ( x i ) and this subset is called elite set. Without loss of generality, we may consider the first k samples in D ( t + 1 ) to belong to the elite group — Let’s label them as D ( t + 1 ) _ elite = x ( t + 1 ) _ i ∣ x ( t + 1 ) _ i ∈ D ( t + 1 ) , i = 1 , … , λ , λ ≤ Λ Then we estimate the new mean and std for the next generation using the elite set: μ ( t + 1 ) = avg ( D elite ( t + 1 ) ) = 1 λ ∑ i = 1 λ x i ( t + 1 ) σ ( t + 1 ) 2 = var ( D elite ( t + 1 ) ) = 1 λ ∑ i = 1 λ ( x i ( t + 1 ) − μ ( t ) ) 2 Repeat steps (2)-(4) until the result is good enough ✌️
# Covariance Matrix Adaptation Evolution Strategies (CMA-ES) [#](#covariance-matrix-adaptation-evolution-strategies-cma-es)
The standard deviation σ accounts for the level of exploration: the larger σ the bigger search space we can sample our offspring population. In [vanilla ES](#simple-gaussian-evolution-strategies) , σ ( t + 1 ) is highly correlated with σ ( t ) , so the algorithm is not able to rapidly adjust the exploration space when needed (i.e. when the confidence level changes).
[CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) , short for “Covariance Matrix Adaptation Evolution Strategy” , fixes the problem by tracking pairwise dependencies between the samples in the distribution with a covariance matrix C . The new distribution parameter becomes:
θ = ( μ , σ , C ) , p θ ( x ) ∼ N ( μ , σ 2 C ) ∼ μ + σ N ( 0 , C )
where σ controls for the overall scale of the distribution, often known as step size .
Before we dig into how the parameters are updated in CMA-ES, it is better to review how the covariance matrix works in the multivariate Gaussian distribution first. As a real symmetric matrix, the covariance matrix C has the following nice features (See [proof](http://s3.amazonaws.com/mitsloan-php/wp-faculty/sites/30/2016/12/15032137/Symmetric-Matrices-and-Eigendecomposition.pdf) & [proof](http://control.ucsd.edu/mauricio/courses/mae280a/lecture11.pdf) ):
It is always diagonalizable. Always positive semi-definite. All of its eigenvalues are real non-negative numbers. All of its eigenvectors are orthogonal. There is an orthonormal basis of R n consisting of its eigenvectors.
Let the matrix C have an orthonormal basis of eigenvectors B = [ b 1 , … , b n ] , with corresponding eigenvalues λ 1 2 , … , λ n 2 . Let D = diag ( λ 1 , … , λ n ) .
C = B ⊤ D 2 B = [ ∣ ∣ ∣ b 1 b 2 … b n ∣ ∣ ∣ ] [ λ 1 2 0 … 0 0 λ 2 2 … 0 ⋮ … ⋱ ⋮ 0 … 0 λ n 2 ] [ − b 1 − − b 2 − … − b n − ]
The square root of C is:
C 1 2 = B ⊤ D B
| Symbol | Meaning |
|---|---|
| x i ( t ) ∈ R n | the i -th samples at the generation (t) |
| y i ( t ) ∈ R n | x i ( t ) = μ ( t − 1 ) + σ ( t − 1 ) y i ( t ) |
| μ ( t ) | mean of the generation (t) |
| σ ( t ) | step size |
| C ( t ) | covariance matrix |
| B ( t ) | a matrix of C ’s eigenvectors as row vectors |
| D ( t ) | a diagonal matrix with C ’s eigenvalues on the diagnose. |
| p σ ( t ) | evaluation path for σ at the generation (t) |
| p c ( t ) | evaluation path for C at the generation (t) |
| α μ | learning rate for μ ’s update |
| α σ | learning rate for p σ |
| d σ | damping factor for σ ’s update |
| α c p | learning rate for p c |
| α c λ | learning rate for C ’s rank-min(λ, n) update |
| α c 1 | learning rate for C ’s rank-1 update |
## Updating the Mean [#](#updating-the-mean)
μ ( t + 1 ) = μ ( t ) + α μ 1 λ ∑ i = 1 λ ( x i ( t + 1 ) − μ ( t ) )
CMA-ES has a learning rate α μ ≤ 1 to control how fast the mean μ should be updated. Usually it is set to 1 and thus the equation becomes the same as in vanilla ES, μ ( t + 1 ) = 1 λ ∑ i = 1 λ ( x i ( t + 1 ) .
## Controlling the Step Size [#](#controlling-the-step-size)
The sampling process can be decoupled from the mean and standard deviation:
x i ( t + 1 ) = μ ( t ) + σ ( t ) y i ( t + 1 ) , where y i ( t + 1 ) = x i ( t + 1 ) − μ ( t ) σ ( t ) ∼ N ( 0 , C )
The parameter σ controls the overall scale of the distribution. It is separated from the covariance matrix so that we can change steps faster than the full covariance. A larger step size leads to faster parameter update. In order to evaluate whether the current step size is proper, CMA-ES constructs an evolution path p σ by summing up a consecutive sequence of moving steps, 1 λ ∑ i λ y i ( j ) , j = 1 , … , t . By comparing this path length with its expected length under random selection (meaning single steps are uncorrelated), we are able to adjust σ accordingly (See Fig. 2).
Three scenarios of how single steps are correlated in different ways and their impacts on step size update. (Image source: additional annotations on Fig 5 in [CMA-ES tutorial](https://arxiv.org/abs/1604.00772) paper)
Each time the evolution path is updated with the average of moving step y i in the same generation.
1 λ ∑ i = 1 λ y i ( t + 1 ) = 1 λ ∑ i = 1 λ x i ( t + 1 ) − λ μ ( t ) σ ( t ) = μ ( t + 1 ) − μ ( t ) σ ( t ) 1 λ ∑ i = 1 λ y i ( t + 1 ) ∼ 1 λ N ( 0 , λ C ( t ) ) ∼ 1 λ C ( t ) 1 2 N ( 0 , I ) Thus λ C ( t ) − 1 2 μ ( t + 1 ) − μ ( t ) σ ( t ) ∼ N ( 0 , I )
By multiplying with C − 1 2 , the evolution path is transformed to be independent of its direction. The term C ( t ) − 1 2 = B ( t ) ⊤ D ( t ) − 1 2 B ( t ) transformation works as follows:
B ( t ) contains row vectors of C ’s eigenvectors. It projects the original space onto the perpendicular principal axes. Then D ( t ) − 1 2 = diag ( 1 λ 1 , … , 1 λ n ) scales the length of principal axes to be equal. B ( t ) ⊤ transforms the space back to the original coordinate system.
In order to assign higher weights to recent generations, we use polyak averaging to update the evolution path with learning rate α σ . Meanwhile, the weights are balanced so that p σ is [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior) , ∼ N ( 0 , I ) both before and after one update.
p σ ( t + 1 ) = ( 1 − α σ ) p σ ( t ) + 1 − ( 1 − α σ ) 2 λ C ( t ) − 1 2 μ ( t + 1 ) − μ ( t ) σ ( t ) = ( 1 − α σ ) p σ ( t ) + c σ ( 2 − α σ ) λ C ( t ) − 1 2 μ ( t + 1 ) − μ ( t ) σ ( t )
The expected length of p σ under random selection is E | N ( 0 , I ) | , that is the expectation of the L2-norm of a N ( 0 , I ) random variable. Following the idea in Fig. 2, we adjust the step size according to the ratio of | p σ ( t + 1 ) | / E | N ( 0 , I ) | :
ln ⁡ σ ( t + 1 ) = ln ⁡ σ ( t ) + α σ d σ ( ‖ p σ ( t + 1 ) ‖ E ‖ N ( 0 , I ) ‖ − 1 ) σ ( t + 1 ) = σ ( t ) exp ⁡ ( α σ d σ ( ‖ p σ ( t + 1 ) ‖ E ‖ N ( 0 , I ) ‖ − 1 ) )
where d σ ≈ 1 is a damping parameter, scaling how fast ln ⁡ σ should be changed.
## Adapting the Covariance Matrix [#](#adapting-the-covariance-matrix)
For the covariance matrix, it can be estimated from scratch using y i of elite samples (recall that y i ∼ N ( 0 , C ) ):
C λ ( t + 1 ) = 1 λ ∑ i = 1 λ y i ( t + 1 ) y i ( t + 1 ) ⊤ = 1 λ σ ( t ) 2 ∑ i = 1 λ ( x i ( t + 1 ) − μ ( t ) ) ( x i ( t + 1 ) − μ ( t ) ) ⊤
The above estimation is only reliable when the selected population is large enough. However, we do want to run fast iteration with a small population of samples in each generation. That’s why CMA-ES invented a more reliable but also more complicated way to update C . It involves two independent routes,
Rank-min(λ, n) update : uses the history of { C λ } , each estimated from scratch in one generation. Rank-one update : estimates the moving steps y i and the sign information from the history.
The first route considers the estimation of C from the entire history of { C λ } . For example, if we have experienced a large number of generations, C ( t + 1 ) ≈ avg ( C λ ( i ) ; i = 1 , … , t ) would be a good estimator. Similar to p σ , we also use polyak averaging with a learning rate to incorporate the history:
C ( t + 1 ) = ( 1 − α c λ ) C ( t ) + α c λ C λ ( t + 1 ) = ( 1 − α c λ ) C ( t ) + α c λ 1 λ ∑ i = 1 λ y i ( t + 1 ) y i ( t + 1 ) ⊤
A common choice for the learning rate is α c λ ≈ min ( 1 , λ / n 2 ) .
The second route tries to solve the issue that y i y i ⊤ = ( − y i ) ( − y i ) ⊤ loses the sign information. Similar to how we adjust the step size σ , an evolution path p c is used to track the sign information and it is constructed in a way that p c is conjugate, ∼ N ( 0 , C ) both before and after a new generation.
We may consider p c as another way to compute avg i ( y i ) (notice that both ∼ N ( 0 , C ) ) while the entire history is used and the sign information is maintained. Note that we’ve known k μ ( t + 1 ) − μ ( t ) σ ( t ) ∼ N ( 0 , C ) in the [last section](#controlling-the-step-size) ,
p c ( t + 1 ) = ( 1 − α c p ) p c ( t ) + 1 − ( 1 − α c p ) 2 λ μ ( t + 1 ) − μ ( t ) σ ( t ) = ( 1 − α c p ) p c ( t ) + α c p ( 2 − α c p ) λ μ ( t + 1 ) − μ ( t ) σ ( t )
Then the covariance matrix is updated according to p c :
C ( t + 1 ) = ( 1 − α c 1 ) C ( t ) + α c 1 p c ( t + 1 ) p c ( t + 1 ) ⊤
The rank-one update approach is claimed to generate a significant improvement over the rank-min(λ, n)-update when k is small, because the signs of moving steps and correlations between consecutive steps are all utilized and passed down through generations.
Eventually we combine two approaches together,
C ( t + 1 ) = ( 1 − α c λ − α c 1 ) C ( t ) + α c 1 p c ( t + 1 ) p c ( t + 1 ) ⊤ ⏟ rank-one update + α c λ 1 λ ∑ i = 1 λ y i ( t + 1 ) y i ( t + 1 ) ⊤ ⏟ rank-min(lambda, n) update
In all my examples above, each elite sample is considered to contribute an equal amount of weights, 1 / λ . The process can be easily extended to the case where selected samples are assigned with different weights, w 1 , … , w λ , according to their performances. See more detail in [tutorial](https://arxiv.org/abs/1604.00772) .
Illustration of how CMA-ES works on a 2D optimization problem (the lighter color the better). Black dots are samples in one generation. The samples are more spread out initially but when the model has higher confidence in finding a good solution in the late stage, the samples become very concentrated over the global optimum. (Image source: [Wikipedia CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) )
# Natural Evolution Strategies [#](#natural-evolution-strategies)
Natural Evolution Strategies ( NES ; [Wierstra, et al, 2008](https://arxiv.org/abs/1106.4487) ) optimizes in a search distribution of parameters and moves the distribution in the direction of high fitness indicated by the natural gradient .
## Natural Gradients [#](#natural-gradients)
Given an objective function J ( θ ) parameterized by θ , let’s say our goal is to find the optimal θ to maximize the objective function value. A plain gradient finds the steepest direction within a small Euclidean distance from the current θ ; the distance restriction is applied on the parameter space. In other words, we compute the plain gradient with respect to a small change of the absolute value of θ . The optimal step is:
d ∗ = argmax ‖ d ‖ = ϵ J ( θ + d ) , where ϵ → 0
Differently, natural gradient works with a probability [distribution](https://arxiv.org/abs/1301.3584v7) [space](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) parameterized by θ , p θ ( x ) (referred to as “search distribution” in NES [paper](https://arxiv.org/abs/1106.4487) ). It looks for the steepest direction within a small step in the distribution space where the distance is measured by KL divergence. With this constraint we ensure that each update is moving along the distributional manifold with constant speed, without being slowed down by its curvature.
d N ∗ = argmax KL [ p θ ‖ p θ + d ] = ϵ J ( θ + d )
## Estimation using Fisher Information Matrix [#](#estimation-using-fisher-information-matrix)
But, how to compute KL [ p θ | p θ + Δ θ ] precisely? By running Taylor expansion of log ⁡ p θ + d at θ , we get:
KL [ p θ ‖ p θ + d ] = E x ∼ p θ [ log ⁡ p θ ( x ) − log ⁡ p θ + d ( x ) ] ≈ E x ∼ p θ [ log ⁡ p θ ( x ) − ( log ⁡ p θ ( x ) + ∇ θ log ⁡ p θ ( x ) d + 1 2 d ⊤ ∇ θ 2 log ⁡ p θ ( x ) d ) ] ; Taylor expand log ⁡ p θ + d ≈ − E x [ ∇ θ log ⁡ p θ ( x ) ] d − 1 2 d ⊤ E x [ ∇ θ 2 log ⁡ p θ ( x ) ] d
where
E x [ ∇ θ log ⁡ p θ ] d = ∫ x ∼ p θ p θ ( x ) ∇ θ log ⁡ p θ ( x ) = ∫ x ∼ p θ p θ ( x ) 1 p θ ( x ) ∇ θ p θ ( x ) = ∇ θ ( ∫ x p θ ( x ) ) ; note that p θ ( x ) is probability distribution. = ∇ θ ( 1 ) = 0
Finally we have,
KL [ p θ ‖ p θ + d ] = − 1 2 d ⊤ F θ d , where F θ = E x [ ( ∇ θ log ⁡ p θ ) ( ∇ θ log ⁡ p θ ) ⊤ ]
where F θ is called the [Fisher Information Matrix](http://mathworld.wolfram.com/FisherInformationMatrix.html) and [it is](https://wiseodd.github.io/techblog/2018/03/11/fisher-information/) the covariance matrix of ∇ θ log ⁡ p θ since E [ ∇ θ log ⁡ p θ ] = 0 .
The solution to the following optimization problem:
max J ( θ + d ) ≈ max ( J ( θ ) + ∇ θ J ( θ ) ⊤ d ) s.t. KL [ p θ ‖ p θ + d ] − ϵ = 0
can be found using a Lagrangian multiplier,
L ( θ , d , β ) = J ( θ ) + ∇ θ J ( θ ) ⊤ d − β ( 1 2 d ⊤ F θ d + ϵ ) = 0 s.t. β > 0 ∇ d L ( θ , d , β ) = ∇ θ J ( θ ) − β F θ d = 0 Thus d N ∗ = ∇ θ N J ( θ ) = F θ − 1 ∇ θ J ( θ )
where d N ∗ only extracts the direction of the optimal moving step on θ , ignoring the scalar β − 1 .
The natural gradient samples (black solid arrows) in the right are the plain gradient samples (black solid arrows) in the left multiplied by the inverse of their covariance. In this way, a gradient direction with high uncertainty (indicated by high covariance with other samples) are penalized with a small weight. The aggregated natural gradient (red dash arrow) is therefore more trustworthy than the natural gradient (green solid arrow). (Image source: additional annotations on Fig 2 in [NES](https://arxiv.org/abs/1106.4487) paper)
## NES Algorithm [#](#nes-algorithm)
The fitness associated with one sample is labeled as f ( x ) and the search distribution over x is parameterized by θ . NES is expected to optimize the parameter θ to achieve maximum expected fitness:
J ( θ ) = E x ∼ p θ ( x ) [ f ( x ) ] = ∫ x f ( x ) p θ ( x ) d x
Using the same log-likelihood [trick](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/) in [REINFORCE](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#reinforce) :
∇ θ J ( θ ) = ∇ θ ∫ x f ( x ) p θ ( x ) d x = ∫ x f ( x ) p θ ( x ) p θ ( x ) ∇ θ p θ ( x ) d x = ∫ x f ( x ) p θ ( x ) ∇ θ log ⁡ p θ ( x ) d x = E x ∼ p θ [ f ( x ) ∇ θ log ⁡ p θ ( x ) ]
Besides natural gradients, NES adopts a couple of important heuristics to make the algorithm performance more robust.
NES applies rank-based fitness shaping , that is to use the rank under monotonically increasing fitness values instead of using f ( x ) directly. Or it can be a function of the rank (“utility function”), which is considered as a free parameter of NES. NES adopts adaptation sampling to adjust hyperparameters at run time. When changing θ → θ ′ , samples drawn from p θ are compared with samples from p θ ′ using [Mann-Whitney U-test(https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)]; if there shows a positive or negative sign, the target hyperparameter decreases or increases by a multiplication constant. Note the score of a sample x i ′ ∼ p θ ′ ( x ) has importance sampling weights applied w i ′ = p θ ( x ) / p θ ′ ( x ) .
# Applications: ES in Deep Reinforcement Learning [#](#applications-es-in-deep-reinforcement-learning)
## OpenAI ES for RL [#](#openai-es-for-rl)
The concept of using evolutionary algorithms in reinforcement learning can be traced back [long ago](https://arxiv.org/abs/1106.0221) , but only constrained to tabular RL due to computational limitations.
Inspired by [NES](#natural-evolution-strategies) , researchers at OpenAI ( [Salimans, et al. 2017](https://arxiv.org/abs/1703.03864) ) proposed to use NES as a gradient-free black-box optimizer to find optimal policy parameters θ that maximizes the return function F ( θ ) . The key is to add Gaussian noise ϵ on the model parameter θ and then use the log-likelihood trick to write it as the gradient of the Gaussian pdf. Eventually only the noise term is left as a weighting scalar for measured performance.
Let’s say the current parameter value is θ ^ (the added hat is to distinguish the value from the random variable θ ). The search distribution of θ is designed to be an isotropic multivariate Gaussian with a mean θ ^ and a fixed covariance matrix σ 2 I ,
θ ∼ N ( θ ^ , σ 2 I ) equivalent to θ = θ ^ + σ ϵ , ϵ ∼ N ( 0 , I )
The gradient for θ update is:
∇ θ E θ ∼ N ( θ ^ , σ 2 I ) F ( θ ) = ∇ θ E ϵ ∼ N ( 0 , I ) F ( θ ^ + σ ϵ ) = ∇ θ ∫ ϵ p ( ϵ ) F ( θ ^ + σ ϵ ) d ϵ ; Gaussian p ( ϵ ) = ( 2 π ) − n 2 exp ⁡ ( − 1 2 ϵ ⊤ ϵ ) = ∫ ϵ p ( ϵ ) ∇ ϵ log ⁡ p ( ϵ ) ∇ θ ϵ F ( θ ^ + σ ϵ ) d ϵ ; log-likelihood trick = E ϵ ∼ N ( 0 , I ) [ ∇ ϵ ( − 1 2 ϵ ⊤ ϵ ) ∇ θ ( θ − θ ^ σ ) F ( θ ^ + σ ϵ ) ] = E ϵ ∼ N ( 0 , I ) [ ( − ϵ ) ( 1 σ ) F ( θ ^ + σ ϵ ) ] = 1 σ E ϵ ∼ N ( 0 , I ) [ ϵ F ( θ ^ + σ ϵ ) ] ; negative sign can be absorbed.
In one generation, we can sample many e p s i l o n i , i = 1 , … , n and evaluate the fitness in parallel . One beautiful design is that no large model parameter needs to be shared. By only communicating the random seeds between workers, it is enough for the master node to do parameter update. This approach is later extended to adaptively learn a loss function; see my previous post on [Evolved Policy Gradient](https://lilianweng.github.io/posts/2019-06-23-meta-rl/#meta-learning-the-loss-function) .
The algorithm for training a RL policy using evolution strategies. (Image source: [ES-for-RL](https://arxiv.org/abs/1703.03864) paper)
To make the performance more robust, OpenAI ES adopts virtual batch normalization (BN with mini-batch used for calculating statistics fixed), mirror sampling (sampling a pair of ( − ϵ , ϵ ) for evaluation), and [fitness shaping](#fitness-shaping) .
## Exploration with ES [#](#exploration-with-es)
Exploration ( [vs exploitation](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/#exploitation-vs-exploration) ) is an important topic in RL. The optimization direction in the ES algorithm [above](TBA) is only extracted from the cumulative return F ( θ ) . Without explicit exploration, the agent might get trapped in a local optimum.
Novelty-Search ES ( NS-ES ; [Conti et al, 2018](https://arxiv.org/abs/1712.06560) ) encourages exploration by updating the parameter in the direction to maximize the novelty score. The novelty score depends on a domain-specific behavior characterization function b ( π θ ) . The choice of b ( π θ ) is specific to the task and seems to be a bit arbitrary; for example, in the Humanoid locomotion task in the paper, b ( π θ ) is the final ( x , y ) location of the agent.
Every policy’s b ( π θ ) is pushed to an archive set A . Novelty of a policy π θ is measured as the k-nearest neighbor score between b ( π θ ) and all other entries in A .
(The use case of the archive set sounds quite similar to [episodic memory](https://lilianweng.github.io/posts/2019-06-23-meta-rl/#episodic-control) .) N ( θ , A ) = 1 λ ∑ i = 1 λ ‖ b ( π θ ) , b i knn ‖ 2 , where b i knn ∈ kNN ( b ( π θ ) , A )
The ES optimization step relies on the novelty score instead of fitness:
∇ θ E θ ∼ N ( θ ^ , σ 2 I ) N ( θ , A ) = 1 σ E ϵ ∼ N ( 0 , I ) [ ϵ N ( θ ^ + σ ϵ , A ) ]
NS-ES maintains a group of M independently trained agents (“meta-population”), M = { θ 1 , … , θ M } and picks one to advance proportional to the novelty score. Eventually we select the best policy. This process is equivalent to ensembling; also see the same idea in [SVPG](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#svpg) .
m ← pick i = 1 , … , M according to probability N ( θ i , A ) ∑ j = 1 M N ( θ j , A ) θ m ( t + 1 ) ← θ m ( t ) + α 1 σ ∑ i = 1 N ϵ i N ( θ m ( t ) + ϵ i , A ) where ϵ i ∼ N ( 0 , I )
where N is the number of Gaussian perturbation noise vectors and α is the learning rate.
NS-ES completely discards the reward function and only optimizes for novelty to avoid deceptive local optima. To incorporate the fitness back into the formula, another two variations are proposed.
NSR-ES :
θ m ( t + 1 ) ← θ m ( t ) + α 1 σ ∑ i = 1 N ϵ i N ( θ m ( t ) + ϵ i , A ) + F ( θ m ( t ) + ϵ i ) 2
NSRAdapt-ES (NSRA-ES) : the adaptive weighting parameter w = 1.0 initially. We start decreasing w if performance stays flat for a number of generations. Then when the performance starts to increase, we stop decreasing w but increase it instead. In this way, fitness is preferred when the performance stops growing but novelty is preferred otherwise.
θ m ( t + 1 ) ← θ m ( t ) + α 1 σ ∑ i = 1 N ϵ i ( ( 1 − w ) N ( θ m ( t ) + ϵ i , A ) + w F ( θ m ( t ) + ϵ i ) ) (Left) The environment is Humanoid locomotion with a three-sided wall which plays a role as a deceptive trap to create local optimum. (Right) Experiments compare ES baseline and other variations that encourage exploration. (Image source: [NS-ES](https://arxiv.org/abs/1712.06560) paper)
## CEM-RL [#](#cem-rl)
Architectures of the (a) CEM-RL and (b) [ERL](https://papers.nips.cc/paper/7395-evolution-guided-policy-gradient-in-reinforcement-learning.pdf) algorithms (Image source: [CEM-RL](https://arxiv.org/abs/1810.01222) paper)
The CEM-RL method ( [Pourchot & Sigaud, 2019](https://arxiv.org/abs/1810.01222) ) combines Cross Entropy Method (CEM) with either [DDPG](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#ddpg) or [TD3](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#td3) . CEM here works pretty much the same as the simple Gaussian ES described [above](#simple-gaussian-evolution-strategies) and therefore the same function can be replaced using CMA-ES. CEM-RL is built on the framework of Evolutionary Reinforcement Learning ( ERL ; [Khadka & Tumer, 2018](https://papers.nips.cc/paper/7395-evolution-guided-policy-gradient-in-reinforcement-learning.pdf) ) in which the standard EA algorithm selects and evolves a population of actors and the rollout experience generated in the process is then added into reply buffer for training both RL-actor and RL-critic networks.
Workflow:
The mean actor of the CEM population is π μ is initialized with a random actor network. The critic network Q is initialized too, which will be updated by DDPG/TD3. Repeat until happy: a. Sample a population of actors ∼ N ( π μ , Σ ) . b. Half of the population is evaluated. Their fitness scores are used as the cumulative reward R and added into replay buffer. c. The other half are updated together with the critic. d. The new π m u and Σ is computed using top performing elite samples. [CMA-ES](#covariance-matrix-adaptation-evolution-strategies-cma-es) can be used for parameter update too.
# Extension: EA in Deep Learning [#](#extension-ea-in-deep-learning)
(This section is not on evolution strategies, but still an interesting and relevant reading.)
The Evolutionary Algorithms have been applied on many deep learning problems. POET ( [Wang et al, 2019](https://arxiv.org/abs/1901.01753) ) is a framework based on EA and attempts to generate a variety of different tasks while the problems themselves are being solved. POET has been introduced in my [last post](https://lilianweng.github.io/posts/2019-06-23-meta-rl/#task-generation-by-domain-randomization) on meta-RL. Evolutionary Reinforcement Learning (ERL) is another example; See Fig. 7 (b).
Below I would like to introduce two applications in more detail, Population-Based Training (PBT) and Weight-Agnostic Neural Networks (WANN) .
## Hyperparameter Tuning: PBT [#](#hyperparameter-tuning-pbt)
Paradigms of comparing different ways of hyperparameter tuning. (Image source: [PBT](https://arxiv.org/abs/1711.09846) paper)
Population-Based Training ( [Jaderberg, et al, 2017](https://arxiv.org/abs/1711.09846) ), short for PBT applies EA on the problem of hyperparameter tuning. It jointly trains a population of models and corresponding hyperparameters for optimal performance.
PBT starts with a set of random candidates, each containing a pair of model weights initialization and hyperparameters, { ( θ i , h i ) ∣ i = 1 , … , N } . Every sample is trained in parallel and asynchronously evaluates its own performance periodically. Whenever a member deems ready (i.e. after taking enough gradient update steps, or when the performance is good enough), it has a chance to be updated by comparing with the whole population:
exploit() : When this model is under-performing, the weights could be replaced with a better performing model. explore() : If the model weights are overwritten, explore step perturbs the hyperparameters with random noise.
In this process, only promising model and hyperparameter pairs can survive and keep on evolving, achieving better utilization of computational resources.
The algorithm of population-based training. (Image source: [PBT](https://arxiv.org/abs/1711.09846) paper)
## Network Topology Optimization: WANN [#](#network-topology-optimization-wann)
Weight Agnostic Neural Networks (short for WANN ; [Gaier & Ha 2019](https://arxiv.org/abs/1906.04358) ) experiments with searching for the smallest network topologies that can achieve the optimal performance without training the network weights. By not considering the best configuration of network weights, WANN puts much more emphasis on the architecture itself, making the focus different from [NAS](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf) . WANN is heavily inspired by a classic genetic algorithm to evolve network topologies, called NEAT (“Neuroevolution of Augmenting Topologies”; [Stanley & Miikkulainen 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.gecco02_1.pdf) ).
The workflow of WANN looks pretty much the same as standard GA:
Initialize: Create a population of minimal networks. Evaluation: Test with a range of shared weight values. Rank and Selection: Rank by performance and complexity. Mutation: Create new population by varying best networks. mutation operations for searching for new network topologies in WANN (Image source: [WANN](https://arxiv.org/abs/1906.04358) paper)
At the “evaluation” stage, all the network weights are set to be the same. In this way, WANN is actually searching for network that can be described with a minimal description length. In the “selection” stage, both the network connection and the model performance are considered.
Performance of WANN found network topologies on different RL tasks are compared with baseline FF networks commonly used in the literature. "Tuned Shared Weight" only requires adjusting one weight value. (Image source: [WANN](https://arxiv.org/abs/1906.04358) paper)
As shown in Fig. 11, WANN results are evaluated with both random weights and shared weights (single weight). It is interesting that even when enforcing weight-sharing on all weights and tuning this single parameter, WANN can discover topologies that achieve non-trivial good performance.
Cited as:
@article{weng2019ES,
title = "Evolution Strategies" ,
author = "Weng, Lilian" ,
journal = "lilianweng.github.io" ,
year = "2019" ,
url = "https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/" } copy
# References [#](#references)
[1] Nikolaus Hansen. [“The CMA Evolution Strategy: A Tutorial”](https://arxiv.org/abs/1604.00772) arXiv preprint arXiv:1604.00772 (2016).
[2] Marc Toussaint. [Slides: “Introduction to Optimization”](https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/06-blackBoxOpt.pdf)
[3] David Ha. [“A Visual Guide to Evolution Strategies”](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) blog.otoro.net. Oct 2017.
[4] Daan Wierstra, et al. [“Natural evolution strategies.”](https://arxiv.org/abs/1106.4487) IEEE World Congress on Computational Intelligence, 2008.
[5] Agustinus Kristiadi. [“Natural Gradient Descent”](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) Mar 2018.
[6] Razvan Pascanu & Yoshua Bengio. [“Revisiting Natural Gradient for Deep Networks.”](https://arxiv.org/abs/1301.3584v7) arXiv preprint arXiv:1301.3584 (2013).
[7] Tim Salimans, et al. [“Evolution strategies as a scalable alternative to reinforcement learning.”](https://arxiv.org/abs/1703.03864) arXiv preprint arXiv:1703.03864 (2017).
[8] Edoardo Conti, et al. [“Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents.”](https://arxiv.org/abs/1712.06560) NIPS. 2018.
[9] Aloïs Pourchot & Olivier Sigaud. [“CEM-RL: Combining evolutionary and gradient-based methods for policy search.”](https://arxiv.org/abs/1810.01222) ICLR 2019.
[10] Shauharda Khadka & Kagan Tumer. [“Evolution-guided policy gradient in reinforcement learning.”](https://papers.nips.cc/paper/7395-evolution-guided-policy-gradient-in-reinforcement-learning.pdf) NIPS 2018.
[11] Max Jaderberg, et al. [“Population based training of neural networks.”](https://arxiv.org/abs/1711.09846) arXiv preprint arXiv:1711.09846 (2017).
[12] Adam Gaier & David Ha. [“Weight Agnostic Neural Networks.”](https://arxiv.org/abs/1906.04358) arXiv preprint arXiv:1906.04358 (2019).
[Evolution](https://lilianweng.github.io/tags/evolution/) [Reinforcement-Learning](https://lilianweng.github.io/tags/reinforcement-learning/) [« Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) [» Meta Reinforcement Learning](https://lilianweng.github.io/posts/2019-06-23-meta-rl/) © 2025 [Lil'Log](https://lilianweng.github.io/) Powered by [Hugo](https://gohugo.io/) & [PaperMod](https://git.io/hugopapermod)