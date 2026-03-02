[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START Unlocking Agentic RL Training for GPT-OSS: A Practical Retrospective HTML_TAG_END
[Team Article](/blog) Published January 27, 2026 [Upvote 59](/login?next=%2Fblog%2FLinkedIn%2Fgpt-oss-agentic-rl) +53 [Jason Zhu JasonZhu13 Follow](/JasonZhu13) [LinkedIn](/LinkedIn) [Hejian Sang pb09204048 Follow](/pb09204048) [LinkedIn](/LinkedIn) [Arup De arde171 Follow](/arde171) [LinkedIn](/LinkedIn) [Rohit Jain rohjain Follow](/rohjain) [LinkedIn](/LinkedIn) [Yanning Chen m0m0chen Follow](/m0m0chen) [LinkedIn](/LinkedIn) HTML_TAG_START
[HTML_TAG_START Challenges of GPT-OSS RL Training HTML_TAG_END](#challenges-of-gpt-oss-rl-training) [HTML_TAG_START A Practical Debugging Journey in verl: Restoring PPO On-Policy Integrity HTML_TAG_END](#a-practical-debugging-journey-in-verl-restoring-ppo-on-policy-integrity) [HTML_TAG_START Restoring PPO On-Policy Integrity: A Fix for MoE Log-Probability Mismatch HTML_TAG_END](#restoring-ppo-on-policy-integrity-a-fix-for-moe-log-probability-mismatch) [HTML_TAG_START Correcting Training–Inference Mismatch HTML_TAG_END](#correcting-traininginference-mismatch) [HTML_TAG_START Attention Sink Support in FlashAttentionV3 HTML_TAG_END](#attention-sink-support-in-flashattentionv3) [HTML_TAG_START Standard Attention HTML_TAG_END](#standard-attention) [HTML_TAG_START Attention with Sinks (GPT-OSS) HTML_TAG_END](#attention-with-sinks-gpt-oss) [HTML_TAG_START Mathematical Formulation HTML_TAG_END](#mathematical-formulation) [HTML_TAG_START Backward Pass HTML_TAG_END](#backward-pass) [HTML_TAG_START Results HTML_TAG_END](#results) [HTML_TAG_START Memory-Efficient Training HTML_TAG_END](#memory-efficient-training) [HTML_TAG_START Mitigating FSDP Memory Blow-Ups Caused by Repeated MoE Expert Materialization HTML_TAG_END](#mitigating-fsdp-memory-blow-ups-caused-by-repeated-moe-expert-materialization) [HTML_TAG_START Sequence Parallel with Flash Attention V3 HTML_TAG_END](#sequence-parallel-with-flash-attention-v3) [HTML_TAG_START Conclusion HTML_TAG_END](#conclusion) [HTML_TAG_START Acknowledgments HTML_TAG_END](#acknowledgments) [HTML_TAG_START References HTML_TAG_END](#references) Agentic reinforcement learning (RL) extends traditional LLM training by optimizing not just a single-turn response, but an entire decision-making process learned through direct interaction with an environment during training. Unlike traditional single-turn reinforcement learning or offline preference-based methods that rely on static datasets, agentic RL trains policies by actively collecting on-policy data as the agent plans actions, invokes tools, observes outcomes, and adapts its behavior over multi-step trajectories in either simulated or real environments. This interaction-driven optimization assigns credit across long-horizon decisions, where intermediate choices such as query reformulation, tool selection, and execution order directly influence downstream success. Training follows an iterative closed loop in which the agent interacts with the environment to collect rollout trajectories, computes rewards over these trajectories, updates the policy based on observed outcomes, and then uses the updated policy to drive the next round of interaction and data collection such as GRPO or PPO algorithms..
LinkedIn is an AI-first company that's built agents to help professionals be more successful. In this setting, models must reason over incomplete information, interact with structured services, and adapt to evolving user intent across multiple steps rather than produce a single static response. These capabilities are especially critical for agents that support the goals of recruiters, job and knowledge seekers, and learners end users, such as retrieving information, refining queries, coordinating tools, and executing multi-step workflows. By learning robust decision policies through interaction, agentic RL provides a principled foundation for building scalable, reliable, and adaptable AI systems through end-to-end optimization.
The GPT-OSS model has shown comparable performance to OpenAI o3-mini and o4-mini [ [ref](https://openai.com/index/introducing-gpt-oss/) ], but its suitability for agentic reinforcement learning training has not yet been validated. Most recent work focuses on fine-tuning without tool calling, such as: [Fine-tuning with gpt-oss and Hugging Face Transformers](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) and [unsloth tutorial: how to fine-tune gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss) . This blog explores the journey to unlock [agentic RL](https://github.com/volcengine/verl/issues/3794) training for GPT-OSS as a potential backbone model for agentic applications.
In our experiments, we use [verl](https://github.com/volcengine/verl) as our training framework since it is one of the most popular adopted frameworks in the open source community. We use gsm8k, [Retool](https://github.com/verl-project/verl-recipe/tree/21892b9276936efab5375c3f6b8415e472ef7118/retool) task, [verifiable instruction following task](https://arxiv.org/abs/2507.02833) , which are commonly used in RL training. We focus on presenting experimental results for the GPT-OSS-20B model, and our attention-sink fix also works for GPT-OSS-120B. The Qwen-2.5-32B model is additionally used to benchmark standard metric trends during RL training.
## Challenges of GPT-OSS RL Training
verl has been an OSS framework used by the team, and the team has previously collaborated and contributed to it to help democratize agentic reinforcement learning training. With the introduction of the new Harmony chat template in GPT-OSS, the first step is to ensure that the training framework fully supports the updated message format and conversation semantics required by Harmony. This step helps rollout generation, trajectory construction, and tool parsing remain consistent and correct under the new template.
The team uses ReTool as a representative example to verify code correctness. ReTool is an agentic coding task in which the model is asked to solve a math problem with the assistance of a code compiler tool. This setup allows the model to focus on core reasoning and algorithmic logic, while delegating the actual arithmetic and execution to the tool. During an episode, the model interacts with the code tool multiple times, using execution results as feedback to refine its solution. At the end of the trajectory, the model produces a final answer, on which the reward is computed.
During the initial training runs, we observed exploding KL divergence and entropy, along with non-increasing rewards, indicating underlying issues in the GPT-OSS training setup, as shown in Figure 1.
| Average Gradient Norm | Average Reward |
|---|---|
| | |
Figure 1. Left: Qwen32b has significantly higher rewards compared to GPT-OSS 20B; Right: The gradient norm exploded as training progressed.
## A Practical Debugging Journey in verl: Restoring PPO On-Policy Integrity
### Restoring PPO On-Policy Integrity: A Fix for MoE Log-Probability Mismatch
Figure 2. Non-zero importance sampling clip value even for on-policy training.
We focus on on-policy methods because they provide greater stability and more reliable convergence. The foundation of pure on-policy Proximal Policy Optimization (PPO) mandates that the importance sampling ratio must be exactly 1. The mathematical definition of the importance ratio is:
ratio = π ( a ∣ s ) π old ( a ∣ s ) \text{ratio} = \frac{\pi(a \mid s)}{\pi_{\text{old}}(a \mid s)} ratio = π old ​ ( a ∣ s ) π ( a ∣ s ) ​
This requirement ensures that the policy update is executed only on the data generated by the current policy π(a | s) = π old (a | s), preventing unintended clipping.
We have observed the non-zero clipping value in our ReTool training, as shown in Figure 2, stemming from a mismatch between the two log-probabilities:
Current log-probability log_prob : log(π(a | s)) Old log-probability old_log_prob : log(π old (a | s))
Root Cause: The Dual Forward Pass and MoE Architecture
Prior to verl 0.3.0, the implementation relied on two separate forward passes (one to compute the current log_prob and one to retrieve the stored old_log_prob ) for the same state-action pair.
In a Mixture of Experts (MoE) architecture like GPT-OSS, the gating network routes the input to different experts. Due to implementation factors (e.g., subtle floating-point differences or explicit stochasticity), the expert routing can differ slightly between the two passes. Readers who are interested can further read [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/pdf/2510.11370v1) .
This difference in routing leads to:
log ⁡ ( π ( a ∣ s ) ) ≠ log ⁡ ( π old ( a ∣ s ) ) \log(\pi(a \mid s)) \neq \log(\pi_{\text{old}}(a \mid s)) lo g ( π ( a ∣ s ))  = lo g ( π old ​ ( a ∣ s ))
The resulting ratio deviates from 1, falsely triggering the PPO clip and violating the core on-policy assumption.
Solution: Enforcing Ratio = 1 via Log-Probability Substitution
The fix resolves the issue by logically overriding the flawed computation when the environment is known to be on-policy (i.e., when the minibatch size equals the global batch size):
if on_policy:
old_log_prob = log_prob.detach() else :
old_log_prob = model_inputs[ "old_log_probs" ]
By setting old_log_prob equal to the newly computed log_prob (detached to prevent gradient flow through the reference value), the importance ratio is mathematically forced back to 1. This strategy bypasses the instability caused by MoE's non-deterministic routing and guarantees strict on-policy behavior during PPO training.
### Correcting Training–Inference Mismatch
Although fixing the log-probability mismatch reduced the importance-sampling clip ratio to zero, gradient norms continued to explode and rewards failed to improve. To isolate the issue, we simplified training to GSM8K, a single-step task without agentic tool use. The same instability persisted, as shown in the green curves in Figure 3, indicating a fundamental issue in basic RL training with GPT-OSS under verl.
We hypothesize that training–inference mismatch could be a potential cause: discrepancies between inference-time execution—where engines such as vLLM and SGLang aggressively optimize for throughput—and training-time execution under FSDP, which prioritizes numerical precision and stability, can effectively turn otherwise on-policy RL into off-policy optimization.
This [blog](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) details why such mismatches lead to unstable gradients and non-improving rewards. Figure 3 compares training runs with and without rollout correction (see this [verl blog](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) for details). After applying rollout correction, training dynamics improve significantly, with gradient norms remaining stable rather than exploding.
However, as shown in the left plot of Figure 4, the reward increases only modestly, and convergence on the simple GSM8K task remains substantially slower compared to smaller dense model variants.
| Average Entropy | Average Gradient Norm | Average KL Loss |
|---|---|---|
| | | |
Figure 3. Gradient norm behavior under different training configurations. Green: Training without rollout correction, exhibiting unstable gradients. Red: Training with the attention layer frozen to isolate the issue to the attention mechanism, resulting in partial stabilization. Blue: Training with rollout correction enabled (sequence-level importance sampling), yielding stable gradient norms.
| Average Reward | Max Log-Perplexity Difference |
|---|---|
| | |
Figure 4. Left: Reward improvement on GSM8K remains slow even after applying rollout correction, with performance comparable to runs where the attention layer is frozen during training. Right: A substantial log-ppl mismatch is observed between the inference engine (SGLang with Triton kernels supporting attention-sink forward passes) and the training stack (FSDP with FlashAttention-v2), indicating a large training–inference inconsistency.
To further isolate the root cause, we freeze the attention layers during training and observe reward dynamics similar to those of runs without freezing (blue curve vs yellow curve in Figure 4). This indicates that learning is primarily driven by the MoE layers, while the attention mechanism contributes less effectively than expected. In addition, we observe a substantial token-level probability mismatch between the inference engine and the distributed training stack which are using different attention kernels. Together, these observations motivate a deeper investigation into the attention mechanism.
## Attention Sink Support in FlashAttentionV3
Attention sinks used in GPT-OSS are learnable scalar parameters (one per attention head) that act as "virtual tokens" in the softmax computation. They allow the model to allocate attention mass to a learned sink rather than forcing all attention to content tokens, which has been shown to improve attention stability in streaming inference and training with sliding-window attention.
After a deeper investigation, we identified several major issues:
verl hard-codes FlashAttention v2 in fsdp_worker , which does not support attention sinks. The attention sink backward pass is not supported in FlashAttention v2 and v3, so it does not work as expected even when FlashAttention v3 is enabled. Since the forward pass has not yet been merged into the original FlashAttention v3 repository, we leveraged the forward pass from the vLLM FlashAttention fork ( [PR #75](https://github.com/vllm-project/flash-attention/pull/75) ) and implemented the backward pass to compute the sink gradient.
### Standard Attention
scores = QK^T / sqrt(d) # [B, H, N_q, N_k] probs = softmax(scores, dim=- 1 ) # Σ_j P_ij = 1 output = probs @ V # [B, H, N_q, d_v]
### Attention with Sinks (GPT-OSS)
scores = QK^T / sqrt(d) # [B, H, N_q, N_k] combined = concat([scores, sink_param], dim=- 1 ) # [B, H, N_q, N_k+1] probs = softmax(combined, dim=- 1 ) # Σ_j P_ij + P_sink = 1 probs_content = probs[..., :- 1 ] # Drop sink component output = probs_content @ V # [B, H, N_q, d_v]
Key difference: The sink participates in softmax normalization but doesn't contribute to the output.
### Mathematical Formulation
The attention weight for content token j in row i is defined as:
P i j = exp ⁡ ( S i j ) ∑ j ′ = 1 N k exp ⁡ ( S i j ′ ) + exp ⁡ ( S h ) P_{ij}
=
\frac{\exp(S_{ij})}
{\sum_{j'=1}^{N_k} \exp(S_{ij'}) + \exp(S_h)} P ij ​ = ∑ j ′ = 1 N k ​ ​ exp ( S i j ′ ​ ) + exp ( S h ​ ) exp ( S ij ​ ) ​
Where:
S ij = Q i K j ⊤ / √d are the attention scores P ij are the attention weights for the content tokens S h is the learnable sink parameter for head h
Sink Probability:
The sink probability is computed but not used in the output:
P i , h = exp ⁡ ( S h ) ∑ j ′ = 1 N k exp ⁡ ( S i j ′ ) + exp ⁡ ( S h ) P_{i,h}
=
\frac{\exp(S_h)}
{\sum_{j'=1}^{N_k} \exp(S_{ij'}) + \exp(S_h)} P i , h ​ = ∑ j ′ = 1 N k ​ ​ exp ( S i j ′ ​ ) + exp ( S h ​ ) exp ( S h ​ ) ​
### Backward Pass
The gradient of the loss L with respect to the sink parameter S h is:
∂ L ∂ S h = − ∑ i P i , h ( ∂ L ∂ S i , h − ∑ j ∈ { 1 , … , N k } P i j ∂ L ∂ S i j ) \frac{\partial L}{\partial S_h}
=
-
\sum_i
P_{i,h}
\left(
\frac{\partial L}{\partial S_{i,h}}
-
\sum_{j \in \{1,\ldots,N_k\}}
P_{ij}
\frac{\partial L}{\partial S_{ij}}
\right) ∂ S h ​ ∂ L ​ = − i ∑ ​ P i , h ​ ​ ∂ S i , h ​ ∂ L ​ − j ∈ { 1 , … , N k ​ } ∑ ​ P ij ​ ∂ S ij ​ ∂ L ​ ​
Where:
P i,h is the sink attention probability for row i ∂L/∂S ij is the gradient with respect to the attention scores, including the sink
Simplified Gradient:
Since the sink is computed but not used in the output, its gradient ∂L/∂S i,h = 0.
Therefore, the backward equation simplifies to:
∂ L ∂ S h = − ∑ i P i , h ( ∑ j ∈ { 1 , … , N k } P i j ∂ L ∂ S i j ) \frac{\partial L}{\partial S_h}
=
-
\sum_i
P_{i,h}
\left(
\sum_{j \in \{1,\ldots,N_k\}}
P_{ij}
\frac{\partial L}{\partial S_{ij}}
\right) ∂ S h ​ ∂ L ​ = − i ∑ ​ P i , h ​ ​ j ∈ { 1 , … , N k ​ } ∑ ​ P ij ​ ∂ S ij ​ ∂ L ​ ​
The forward pass was adapted from vLLM's FlashAttention fork, and we implemented the backward pass to compute gradients for the sink parameters. The implementation will be released following the internal review process.
### Results
After applying the fix in FlashAttention v3, we observe substantially faster convergence for GPT-OSS-20B across a range of reinforcement learning tasks. These include single-turn RL on math reasoning (GSM8K — red curve in Figure 5), instruction following (VerifyIf, evaluated on an out-of-domain multi-if benchmark — Figure 6), and multi-turn agentic RL with tool use (ReTool — Figure 7).
Across all settings, training becomes stable and exhibits steady reward improvement.
Figure 5. . Single Turn GSM8K, the red curve converges much faster than the rest without the fix
| Average Entropy | Average Gradient Norm | Average Reward |
|---|---|---|
| | | |
Figure 6 . On verifiable instruction following the task, the run without the fix collapsed (blue), and the run with fix showed steady reward improvement.
| Average Gradient Norm | Average Reward | Validation Accuracy |
|---|---|---|
| | | |
Figure 7 . On the Retool task, the run with fix showed steady reward improvement and no gradient exploding (fa2 is the flash attention 2 without the fix while fa3 is the flash attention 3 with the fix). After the fix, the validation accuracy score goes up now.
## Memory-Efficient Training
### Mitigating FSDP Memory Blow-Ups Caused by Repeated MoE Expert Materialization
One issue we consistently encountered was excessive memory allocation during the FSDP forward pass, which led to repeated out-of-memory (OOM) failures when training GPT-OSS-20B bf16 models on 16 H200 nodes (max response length: 16k, prompt length: 8k). This behavior is highly unexpected for a 20B-parameter MoE model.
2025-11-27T11:15:27.927Z [36m(TaskRunner pid=32081)[0m File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py", line 123, in forward
2025-11-27T11:15:27.927Z [36m(TaskRunner pid=32081)[0m hidden_states = hidden_states.repeat(num_experts, 1)
2025-11-27T11:15:27.927Z [36m(TaskRunner pid=32081)[0m torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 180.00 GiB. GPU 0 has a total capacity of 139.72 GiB of which 110.94 GiB is free. Process 685851 has 24.88 GiB memory in use. Process 692458 has 3.87 GiB memory in use. Of the allocated memory 23.28 GiB is allocated by PyTorch, and 84.43 MiB is reserved by PyTorch but unallocated.
We identified the issue as originating from two different implementations of the MoE forward path in Hugging Face Transformers. This issue has also been reported by other users: [https://github.com/huggingface/transformers/issues/40073](https://github.com/huggingface/transformers/issues/40073) ; When verl computes log-probabilities under FSDP, the inference forward path is triggered. In the current Hugging Face implementation, this path duplicates hidden states for all experts and performs batched matrix multiplication, materializing extremely large tensors in GPU memory. By contrast, the training forward path uses a for-loop to process each expert sequentially and then combines the results. While slower, this approach is significantly more memory efficient.
@GPUMemoryLogger( role= "dp actor" , logger=logger ) def compute_log_prob ( self, data: DataProto, calculate_entropy= False ) -> torch.Tensor: """ .... """ # set to eval, this essentially prioritizes parallelism at the cost of memory efficiency self.actor_module. eval ()
...
We patched the Hugging Face implementation to use a more memory-efficient execution path, avoiding repeated materialization of experts.
### Sequence Parallel with Flash Attention V3
Agentic RL requires the agent to interact with the environment over multiple steps while maintaining an ever-expanding context. Observations and environment feedback from each step are appended to the context and used as input for subsequent decision-making, which introduces significant challenges for memory efficiency and scalability during training.
Under fully sharded data parallelism (FSDP), model parameters, optimizer states, and gradients are sharded across the entire world size (i.e., all GPUs in the training cluster). Each GPU stores and updates only its assigned parameter shards, while rollout data are replicated across all GPUs—meaning every GPU processes the full agent interaction history for each rollout.
During the forward pass, when computation reaches a layer whose parameters are not locally available, an all_gather operation is triggered to materialize the full parameters across GPUs. During the backward pass, a corresponding reduce_scatter operation aggregates gradients and ensures that each GPU retains only its local shard. This provides a degree of scaling: as the number of GPUs increases, the per-GPU memory footprint decreases.
FSDP provides model-level scaling by sharding model parameters, gradients, and optimizer states across GPUs. Sequence parallelism (or context parallelism) further reduces per-GPU memory consumption by partitioning the input sequence across devices, thereby lowering the peak activation memory on each GPU.
As the number of sequence-parallel dimensions increases, the maximum activation memory per GPU correspondingly decreases. We have implemented sequence parallelism to be attention-sink-aware and compatible with FlashAttention v3 (Figure 8, right).
Figure 8 . Left: Inference without sequence parallelism. Right: Inference with sequence parallelism, where additional all-to-all communication is performed before and after the attention layer. This partitions the sequence across parallel workers and reduces the peak memory footprint of attention computation by a factor proportional to the sequence-parallelism degree.
Sequence parallelism scales along the sequence dimension to reduce the per-GPU activation footprint. Input tokens from all sequences are packed into a single contiguous list by removing padding tokens, while position IDs are used to distinguish tokens belonging to different sequences. This design naturally benefits from FlashAttention’s variable-length support. For sequence parallelism, layers other than the attention layer do not have inter-position dependencies; therefore, they do not require each GPU to hold a complete sequence shard, and no additional communication is needed for these layers.
The attention layer, however, requires all tokens belonging to the same sequence to be present on the same GPU in order to compute attention weights correctly. To satisfy this constraint, an all-to-all communication is performed to gather sequence elements, with the split performed at the attention-head level. This design avoids communication within the attention computation itself, which would otherwise be prohibitively expensive. After the attention layer, a single all-to-all communication redistributes the outputs back to their original sequence-parallel layout, after which the remaining non-attention layers can proceed without further synchronization.
## Conclusion
Our journey to enable agentic RL training for the GPT-OSS backbone model was a practical retrospective, highlighting that unlocking advanced capabilities in open-source LLMs requires meticulous, deep-dive engineering.
We made contributions that transformed the viability of GPT-OSS for agentic applications, specifically by:
Stabilizing PPO: We contributed a fix to restore on-policy integrity, overriding the log-probability mismatch caused by the MoE architecture’s non-determinism (Figure 2).
Enabling Attention Sink Support: We successfully implemented and integrated the attention sink backward pass into FlashAttention v3, correcting the catastrophic training–inference mismatch that had previously caused instability and slow convergence (Figures 5, 6, and 7).
Scaling Memory Efficiency: We introduced crucial memory optimizations, including patching the MoE materialization process and integrating sequence parallelism with the new attention sink support, enabling training with the long context windows essential for multi-step agents (Figure 8).
These engineering efforts validate GPT-OSS as a scalable and high-performance backbone for building the next generation of intelligent, multi-step decision-making agents.
## Acknowledgments
Thanks to Deepak Agarwal, Bee-Chung Chen, Animesh Singh, Gungor Polatkan, Balaji Krishnapuram, and Jitendra Agarwal for their leadership support.
## References
Feng, Jiazhan, et al. Retool: Reinforcement Learning for Strategic Tool Use in LLMs. arXiv preprint arXiv:2504.11536 (2025).
Xiao, Guangxuan, et al. Efficient Streaming Language Models with Attention Sinks. arXiv preprint arXiv:2309.17453 (2023).
When Speed Kills Stability: Demystifying RL Collapse from the Training–Inference Mismatch. [https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
HTML_TAG_END
### Community
[sseymens](/sseymens) [Jan 27](#6978226bb5032ff34675b51f) • [edited Jan 27](#6978226bb5032ff34675b51f)
Thank you for this detailed retrospective on enabling agentic RL training for GPT-OSS. The article clearly documents important challenges and practical solutions. I have a few questions and observations that might help clarify some aspects.
Mathematical Clarification on Attention Sink Gradient
Regarding the attention sink backward pass derivation (Section 4), I'd appreciate clarification on the notation and justification. The article presents the general formula: ∂L/∂S_h = -Σ_i P_{i,h} (∂L/∂S_{i,h} - Σ_{j∈{1,...,N_k}} P_{ij} ∂L/∂S_{ij}). Then states that ∂L/∂S_{i,h} = 0 because "the sink is computed but not used in the output," leading to the simplified form: ∂L/∂S_h = -Σ_i P_{i,h} (Σ_j P_{ij} ∂L/∂S_{ij}).
I believe the final formula is correct, but the notation ∂L/∂S_{i,h} is unclear. If this refers to the gradient of the loss with respect to the sink's contribution to the output at position i (i.e., ∂L/∂(P_{i,h} V_sink)), then indeed it would be zero since there's no V_sink term. However, the sink parameter S_h itself does affect the output through softmax normalization: P_{ij} = exp(S_{ij}) / (Σ_{j'} exp(S_{ij'}) + exp(S_h)). When S_h changes, all P_{ij} change, affecting O_i = Σ_j P_{ij} V_j. The gradient ∂L/∂S_h is non-zero and flows through this normalization.
Could you clarify what ∂L/∂S_{i,h} represents in your notation? Also, could you provide the full derivation showing how you arrive at the final form from the chain rule through the softmax operation? A step-by-step derivation would help readers verify the implementation and understand how the gradient flows through the attention mechanism.
On-Policy Fix: Clarification Needed
The solution for the MoE log-probability mismatch (Section 2) sets old_log_prob = log_prob.detach() when on_policy=True. While this mathematically forces the importance sampling ratio to 1, I'm curious about the implications. If the policy parameters change between rollout collection and the training step (even slightly), this fix would still assume ratio = 1. How do you ensure true on-policy conditions? Do you collect rollouts and train in the same step, or is there a mechanism to detect policy drift?
Have you considered storing the actual routing decisions during rollout to ensure deterministic replay, or using deterministic routing during training? This might address the root cause rather than working around it. Also, the fix only applies when on_policy=True. How do you handle cases where the policy has drifted? Is there a threshold or mechanism to detect when to switch to off-policy mode?
Experimental Details
Several aspects would benefit from additional detail. Which fix had the largest impact? Are all three fixes necessary, or would fixing attention sinks alone have been sufficient? A table showing results with/without each fix would be very helpful. You mention the fixes work for GPT-OSS-120B, but no experimental results are shown. Could you share learning curves or metrics for the 120B model?
The article mentions both vLLM and SGLang. Which engine is used for rollouts in the final experiments? If different engines are used for different experiments, how does this affect the training-inference mismatch analysis? You also reference "rollout correction (sequence-level importance sampling)" and mention "this verl blog," but the details aren't explained. Could you clarify what this correction does and how it relates to the attention sink fix?
Implementation and Reproducibility
The article mentions that "The implementation will be released following the internal review process." For reproducibility and verification, it would be valuable to know an approximate timeline for release, whether the backward pass implementation will be contributed to the main FlashAttention repository, and if there are any interim workarounds or partial implementations that could be shared.
Positive Aspects
Despite these questions, the article makes valuable contributions. It clearly identifies real-world challenges with MoE + RL training, documents the engineering journey transparently, shows substantial improvements in training stability, and raises awareness of training-inference mismatch issues. The fixes appear to work well for your use case, and the results demonstrate clear improvements. My questions are aimed at understanding the details better and ensuring the solutions are robust and reproducible.
Thank you again for sharing this work with the community!
See translation 👍 2 2 + Reply [pb09204048](/pb09204048) Article author [about 1 month ago](#697c319c875c28e240bec134)
Hi [@ sseymens](/sseymens) Thank you for your comments. I can help to reply your question about MOE on policy part.
Yeah, forcing old_log_prob = log_prob.detach() does not solve the on policy issue since the prob is using current policy but sampling distribution can be different due to expert selection. When we explored the agentic issues for gpt-oss training, we did not root the cause at the beginning. One hypothesis is due to inference-training inconsistency. After we apply the importance sampling, it does not help. So we test if forcing old_log_prob = log_prob.detach() will alleviate the issue if this is the root cause. This is just for hypothesis testing. When we explored the agentic issues for gpt-oss training, verl has not supported expert router replay yet. So we cannot test this idea. [https://arxiv.org/pdf/2510.11370v1](https://arxiv.org/pdf/2510.11370v1) . Now we tested the relay. But this is not the root cause too. The root cause is attention sink. See translation 👍 1 1 + Reply [sseymens](/sseymens) [about 1 month ago](#697d04fc71e2eeba619f847b)
Thank you for taking the time to reply and for clarifying the MoE/on-policy part—that helps a lot.
What you said (and how I'm reading it):
You confirmed that forcing old_log_prob = log_prob.detach() does not resolve the on-policy issue: the evaluation uses the current policy π(a|s), but the sampling distribution can still differ because expert selection at rollout time can differ from the training forward pass. Forcing the importance ratio to 1 in the PPO formula therefore doesn't fix the underlying problem—the data can still be effectively off-policy. I agree with that distinction.
You clarified the debugging sequence: root cause was unknown at first; you hypothesized inference–training inconsistency; importance sampling (rollout correction) didn't fix it; then you tried forcing the detach as a hypothesis test to see whether the MoE log-prob mismatch was the cause. That makes it clear the detach was diagnostic, not the actual fix.
You noted that expert router replay wasn't available in verl when you were debugging; you've since tested it (Stabilizing MoE RL, arxiv 2510.11370) and it was not the root cause—the root cause is attention sink. That fits the story: the real mismatch is at the attention layer (training on FA2 without sink support vs. inference on sink-aware kernels), so token-level log-probs diverge between rollout and training. That explains why rollout correction alone didn't fix things (it corrects at sequence level but not at the attention computation) and why freezing attention gave similar reward curves (attention wasn't learning correctly without the sink backward). I'm satisfied on the on-policy/MoE part.
What wasn't addressed: You addressed the MoE/on-policy part; the following from my original comment were not covered. If you have bandwidth, they would help for implementation and reproducibility:
Attention sink backward (Section 4)—notation and derivation
The article gives the general form ∂L/∂S_h = -Σ_i P_{i,h} (∂L/∂S_{i,h} - Σ_j P_{ij} ∂L/∂S_{ij}), then states that ∂L/∂S_{i,h} = 0 because "the sink is computed but not used in the output," yielding ∂L/∂S_h = -Σ_i P_{i,h} Σ_j P_{ij} ∂L/∂S_{ij}.
For implementers, two things would help:
Definition of ∂L/∂S_{i,h}: In the extended softmax, the logits are (S_{i,1}, …, S_{i,N_k}, S_h). The output is O_i = Σ_j P_{ij} V_j with no V_sink term, so the sink affects O_i only through the normalization Z_i = Σ_{j'} exp(S_{ij'}) + exp(S_h). The direct contribution of the sink to the output (e.g. a P_{i,h} V_sink term) is therefore zero, so ∂L/∂S_{i,h} in that sense is 0. If ∂L/∂S_{i,h} is defined differently in your derivation, a one-sentence definition in the article would remove ambiguity.
Chain rule through softmax: A short derivation showing how ∂L/∂S_h is obtained from ∂L/∂P_{ij} (and possibly ∂L/∂O_i) via ∂P_{ij}/∂S_h and ∂P_{i,h}/∂S_h would let readers verify the sign and aggregation over query positions i and match it to the FlashAttention backward API.
Experimental setup and ablations
To reproduce and extend the results, it would be helpful to have:
Ablation: Was the FA3 + sink backward sufficient for stable training (with MoE materialization and sequence parallelism only for correctness/memory), or did you still rely on rollout correction or the on-policy detach in the final runs? A small table (e.g. FA2 vs FA3+sink, with/without rollout correction) would make the relative impact clear.
Rollout engine: The article mentions both vLLM and SGLang; Figure 4 references SGLang with Triton kernels for the sink forward. Which engine was used for rollouts in the final experiments (Figures 5–7), and was it the same across GSM8K, VerifyIf, and ReTool? This matters for training–inference consistency.
Rollout correction vs sink fix: A one-sentence summary of what "rollout correction" (sequence-level importance sampling) does in your pipeline and whether it is still enabled in the FA3+sink runs would clarify how the two mechanisms interact.
GPT-OSS-120B: You mention the sink fix also works for 120B; any learning curves or metrics (even high-level) would be valuable for scaling.
Implementation and release
For verification and adoption: any update on timeline for releasing the sink backward implementation, whether you plan to contribute it to the main FlashAttention repo or keep it in a fork, and whether there are interim options (e.g. a reference PyTorch implementation or a patch description) would help the community reproduce the results and align training with inference.
Thanks again for the clarification; the narrative (hypothesis testing → rollout correction → attention sink as root cause) is much clearer, and the work is clearly valuable for agentic RL on GPT-OSS.
See translation 👍 1 1 + Reply deleted [24 days ago](#69858dfdc7d8711a1d28ff03) This comment has been hidden [MatthewFrank](/MatthewFrank) [20 days ago](#698a281ae23462f32261330d)
Excellent retrospective on the practical challenges of agentic RL training at scale! The architectural decisions you made around distributed training and infrastructure are really valuable insights. For documenting complex training infrastructures like this, I've found InfraSketch ( [https://www.infrasketch.net/](https://www.infrasketch.net/) ) to be a game-changer—you can describe your system in plain English and get architecture diagrams in seconds, then refine them conversationally. It's been invaluable for creating design docs that actually get used and updated.
See translation Reply [jbollenbacher](/jbollenbacher) [10 days ago](#6997d0d14d77f946f4468d8e)
Will you release the weights of the gpt-oss-agentic models you trained? I'm sure they'd be very popular.
See translation Reply [kirankate06](/kirankate06) [5 days ago](#699e2fafd08eda32be23b6ef) • [edited 5 days ago](#699e2fafd08eda32be23b6ef)
Hello, thank you for your work and for explaining the changes required. Are these changes merged to verl and FlashAttention V3?
See translation Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2FLinkedIn%2Fgpt-oss-agentic-rl) or [log in](/login?next=%2Fblog%2FLinkedIn%2Fgpt-oss-agentic-rl) to comment
[Upvote 59](/login?next=%2Fblog%2FLinkedIn%2Fgpt-oss-agentic-rl) +47 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe