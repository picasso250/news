[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START Transformers v5: Simple model definitions powering the AI ecosystem HTML_TAG_END
Published December 1, 2025 [Update on GitHub](https://github.com/huggingface/blog/blob/main/transformers-v5.md) [Upvote 302](/login?next=%2Fblog%2Ftransformers-v5) +296 [Lysandre lysandre Follow](/lysandre) [Arthur Zucker ArthurZ Follow](/ArthurZ) [Cyril Vallez cyrilvallez Follow](/cyrilvallez) [Vaibhav Srivastav reach-vb Follow](/reach-vb) HTML_TAG_START
[HTML_TAG_START Simplicity HTML_TAG_END](#simplicity) [HTML_TAG_START Model Additions HTML_TAG_END](#model-additions) [HTML_TAG_START Code Reduction HTML_TAG_END](#code-reduction) [HTML_TAG_START Training HTML_TAG_END](#training) [HTML_TAG_START Pre-training at scale HTML_TAG_END](#pre-training-at-scale) [HTML_TAG_START Fine-tuning & Post-training HTML_TAG_END](#fine-tuning--post-training) [HTML_TAG_START Inference HTML_TAG_END](#inference) [HTML_TAG_START Production & Local HTML_TAG_END](#production--local) [HTML_TAG_START Quantization HTML_TAG_END](#quantization) [HTML_TAG_START Conclusion HTML_TAG_END](#conclusion) Transformers' version v4.0.0rc-1, the initial release candidate for version 4, was released on November 19th, 2020. Five years later, we now release v5.0.0rc-0.
Today, as we launch v5, Transformers is installed more than 3 million times each day via pip - up from 20,000/day in v4 🤯. Altogether, it has now surpassed 1.2 billion installs !
The ecosystem has expanded from 40 model architectures in v4 to over 400 today , and the community has contributed more than 750,000 model checkpoints on the Hub compatible with Transformers, up from roughly 1,000 at the time of v4.
This growth is powered by the evolution of the field and the now mainstream access to AI. As a leading model-definition library in the ecosystem, we need to continuously evolve and adapt the library to continue being relevant. Reinvention is key for longevity in AI.
We’re fortunate to collaborate with many libraries and apps built on transformers, in no specific order: llama.cpp, MLX, onnxruntime, Jan, LMStudio, vLLM, SGLang, Unsloth, LlamaFactory, dLLM, MaxText, TensorRT, Argmax, among many other friends.
For v5, we wanted to work on several notable aspects: simplicity, training, inference, and production. We detail the work that went into them in this post.
## Simplicity
The first focus of the team was on simplicity. Working on transformers, we see the code as the product. We want our model integrations to be clean, so that the ecosystem may depend on our model definitions and understand what’s really happening under the hood, how models differ from each other, and the key features of each new model. Simplicity results in wider standardization, generality, and wider support.
### Model Additions
Transformers is the backbone of hundreds of thousands of projects, Unsloth included. We build on Transformers to
help people fine-tune and train models efficiently, whether that’s BERT, text-to-speech (TTS), or others; to run
fast inference for reinforcement learning (RL) even when models aren’t yet supported in other libraries. We're
excited for Transformers v5 and are super happy to be working with the Hugging Face team!
-- Michael Han at Unsloth
Transformers, at the core, remains a model architecture toolkit. We aim to have all recent architectures and to be the “source of truth” for model definitions. We’ve been adding between 1 - 3 new models every week for 5 years, shown in the timeline below:
We’ve worked on improving that model-addition process.
#### Modular Approach
Over the past year, we’ve heavily pushed our modular design as a significant step forward. This allows for easier maintenance, faster integration, and better collaboration across the community.
We give a deeper overview in our [Maintain the Unmaintainable](https://huggingface.co/spaces/transformers-community/Transformers-tenets) blog post. For brevity, we aim to achieve a much easier model contribution process, as well as a lower maintenance burden. One metric we can highlight is that the number of lines of code to contribute (and review), drop significantly when [modular](https://huggingface.co/docs/transformers/en/modular_transformers) is used:
While we respect the “One model, one file” philosophy, we continue introducing some abstractions making the management of common helpers simpler. The prime example of this is the introduction of the AttentionInterface , which offers a centralized abstraction for attention methods. The eager method will remain in the modeling file; others, such as FA1/2/3, FlexAttention, or SDPA, are moved to the interface.
Over the past couple of years, the increasing amount of 0-day support for new model architectures and
standardization of attention handling has helped to simplify our support for post-training modern LLMs.
-- Wing Lian, Axolotl
#### Tooling for Model Conversion
We’re building tooling to help us identify which existing model architecture a new model resembles. This feature uses machine learning to find code similarities between independent modeling files. Going further, we aim to automate the conversion process by opening a draft PR for the model to be integrated into our transformers format. This process reduces manual effort and ensures consistency.
### Code Reduction
#### Streamlining Modeling & Tokenization/Processing Files
We’ve significantly refactored the modeling and tokenization files. Modeling files have been greatly improved thanks to the modular approach mentioned above, on top of standardization across models. Standardization contributes to abstracting most of the tools that don’t make up a model, so that the modeling code only contains the relevant parts for a model’s forward/backward passes.
Alongside this work, we’re simplifying the tokenization and processing files: going forward, we’ll only focus on the tokenizers backend, removing the concept of “Fast” and “Slow” tokenizers.
We'll use tokenizers as our main tokenization backend, just as we do for PyTorch-based models. We’ll offer alternatives for Sentencepiece or MistralCommon backed tokenizers, which will be non-default but will be supported. Image processors will now only exist with their fast variant, which depends on the torchvision backend.
Finally, we’re sunsetting our Flax/TensorFlow support in favor of focusing on PyTorch as the sole backend; however,
we're also working with partners in the Jax ecosystem to ensure we have compatibility between our models and this
ecosystem.
With its v5 release, transformers is going all in on PyTorch. Transformers acts as a source of truth and
foundation for modeling across the field; we've been working with the team to ensure good performance
across the stack.
We're excited to continue pushing for this in the future across training, inference, and deployment.
-- Matt White, Executive Director, PyTorch Foundation. GM of AI, Linux Foundation
## Training
Training remains a big focus of the team as we head into v5: whereas previously we would focus heavily on fine-tuning rather than pre-training/full-training at scale, we’ve recently done significant work to improve our support for the latter as well.
### Pre-training at scale
Supporting pre-training meant reworking the initialization of our models, ensuring that they worked at scale with different parallelism paradigms, and shipping support for optimized kernels for both the forward and backward passes.
Going forward, we’re excited to have extended compatibility with torchtitan, megatron, nanotron, as well as any other pre-training tool that is interested in collaborating with us.
### Fine-tuning & Post-training
We continue collaborating closely with all fine-tuning tools in the Python ecosystem. We aim to continue providing model implementations compatible with [Unsloth](https://huggingface.co/unsloth) , [Axolotl](https://huggingface.co/axolotl-ai-co) , [LlamaFactory](https://huggingface.co/llamafactory) , [TRL](https://huggingface.co/docs/trl/en/index) and others in the PyTorch ecosystem; but we are also working with tools such as [MaxText](https://github.com/AI-Hypercomputer/maxtext) , in the JAX ecosystem, to have good interoperability between their frameworks and transformers .
All fine-tuning and post-training tools can now rely on transformers for model definitions; further enabling Agentic use-cases through [OpenEnv](https://huggingface.co/openenv) or the Prime Environment Hub.
## Inference
We’re putting a significant focus on inference for v5, with several paradigm changes: the introduction of specialized kernels, cleaner defaults, new APIs, support for optimized inference engines.
Similarly to training, we’ve been putting some effort in packaging kernels so that they’re automatically used in case your hardware and software permits it. If you haven’t heard of kernels before, we recommend taking a look at this [doc](https://huggingface.co/docs/kernels/basic-usage) .
Alongside this effort, we ship two new APIs dedicated to inference:
We ship support for continuous batching and paged attention mechanisms. This has now been used internally for some time, and we’re working on finalizing the rough edges and writing usage guides. We introduce transformers serve as the new transformers-specific serving system, which deploys an OpenAI API-compatible server.
We see this as a major step forward for use-cases such as evaluation, where a great number of inference requests are done simultaneously. We don’t aim to do specialized optimizations like the dedicated inference engines (vLLM, SGLang, TensorRT LLM). Instead, we aim to be perfectly inter-compatible with these, as detailed in the next section.
The Transformers backend in vLLM has been very enabling to get more architectures, like BERT and other encoders, available to more users. We've been working with the Transformers team to ensure many models are available across modalities with the best performance possible. This is just the start of our collaboration: we're happy to see the Transformers team will have this as a focus going into version 5.
-- Simon Mo, Harry Mellor at vLLM
Standardization is key to accelerating AI innovation. Transformers v5 empowers the SGLang team to spend less time on model reimplementation and more time on kernel optimization. We look forward to building a more efficient and unified AI ecosystem together!
-- Chenyang Zhao at SGLang
### Production & Local
Recently, we've been working hand in hand with the most popular inference engines for them to use transformers as a backend. The value added is significant: as soon as a model is added to transformers , it becomes available in these inference engines, while taking advantage of the strengths each engine provides : inference optimizations, specialized kernels, dynamic batching, etc.
We've also been working very closely with ONNXRuntime, [llama.cpp](https://github.com/ggml-org/llama.cpp) and [MLX](https://github.com/ml-explore/mlx) so that the implementations between transformers and these modeling libraries have great interoperability. For example, thanks to a significant community effort, it's now very easy to [load GGUF files in transformers](https://huggingface.co/docs/transformers/en/gguf) for further fine-tuning. Conversely, transformers models can be easily [converted to GGUF files](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) for use with llama.cpp.
The Transformers framework is the go-to place for reference AI model implementations. The framework plays a crucial role in enabling modern AI across the entire stack. The team and the community behind the project truly understand and embrace the spirit of the open-source development and collaboration.
-- Georgi Gerganov, ggml-org
The same is true for MLX, where the transformers' safetensors files are directly compatible with MLX's models.
It’s hard to overstate the importance of Transformers (and datasets, tokenizers, etc) to the open-source and
overall AI ecosystem. I can’t count the number of times I’ve personally used Transformers as a source-of-truth.
-- Awni Hannun, MLX
Finally, we’re pushing the boundaries of local inference and are working hand-in-hand with the [executorch](https://github.com/pytorch/executorch) team to get the transformers models to be available on-device. We’re expanding the coverage to multimodal models (vision, audio) through [optimum](https://github.com/huggingface/optimum-executorch) .
## Quantization
Quantization is quickly emerging as the standard for state-of-the-art model development. Many SOTA models are now released in low-precision formats such as 8-bit and 4-bit (e.g., gpt-oss, Kimi-K2, Deepseek-r1), hardware is increasingly optimized for low-precision workloads, and the community is actively sharing high-quality quantized checkpoints. In v5, we're making quantization a central focus of Transformers support, ensuring full compatibility with all major features, and delivering a reliable framework for training and inference.
We introduce a significant change to the way we load weights in our models; and with this, we move to quantization being a first-class citizen.
Our collaboration with the Transformers team was highly productive, marked by their proactive code reviews,
feedback, and technical expertise. Their support was crucial in integrating TorchAO, expanding quantization
features, and improving documentation for broader adoption in the V5.
-- Jerry Zhang at TorchAO
We're excited that v5 has made quantization a first-class citizen. It provides the foundation for bitsandbytes to better support key features like TP and MoEs, and also makes it easier to integrate new quantization methods.
-- Matthew Douglas & Titus von Koeller, bitsandbytes
## Conclusion
The overarching theme of this version 5 release is “interoperability”. All refactors, performance improvements, and standardization are aligned with this theme. v5 plays nicely and end-to-end with the growing ecosystem: train a model with Unsloth/Axolotl/LlamaFactory/MaxText deploy it with vLLM/SGLang, and export it to llama.cpp/executorch/MLX to run locally!
Version 5 is undeniably an accomplishment of the past five years by a very large number of people in our community. We also see it as a promise, and as a beacon of the direction we want to go.
We took it as an opportunity to clean up the toolkit and isolate what mattered; we now have a clean slate on top of which to build. Thanks to the many changes from the community and team, improvements in performance, usability, and readability, will be simpler to ship.
Now that v5.0.0's first RC is out there, we'll be eagerly awaiting your feedback. Please check our [release notes](https://github.com/huggingface/transformers/releases/tag/v5.0.0rc0) for all the technical details, and we'll be awaiting your feedback in our [GitHub issues](https://github.com/huggingface/transformers/issues) !
HTML_TAG_END
More Articles from our Blog
[announcement open-source community
## OpenEnv in Practice: Evaluating Tool-Using Agents in Real-World Environments
+1 30 February 12, 2026 christian-washington, et. al.](/blog/openenv-turing) [huggingface_hub python announcement
## huggingface_hub v1.0: Five Years of Building the Foundation of Open Machine Learning
75 October 27, 2025 Wauplin, et. al.](/blog/huggingface-hub-v1)
### Community
[julien-c](/julien-c) [Dec 1, 2025](#692ddf683b93883b48f7ca14)
first 🔥
🔥 20 20 🚀 3 3 + Reply [LagOps](/LagOps) [Dec 1, 2025](#692ded6d3251be9f74d4e660)
Awesome work, great job of maintaining the library, it sure isn't easy with all the different architectures and advancements.
See translation ❤️ 3 3 + Reply [itseffi](/itseffi) [Dec 1, 2025](#692df76a9bd61fb1eb8e1c4f)
👏👏👏
🤝 1 1 + Reply [ashim](/ashim) [Dec 1, 2025](#692dfce2a7a22508e43fe73c)
letssss gooooo
🚀 1 1 + Reply [victordata](/victordata) [Dec 1, 2025](#692dffb09bd61fb1eb8e1c57)
Congratulations for the launch !
See translation 🤗 1 1 + Reply [saggl](/saggl) [Dec 1, 2025](#692e03e9d80bcfd34e86e35a)
🤗🥳
🤗 1 1 + Reply [yukiarimo](/yukiarimo) [Dec 1, 2025](#692e2133e690cfc4e59d41bb)
No, you don’t! It’s just pure control. Researchers would love to have direct PyTorch only access to the architecture, training, and inference code for different models without this Transformers blab
See translation Reply [Neilblaze](/Neilblaze) [Dec 1, 2025](#692e21b967a78011a47bfedb)
🤗💛
❤️ 1 1 + Reply [mst272](/mst272) [Dec 2, 2025](#692ea91ea7a22508e43fe757)
🤗！！！
🤗 1 1 + Reply [Avihu](/Avihu) [Dec 2, 2025](#692eec87f0eec284ae5e39cf)
awesome work!
🚀 1 1 + Reply [firobeid](/firobeid) [Dec 3, 2025](#69306b3056521c4665f3989c)
Is there a spaces link to the " Transformers Models Timeline" dashboard? I want to add it in my course
See translation Reply [melvindave](/melvindave) [Dec 8, 2025](#69372f65b8f3ce7a697f03f4)
will transformers serve be a good enough alternative to llama.cpp?
See translation Reply [cveavy](/cveavy) [Jan 7](#695e9209bb2a202de9ae6352)
Very nice to have V5.
See translation Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Ftransformers-v5) or [log in](/login?next=%2Fblog%2Ftransformers-v5) to comment
[Upvote 302](/login?next=%2Fblog%2Ftransformers-v5) +290 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe