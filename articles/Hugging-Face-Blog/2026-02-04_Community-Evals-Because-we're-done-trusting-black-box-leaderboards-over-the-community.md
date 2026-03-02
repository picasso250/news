[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START Community Evals: Because we're done trusting black-box leaderboards over the community HTML_TAG_END
Published February 4, 2026 [Update on GitHub](https://github.com/huggingface/blog/blob/main/community-evals.md) [Upvote 82](/login?next=%2Fblog%2Fcommunity-evals) +76 [ben burtenshaw burtenshaw Follow](/burtenshaw) [Nathan Habib SaylorTwift Follow](/SaylorTwift) [Bertrand Chevrier kramp Follow](/kramp) [merve merve Follow](/merve) [Daniel van Strien davanstrien Follow](/davanstrien) [Niels Rogge nielsr Follow](/nielsr) [Julien Chaumond julien-c Follow](/julien-c) HTML_TAG_START
[HTML_TAG_START Evaluation is broken HTML_TAG_END](#evaluation-is-broken) [HTML_TAG_START What We're Shipping HTML_TAG_END](#what-were-shipping) [HTML_TAG_START Why This Matters HTML_TAG_END](#why-this-matters) [HTML_TAG_START Get Started HTML_TAG_END](#get-started)
TL;DR: Benchmark datasets on Hugging Face can now host leaderboards. Models store their own eval scores. Everything links together. The community can submit results via PR. Verified badges prove that the results can be reproduced.
## Evaluation is broken
Let's be real about where we are with evals in 2026. MMLU is saturated above 91%. GSM8K hit 94%+. HumanEval is conquered. Yet some models that ace benchmarks still can't reliably browse the web, write production code, or handle multi-step tasks without hallucinating, based on usage reports. There is a clear gap between benchmark scores and real-world performance.
Furthermore, there is another gap within reported benchmark scores. Multiple sources report different results. From Model Cards, to papers, to evaluation platforms, there is no alignment in reported scores. The result is that the community lacks a single source of truth.
## What We're Shipping
Decentralized and transparent evaluation reporting.
We are going to take evaluations on the Hugging Face Hub in a new direction by decentralizing reporting and allowing the entire community to openly report scores for benchmarks. At first, we will start with a shortlist of 4 benchmarks and over time we’ll expand to the most relevant benchmarks.
For Benchmarks: Dataset repos can now register as benchmarks ( [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) , [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) , [HLE](https://huggingface.co/datasets/cais/hle) are already live). They automatically aggregate reported results from across the Hub and display leaderboards in the dataset card. The benchmark defines the eval spec via eval.yaml , based on the [Inspect AI](https://inspect.aisi.org.uk/) format, so anyone can reproduce it. The reported results need to align with the task definition.
For Models: Eval scores live in .eval_results/*.yaml in the model repo. They appear on the model card and are fed into benchmark datasets. Both the model author’s results and open pull requests for results will be aggregated. Model authors will be able to close score PR and hide results.
For the Community: Any user can submit evaluation results for any model via a PR. Results get shown as "community", without waiting for model authors to merge or close. The community can link to sources like a paper, Model Card, third-party evaluation platform, or inspect eval logs. The community can discuss scores like any PR. Since the Hub is Git based, there is a history of when evals were added, when changes were made, etc. The sources look like below.
To learn more about evaluation results, check out the [docs](https://huggingface.co/docs/hub/eval-results) .
Model scores in the Hub
## Why This Matters
Decentralizing evaluation will expose scores that already exist across the community in sources like model cards and papers. By exposing these scores, the community can build on top of them to aggregate, track, and understand scores across the field. Also, all scores will be exposed via Hub APIs, making it easy to aggregate and build curated leaderboards, dashboards, etc.
Community evals do not replace benchmarks so leaderboards and closed evals with published results are still crucial. However, we believe it's important to contribute to the field with open eval results based on reproducible eval specs.
This won't solve benchmark saturation or close the benchmark-reality gap. Nor will it stop training on test sets. But it makes the game visible by exposing what is evaluated, how, when, and by whom.
Mostly, we hope to make the Hub an active place to build and share reproducible benchmarks. Particularly focusing on new tasks and domains that challenge SOTA models more.
## Get Started
Read the docs: To learn more about evaluation results, check out the [docs](https://huggingface.co/docs/hub/eval-results) .
Add eval results: Publish the evals you conducted as YAML files in .eval_results/ on any model repo.
Check out the scores on the [benchmark dataset](https://huggingface.co/datasets?benchmark=benchmark:official&sort=trending) .
Register a new benchmark: Add eval.yaml to your dataset repo and [contact us to be included in the shortlist.](https://huggingface.co/spaces/OpenEvals/README/discussions/2)
The feature is in beta. We're building in the open. [Feedback welcome.](https://huggingface.co/spaces/OpenEvals/README/discussions/1)
HTML_TAG_END
More Articles from our Blog
[swift hub open-source
## Introducing swift-huggingface: The Complete Swift Client for Hugging Face
mattt 43 December 5, 2025 mattt](/blog/swift-huggingface) [open-source LLM community
## 🇵🇭 FilBench - Can LLMs Understand and Generate Filipino?
+5 23 August 12, 2025 ljvmiranda921, et. al.](/blog/filbench)
### Community
deleted [24 days ago](#698564ce944e460ededeede3) This comment has been hidden [mlabonne](/mlabonne) [23 days ago](#69860262b5e32178c2e20721)
Great initiative, aggregating multiple signals is the way to go!
See translation ❤️ 3 3 + Reply [NJX-njx](/NJX-njx) [23 days ago](#6986c939e1b677c5df5c3afb)
Although such a measure has not solved the problems encountered in the current evaluation, at least it is indeed a very good measure in terms of decentralization and mobilizing the power of the community for co-construction.
See translation Reply [naufalso](/naufalso) [22 days ago](#69880808260cce8b14247341) • [edited 22 days ago](#69880808260cce8b14247341)
Will there be the integration with existing huggingface [lighteval](https://github.com/huggingface/lighteval) ?
See translation Reply [SaylorTwift](/SaylorTwift) Article author [21 days ago](#6989dfe0573fbffd10a8fbd6)
hi [@ naufalso](/naufalso) ! Lighteval now suport inspect-ai as a backend, so everything supported by inspect is integrrated in lighteval 🔥
See translation 👍 1 1 + Reply [hengloose](/hengloose) [20 days ago](#698a04bae23462f322613301)
Amazing
Reply [MatthewFrank](/MatthewFrank) [20 days ago](#698a283c3add708a82c1f903)
This is such an important initiative for transparency in model evaluation! Building trustworthy evaluation infrastructure requires careful architectural design. For anyone building evaluation systems or ML infrastructure, clear documentation is critical. I've been using InfraSketch ( [https://www.infrasketch.net/](https://www.infrasketch.net/) ) to document our evaluation pipelines—you describe the system architecture in plain English and get visual diagrams that you can iterate on conversationally. Makes it much easier to communicate how evaluation systems work and maintain documentation as they evolve.
See translation Reply [harshakokel](/harshakokel) [18 days ago](#698cc3459017bb9609f24fc4) • [edited 18 days ago](#698cc3459017bb9609f24fc4)
This is a very important and timely initiative. It’s easy to get lost in the sea of leaderboards, each with its own format and reporting style. The Inspect AI log format brings much‑needed standardization, and having Hugging Face host evaluation logs is a real game changer. One reason many valuable benchmarks fade away is that original contributors often lack the resources to continuously maintain leaderboards. The Community Evals initiative has tremendous potential to address this gap, and I truly appreciate the effort behind it.
We’re hoping to include our planning benchmark, [ACPBench](https://ibm.github.io/ACPBench/index.html) , as part of this ecosystem—it's fully compatible with Inspect AI, the [evaluation scripts](https://github.com/IBM/ACPBench/blob/main/GettingStarted.md) are available on our GitHub.
### References
ACPBench: Reasoning About Action, Change, and Planning, Harsha Kokel, Michael Katz, Kavitha Srinivas, Shirin Sohrabi , [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34857) ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning, Harsha Kokel, Michael Katz, Kavitha Srinivas, Shirin Sohrabi , [ICLR 2026](https://openreview.net/forum?id=WIXohR7mEo) See translation ➕ 1 1 + Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Fcommunity-evals) or [log in](/login?next=%2Fblog%2Fcommunity-evals) to comment
[Upvote 82](/login?next=%2Fblog%2Fcommunity-evals) +70 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe