[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START We got Claude to teach open models how to write CUDA kernels! HTML_TAG_END
Published January 28, 2026 [Update on GitHub](https://github.com/huggingface/blog/blob/main/upskill.md) [Upvote 144](/login?next=%2Fblog%2Fupskill) +138 [ben burtenshaw burtenshaw Follow](/burtenshaw) [shaun smith evalstate Follow](/evalstate) [merve merve Follow](/merve) [Pedro Cuenca pcuenq Follow](/pcuenq) HTML_TAG_START
[HTML_TAG_START What are agent skills? HTML_TAG_END](#what-are-agent-skills) [HTML_TAG_START 1. Get the teacher (Claude Opus 4.5) to build a kernel HTML_TAG_END](#1-get-the-teacher-claude-opus-45-to-build-a-kernel) [HTML_TAG_START 2. Make an agent skill from the trace HTML_TAG_END](#2-make-an-agent-skill-from-the-trace) [HTML_TAG_START 3. Take your skill to an open source, smaller, or cheaper model HTML_TAG_END](#3-take-your-skill-to-an-open-source-smaller-or-cheaper-model) [HTML_TAG_START Deep dive tutorial into building kernels with agent skills HTML_TAG_END](#deep-dive-tutorial-into-building-kernels-with-agent-skills) [HTML_TAG_START Setup and Install HTML_TAG_END](#setup-and-install) [HTML_TAG_START Skill Generation HTML_TAG_END](#skill-generation) [HTML_TAG_START Generate the Skill HTML_TAG_END](#generate-the-skill) [HTML_TAG_START Evaluate on a Different Model HTML_TAG_END](#evaluate-on-a-different-model) [HTML_TAG_START How the evaluation in upskill works HTML_TAG_END](#how-the-evaluation-in-upskill-works) [HTML_TAG_START What's Next HTML_TAG_END](#whats-next) [HTML_TAG_START Resources HTML_TAG_END](#resources) The best thing about agent skills is upskilling your agents on hard problems. There are two ways to look at that:
You can take Opus 4.5 or other SOTA models and tackle the hardest problems out there. You can take models that run on your laptop and upskill them to harder problems. In this blog post, we’ll show you how to take on the latter.
This blog post walks through the process of using a new tool, upskill , to generate and evaluate agent skills with large models and use them with smaller models. We will benchmark upskill on the task of writing CUDA kernels for [diffusers](https://huggingface.co/docs/diffusers/en/index) models, but the process is generally useful for cutting costs, or using smaller models on hard and domain-specific problems.
## What are agent skills?
In case you missed it, agent skills are taking the coding agent game by storm. In fact, they’re a straightforward concept to define model context as files, like instructions as markdown and code as scripts. The file format makes them easy to generate, share, and review. In short, they’re a practical medium to share capabilities across models and tools, and they're most useful in specific domains or hard problems. Not stuff the model can do well anyway.
This post showcases this process by using Claude to generate a Skill file that can be used by open source models for a complex and specialized task: write CUDA kernels.
We first tried a simple skill based on existing documentation, and we found that it improved performance for some others, but not all. In fact, it could even degrade performance or increase token usage for some models. Check out the plot below to see the performance of the model with and without the basic skill.
Now, let's walk through how you can use upskill to upskill your agents on hard problems, and measure performance.
## 1. Get the teacher (Claude Opus 4.5) to build a kernel
First, we use Claude Code to build a kernel interactively and export the trace. We worked through the process by instructing, validating, and adding documentation links. This somewhat naive process is important to reveal the models' initial challenges. In fact, you can iterate on this multiple times, by trying to solve the task with draft versions of the skill, and experimenting with smaller models. Each time, you can instruct the agent to improve the skill and test it on the smaller model.
Here's an [example of the skill](https://huggingface.co/hf-skills/h100-diffusers-kernel-builder) that we created and have been using to build kernels. We started from this [agent trace](https://huggingface.co/hf-skills/h100-diffusers-kernel-builder/blob/main/agent-trace.txt) where the agent was able to build a kernel, but not without some help.
## 2. Make an agent skill from the trace
Once the teacher model has performed the task, we need them to make a skill. There are a number of effective ways to do this.
Within the same session, instruct the agent to create a skill file for the task it just completed. Use [Anthropic ‘skill creator’ skill](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md) either within the agent session or with an exported trace and a new agent session. Use the upskill tool to create a skill based on the trace.
In most cases, the first 2 options result in functional skills. However, the performance of an agent with the skill is unknown. That’s where upskill is useful, because it will also generate test cases for your skill based on the trace. It then compares the results under both scenarios: using the trace, or applying the skill. We see below that the original model (Claude Opus) met the same performance with and without the skill. This means the skill captured the task for this model . Great!
## 3. Take your skill to an open source, smaller, or cheaper model
Finally, we need to transfer our newly created skill to the tool or model we intend to use. Most tools like codex , cursor , and opencode have settled on a consistent format for skills, which is a directory at {agent}/skills/{skill_name}/SKILL.md , so we just need to copy the skill directory to this location.
With upskill we can pass a skill and a set of models to the eval command and upskill will run the test cases on those models with and without the skill to compare performance. We can see here that the skill increases accuracy on some open models, but not on all.
In this case, we might want to iterate further on the gpt-oss skills by regenerating the skill. We can do upskill generate --from {skill} .
There is more to agent skills than model performance. Often agents can reach a given accuracy with or without a skill, they just need to consume more tokens to get there. For recurring tasks, we want to optimize agents to use less tokens to achieve the same accuracy. The results below reveal another dimension to the skill. Some models are significantly reducing their performance token usage, whilst others are using more tokens with the skill. For example, with [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) the skill is clearly effective in terms of accuracy and token usage. However, for Claude Opus 4.5 there is no clear performance increase and an increase in token usage, so you would not want to use this skill with Claude Opus 4.5.
tldr; try out and evaluate models with the skills you create. Use upskill eval or a similar tool to evaluate the models performance with and without skills.
That’s the high level end to end of upskilling your coding agents on hard problems. Try out upskill now like this:
# install upskill pip install upskill # or use uvx uvx upskill --help # generate a skill based on an agent trace upskill generate "write nvidia kernels" --from ./trace.md # evaluate models on a skill upskill eval ./skills/my-skill/ --model haiku --model sonnet # generate skills for local models upskill generate "parse YAML"
--model opus
--eval-model "unsloth/GLM-4.7-Flash-GGUF:Q4_0"
--eval-base-url http://localhost:8080/v1
## Deep dive tutorial into building kernels with agent skills
We have a high level understanding of how we can upskill an agent. Let’s now look at the use case we solved for writing CUDA kernels.
We didn’t just want to write kernel code, but understand the full kernel-builder workflow: project structure, build.toml configuration, architecture-specific optimizations, and PyTorch bindings. This tutorial shows how upskill creates validated skills that actually work.
The [kernel-builder-cuda-kernels](https://github.com/burtenshaw/kernel-skill) skill teaches Claude everything it needs to know about CUDA development: which GPU architecture to target, how to structure a kernel-builder project, when to use shared memory versus registers, and how to write PyTorch bindings.
With this skill, you can tell Claude things like:
Build a fused LayerNorm + GELU kernel optimized for H100.
And Claude will create the complete project structure, CUDA implementation, and build configuration—following the exact conventions that kernel-builder expects.
This isn't about generating boilerplate. The skill encodes domain expertise: H100 uses compute capability 9.0, shared memory should be aligned to 128 bytes, async memory copies require __CUDA_ARCH__ >= 900 . Knowledge that would take hours to gather from documentation gets packaged into ~500 tokens that load on demand.
## Setup and Install
Install upskill:
pip install upskill # or use uvx for one-off runs uvx upskill --help
Set your API key:
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...
That's it. Upskill uses Anthropic Claude Opus-4.5 model by default but also supports OpenAI and local models via OpenAI-compatible endpoints as generators. We want to use the more expensive and higher quality models to generate skills, and the smaller ones to use them. Think robin hood.
## Skill Generation
Let's walk through generating a skill that teaches agents how to build CUDA kernels with HuggingFace's [kernels](https://github.com/huggingface/kernels) library.
### Generate the Skill
Start with a clear task description:
upskill generate "build optimized CUDA kernels for PyTorch using HuggingFace kernel-builder"
Above we used upskill, but it could in fact be any agent or chat tool and an exported trace.
upskill generate "write kernels" --from <agent-trace>.md
Also, we could start from an existing skill and add to it:
upskill generate "add more error handling and edge cases"
--from ./skills/kernel-builder-cuda-kernels/
upskill loads the existing skill, applies your improvements, and re-evaluates to ensure the changes help.
upskill creates a skill, generates test cases, evaluates performance, and refines based on failures:
Generating skill with sonnet...
Generating test cases...
Evaluating on sonnet... (attempt 1)
60% -> 95% (+35%) OK
kernel-builder-cuda-kernels
Build optimized CUDA kernels for PyTorch using HuggingFace kernel-builder.
SKILL.md ~520 tokens
baseline ████████████ 60%
with skill ███████████████████ 95% (+35%)
Saved to ./skills/kernel-builder-cuda-kernels
The baseline shows how the model performs without any skill. The "with skill" result shows performance after the skill is injected into context. A 35% improvement means the skill is working.
The skill is saved as a directory following the [Agent Skills specification](https://agentskills.io) :
./skills/kernel-builder-cuda-kernels/
├── SKILL.md # Main instructions (~520 tokens)
└── skill_meta.json # Metadata and test cases Open `SKILL.md` to see what upskill generated: ---
name: kernel-builder-cuda-kernels
description: Build optimized CUDA kernels for PyTorch using HuggingFace kernel-builder.
---
# Building CUDA Kernels with kernel-builder
## Overview
This guide explains how to create optimized CUDA kernels for PyTorch models
using HuggingFace's kernel-builder. It covers project setup, kernel implementation,
and building for specific GPU architectures like NVIDIA H100.
## Project Structure
project/
├── build.toml # Build configuration
├── kernel_src/ # CUDA kernel implementations
│ ├── attention.cu
│ ├── layernorm.cu
│ └── geglu.cu
└── torch-ext/ # PyTorch C++ bindings
└── torch_binding.cpp
## Build Configuration
Create `build.toml` to define your kernel package:
[general]
name = "diffuser_kernels"
backends = ["cuda"]
[general.cuda]
# H100 is compute capability 9.0
capabilities = ["9.0"]
...
### Evaluate on a Different Model
The important test is: does this skill help local or cheaper models to build kernels?
# Start a local OpenAI-compatible server with a web UI: llama-server -hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M # Evaluate on local model (llama.cpp server) upskill eval ./skills/my-skill/
--model "unsloth/GLM-4.7-Flash-GGUF:Q4_0"
--base-url http://localhost:8080/v1 Generating skill with sonnet...
Generating test cases...
Evaluating on "unsloth/GLM-4.7-Flash-GGUF:Q4_0"... (attempt 1)
40% -> 85% (+45%) OK
baseline ████████░░░░░░░░░░░░ 40%
with skill █████████████████░░░ 85% (+45%)
Saved to ./skills/kernel-builder-cuda-kernels
A 45% improvement on "unsloth/GLM-4.7-Flash-GGUF:Q4_0" means the skill successfully transfers domain knowledge from a capable model to a faster, cheaper one. Skills that work on weaker models will definitely work on stronger ones.
This is the core value proposition: use expensive models to create skills, then deploy those skills with cheap or local models.
## How the evaluation in upskill works
upskill uses a teacher-student approach to evaluate models where the teacher model generates test cases for the student model to be evaluated on.
Teacher model (Opus) generates the skill Test cases (Opus) are generated automatically from the task description Student model (local) is evaluated with and without the skill Skill lift measures the improvement
If you pass an existing skill to upskill eval , it will generate test cases for the skill and evaluate the model on them. Test cases are simple input/output pairs that verify the agent understands the task:
{ "cases" : [ { "input" : "Create a build.toml for a CUDA kernel targeting H100" , "expected" : { "contains" : "9.0" } } , { "input" : "Write a basic CUDA kernel template with proper includes" , "expected" : { "contains" : "cuda_runtime.h" } } ] }
We can also test how a skill performs across different models:
upskill eval ./skills/kernel-builder-cuda-kernels/
--model haiku --m kimi --runs 5 Evaluating kernel-builder-cuda-kernels across 2 model(s)
3 test case(s), 5 run(s) per model
haiku
Pass rate: 4/5 (80%) Avg assertions: 2.8/3
sonnet
Pass rate: 5/5 (100%) Avg assertions: 3.0/3
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Model ┃ Pass Rate ┃ Avg Assertions ┃ Avg Tokens ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ haiku │ 4/5 │ 2.8/3 │ 1250 │
│ kimi │ 5/5 │ 3.0/3 │ 1890 │
└────────┴───────────┴────────────────┴────────────┘
This helps you find the cost-performance sweet spot: maybe Haiku with the skill is good enough for your use case, saving significant API costs.
## What's Next
We've shown that upskill can create validated skills that transfer domain expertise from powerful models to cheaper ones. The kernel-builder skill is just one example of what's possible.
Some things to try:
Generate skills for your internal tools Build a skill library for your codebase Capture tribal knowledge Benchmark across models
The approach works for any specialized task where you'd otherwise write detailed prompts repeatedly. Skills are portable across [Claude Code](https://docs.anthropic.com/en/docs/claude-code) , [Codex](https://github.com/openai/codex) , [Cursor](https://cursor.com/docs/context/skills) , and other tools that support the [Agent Skills specification](https://agentskills.io) .
## Resources
[Upskill repo](https://github.com/huggingface/upskill) [Agent Skills Specification](https://agentskills.io) [HuggingFace kernel-builder](https://github.com/huggingface/kernels/tree/main/builder) HTML_TAG_END
More Articles from our Blog
[open-source cuda kernels
## Custom Kernels for All from Codex and Claude
64 February 13, 2026 burtenshaw, et. al.](/blog/custom-cuda-kernels-agent-skills) [agents gui vlm
## Smol2Operator: Post-Training GUI Agents for Computer Use
+1 137 September 23, 2025 A-Mahla, et. al.](/blog/smol2operator)
### Community
[NJX-njx](/NJX-njx) [Jan 29](#697b1b26ed634b59769093d0)
This article is very inspiring to me.
Since skills have become a great boost to the improvement of model capabilities, can we try to distill skills, just like we did model distillation before? I think this can be achieved through multiple iterations. The current functions of upskill are actually quite complete, but I wonder if we can try to make it generate a compatibility matrix between multiple skills, so that the combined effect is greater than the sum of the parts. In addition, Model A generates skills, and Model B looks for counterexamples, so that they can evolve together. See translation ❤️ 3 3 🔥 1 1 + Reply [ClementeH](/ClementeH) [Jan 29](#697b7c3e0812deb10efeedcc)
Great post! Haven't tried upskill yet, but I'll give it a shot. I think the agent trace link is broken — it points to trace.md but the actual file is agent-trace.txt: [https://huggingface.co/hf-skills/h100-diffusers-kernel-builder/blob/main/agent-trace.txt](https://huggingface.co/hf-skills/h100-diffusers-kernel-builder/blob/main/agent-trace.txt)
See translation 3 replies · [pcuenq](/pcuenq) Article author [Jan 29](#697bb4253463b1f0246b6338)
Thank you [@ ClementeH](/ClementeH) ! Would you like to open a PR to fix it? :) If you don't have time we'll do it!
See translation Expand 2 replies [one-man-won](/one-man-won) [28 days ago](#698081acfdf9393f758115b2)
Thanks for this article! Can this be replicated to use open-source models such as Qwen/Qwen3-30B-A3B-Thinking-2507 to generate the agent trace and the skill file? From my initial exploration, I assume it requires an API key from Anthropic to generate and evaluate skills.
See translation 1 reply · [evalstate](/evalstate) Article author [27 days ago](#6980b17491589894cb3c9159) • [edited 27 days ago](#6980b17491589894cb3c9159)
It can -- the only place where a model string is included is here: [https://github.com/huggingface/upskill/blob/16a688fa32e03ee6859f01f3e13f23986e67006f/src/upskill/agent_cards/test_gen.md?plain=1#L5](https://github.com/huggingface/upskill/blob/16a688fa32e03ee6859f01f3e13f23986e67006f/src/upskill/agent_cards/test_gen.md?plain=1#L5) where we use Opus' structured output to generate test cases. All else should be configurable from the --model switch.
See translation [aim143](/aim143) [27 days ago](#6980d615e47946b78717eb85)
We need to talk about the 'magic' behind Claude’s CUDA kernels. Is it superior synthetic data, or did Anthropic find a better way to teach LLMs hardware-level logic? Open to all technical theories
See translation Reply [Insuuuu](/Insuuuu) [26 days ago](#69834a30f5cf32fd8db79fca)
Upskill generate worked mostly as intended, but when I tried to do the actually interesting part and ran upskill eval , everything started falling apart.
A few concrete issues I hit (Windows + Ubuntu/WSL):
Skill path chaos:
Ubuntu: Access denied … SKILL.md is not within an allowed skill directory even when the file exists. Windows: Path must be absolute… and then fast-agent tries to read_skill from /tmp/.../SKILL.md (!), which is “absolute” on Linux but not on Windows. I ended up chasing /skills, /tmp/skills, .fast-agent/skills, .claude/skills, absolute paths… nothing was consistently accepted across runs.
Model selection confusion / fast-agent flag collision
-m means something else in fast-agent, so the tool evaluates the wrong thing unless you use the right flag. It’s very easy to think you’re running Haiku/Sonnet but you’re not. -m haiku actually made the program write a literal haiku 😭😭
Provider / model parameter mismatches
OpenAI GPT-5 models error because the request uses max_tokens instead of max_completion_tokens (hard 400).
Weird UI/logging behavior
Terminal output spams repeated “Chatting … turn 1” lines, then never prints the result table even though run logs exist.
Token usage
Can blow up fast (I saw huge output spikes)
And in at least one case the “agents” wrote a bunch of .md artifacts into my actual filesystem, which was scary.
Overall: the concept is awesome, but eval is super brittle right now. Did anyone get eval actually working reliably?
See translation 2 replies · [evalstate](/evalstate) Article author [25 days ago](#69835fd685cfc86683705fc9)
Hi --
The most reliable way to do this is by setting the skills directory - I'll do a few runs to confirm - the system prompt should specify the absolute path ({{availableSkills}} expands to this) . Noted 🤗 gpt-5-nano is mapped to responses; I'll add the date alias but to use it directly specify responses.gpt-5-nano-2025-08-07 . I think this ships with that display off by default. You see so many as we run in parallel for performance. I'll make this switchable and ensure that it doesn't collide with the console output. Different runs may have wildly different token usage depending on the trajectory - will take a look at making this easier to report.
Thanks for the feedback - we're planning a sweep of PR/Issues early next week, and considering incorporating a sandbox mechanism!
See translation ❤️ 1 1 + Expand 1 reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Fupskill) or [log in](/login?next=%2Fblog%2Fupskill) to comment
[Upvote 144](/login?next=%2Fblog%2Fupskill) +132 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe