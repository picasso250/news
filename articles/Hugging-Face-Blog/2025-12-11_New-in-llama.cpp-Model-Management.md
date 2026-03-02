[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START New in llama.cpp: Model Management HTML_TAG_END
[Team Article](/blog) Published December 11, 2025 [Upvote 126](/login?next=%2Fblog%2Fggml-org%2Fmodel-management-in-llamacpp) +120 [Xuan-Son Nguyen ngxson Follow](/ngxson) [ggml-org](/ggml-org) [Victor Mustar victor Follow](/victor) [ggml-org](/ggml-org) HTML_TAG_START
[HTML_TAG_START Quick Start HTML_TAG_END](#quick-start) [HTML_TAG_START Features HTML_TAG_END](#features) [HTML_TAG_START Examples HTML_TAG_END](#examples) [HTML_TAG_START Chat with a specific model HTML_TAG_END](#chat-with-a-specific-model) [HTML_TAG_START List available models HTML_TAG_END](#list-available-models) [HTML_TAG_START Manually load a model HTML_TAG_END](#manually-load-a-model) [HTML_TAG_START Unload a model to free VRAM HTML_TAG_END](#unload-a-model-to-free-vram) [HTML_TAG_START Key Options HTML_TAG_END](#key-options) [HTML_TAG_START Also available in the Web UI HTML_TAG_END](#also-available-in-the-web-ui) [HTML_TAG_START Join the Conversation HTML_TAG_END](#join-the-conversation) [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) now ships with router mode , which lets you dynamically load, unload, and switch between multiple models without restarting.
Reminder: llama.cpp server is a lightweight, OpenAI-compatible HTTP server for running LLMs locally.
This feature was a popular request to bring Ollama-style model management to llama.cpp. It uses a multi-process architecture where each model runs in its own process, so if one model crashes, others remain unaffected.
## Quick Start
Start the server in router mode by not specifying a model :
llama-server
This auto-discovers models from your llama.cpp cache ( LLAMA_CACHE or ~/.cache/llama.cpp ). If you've previously downloaded models via llama-server -hf user/model , they'll be available automatically.
You can also point to a local directory of GGUF files:
llama-server --models-dir ./my-models
## Features
Auto-discovery : Scans your llama.cpp cache (default) or a custom --models-dir folder for GGUF files On-demand loading : Models load automatically when first requested LRU eviction : When you hit --models-max (default: 4), the least-recently-used model unloads Request routing : The model field in your request determines which model handles it
## Examples
### Chat with a specific model
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{ "model": "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M", "messages": [{"role": "user", "content": "Hello!"}] }'
On the first request, the server automatically loads the model into memory (loading time depends on model size). Subsequent requests to the same model are instant since it's already loaded.
### List available models
curl http://localhost:8080/models
Returns all discovered models with their status ( loaded , loading , or unloaded ).
### Manually load a model
curl -X POST http://localhost:8080/models/load \
-H "Content-Type: application/json" \
-d '{"model": "my-model.gguf"}'
### Unload a model to free VRAM
curl -X POST http://localhost:8080/models/unload \
-H "Content-Type: application/json" \
-d '{"model": "my-model.gguf"}'
## Key Options
| Flag | Description |
|---|---|
| --models-dir PATH | Directory containing your GGUF files |
| --models-max N | Max models loaded simultaneously (default: 4) |
| --no-models-autoload | Disable auto-loading; require explicit /models/load calls |
All model instances inherit settings from the router:
llama-server --models-dir ./models -c 8192 -ngl 99
All loaded models will use 8192 context and full GPU offload. You can also define per-model settings using [presets](https://github.com/ggml-org/llama.cpp/pull/17859) :
llama-server --models-preset config.ini [my-model] model = /path/to/model.gguf ctx-size = 65536 temp = 0.7
## Also available in the Web UI
The [built-in web UI](https://github.com/ggml-org/llama.cpp/tree/master/tools/server/webui) also supports model switching. Just select a model from the dropdown and it loads automatically.
## Join the Conversation
We hope this feature makes it easier to A/B test different model versions, run multi-tenant deployments, or simply switch models during development without restarting the server.
Have questions or feedback? Drop a comment below or open an issue on [GitHub](https://github.com/ggml-org/llama.cpp/issues) .
HTML_TAG_END
### Community
[bukit](/bukit) [Dec 11, 2025](#693b17fb30a8fb1087cfa5f0)
Mmproj support?
4 replies · 👀 2 2 👍 1 1 + [sbeltz](/sbeltz) [Dec 12, 2025](#693bd7e5429f88442aba1ae0)
Supported via presets.ini, where you can specify the mmproj (and other long and short arguments) per model.
See translation 🔥 3 3 + Expand 3 replies [sbeltz](/sbeltz) [Dec 12, 2025](#693bd9ffdb3bf2535fd0cf48)
Awesome new feature! Can model selection be done on something other than requested model name? Like maybe specify the ranking in presets.ini, and then the highest ranked model that can satisfy the request will be the default. So maybe one model is best for short context, another (or the same with other settings) for when the context gets too long, and another when image input is required.
See translation ➕ 2 2 + Reply [xbruce22](/xbruce22) [Dec 12, 2025](#693c387ddb3bf2535fd0cf56)
This is good addition, Thank you.
See translation 👍 1 1 + Reply [etemiz](/etemiz) [Dec 12, 2025](#693c57ea6107ec9c17bb2879) • [edited Dec 12, 2025](#693c57ea6107ec9c17bb2879)
what is the best way to get <think> </think> and the tokens in between? openAI library is removing them.. i want to run llama-server in console and talk to it using a python library that does not remove the thinking tokens.
i checked the llama-cpp-python but it does not have that.
See translation 1 reply · [xbruce22](/xbruce22) [Dec 16, 2025](#69418941cd121096018fdaed)
llama-server by default in most implementation keeps the reasoning content in reasoning_content variable in response attribute. You can get it from there. Otherwise use reasoning-format flag and pass DeepSeek value to get pure tokens
See translation ❤️ 2 2 + [razvanab](/razvanab) [Dec 13, 2025](#693cd851a1d453e27f52b22c)
Now I can use llama.cpp all the time. A big thank you to the devs.
See translation 😎 1 1 + Reply [sbeltz](/sbeltz) [Dec 13, 2025](#693ce302a45cf6ced1783833)
Is there currently a way to have a "default" model if the request doesn't specify? Could be the currently loaded model or a specific model. (Just noticed one of my apps broke because it's used to llama-server not requiring a model name.)
See translation 1 reply · ➕ 1 1 + [milksteak1111](/milksteak1111) [Jan 14](#69670316648f6122d5acfb82)
# This seems to work
[DEFAULT] port = 8080 n-gpu-layers = -1 device = 0 flash-attn = on chat-template = jinja models-max = 4
See translation [eribob](/eribob) [Dec 14, 2025](#693eada2a45cf6ced1783860)
Does it unload the current model if VRAM is full, to allow swapping to a new model?
See translation 👀 1 1 👍 1 1 + Reply [21world](/21world) [Dec 15, 2025](#694037884d3a55b8d4ec7c65)
fun ideas , add personal avatar and p2p social network also emule p2p models storage
See translation Reply [21world](/21world) [Dec 15, 2025](#6940387bc7e128b6723e5798) This comment has been hidden (marked as Off-Topic) [JLouisBiz](/JLouisBiz) [Dec 26, 2025](#694e6381cd7bb2956d912b9a)
Hey there! Just wanted to drop a quick note saying I'm really digging the new router mode in llama.cpp server. It's a game-changer for me, especially when I need to switch between different models. The auto-discovery of models and LRU eviction is pretty neat – no more manual updates or restarts needed. It's like having a dynamic model manager on-the-fly. And the request routing part? Brilliant! Makes my workflow with dmenu smoother. Check out the full experience and check out my dmenu launcher script on the project's GitHub: [https://gitea.com/gnusupport/LLM-Helpers/src/branch/main/bin/rcd-llm-dmenu-launcher.sh](https://gitea.com/gnusupport/LLM-Helpers/src/branch/main/bin/rcd-llm-dmenu-launcher.sh)
It's a win for sure.
See translation Reply [melvindave](/melvindave) [Jan 3](#695927029918266addc03a7e)
thanks for the update! does it now behave like ollama?
See translation Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Fggml-org%2Fmodel-management-in-llamacpp) or [log in](/login?next=%2Fblog%2Fggml-org%2Fmodel-management-in-llamacpp) to comment
[Upvote 126](/login?next=%2Fblog%2Fggml-org%2Fmodel-management-in-llamacpp) +114 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe