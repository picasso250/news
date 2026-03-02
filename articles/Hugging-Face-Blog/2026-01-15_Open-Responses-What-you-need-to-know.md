[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START Open Responses: What you need to know HTML_TAG_END
Published January 15, 2026 [Update on GitHub](https://github.com/huggingface/blog/blob/main/open-responses.md) [Upvote 108](/login?next=%2Fblog%2Fopen-responses) +102 [shaun smith evalstate Follow](/evalstate) [ben burtenshaw burtenshaw Follow](/burtenshaw) [merve merve Follow](/merve) [Pedro Cuenca pcuenq Follow](/pcuenq) HTML_TAG_START
[HTML_TAG_START What is Open Responses? HTML_TAG_END](#what-is-open-responses) [HTML_TAG_START What do we need to know to build with Open Responses? HTML_TAG_END](#what-do-we-need-to-know-to-build-with-open-responses) [HTML_TAG_START Client Requests to Open Responses HTML_TAG_END](#client-requests-to-open-responses) [HTML_TAG_START Changes for Inference Clients and Providers HTML_TAG_END](#changes-for-inference-clients-and-providers) [HTML_TAG_START Open Responses for Routing HTML_TAG_END](#open-responses-for-routing) [HTML_TAG_START Tools HTML_TAG_END](#tools) [HTML_TAG_START Sub Agent Loops HTML_TAG_END](#sub-agent-loops) [HTML_TAG_START Next Steps HTML_TAG_END](#next-steps) Open Responses is a new and open inference standard. Initiated by OpenAI, built by the open source AI community, and backed by the Hugging Face ecosystem, Open Responses is based on the Responses API and is designed for the future of Agents. In this blog post, we’ll look at how Open Responses works and why the open source community should use Open Responses.
The era of the chatbot is long gone, and agents dominate inference workloads. Developers are shifting toward autonomous systems that reason, plan, and act over long-time horizons. Despite this shift, much of the ecosystem still uses the Chat Completion format, which was designed for turn-based conversations and falls short for agentic use cases. The Responses format was designed to address these limitations, but it is closed and not as widely adopted. The Chat Completion format is still the de facto standard despite the alternatives.
This mismatch between the agentic workflow requirements and entrenched interfaces motivates the need for an open inference standard. Over the coming months, we will collaborate with the community and inference providers to implement and adapt Open Responses to a shared format, practically capable of replacing chat completions.
Open Responses builds on the direction OpenAI has set with their [Responses API](https://platform.openai.com/docs/api-reference/responses) launched in March 2025, which superseded the existing Completion and Assistants APIs with a consistent way to:
Generate Text, Images, and JSON structured outputs Create Video content through a separate task-based endpoint Run agentic loops on the provider side, executing tool calls autonomously and returning the final result.
## What is Open Responses?
Open Responses extends and open-sources the Responses API, making it more accessible for builders and routing providers to interoperate and collaborate on shared interests.
Some of the key points are:
Stateless by default, supporting encrypted reasoning for providers that require it. Standardized model configuration parameters. Streaming is modeled as a series of semantic events, not raw text or object deltas. Extensible via configurable parameters specific to certain model providers.
## What do we need to know to build with Open Responses?
We’ll briefly explore the core changes that impact most community members. If you want to deep dive into the specification, check out the [Open Responses documentation](https://www.openresponses.org/) .
### Client Requests to Open Responses
Client requests to Open Responses are similar to the existing Responses API. Below we demonstrate a request to the Open Responses API using curl. We're calling a proxy endpoint that routes to Inference Providers using the Open Responses API schema.
curl https://evalstate-openresponses.hf.space/v1/responses \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $HF_TOKEN" \ + -H "OpenResponses-Version: latest" \ -N \
-d '{
"model": "moonshotai/Kimi-K2-Thinking:nebius",
"input": "explain the theory of life"
}'
### Changes for Inference Clients and Providers
Clients that already support the Responses API can migrate to Open Responses with relatively little effort. The main
changes involve how reasoning content is exposed:
Expanded reasoning visibility: Open Responses formalizes three optional fields for reasoning items: content (raw
reasoning traces), encrypted_content (provider-specific protected content), and summary (sanitized from raw traces).
OpenAI models used to only expose summary and encrypted_content . With Open Responses, providers may expose their raw reasoning via the API.
Clients migrating from providers that previously returned only summaries and encrypted content will now have the
opportunity to receive and handle raw reasoning streams when supported by their chosen provider.
Implementing richer state changes and payloads, including more detailed observability—for example, a hosted Code Interpreter can send a specific interpreting state to improve agent and user visibility during long-running operations.
For Model Providers, implementing the changes for Open Responses should be straightforward if they already adhere to the Responses API specification. For Routers, there is now the opportunity to standardize on a consistent endpoint and support configuration options for customization where needed.
Over time, as Providers continue to innovate, certain features will become standardized in the base specification.
In summary, migrating to Open Responses will make the inference experience more consistent and improve quality as undocumented extensions, interpretations, and workarounds of the legacy Completions API are normalized in Open Responses.
You can see how to stream reasoning chunks below.
{ "model" : "moonshotai/Kimi-K2-Thinking:together" , "input" : [ { "type" : "message" , "role" : "user" , "content" : "explain photosynthesis." } ] , "stream" : true }
Here’s the difference between getting an Open Response and using OpenAI Responses for reasoning deltas:
// Open weight models stream raw reasoning event : response.reasoning.delta
data : { "delta" : "User asked: 'Where should I eat...' Step 1: Parse location..." , ... } // Models with encrypted reasoning send summaries, or sent as a convenience by Open Weight models event : response.reasoning_summary_text.delta
data : { "delta" : "Determined user wants restaurant recommendations" , ... }
### Open Responses for Routing
Open Responses distinguishes between “Model Providers” — those who provide inference — and “Routers” — intermediaries who orchestrate between multiple providers.
Clients can now specify a Provider along with provider-specific API options when making requests, allowing intermediary Routers to orchestrate requests between upstream providers.
### Tools
Open Responses natively supports two categories of tools: internal and external. Externally hosted tools are implemented outside the model provider’s system. For example, client side functions to be executed, or MCP servers. Internally hosted tools are within the model provider’s system. For example, OpenAI’s file search or Google Drive integration. The model calls, executes, and retrieves results entirely within the provider's infrastructure, requiring no developer intervention.
### Sub Agent Loops
Open Responses formalizes the agentic loop which is usually made up of a repeating cycle of reasoning, tool invocation, and response generation that enables models to autonomously complete multi-step tasks.
[image source: openresponses.org](https://www.openresponses.org/specification#the-agentic-loop)
The loop operates as follows:
The API receives a user request and samples from the model If the model emits a tool call, the API executes it (internally or externally) Tool results are fed back to the model for continued reasoning The loop repeats until the model signals completion
For internally-hosted tools, the provider manages the entire loop; executing tools, returning results to the model, and streaming output. This means that multi-step workflows like "search documents, summarize findings, then draft an email" use a single request.
Clients control loop behavior via max_tool_calls to cap iterations and tool_choice to constrain which tools are invocable:
{ "model" : "zai-org/GLM-4.7" , "input" : "Find Q3 sales data and email a summary to the team" , "tools" : [ ... ] , "max_tool_calls" : 5 , "tool_choice" : "auto" }
The response contains all intermediate items: tool calls, results, reasoning.
## Next Steps
Open Responses extends and improves the Responses API, providing richer and more detailed content definitions, compatibility, and deployment options. It also provides a standard way to execute sub-agent loops during primary inference calls, opening up powerful capabilities for AI Applications. We are looking forward to working with the Open Responses team and the community at large on future development of the specification.
You can try Open Responses with [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) today. We have an early access version available for use on [Hugging Face Spaces](https://huggingface.co/spaces/evalstate/openresponses) - try it with your Client and Open Responses Compliance Tool today!
HTML_TAG_END
More Articles from our Blog
[agents gui community
## ScreenEnv: Deploy your full stack Desktop Agent
A-Mahla, m-ric 76 July 10, 2025 A-Mahla, m-ric](/blog/screenenv) [open-source cuda kernels
## Custom Kernels for All from Codex and Claude
64 February 13, 2026 burtenshaw, et. al.](/blog/custom-cuda-kernels-agent-skills)
### Community
[ashim](/ashim) [Jan 15](#6969556d1a23b47af263cc0b)
Does this mean that the next step for local llm endpoint providers (like vLLM) is to support hosted tools?
See translation Reply [evalstate](/evalstate) Article author [Jan 15](#696960b93d33c37c7ff21b93)
Yes - I think we will see a lot more of this pattern especially for Agents offloading work to sub-agent Tool Loops via Open Responses.
See translation 🚀 3 3 👍 3 3 + Reply [jimazmarin](/jimazmarin) [Jan 19](#696e2bb7623a66553a368de7)
Are you going to open source the code of the API?
See translation Reply [njbrake](/njbrake) [Jan 26](#6977bfa7d4a00deea21b389c) This comment has been hidden (marked as Resolved) [akwako](/akwako) [about 1 month ago](#697c4c5b45dc2c5819afbb55) • [edited about 1 month ago](#697c4c5b45dc2c5819afbb55)
I hope this doesn't normalize the obscuring of raw output.
See translation Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Fopen-responses) or [log in](/login?next=%2Fblog%2Fopen-responses) to comment
[Upvote 108](/login?next=%2Fblog%2Fopen-responses) +96 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe