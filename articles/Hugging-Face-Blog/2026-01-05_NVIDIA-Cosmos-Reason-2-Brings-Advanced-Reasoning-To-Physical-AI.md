[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START NVIDIA Cosmos Reason 2 Brings Advanced Reasoning To Physical AI HTML_TAG_END
[Enterprise + Article](/blog) Published January 5, 2026 [Upvote 64](/login?next=%2Fblog%2Fnvidia%2Fnvidia-cosmos-reason-2-brings-advanced-reasoning) +58 [Tsung-Yi Lin tsungyi Follow](/tsungyi) [nvidia](/nvidia) [Debraj Sinha debrajsinha Follow](/debrajsinha) [nvidia](/nvidia) HTML_TAG_START
[HTML_TAG_START NVIDIA Cosmos Reason 2: Reasoning Vision Language Model for Physical AI HTML_TAG_END](#nvidia-cosmos-reason-2-reasoning-vision-language-model-for-physical-ai) [HTML_TAG_START ✨ Key Highlights HTML_TAG_END](#✨-key-highlights) [HTML_TAG_START 🤖 Popular Use Cases HTML_TAG_END](#🤖-popular-use-cases) [HTML_TAG_START Other Models From The Cosmos Family: HTML_TAG_END](#other-models-from-the-cosmos-family) [HTML_TAG_START 🔮 Cosmos Predict 2.5 HTML_TAG_END](#🔮-cosmos-predict-25) [HTML_TAG_START Resources HTML_TAG_END](#resources)
NVIDIA today released [Cosmos Reason 2](https://huggingface.co/nvidia/Cosmos-Reason2-8B) , the latest advancement in open, reasoning vision language models for physical AI. Cosmos Reason 2 surpasses its previous version in accuracy and tops the [Physical AI Bench](https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard) and [Physical Reasoning](https://huggingface.co/spaces/facebook/physical_reasoning_leaderboard) leaderboards as the #1 open model for visual understanding.
## NVIDIA Cosmos Reason 2: Reasoning Vision Language Model for Physical AI
Since their introduction, [vision-language models](https://www.nvidia.com/en-us/glossary/vision-language-models/) have rapidly improved at tasks like object and pattern recognition in images. But they still struggle with tasks humans find natural, like planning several steps ahead, dealing with uncertainty or adapting to new situations. Cosmos Reason is designed to close this gap by giving robots and AI agents stronger common sense and reasoning to solve complex problems step by step.
Cosmos Reason 2 is a state-of-the-art, open reasoning vision-language model (VLM) that enables robots and AI agents to see, understand, plan, and act in the physical world like humans. It uses common sense, physics, and prior knowledge to recognize how objects move across space and time to handle complex tasks, adapt to new situations, and figure out how to solve problems step by step.
### ✨ Key Highlights
Improved spatio-temporal understanding and timestamp precision.
Optimized performance with flexible deployment options from edge to cloud with 2B and 8B parameters model sizes.
Support for expanded set of spatial understanding and visual perception capabilities — 2D/3D point localization, bounding box coordinates, trajectory data, and OCR support.
Improved long-context understanding with 256K input tokens, up from 16K with Cosmos Reason 1.
Adaptable to multiple use cases with easy-to-use [Cosmos Cookbook recipes](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html) .
### 🤖 Popular Use Cases
Video analytics AI agents — These agents can extract valuable insights from massive volumes of video data to optimize processes. Cosmos Reason 2 builds on the capabilities of Cosmos Reason 1 and now provides OCR support, as well as 2D/3D point localization and a set of mark understanding.
Example of how Cosmos Reason can understand text embedded within a video to determine the condition of the road during a rainstorm.
Developers can jumpstart development of video analytics AI agents by using the [NVIDIA blueprint for video search and summarization (VSS)](https://build.nvidia.com/nvidia/video-search-and-summarization) with Cosmos Reason as the VLM.
[Salesforce](https://salesforce.com/blog/the-new-frontier-of-physical-ai-how-salesforce-and-nvidia-turn-robots-into-enterprise-agents/) is transforming workplace safety and compliance by analyzing video footage captured by Cobalt robots with Agentforce and VSS blueprint with Cosmos Reason as the VLM.
Data annotation and critique — Enable developers to automate high-quality annotation and critique of massive, diverse training datasets. Cosmos Reason provides time stamps and detailed descriptions for real or synthetically generated training videos.
Example of a sample prompt to generate detailed, time-stamped captions for a race car video.
[Uber is exploring Cosmos Reason 2](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/video_caption_vqa/post_training.html) to deliver accurate, searchable video captions for autonomous vehicle (AV) training data, enabling efficient identification of critical driving scenarios. This [co-authored Reason 2 for AV Video Captioning and VQA recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/video_caption_vqa/post_training.html) demonstrates how to fine-tune and evaluate Cosmos Reason 2-8B on annotated AV videos. Across multiple evaluation metrics, measurable improvements were achieved: BLEU scores improved 10.6% (0.113 → 0.125), MCQ-based VQA gained 0.67 percentage points (80.18% → 80.85%), and LingoQA increased 13.8% (63.2% → 77.0%). These gains demonstrate effective domain adaptation for AV applications.
Robot planning and reasoning — Act as the brain for deliberate, methodical decision-making in a robot vision language action (VLA) model. Cosmos Reason 2 now provides trajectory coordinates in addition to determining next steps.
Example of the prompt and JSON output from Cosmos Reason 2 to provide the steps and trajectory the robot gripper needs to take to move the painter’s tape into the basket.
[Encord](https://encord.com/blog/data-agents) provides native support for Cosmos Reason 2 in its [Data Agent](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fencord.com%2Fdata-agents%2F&data=05%7C02%7Ckrumley%40nvidia.com%7C42a86cea3d0c445e973208de44e9bea2%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C639023968198962586%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C40000%7C%7C%7C&sdata=%2Fqw0guXgke6MRaqZShEkwhyYu%2FohI314%2Fju3A4%2F5pao%3D&reserved=0) library and AI data platform, enabling developers to leverage Cosmos Reason 2 as a VLA for robotics and other physical AI use cases.
Companies like Hitachi, [Milestone](https://www.milestonesys.com/company/news/press-releases/milestone-launches-vision-language-model/) and [VAST Data](https://www.vastdata.com/blog/vast-nvidia-cosmos-reason-smart-cities) are using Cosmos Reason to advance robotics, autonomous driving, and video analytics AI agents for traffic and workplace safety.
Try [Cosmos Reason 2 on build.nvidia.com](https://build.nvidia.com/nvidia/cosmos-reason2-8b) and experience the latest features with sample prompts for generating bounding boxes and robot trajectories. Upload your own videos and images for further analysis.
Download Cosmos Reason 2 models ( [2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) and [8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) ) on Hugging Face or [use Cosmos Reason 2 in the cloud](https://nvidia-cosmos.github.io/cosmos-cookbook/getting_started/brev/reason2/reason2_on_brev.html) . The model will be available soon on Amazon Web Services, Google Cloud and Microsoft Azure. To get started, check out [Cosmos Reason 2 documentation](https://docs.nvidia.com/cosmos/latest/reason2/index.html) and the [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html) .
## Other Models From The Cosmos Family:
### 🔮 Cosmos Predict 2.5
Cosmos Predict is a generative AI model that predicts future states of the physical world as video, based on text, image, or video inputs.
[Physical AI Bench](https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard) leader for quality, accuracy and overall consistency. Up to 30 seconds of physically and temporally consistent clip per generation. Supports multiple framerates and resolution. Pre-trained on 200 million clips. Available as 2B and 14B pre-trained models and various 2B post-trained models for multiview, action conditioning and autonomous vehicle training.
[Check out model card](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) >>
🔁 Cosmos Transfer 2.5
Cosmos Transfer is our lightest multicontrol model built for video to world style transfer.
Scale a single simulation or spatial video across various environments and lighting conditions. Improved prompt adherence and physics alignment. Use with [NVIDIA Isaac Sim™](https://developer.nvidia.com/isaac/sim) or [NVIDIA Omniverse NuRec](https://developer.nvidia.com/blog/accelerating-av-simulation-with-neural-reconstruction-and-world-foundation-models/) for simulation to real transformation.
[Check out model card>>](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
🤖 NVIDIA GR00T N1.6
[NVIDIA GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) is an open reasoning vision language action (VLA) model, purpose-built for humanoid robots, that unlocks full body control and uses NVIDIA Cosmos Reason for better reasoning and contextual understanding.
## Resources
▶️ Watch a demo of Cosmos → [https://youtu.be/iWs-2TD5Dcc](https://youtu.be/iWs-2TD5Dcc)
🧑🏻‍🍳 Read the Cosmos Cookbook → [https://nvda.ws/4qevli8](https://nvda.ws/4qevli8)
📚 Explore Models & Datasets → [https://github.com/nvidia-cosmos](https://github.com/nvidia-cosmos)
⬇️ Try Cosmos Models in our Hosted Catalog → [https://nvda.ws/3Yg0Dcx](https://nvda.ws/3Yg0Dcx)
💻 Join the Cosmos Community → [https://discord.gg/u23rXTHSC9](https://discord.gg/u23rXTHSC9)
🗳️ Contribute to the Cosmos Cookbook → [https://nvda.ws/4aQcBkk](https://nvda.ws/4aQcBkk)
HTML_TAG_END
### Community
[merve](/merve) [Jan 6](#695ce3aa4fa7fd6f0077787d)
very bullish on embodied VLA/VLMs this year 🔥
See translation 🚀 1 1 + Reply [mindchain](/mindchain) [Jan 6](#695d2e288ff7a3ae33f4e771)
looks so cool!
🤗 1 1 + Reply deleted [Jan 8](#695f85622e53e8fe83bf6e3c) This comment has been hidden [Daniel6316](/Daniel6316) [Jan 12](#6964edc0df922e713eadb626)
Great update from NVIDIA—Cosmos Reason 2 looks like a big step forward for physical AI, especially with its improved reasoning and decision-making capabilities. It’ll be exciting to see how this advances real-world robotics and autonomous systems.
See translation Reply deleted [Jan 13](#69669af70a7b247f24f03507)
dsa
Reply deleted [Jan 13](#69669b0742220d95b4e009e3)
ads
Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Fnvidia%2Fnvidia-cosmos-reason-2-brings-advanced-reasoning) or [log in](/login?next=%2Fblog%2Fnvidia%2Fnvidia-cosmos-reason-2-brings-advanced-reasoning) to comment
[Upvote 64](/login?next=%2Fblog%2Fnvidia%2Fnvidia-cosmos-reason-2-brings-advanced-reasoning) +52 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe