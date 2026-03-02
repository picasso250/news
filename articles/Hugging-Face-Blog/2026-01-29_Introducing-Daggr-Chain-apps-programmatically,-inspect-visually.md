[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START Introducing Daggr: Chain apps programmatically, inspect visually HTML_TAG_END
Published January 29, 2026 [Update on GitHub](https://github.com/huggingface/blog/blob/main/daggr.md) [Upvote 103](/login?next=%2Fblog%2Fdaggr) +97 [merve merve Follow](/merve) [yuvraj sharma ysharma Follow](/ysharma) [Abubakar Abid abidlabs Follow](/abidlabs) [hysts hysts Follow](/hysts) [Pedro Cuenca pcuenq Follow](/pcuenq) HTML_TAG_START
[HTML_TAG_START Table of Contents HTML_TAG_END](#table-of-contents) [HTML_TAG_START Background HTML_TAG_END](#background) [HTML_TAG_START Getting Started HTML_TAG_END](#getting-started) [HTML_TAG_START Node Types HTML_TAG_END](#node-types) [HTML_TAG_START Sharing Your Workflows HTML_TAG_END](#sharing-your-workflows) [HTML_TAG_START End-to-End Example with Different Nodes HTML_TAG_END](#end-to-end-example-with-different-nodes) [HTML_TAG_START Next Steps HTML_TAG_END](#next-steps) TL;DR: [Daggr](https://github.com/gradio-app/daggr) is a new, open-source Python library for building AI workflows that connect Gradio apps, ML models, and custom functions. It automatically generates a visual canvas where you can inspect intermediate outputs, rerun individual steps, and manage state for complex pipelines, all in a few lines of Python code!
## Table of Contents
[Background](#background) [Getting Started](#getting-started) [Sharing Your Workflows](#sharing-your-workflows) [End-to-End Example with Different Nodes](#end-to-end-example-with-different-nodes) [Next Steps](#next-steps)
## Background
If you've built AI applications that combine multiple models or processing steps, you know the pain: chaining API calls, debugging pipelines, and losing track of intermediate results. When something goes wrong in step 5 of a 10-step workflow, you often have to re-run everything just to see what happened.
Most developers either build fragile scripts that are hard to debug or turn to heavy orchestration platforms designed for production pipelines—not rapid experimentation.
We've been working on Daggr to solve problems we kept running into when building AI demos and workflows:
Visualize your code flow : Unlike node-based GUI editors, where you drag and connect nodes visually, Daggr takes a code-first approach. You define workflows in Python, and a visual canvas is generated automatically. This means you get the best of both worlds: version-controllable code and visual inspection of intermediate outputs.
Inspect and Rerun Any Step : The visual canvas isn't just for show. You can inspect the output of any node, modify inputs, and rerun individual steps without executing the entire pipeline. This is invaluable when you're debugging a 10-step workflow and only step 7 is misbehaving. You can even provide “backup nodes” – replacing one model or Space with another – to build resilient workflows.
First-Class Gradio Integration : Since Daggr is built by the Gradio team, it works seamlessly with Gradio Spaces. Point to any public (or private) Space and you can use it as a node in your workflow. No adapters, no wrappers—just reference the Space name and API endpoint.
State Persistence : Daggr automatically saves your workflow state, input values, cached results, canvas position—so you can pick up where you left off. Use "sheets" to maintain multiple workspaces within the same app.
## Getting Started
Install daggr with pip or uv, it just requires Python 3.10 or higher:
pip install daggr
uv pip install daggr
Here's a simple example that generates an image and removes its background. Check out [this Space’s API reference](https://huggingface.co/spaces/hf-applications/Z-Image-Turbo) from the bottom of the Space to see which inputs it takes and which outputs it yields. In this example, the Space returns both original image and the edited image, so we return only the edited image.
import random import gradio as gr from daggr import GradioNode, Graph # Generate an image using a Gradio Space image_gen = GradioNode( "hf-applications/Z-Image-Turbo" ,
api_name= "/generate_image" ,
inputs={ "prompt" : gr.Textbox(
label= "Prompt" ,
value= "A cheetah sprints across the grassy savanna." ,
lines= 3 ,
), "height" : 1024 , "width" : 1024 , "seed" : random.random,
},
outputs={ "image" : gr.Image(label= "Generated Image" ),
},
) # Remove background using another Gradio Space bg_remover = GradioNode( "hf-applications/background-removal" ,
api_name= "/image" ,
inputs={ "image" : image_gen.image, # Connect to previous node's output },
outputs={ "original_image" : None , # Hide this output "final_image" : gr.Image(label= "Final Image" ),
},
)
graph = Graph(
name= "Transparent Background Generator" ,
nodes=[image_gen, bg_remover]
)
graph.launch()
That's it. Run this script and you get a visual canvas served on port 7860 launched automatically, as well as a shareable live link, showing both nodes connected, with inputs you can modify and outputs you can inspect at each step.
### Node Types
Daggr supports three types of nodes:
GradioNode calls a Gradio Space API endpoint or locally served Gradio app. Passing run_locally=True , Daggr automatically clones the Space, creates an isolated virtual environment, and launches the app. If local execution fails, it gracefully falls back to the remote API.
node = GradioNode( "username/space-name" ,
api_name= "/predict" ,
inputs={ "text" : gr.Textbox(label= "Input" )},
outputs={ "result" : gr.Textbox(label= "Output" )},
) # clone a Space locally and serve node = GradioNode( "hf-applications/background-removal" ,
api_name= "/image" ,
run_locally= True ,
inputs={ "image" : gr.Image(label= "Input" )},
outputs={ "final_image" : gr.Image(label= "Output" )},
FnNode — runs a custom Python function:
def process ( text: str ) -> str : return text.upper()
node = FnNode(
fn=process,
inputs={ "text" : gr.Textbox(label= "Input" )},
outputs={ "result" : gr.Textbox(label= "Output" )},
)
InferenceNode — calls a model via Hugging Face Inference Providers:
node = InferenceNode(
model= "moonshotai/Kimi-K2.5:novita" ,
inputs={ "prompt" : gr.Textbox(label= "Prompt" )},
outputs={ "response" : gr.Textbox(label= "Response" )},
)
### Sharing Your Workflows
Generate a public URL with Gradio's tunneling:
graph.launch(share= True )
For permanent hosting, deploy on Hugging Face Spaces using the Gradio SDK—just add daggr to your requirements.txt .
## End-to-End Example with Different Nodes
We will now develop an app that takes in an image and generates a 3D asset. This demo can run on daggr 0.4.3. Here are the steps:
Take an image, remove the background: For this, we will clone the [BiRefNet Space](https://huggingface.co/spaces/merve/background-removal) and run it locally. Downscale the image for efficiency: We will write a simple function for this with FnNode. Generate an image in 3D asset style for better results: We will use InferenceNode with [Flux.2-klein-4B model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Inference Providers. Pass the output image to a 3D generator: We will send the output image to the Trellis.2 Space hosted on Spaces.
Spaces that are run locally might take models to CUDA (with to.(“cuda”) ) or ZeroGPU within the application file. To disable this behavior to run the model on CPU (useful if you have a device with no NVIDIA GPU) duplicate the Space you want to use and clone it.
The resulting graph looks like below.
Let’s write the first step, which is the background remover. We will clone and run [this Space](https://huggingface.co/spaces/merve/background-removal) locally. This Space runs on CPU, and takes ~13 seconds to run. You can swap with [this app](https://huggingface.co/spaces/hf-applications/background-removal) if you have an NVIDIA GPU.
from daggr import FnNode, GradioNode, InferenceNode, Graph
background_remover = GradioNode( "merve/background-removal" ,
api_name= "/image" ,
run_locally= True ,
inputs={ "image" : gr.Image(),
},
outputs={ "original_image" : None , "final_image" : gr.Image(
label= "Final Image" ),
},
)
For the second step, we need to write a helper function to downscale the image and pass it to FnNode .
from PIL import Image from daggr.state import get_daggr_files_dir def downscale_image_to_file ( image: Any , scale: float = 0.25 ) -> str | None :
pil_img = Image. open (image)
scale_f = max ( 0.05 , min ( 1.0 , float (scale)))
w, h = pil_img.size
new_w = max ( 1 , int (w * scale_f))
new_h = max ( 1 , int (h * scale_f))
resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)
out_path = get_daggr_files_dir() / f" {uuid.uuid4()} .png" resized.save(out_path) return str (out_path)
We can now pass in the function to initialize the FnNode .
downscaler = FnNode(
downscale_image_to_file,
name= "Downscale image for Inference" ,
inputs={ "image" : background_remover.final_image, "scale" : gr.Slider(
label= "Downscale factor" ,
minimum= 0.25 ,
maximum= 0.75 ,
step= 0.05 ,
value= 0.25 ,
),
},
outputs={ "image" : gr.Image(label= "Downscaled Image" , type = "filepath" ),
},
)
We will now write the InferenceNode with the Flux model.
flux_enhancer = InferenceNode(
model= "black-forest-labs/FLUX.2-klein-4B:fal-ai" ,
inputs={ "image" : downscaler.image, "prompt" : gr.Textbox(
label= "prompt" ,
value=( "Transform this into a clean 3D asset render" ),
lines= 3 ,
),
},
outputs={ "image" : gr.Image(label= "3D-Ready Enhanced Image" ),
},
)
When deploying apps with InferenceNode to Hugging Face Spaces, use a fine-grained Hugging Face access token with the option "Make calls to Inference Providers" only.
Last node is 3D generation with querying the Trellis.2 Space on Hugging Face.
trellis_3d = GradioNode( "microsoft/TRELLIS.2" ,
api_name= "/image_to_3d" ,
inputs={ "image" : flux_enhancer.image, "ss_guidance_strength" : 7.5 , "ss_sampling_steps" : 12 ,
},
outputs={ "glb" : gr.HTML(label= "3D Asset (GLB preview)" ),
},
)
Chaining them together and launching the app is as simple as follows.
graph = Graph(
name= "Image to 3D Asset Pipeline" ,
nodes=[background_remover, downscaler, flux_enhancer, trellis_3d],
) if __name__ == "__main__" :
graph.launch()
You can find the complete example running in [this Space](https://huggingface.co/spaces/merve/daggr-image-to-3d) , to run locally you just need to take app.py, install requirements and login to Hugging Face Hub.
## Next Steps
Daggr is in beta and intentionally lightweight. APIs may change between versions, and while we persist workflow state locally, data loss is possible during updates. If you have feature requests or find bugs, please open an issue [here](https://github.com/gradio-app/daggr/issues) . We’re looking forward to your feedback! Share your daggr workflows on socials with Gradio for a chance to be featured. Check out all the featured works [here](https://huggingface.co/collections/ysharma/daggr-hf-spaces) .
HTML_TAG_END
More Articles from our Blog
[gradio claude html
## One-Shot Any Web App with Gradio's gr.HTML
21 February 18, 2026 ysharma, et. al.](/blog/gradio-html-one-shot-apps) [gradio mcp community
## Implementing MCP Servers in Python: An AI Shopping Assistant with Gradio
freddyaboulton 60 July 31, 2025 freddyaboulton](/blog/gradio-vton-mcp)
### Community
[ArseniyPerchik](/ArseniyPerchik) [30 days ago](#697dbe0690157f3510070d35) • [edited 30 days ago](#697dbe0690157f3510070d35)
What about conditional nodes? I mean the node where the next selected node is determined based on the current node's output.
See translation 1 reply · [abidlabs](/abidlabs) Article author [25 days ago](#6983fdbae4ee336f7023ed1f)
This is a great point. We'd like to support them, but are thinking about the right API because we'd also like programmatic inspection / API usage, which requires some level of determinism. If you have any suggestions, please feel free to open an issue here: [https://github.com/gradio-app/daggr](https://github.com/gradio-app/daggr)
See translation [hxgdzyuyi](/hxgdzyuyi) [25 days ago](#6983fd582e4e1cdc1c18bd66)
why not jupyternotebook
See translation 1 reply · [abidlabs](/abidlabs) Article author [25 days ago](#6983fdd1e93bfbf68937327b)
Do you have any issues running Daggr in a jupyter notebook? It should be supported, but if run into issues, please feel free to open an issue here: [https://github.com/gradio-app/daggr](https://github.com/gradio-app/daggr)
See translation 👍 1 1 + [MatthewFrank](/MatthewFrank) [20 days ago](#698a27cfd3b0997926a91c38)
The visual inspection capability for chained apps is brilliant! Being able to see the flow programmatically while maintaining visibility is so important for debugging and understanding system behavior. Speaking of visualizing system flows, I've been using InfraSketch ( [https://www.infrasketch.net/](https://www.infrasketch.net/) ) for documenting our application architectures—it generates diagrams from plain English descriptions and lets you refine through conversation. It's been a great complement to tools like Daggr for communicating the overall system design to the broader team.
See translation Reply [Funnelsflex](/Funnelsflex) [8 days ago](#6999ea6e415aea0b7f3b2531)
This is a solid approach to pipeline transparency. Working with Funnelsflex deployments, we often see a friction point between rigid UI funnel builders and the flexibility of raw Python scripts. Daggr seems to bridge that gap by keeping the logic in the code where it belongs, while providing the visual state persistence that’s usually missing in standard Gradio chains.
The ability to rerun individual nodes without triggering the entire inference stack is a huge win for debugging complex flexible workflows. I’m curious to see how it handles state management when scaling to high-concurrency environments, but for rapid prototyping and 'inspectable' AI apps, it’s a very clean implementation.
See translation Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Fdaggr) or [log in](/login?next=%2Fblog%2Fdaggr) to comment
[Upvote 103](/login?next=%2Fblog%2Fdaggr) +91 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe