[Hugging Face](/) [Models](/models) [Datasets](/datasets) [Spaces](/spaces) Community [Docs](/docs) [Enterprise](/enterprise) [Pricing](/pricing) [Log In](/login) [Sign Up](/join) [Back to Articles](/blog)
# HTML_TAG_START Deploying Open Source Vision Language Models (VLM) on Jetson HTML_TAG_END
[Enterprise + Article](/blog) Published February 24, 2026 [Upvote 25](/login?next=%2Fblog%2Fnvidia%2Fcosmos-on-jetson) +19 [Mitesh Patel mitp Follow](/mitp) [nvidia](/nvidia) [Johnny Nuñez Cano johnnynv Follow](/johnnynv) [nvidia](/nvidia) [Raymond Lo raymondlo84-nvidia Follow](/raymondlo84-nvidia) [nvidia](/nvidia) HTML_TAG_START
[HTML_TAG_START Prerequisites HTML_TAG_END](#prerequisites) [HTML_TAG_START Overview HTML_TAG_END](#overview) [HTML_TAG_START Step 1: Install the NGC CLI HTML_TAG_END](#step-1-install-the-ngc-cli) [HTML_TAG_START Download and install HTML_TAG_END](#download-and-install) [HTML_TAG_START Configure the CLI HTML_TAG_END](#configure-the-cli) [HTML_TAG_START Step 2: Download the Model HTML_TAG_END](#step-2-download-the-model) [HTML_TAG_START Step 3: Pull the vLLM Docker Image HTML_TAG_END](#step-3-pull-the-vllm-docker-image) [HTML_TAG_START For Jetson AGX Thor HTML_TAG_END](#for-jetson-agx-thor) [HTML_TAG_START For Jetson AGX Orin / Orin Super Nano HTML_TAG_END](#for-jetson-agx-orin--orin-super-nano) [HTML_TAG_START Step 4: Serve Cosmos Reason 2B with vLLM HTML_TAG_END](#step-4-serve-cosmos-reason-2b-with-vllm) [HTML_TAG_START Option A: Jetson AGX Thor HTML_TAG_END](#option-a-jetson-agx-thor) [HTML_TAG_START Option B: Jetson AGX Orin HTML_TAG_END](#option-b-jetson-agx-orin) [HTML_TAG_START Option C: Jetson Orin Super Nano (memory-constrained) HTML_TAG_END](#option-c-jetson-orin-super-nano-memory-constrained) [HTML_TAG_START Verify the server is running HTML_TAG_END](#verify-the-server-is-running) [HTML_TAG_START Step 5: Test with a Quick API Call HTML_TAG_END](#step-5-test-with-a-quick-api-call) [HTML_TAG_START Step 6: Connect to Live VLM WebUI HTML_TAG_END](#step-6-connect-to-live-vlm-webui) [HTML_TAG_START Install Live VLM WebUI HTML_TAG_END](#install-live-vlm-webui) [HTML_TAG_START Configure the WebUI HTML_TAG_END](#configure-the-webui) [HTML_TAG_START Recommended WebUI settings for Orin HTML_TAG_END](#recommended-webui-settings-for-orin) [HTML_TAG_START Troubleshooting HTML_TAG_END](#troubleshooting) [HTML_TAG_START Out of memory on Orin HTML_TAG_END](#out-of-memory-on-orin) [HTML_TAG_START Model not found in WebUI HTML_TAG_END](#model-not-found-in-webui) [HTML_TAG_START Slow inference on Orin HTML_TAG_END](#slow-inference-on-orin) [HTML_TAG_START vLLM fails to load model HTML_TAG_END](#vllm-fails-to-load-model) [HTML_TAG_START Summary HTML_TAG_END](#summary) [HTML_TAG_START Additional Resources HTML_TAG_END](#additional-resources) Vision-Language Models (VLMs) mark a significant leap in AI by blending visual perception with semantic reasoning. Moving beyond traditional models constrained by fixed labels, VLMs utilize a joint embedding space to interpret and discuss complex, open-ended environments using natural language.
The rapid evolution of reasoning accuracy and efficiency has made these models ideal for edge devices. The [NVIDIA Jetson family](https://marketplace.nvidia.com/en-us/enterprise/robotics-edge/?limit=15) , ranging from the high-performance AGX Thor and AGX Orin to the compact Orin Nano Super is purpose-built to drive accelerated applications for physical AI and robotics, providing the optimized runtime necessary for leading [open source models](https://www.jetson-ai-lab.com/models/) .
In this tutorial, we will demonstrate how to deploy the [NVIDIA Cosmos Reason 2B](https://build.nvidia.com/nvidia/cosmos-reason2-2b) model across the Jetson lineup using the [vLLM](https://vllm.ai/) framework. We will also guide you through connecting this model to the [Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) , enabling a real-time, webcam-based interface for interactive physical AI.
## Prerequisites
Supported Devices:
Jetson AGX Thor Developer Kit Jetson AGX Orin (64GB / 32GB) Jetson Orin Super Nano
JetPack Version:
JetPack 6 (L4T r36.x) — for Orin devices JetPack 7 (L4T r38.x) — for Thor
Storage: NVMe SSD required
~5 GB for the FP8 model weights ~8 GB for the vLLM container image
Accounts:
Create [NVIDIA NGC](https://ngc.nvidia.com/) account(free) to download both the model and vLLM contanier
## Overview
| | Jetson AGX Thor | Jetson AGX Orin | Orin Super Nano |
|---|---|---|---|
| vLLM Container | nvcr.io/nvidia/vllm:26.01-py3 | ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 | ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 |
| Model | FP8 via NGC (volume mount) | FP8 via NGC (volume mount) | FP8 via NGC (volume mount) |
| Max Model Length | 8192 tokens | 8192 tokens | 256 tokens (memory-constrained) |
| GPU Memory Util | 0.8 | 0.8 | 0.65 |
The workflow is the same for both devices:
Download the FP8 model checkpoint via NGC CLI Pull the vLLM Docker image for your device Launch the container with the model mounted as a volume Connect Live VLM WebUI to the vLLM endpoint
## Step 1: Install the NGC CLI
The NGC CLI lets you download model checkpoints from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/?tab=model) .
### Download and install
mkdir -p ~/Projects/CosmosReason
cd ~/Projects/CosmosReason
# Download the NGC CLI for ARM64
# Get the latest installer URL from: https://org.ngc.nvidia.com/setup/installers/cli
wget -O ngccli_arm64.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.13.0/files/ngccli_arm64.zip
unzip ngccli_arm64.zip
chmod u+x ngc-cli/ngc
# Add to PATH
export PATH="$PATH:$(pwd)/ngc-cli"
### Configure the CLI
ngc config set
You will be prompted for:
API Key — generate one at [NGC API Key setup](https://org.ngc.nvidia.com/setup/api-key) CLI output format — choose json or ascii org — press Enter to accept the default
## Step 2: Download the Model
Download the FP8 quantized checkpoint. This is used on all Jetson devices:
cd ~/Projects/CosmosReason
ngc registry model download-version "nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8"
This creates a directory called cosmos-reason2-2b_v1208-fp8-static-kv8/ containing the model weights. Note the full path — you will mount it into the Docker container as a volume.
## Step 3: Pull the vLLM Docker Image
### For Jetson AGX Thor
docker pull nvcr.io/nvidia/vllm:26.01-py3
### For Jetson AGX Orin / Orin Super Nano
docker pull ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04
## Step 4: Serve Cosmos Reason 2B with vLLM
### Option A: Jetson AGX Thor
Thor has ample GPU memory and can run the model with a generous context length.
Set the path to your downloaded model and free cached memory on the host:
MODEL_PATH="$HOME/Projects/CosmosReason/cosmos-reason2-2b_v1208-fp8-static-kv8"
sudo sysctl -w vm.drop_caches=3
Launch the container with the model mounted:
docker run --rm -it \
--runtime nvidia \
--network host \
--ipc host \
-v "$MODEL_PATH:/models/cosmos-reason2-2b:ro" \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
nvcr.io/nvidia/vllm:26.01-py3 \
bash
Inside the container, activate the environment and serve the model:
vllm serve /models/cosmos-reason2-2b \
--max-model-len 8192 \
--media-io-kwargs '{"video": {"num_frames": -1}}' \
--reasoning-parser qwen3 \
--gpu-memory-utilization 0.8
Note: The --reasoning-parser qwen3 flag enables chain-of-thought reasoning extraction. The --media-io-kwargs flag configures video frame handling.
Wait until you see:
INFO: Uvicorn running on http://0.0.0.0:8000
### Option B: Jetson AGX Orin
AGX Orin has enough memory to run the model with the same generous parameters as Thor.
Set the path to your downloaded model and free cached memory on the host:
MODEL_PATH="$HOME/Projects/CosmosReason/cosmos-reason2-2b_v1208-fp8-static-kv8"
sudo sysctl -w vm.drop_caches=3
1. Launch the container:
docker run --rm -it \
--runtime nvidia \
--network host \
-v "$MODEL_PATH:/models/cosmos-reason2-2b:ro" \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 \
bash
2. Inside the container, activate the environment and serve:
cd /opt/
source venv/bin/activate
vllm serve /models/cosmos-reason2-2b \
--max-model-len 8192 \
--media-io-kwargs '{"video": {"num_frames": -1}}' \
--reasoning-parser qwen3 \
--gpu-memory-utilization 0.8
Wait until you see:
INFO: Uvicorn running on http://0.0.0.0:8000
### Option C: Jetson Orin Super Nano (memory-constrained)
The Orin Super Nano has significantly less RAM, so we need aggressive memory optimization flags.
Set the path to your downloaded model and free cached memory on the host:
MODEL_PATH="$HOME/Projects/CosmosReason/cosmos-reason2-2b_v1208-fp8-static-kv8"
sudo sysctl -w vm.drop_caches=3
1. Launch the container:
docker run --rm -it \
--runtime nvidia \
--network host \
-v "$MODEL_PATH:/models/cosmos-reason2-2b:ro" \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 \
bash
2. Inside the container, activate the environment and serve:
cd /opt/
source venv/bin/activate
vllm serve /models/cosmos-reason2-2b \
--host 0.0.0.0 \
--port 8000 \
--trust-remote-code \
--enforce-eager \
--max-model-len 256 \
--max-num-batched-tokens 256 \
--gpu-memory-utilization 0.65 \
--max-num-seqs 1 \
--enable-chunked-prefill \
--limit-mm-per-prompt '{"image":1,"video":1}' \
--mm-processor-kwargs '{"num_frames":2,"max_pixels":150528}'
Key flags explained (Orin Super Nano only):
| Flag | Purpose |
|---|---|
| --enforce-eager | Disables CUDA graphs to save memory |
| --max-model-len 256 | Limits context to fit in available memory |
| --max-num-batched-tokens 256 | Matches the model length limit |
| --gpu-memory-utilization 0.65 | Reserves headroom for system processes |
| --max-num-seqs 1 | Single request at a time to minimize memory |
| --enable-chunked-prefill | Processes prefill in chunks for memory efficiency |
| --limit-mm-per-prompt | Limits to 1 image and 1 video per prompt |
| --mm-processor-kwargs | Reduces video frames and image resolution |
| --VLLM_SKIP_WARMUP=true | Skips warmup to save time and memory |
Wait until you see the server is ready:
INFO: Uvicorn running on http://0.0.0.0:8000
### Verify the server is running
From another terminal on the Jetson:
curl http://localhost:8000/v1/models
You should see the model listed in the response.
## Step 5: Test with a Quick API Call
Before connecting the WebUI, verify the model responds correctly:
curl -s http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/models/cosmos-reason2-2b",
"messages": [
{
"role": "user",
"content": "What capabilities do you have?"
}
],
"max_tokens": 128
}' | python3 -m json.tool
Tip: The model name used in the API request must match what vLLM reports. Verify with curl http://localhost:8000/v1/models .
## Step 6: Connect to Live VLM WebUI
[Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) provides a real-time webcam-to-VLM interface. With vLLM serving Cosmos Reason 2B, you can stream your webcam and get live AI analysis with reasoning.
### Install Live VLM WebUI
The easiest method is pip (Open another terminal):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd ~/Projects/CosmosReason
uv venv .live-vlm --python 3.12
source .live-vlm/bin/activate
uv pip install live-vlm-webui
live-vlm-webui
Or use Docker:
git clone https://github.com/nvidia-ai-iot/live-vlm-webui.git
cd live-vlm-webui
./scripts/start_container.sh
### Configure the WebUI
Open https://localhost:8090 in your browser Accept the self-signed certificate (click Advanced → Proceed ) In the VLM API Configuration section on the left sidebar: Set API Base URL to http://localhost:8000/v1 Click the Refresh button to detect the model Select the Cosmos Reason 2B model from the dropdown Select your camera and click Start
The WebUI will now stream your webcam frames to Cosmos Reason 2B and display the model’s analysis in real-time.
### Recommended WebUI settings for Orin
Since Orin runs with a shorter context length, adjust these settings in the WebUI:
Max Tokens : Set to 100–150 (shorter responses complete faster) Frame Processing Interval : Set to 60+ (gives the model time between frames)
## Troubleshooting
### Out of memory on Orin
Problem: vLLM crashes with CUDA out-of-memory errors.
Solution:
Free system memory before starting:
sudo sysctl -w vm.drop_caches=3
Lower --gpu-memory-utilization (try 0.55 or 0.50 )
Reduce --max-model-len further (try 128 )
Make sure no other GPU-intensive processes are running
### Model not found in WebUI
Problem: The model doesn’t appear in the Live VLM WebUI dropdown.
Solution:
Verify vLLM is running: curl http://localhost:8000/v1/models Make sure the WebUI API Base URL is set to http://localhost:8000/v1 (not https ) If vLLM and WebUI are in separate containers, use http://<jetson-ip>:8000/v1 instead of localhost
### Slow inference on Orin
Problem: Each response takes a very long time.
Solution:
This is expected with the memory-constrained configuration. Cosmos Reason 2B FP8 on Orin prioritizes fitting in memory over speed Reduce max_tokens in the WebUI to get shorter, faster responses Increase the frame interval so the model isn’t constantly processing new frames
### vLLM fails to load model
Problem: vLLM reports that the model path doesn’t exist or can’t be loaded.
Solution:
Verify the NGC download completed successfully: ls ~/Projects/CosmosReason/cosmos-reason2-2b_v1208-fp8-static-kv8/ Make sure the volume mount path is correct in your docker run command Check that the model directory is mounted as read-only ( :ro ) and the path inside the container matches what you pass to vllm serve
## Summary
In this tutorial, we showcased how to deploy NVIDIA Cosmos Reason 2B model on Jetson family of devices using vLLM.
The combination of Cosmos Reason 2B’s chain-of-thought capabilities with Live VLM WebUI’s real-time streaming makes it ideal to prototype and evaluate vision AI applications at the edge.
## Additional Resources
Cosmos Reason 2B on NVIDIA Build : [https://huggingface.co/nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) NGC Model Catalog : [https://catalog.ngc.nvidia.com/](https://catalog.ngc.nvidia.com/) Live VLM WebUI : [https://github.com/NVIDIA-AI-IOT/live-vlm-webui](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) vLLM container for Jetson Thor : [https://ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04](https://ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04) vLLM container for Jetson AGX Orin, and Orin Super Nano : [https://nvcr.io/nvidia/vllm:26.01-py3](https://nvcr.io/nvidia/vllm:26.01-py3) NGC CLI Installers : [https://org.ngc.nvidia.com/setup/installers/cli](https://org.ngc.nvidia.com/setup/installers/cli) Open Models supported on Jetson : [https://www.jetson-ai-lab.com/models/](https://www.jetson-ai-lab.com/models/) Getting started with Jetson : [https://www.jetson-ai-lab.com/tutorials/](https://www.jetson-ai-lab.com/tutorials/) HTML_TAG_END
### Community
[raymondlo84-nvidia](/raymondlo84-nvidia) Article author [3 days ago](#69a0a6c1999ef95990ec3919)
If you run into permission error on Docker, please follow the instruction here! :) [https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html)
See translation Reply [raymondlo84-nvidia](/raymondlo84-nvidia) Article author [3 days ago](#69a0a95617fc449ca63b385a) • [edited 3 days ago](#69a0a95617fc449ca63b385a)
and make sure you install CURL if you have trouble running curl command (did not get installed by default on Jetson).
apt-get install curl See translation Reply [raymondlo84-nvidia](/raymondlo84-nvidia) Article author [3 days ago](#69a0aaafa43960b4fbf6841b)
And this will be how it looks like once it's all working :) cheers!
See translation 🔥 3 3 + Reply [surprisal](/surprisal) [1 day ago](#69a29f419af51d9d05a0c344)
it appears that you have adapted Cosmos Reason 2B, the VLM, to robotic manipulation in your demo. could you please tell us more about your adaptation, like how motion planning and end-effector controlling are implemented? thanks.
See translation Reply Edit Preview Upload images, audio, and videos by dragging in the text input, pasting, or clicking here . Tap or paste here to upload images Comment
· [Sign up](/join?next=%2Fblog%2Fnvidia%2Fcosmos-on-jetson) or [log in](/login?next=%2Fblog%2Fnvidia%2Fcosmos-on-jetson) to comment
[Upvote 25](/login?next=%2Fblog%2Fnvidia%2Fcosmos-on-jetson) +13 System theme Company [TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/) Website [Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs) Stripe