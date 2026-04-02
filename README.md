Explainable Image Captioning 

Transformer Decoders & Attention Maps for Visual Grounding: 
This is an Explainable AI (XAI) project that generates descriptive captions for images while providing visual proof of its "thought process." By extracting cross-attention maps from a Transformer decoder, the system 
generates heatmaps that show exactly which pixels the model prioritized to predict each specific word.

Live Link : http://56.228.59.90/

🏗️ System Architecture
The model follows an Encoder-Decoder architecture, bridging a Convolutional Neural Network (CNN) with a Generative Pre-trained Transformer (GPT).

Phase 1: Encoding the Image (The Vision)
1) Feature Extraction: Raw 224x224 image tensors are processed via ResNet-50. We intercept the output at the final convolutional layer to obtain a 7x7 Spatial Feature Map.

2) The Bridge: A Linear Projection layer compresses the 2048-dimensional ResNet vectors into 768-dimensional latent vectors to match the GPT-2 input size.

3) Positional Encoding: Coordinates are mathematically added to the 49 patches, ensuring the Transformer understands spatial geometry (Top-Left vs. Bottom-Right).


Phase 2: Generating the Caption (The Language)
The model utilizes a GPT-2 Decoder in an autoregressive loop:

1) Self-Attention: GPT-2 analyzes previously generated words to maintain grammatical flow.

2) Cross-Attention (The Neural Bridge): The decoder uses text-based Queries to search the image-based Keys and Values.

3) Prediction: The model fuses visual context with linguistic patterns to predict the next word in the sequence.

Phase 3: Explainability (The XAI)
During the Cross-Attention step, the model calculates a probability distribution across the 49 image patches. We extract these Attention Weights, upsample them, and overlay them as heatmaps to visualize
word-to-pixel grounding.   


🛠️ Technical Challenges & Solutions
Training Optimization
1. The "NaN" Loss Meltdown (Exploding Gradients)

--> Challenge: Deep layers caused gradients to multiply exponentially, leading to numerical overflow (NaN).

--> Solution: Implemented Gradient Clipping (max_norm=0.5) and custom Weight Initialization (std=0.02) to stabilize backpropagation.

2. "Lazy" Heatmaps (Permutation Invariance)

--> Challenge: Initial heatmaps only focused on the image center because the Transformer treated patches as an unordered bag of pixels.

--> Solution: Injected 2D Positional Embeddings into the visual features, forcing the model to learn spatial coordinates.



Deployment Engineering (AWS EC2)
3. The OOM Killer (Memory Management)

--> Challenge: Loading a 712MB model on a 1GB/2GB RAM instance triggered the Linux Out-of-Memory (OOM) Killer.

--> Solution: Created a 4GB Swap File on the SSD and used mmap=True in PyTorch to load weights without a memory spike.

4. Inference Latency (3-Minute Lag)

--> Challenge: CPU-based inference was bottlenecked by 32-bit floating-point math.

--> Solution: Implemented Dynamic 8-bit Quantization (INT8), reducing latency from 180 seconds to <45 seconds.


🚀 Deployment Stats
Infrastructure: AWS EC2 t3.small

Web Server: Flask + Gunicorn (Systemd managed)

Network: Port 80 (Standard HTTP)

OS: Ubuntu 24.04 LTS


📦 Installation & Setup
1) Clone the repository:

Bash

git clone https://github.com/your-username/Caption_Generation_From_Image.git
cd Caption_Generation_From_Image

2) Setup Virtual Environment:

Bash

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3) Run Production Server:

Bash

sudo .venv/bin/python app.py

📄 License
Distributed under the MIT License. See LICENSE for more information.

Developed by Nitesh Kumar Varma — CSE Student & AI Enthusiast

