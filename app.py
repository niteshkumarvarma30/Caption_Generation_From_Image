# app.py
from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from transformers import GPT2Tokenizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import matplotlib
matplotlib.use('Agg') # Prevents server crashes

from models import XAICaptioner

app = Flask(__name__)

# --- Load AI Model ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = XAICaptioner().to(DEVICE)
model.load_state_dict(torch.load("lia_model_full_ep3.pth", map_location=DEVICE, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    raw_image = Image.open(file.stream).convert("RGB")
    vis_image = raw_image.resize((224, 224))
    image_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)

    # 1. AI Generation
    with torch.no_grad():
        features = model.encoder(image_tensor)
        features = features.permute(0, 2, 3, 1).flatten(1, 2)
        encoder_hidden_states = model.vis_norm(model.bridge(features)) * 0.1
        encoder_hidden_states = encoder_hidden_states + model.vis_pos_embed

        input_ids = torch.tensor([[tokenizer.eos_token_id]]).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)

        generated_ids = model.decoder.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            max_length=20, num_beams=1, pad_token_id=tokenizer.pad_token_id
        )[0]
        
        outputs = model.decoder(
            input_ids=generated_ids.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states, output_attentions=True
        )

    # 2. XAI Extraction
    cross_attention = outputs.cross_attentions[-1] 
    avg_attention = cross_attention.mean(dim=1).squeeze(0)

    tokens = []
    for idx, t in enumerate(generated_ids):
        word = tokenizer.decode([t])
        tokens.append(word)
        if idx > 0 and ("." in word or t == tokenizer.eos_token_id): break

    num_words = len(tokens)
    cols = 5
    rows = int(np.ceil(num_words / cols))
    
    # 3. Draw Heatmaps & Convert to Base64 (No saving to disk!)
    fig = plt.figure(figsize=(15, rows * 3.5))
    for i in range(num_words):
        word = tokens[i].strip()
        if not word: word = "[SPACE]"
        ax = plt.subplot(rows, cols, i + 1)
        ax.set_title(word, fontsize=14, fontweight='bold')
        plt.imshow(vis_image)
        attn_map = avg_attention[i].view(7, 7).cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        plt.imshow(attn_map, cmap='jet', alpha=0.5, extent=[0, 224, 224, 0], interpolation='bicubic')
        plt.axis('off')
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # 4. Return Data to Frontend
    caption = tokenizer.decode(generated_ids[:num_words], skip_special_tokens=True).lstrip(", ").replace(" .", "").strip()
    return jsonify({'caption': caption, 'heatmap_image': image_base64})

if __name__ == '__main__':
    app.run(debug=True, port=5000)