import os
import json
import boto3
import torch
import torch.nn.functional as F
from PIL import Image
import io

from src.models.sign_classifier import SignClassifier
from src.config import ModelConfig, OptimConfig

s3 = boto3.client('s3')
model = None

def load_model():
    global model
    if model is None:
        checkpoint_path = "/var/task/models/toy_model.ckpt"
        print(f"Loading model with manual config from {checkpoint_path}...")
        
        m_cfg = ModelConfig()
        o_cfg = OptimConfig()
        
        model = SignClassifier.load_from_checkpoint(
            checkpoint_path,
            model_cfg=m_cfg,
            optim_cfg=o_cfg,
            map_location=torch.device('cpu')
        )
        model.eval()
    return model

def handler(event, context):
    print("--- STARTING HANDLER ---")
    try:
        if 'Records' not in event:
            return {"status": "success", "message": "Import and Init verified!"}

        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        print(f"Processing: {key}")

        current_model = load_model()

        response = s3.get_object(Bucket=bucket, Key=key)
        image = Image.open(io.BytesIO(response['Body'].read())).convert('L')
        
        image = image.resize((28, 28))
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(1, 1, 28, 28) / 255.0
        
        with torch.no_grad():
            logits = current_model(tensor)
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        
        prediction_letter = chr(ord('A') + pred.item())
        print(f"Result: {prediction_letter}")

        result = {"input": key, "letter": prediction_letter, "status": "success"}
        result_key = key.rsplit('.', 1)[0] + ".json"
        
        s3.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=json.dumps(result),
            ContentType='application/json'
        )
        print(f"JSON saved to {result_key}")
        
        return {"status": "success", "prediction": prediction_letter}

    except Exception as e:
        print(f"!!! ERROR: {str(e)}")
        raise e