import json
import httpx
from pydantic import BaseModel, Field
from fastapi import FastAPI
from typing import Dict, Any, Optional
import uvicorn


app = FastAPI()


class PredictRequest(BaseModel):
    hf_pipeline: str
    model_deployed_url: str
    inputs: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

@app.get(path="/status")
async def status():
    
    return {"Hello": "World"}

@app.post(path="/predict")
async def predict(request: PredictRequest):
    print(request)
    # Write your code here to translate input into V2 protocol and send it to model_deployed_url
    if request.hf_pipeline =="text-generation":
        v2_inputs = {
            "inputs": {
                "text": {
                    "dtype": "string",
                    "data": [request.inputs]
                }
            }
        }
    elif request.hf_pipeline == "zero-shot-classification":
        v2_inputs = {
            "inputs": {
                "sequence": {
                    "dtype": "string",
                    "data": [request.inputs]
                },
                "candidate_labels": {
                    "dtype": "string",
                    "data": request.parameters.get("candidate_labels", [])
                }
            }
        }
    elif request.hf_pipeline =="token-classification":
        v2_inputs = {
            "inputs": {
                "tokens": {
                    "dtype": "string",
                    "shape": [-1],
                    "data": request.inputs.split()
                }
            }
        }
    elif request.hf_pipeline == "object-detection":
        image_bytes = await download_image(request.inputs)
        v2_inputs = {
            "inputs": {
                "image": {
                    "dtype": "BYTES",
                    "shape": [len(image_bytes)],
                    "data": list(image_bytes)
                }
            }
        }
    else:
        return {"error": "Unsupported pipeline type"}
    

    response = await httpx.post(request.model_deployed_url, json=v2_inputs)

    # Parse V2 output and convert to native format based on pipeline type
    if response.status_code == 200:
        v2_output = response.json()
        if request.hf_pipeline == "text-generation":
            output = v2_output["outputs"]["generated_text"]["data"][0]
        elif request.hf_pipeline == "zero-shot-classification":
            output = {
                "sequence_classification": {
                    "scores": v2_output["outputs"]["scores"]["data"],
                    "labels": request.parameters.get("candidate_labels", [])
                }
            }
        elif request.hf_pipeline == "token-classification":
            output = {
                "entity_classification": [
                    {"entity": entity, "label": label}
                    for entity, label in zip(v2_output["outputs"]["entities"]["data"], v2_output["outputs"]["tags"]["data"])
                ]
            }
        elif request.hf_pipeline == "object-detection":
            output = {
                "detections": [
                    {
                        "label": label,
                        "confidence": confidence,
                        "bounding_box": {
                            "xmin": bbox[0],
                            "ymin": bbox[1],
                            "xmax": bbox[2],
                            "ymax": bbox[3]
                        }
                    }
                    for label, confidence, bbox in zip(
                        v2_output["outputs"]["detection_classes"]["data"],
                        v2_output["outputs"]["detection_scores"]["data"],
                        v2_output["outputs"]["detection_boxes"]["data"]
                    )
                ]
            }
        else:
            return {"error": "Unsupported pipeline type"}
        
        return output
    else:
        return {"error": "Received error response from model deployment endpoint"}

async def download_image(image_url: str) -> bytes:
    if image_url.startswith("http"):
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
        if response.status_code == 200:
            return response.content
        else:
            raise ValueError(f"Failed to download image from URL {image_url}: {response.text}")
    elif "," in image_url:
        # Assume this is a base64-encoded image
        encoded_image_str = image_url.split(",")[1]
        return encoded_image_str.encode("utf-8")
    else:
        raise ValueError("Invalid image URL") 


