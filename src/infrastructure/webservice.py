import asyncio
import threading
import logging
import json
import base64
import io
import requests
import cv2
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image

logger = logging.getLogger("WebService")

class AnnotationRequest(BaseModel):
    image: str  # Base64 encoded image
    callback_url: str

class WebService:
    def __init__(self, annotation_app):
        self.app = FastAPI()
        self.annotation_app = annotation_app
        self.server = None
        self.server_thread = None
        self.current_request = None
        self.is_running = False
        
        # Enable CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins (you can restrict this to specific origins)
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )
        
        # Setup FastAPI routes
        self.app.post("/annotate")(self.handle_annotation_request)
        
    async def handle_annotation_request(self, request: AnnotationRequest):
        """Handle incoming annotation requests"""
        try:
            logger.info("Received annotation request")
            
            # Check if service is busy
            if self.current_request is not None:
                raise HTTPException(status_code=503, detail="Service is currently busy with another annotation")
            
            # Store the current request
            self.current_request = request
            
            # Decode the base64 image (remove data URL prefix, fix padding, validate)
            b64_image = request.image
            if b64_image.startswith('data:image'):
                b64_image = b64_image.split(',', 1)[-1]
            b64_image = b64_image.strip()
            if not b64_image:
                raise HTTPException(status_code=400, detail="Empty base64 image string")
            missing_padding = len(b64_image) % 4
            if missing_padding:
                b64_image += '=' * (4 - missing_padding)
            try:
                image_data = base64.b64decode(b64_image)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                logger.error(f"Invalid image data: {e}")
                raise HTTPException(status_code=400, detail="Invalid or corrupt base64 image data")
            
            # Save the image temporarily
            temp_image_path = "/tmp/temp_annotation_image.png"
            image.save(temp_image_path)
            
            # Signal the main app to load the image and enter annotation mode
            self.annotation_app.load_web_service_image(temp_image_path, request)
            
            return {"status": "accepted", "message": "Annotation request received"}
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Error handling annotation request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def start_server(self):
        """Start the FastAPI server in a background thread"""
        if self.is_running:
            return

        def run_server():
            config = uvicorn.Config(self.app, host="localhost", port=8000, log_level="info")
            self.server = uvicorn.Server(config)
            self.server.run()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        logger.info("Web service started on http://localhost:8000")
    
    def stop_server(self):
        """Stop the FastAPI server"""
        if not self.is_running:
            return

        if self.server:
            self.server.should_exit = True
            if self.server_thread:
                self.server_thread.join(timeout=5)
        self.is_running = False
        self.current_request = None
        logger.info("Web service stopped")
    
    def submit_annotations(self, annotation_images: Dict[str, np.ndarray]):
        """Submit completed annotations to the callback URL"""
        if not self.current_request:
            logger.error("No current request to submit annotations for")
            return False
        
        try:
            # Prepare the response data
            response_data = {
                "status": "completed",
                "annotations": {}
            }
            
            # Convert annotation images to base64
            for layer_name, annotation_image in annotation_images.items():
                # Convert numpy array to PNG bytes
                success, buffer = cv2.imencode('.png', annotation_image)
                if success:
                    # Convert to base64
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    response_data["annotations"][layer_name] = img_base64
                else:
                    logger.error(f"Failed to encode annotation image for layer {layer_name}")
            
            # Send to callback URL
            response = requests.post(
                self.current_request.callback_url,
                json=response_data,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Annotations submitted successfully")
                self.current_request = None
                return True
            else:
                logger.error(f"Failed to submit annotations: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting annotations: {e}")
            return False
    
    def cancel_current_request(self):
        """Cancel the current annotation request"""
        if not self.current_request:
            return
            
        try:
            # Send cancellation to callback URL
            response_data = {
                "status": "cancelled",
                "annotations": {}
            }
            
            response = requests.post(
                self.current_request.callback_url,
                json=response_data,
                timeout=30
            )
            
            logger.info("Annotation request cancelled")
            
        except Exception as e:
            logger.error(f"Error sending cancellation: {e}")
        finally:
            self.current_request = None