"""
============================================
Fashion Flux - FastAPI Backend
REST API for Fashion Recommendation System
============================================

This module provides a REST API endpoint for fashion recommendations
using a CNN (ResNet50) for feature extraction and Nearest Neighbors
for similarity search.

Author: Fashion Flux Team
"""

import os
import pickle
import io
from typing import List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.linalg import norm
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# TensorFlow imports - optimized for inference
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Reduce TF logging noise

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D


# ==================== CONFIGURATION ====================
UPLOAD_DIR = "uploads"
IMAGES_DIR = "images"
EMBEDDINGS_FILE = "embeddings.pkl"
FILENAMES_FILE = "filenames.pkl"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
NUM_RECOMMENDATIONS = 5

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=2)


# ==================== PYDANTIC MODELS ====================
class RecommendationItem(BaseModel):
    """Single recommendation item with image URL and similarity score"""
    image_url: str
    similarity: float


class RecommendationResponse(BaseModel):
    """API response containing list of recommendations"""
    recommendations: List[RecommendationItem]
    message: str = "Recommendations generated successfully"


# ==================== GLOBAL VARIABLES ====================
model = None
feature_list = None
filenames = None
neighbors = None


# ==================== MODEL & DATA LOADING ====================
def load_model():
    """Load and configure the ResNet50 model for feature extraction."""
    print("Loading ResNet50 model...")

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    print("Model loaded successfully!")
    return model


def load_embeddings():
    """Load precomputed embeddings and filenames from pickle files."""
    print("Loading embeddings and filenames...")

    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

    if not os.path.exists(FILENAMES_FILE):
        raise FileNotFoundError(f"Filenames file not found: {FILENAMES_FILE}")

    with open(EMBEDDINGS_FILE, 'rb') as f:
        feature_list = np.array(pickle.load(f))

    with open(FILENAMES_FILE, 'rb') as f:
        filenames = np.array(pickle.load(f))

    print(f"Loaded {len(filenames)} embeddings successfully!")
    return feature_list, filenames


def initialize_neighbors(feature_list: np.ndarray) -> NearestNeighbors:
    """Initialize and fit the NearestNeighbors model."""
    print("Initializing NearestNeighbors...")
    neighbors = NearestNeighbors(
        n_neighbors=NUM_RECOMMENDATIONS,
        algorithm='brute',
        metric='euclidean'
    )
    neighbors.fit(feature_list)
    print("NearestNeighbors initialized!")
    return neighbors


# ==================== LIFESPAN CONTEXT MANAGER ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - loads model and data on startup."""
    global model, feature_list, filenames, neighbors

    print("\n" + "=" * 50)
    print("Fashion Flux Backend Starting...")
    print("=" * 50 + "\n")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    model = load_model()
    feature_list, filenames = load_embeddings()
    neighbors = initialize_neighbors(feature_list)

    print("\n" + "=" * 50)
    print("Backend Ready! Server is running...")
    print("=" * 50 + "\n")

    yield

    print("\nShutting down Fashion Flux Backend...")
    for f in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, f))
        except Exception:
            pass


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Fashion Flux API",
    description="AI-powered fashion recommendation system using CNN feature extraction",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static images
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ==================== HELPER FUNCTIONS ====================
def validate_file(file: UploadFile) -> None:
    """Validate uploaded file type."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def extract_features(img_data: bytes) -> np.ndarray:
    """Extract feature vector from image bytes using the CNN model."""
    # Load image from bytes for faster processing (no disk I/O)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def get_recommendations(features: np.ndarray) -> List[RecommendationItem]:
    """Find similar items using nearest neighbors search."""
    distances, indices = neighbors.kneighbors([features])

    recommendations = []

    for distance, idx in zip(distances[0], indices[0]):
        # Convert distance to similarity percentage
        similarity = float(np.exp(-distance / 10) * 100)
        similarity = min(99.9, max(0.1, similarity))

        # Get filename and convert to URL path
        original_path = filenames[idx]
        filename = os.path.basename(original_path)
        image_url = f"images/{filename}"

        recommendations.append(RecommendationItem(
            image_url=image_url,
            similarity=round(similarity, 1)
        ))

    return recommendations


# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Fashion Flux API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "embeddings_loaded": feature_list is not None,
        "num_items": len(filenames) if filenames is not None else 0
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(file: UploadFile = File(...)):
    """
    Get fashion recommendations based on uploaded image.

    Args:
        file: Uploaded image file (JPG, JPEG, or PNG)

    Returns:
        RecommendationResponse: List of recommended items with similarity scores
    """
    try:
        validate_file(file)

        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )

        # Verify it's a valid image
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file"
            )

        # Extract features directly from bytes (no file I/O)
        import asyncio
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(executor, extract_features, content)
        recommendations = get_recommendations(features)

        return RecommendationResponse(
            recommendations=recommendations,
            message="Recommendations generated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable (for cloud deployment) or default to 8000
    port = int(os.environ.get("PORT", 8000))

    print("\n" + "=" * 50)
    print("Starting Fashion Flux Backend Server")
    print("=" * 50)
    print(f"\nServer running on port: {port}")
    print("API Documentation available at /docs")
    print("=" * 50 + "\n")

    # Use app object directly to avoid re-importing module
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
