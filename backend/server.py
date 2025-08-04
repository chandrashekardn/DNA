from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import Document, init_beanie
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from enum import Enum
import os
import logging
import uuid
import asyncio
import numpy as np
from contextlib import asynccontextmanager
import torch
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Global variables for model components
model = None
tokenizer = None
explainer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class SequenceAnalysis(Document):
    sequence_id: str = Field(..., description="Unique identifier for the sequence")
    patient_id: Optional[str] = Field(None, description="Patient identifier if applicable")
    sequence_data: str = Field(..., description="Original DNA sequence")
    sequence_length: int = Field(..., description="Length of the sequence")
    sequence_type: str = Field(default="genomic", description="Type of sequence")
    gc_content: float = Field(..., description="GC content percentage")
    complexity_score: float = Field(..., description="Sequence complexity metric")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        name = "sequence_analyses"

class DiseasePredictionResult(Document):
    prediction_id: str = Field(..., description="Unique identifier for the prediction")
    sequence_id: str = Field(..., description="Reference to sequence analysis")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    disease_name: str = Field(..., description="Name of the predicted disease")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence_interval_lower: float = Field(..., description="Lower bound of confidence interval")
    confidence_interval_upper: float = Field(..., description="Upper bound of confidence interval")
    risk_category: RiskLevel = Field(..., description="Risk categorization")
    model_version: str = Field(..., description="Version of the model used")
    variant_effects: List[Dict[str, Any]] = Field(default_factory=list)
    explainability_data: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = Field(..., description="Time taken for prediction in seconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        name = "disease_predictions"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    mongo_url = os.environ['MONGO_URL']
    client = AsyncIOMotorClient(mongo_url)
    database = client[os.environ['DB_NAME']]
    
    await init_beanie(
        database=database,
        document_models=[SequenceAnalysis, DiseasePredictionResult]
    )
    
    # Load Caduceus model (simplified version for demo)
    global model, tokenizer
    logger.info("Loading Caduceus model components...")
    
    try:
        # For this demo, we'll use a placeholder model architecture
        # In production, you would load the actual Caduceus model from HuggingFace
        model = SimpleGenomicModel()
        tokenizer = DNATokenizer()
        logger.info("Model components loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Continue without model for demo purposes
    
    yield
    
    # Cleanup
    client.close()

# Create FastAPI app
app = FastAPI(
    title="Caduceus DNA Disease Prediction API",
    description="Advanced DNA sequence analysis for multi-disease prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
api_router = APIRouter(prefix="/api")

# Simplified model classes for demo (in production, use actual Caduceus)
class SimpleGenomicModel:
    def __init__(self):
        self.device = "cpu"
        
    def predict_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Generate simplified embeddings for sequences"""
        embeddings = []
        for seq in sequences:
            # Calculate basic sequence features
            gc_content = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
            length_norm = min(len(seq) / 1000, 1.0)
            
            # Create simple feature vector
            features = np.array([
                gc_content,
                length_norm,
                seq.count('A') / len(seq) if len(seq) > 0 else 0,
                seq.count('T') / len(seq) if len(seq) > 0 else 0,
                seq.count('C') / len(seq) if len(seq) > 0 else 0,
                seq.count('G') / len(seq) if len(seq) > 0 else 0,
            ])
            embeddings.append(features)
        
        return np.array(embeddings)

class DNATokenizer:
    def __init__(self):
        self.vocab = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    
    def encode_plus(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Tokenize DNA sequence"""
        tokens = [self.vocab.get(nucleotide, 4) for nucleotide in sequence.upper()]
        max_length = kwargs.get('max_length', 4096)
        tokens = tokens[:max_length]  # Truncate if needed
        
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens)
        }

# Input/Output models
class DNASequenceInput(BaseModel):
    sequence: str = Field(..., description="DNA sequence using ATCG nucleotides")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    sequence_type: str = Field("genomic", description="Type of sequence: genomic, coding, regulatory")
    max_length: int = Field(4096, description="Maximum sequence length to process")
    
    @validator('sequence')
    def validate_dna_sequence(cls, v):
        valid_chars = set('ATCGN')
        if not all(c.upper() in valid_chars for c in v):
            raise ValueError('Sequence contains invalid nucleotides. Only A, T, C, G, N allowed.')
        if len(v) < 50:
            raise ValueError('Sequence too short. Minimum 50 nucleotides required.')
        if len(v) > 131072:
            raise ValueError('Sequence too long. Maximum 131k nucleotides supported.')
        return v.upper()

class DiseasePrediction(BaseModel):
    disease_name: str
    probability: float
    confidence_interval: List[float]
    risk_category: str
    variant_effects: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[DiseasePrediction]
    sequence_analysis: Dict[str, Any]
    explainability: Dict[str, Any]
    processing_time: float
    model_version: str

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of the DNA sequence."""
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0

def calculate_sequence_complexity(sequence: str) -> float:
    """Calculate sequence complexity based on nucleotide distribution."""
    counts = Counter(sequence)
    total = len(sequence)
    if total == 0:
        return 0.0
    
    # Calculate Shannon entropy
    entropy = -sum((count/total) * np.log2(count/total) for count in counts.values() if count > 0)
    return entropy / 2.0  # Normalize to 0-1 range

async def perform_disease_prediction(embeddings: np.ndarray, sequence: str) -> List[DiseasePrediction]:
    """Perform disease prediction based on sequence embeddings."""
    diseases = [
        "Cardiovascular Disease",
        "Type 2 Diabetes", 
        "Alzheimer's Disease",
        "Breast Cancer",
        "Colorectal Cancer",
        "Huntington's Disease",
        "Cystic Fibrosis",
        "Sickle Cell Disease"
    ]
    
    predictions = []
    
    # Extract features from embeddings
    if embeddings.ndim > 1:
        features = embeddings[0]  # Take first sample
    else:
        features = embeddings
    
    for i, disease in enumerate(diseases):
        # Generate realistic-looking probabilities based on sequence features
        seed = hash(sequence + disease) % 10000
        np.random.seed(seed)
        
        # Use sequence features to influence probability
        gc_content = features[0] if len(features) > 0 else 0.5
        base_prob = 0.05 + (gc_content * 0.4) + (np.random.random() * 0.3)
        
        # Adjust for specific diseases
        if "Cancer" in disease:
            base_prob *= 1.2  # Slightly higher for cancer types
        elif "Diabetes" in disease:
            base_prob *= 0.8  # Lower for diabetes
        
        probability = min(max(base_prob, 0.01), 0.95)
        
        risk_category = "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"
        
        predictions.append(DiseasePrediction(
            disease_name=disease,
            probability=probability,
            confidence_interval=[max(0, probability - 0.1), min(1, probability + 0.1)],
            risk_category=risk_category,
            variant_effects=[]
        ))
    
    return predictions

async def generate_explainability(sequence: str, predictions: List[DiseasePrediction]) -> Dict[str, Any]:
    """Generate explainability analysis for predictions."""
    # Simplified explainability for demo
    sequence_length = len(sequence)
    gc_content = calculate_gc_content(sequence)
    
    # Identify important regions (simplified)
    important_regions = []
    chunk_size = 50
    
    for i in range(0, min(sequence_length, 200), chunk_size):  # Analyze first 200 bp
        chunk = sequence[i:i+chunk_size]
        chunk_gc = calculate_gc_content(chunk)
        
        # Regions with extreme GC content are "important"
        if chunk_gc < 0.3 or chunk_gc > 0.7:
            importance = abs(chunk_gc - 0.5) * 2  # 0-1 scale
            important_regions.append({
                "start_position": i,
                "end_position": min(i + chunk_size, sequence_length),
                "sequence": chunk,
                "average_attribution": importance,
                "effect_type": "positive" if chunk_gc > 0.5 else "negative",
                "length": len(chunk)
            })
    
    # Sort by importance
    important_regions.sort(key=lambda x: x["average_attribution"], reverse=True)
    
    return {
        "important_regions": important_regions[:5],  # Top 5 regions
        "summary_statistics": {
            "total_positive_attribution": sum(r["average_attribution"] for r in important_regions if r["effect_type"] == "positive"),
            "total_negative_attribution": sum(r["average_attribution"] for r in important_regions if r["effect_type"] == "negative"),
            "sequence_gc_content": gc_content,
            "sequence_complexity": calculate_sequence_complexity(sequence)
        },
        "motif_analysis": {
            "CpG_sites": {
                "occurrences": sequence.count("CG"),
                "average_attribution": 0.15 if sequence.count("CG") > 0 else 0
            },
            "AT_rich_regions": {
                "occurrences": sequence.count("AAAA") + sequence.count("TTTT"),
                "average_attribution": 0.1 if (sequence.count("AAAA") + sequence.count("TTTT")) > 0 else 0
            }
        }
    }

async def log_prediction(patient_id: str, sequence_length: int, 
                        predictions: List[DiseasePrediction], processing_time: float):
    """Log prediction for monitoring and analytics."""
    logger.info(f"Prediction completed: patient={patient_id}, "
                f"length={sequence_length}, time={processing_time:.2f}s, "
                f"predictions={len(predictions)}")

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_disease_risk(
    input_data: DNASequenceInput,
    background_tasks: BackgroundTasks
):
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Generate embeddings using simplified model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        embeddings = model.predict_embeddings([input_data.sequence])
        
        # Perform disease prediction
        predictions = await perform_disease_prediction(embeddings, input_data.sequence)
        
        # Generate explainability analysis
        explainability = await generate_explainability(input_data.sequence, predictions)
        
        # Calculate processing time
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Store sequence analysis
        sequence_analysis = SequenceAnalysis(
            sequence_id=str(uuid.uuid4()),
            patient_id=input_data.patient_id,
            sequence_data=input_data.sequence,
            sequence_length=len(input_data.sequence),
            sequence_type=input_data.sequence_type,
            gc_content=calculate_gc_content(input_data.sequence),
            complexity_score=calculate_sequence_complexity(input_data.sequence)
        )
        await sequence_analysis.insert()
        
        # Store prediction results
        for pred in predictions:
            prediction_result = DiseasePredictionResult(
                prediction_id=str(uuid.uuid4()),
                sequence_id=sequence_analysis.sequence_id,
                patient_id=input_data.patient_id,
                disease_name=pred.disease_name,
                probability=pred.probability,
                confidence_interval_lower=pred.confidence_interval[0],
                confidence_interval_upper=pred.confidence_interval[1],
                risk_category=RiskLevel(pred.risk_category.lower()),
                model_version="caduceus-demo-1.0",
                variant_effects=pred.variant_effects,
                explainability_data=explainability,
                processing_time=processing_time
            )
            await prediction_result.insert()
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            input_data.patient_id,
            len(input_data.sequence),
            predictions,
            processing_time
        )
        
        return PredictionResponse(
            predictions=predictions,
            sequence_analysis={
                "length": len(input_data.sequence),
                "gc_content": calculate_gc_content(input_data.sequence),
                "complexity_score": calculate_sequence_complexity(input_data.sequence),
                "sequence_id": sequence_analysis.sequence_id
            },
            explainability=explainability,
            processing_time=processing_time,
            model_version="caduceus-demo-1.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.get("/history/{patient_id}")
async def get_patient_history(patient_id: str):
    """Get prediction history for a patient."""
    try:
        predictions = await DiseasePredictionResult.find(
            DiseasePredictionResult.patient_id == patient_id
        ).sort(-DiseasePredictionResult.created_at).limit(50).to_list()
        
        return {"history": [pred.dict() for pred in predictions]}
        
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch patient history")

@api_router.get("/")
async def root():
    return {"message": "Caduceus DNA Disease Prediction API", "status": "active"}

# Include the router in the main app
app.include_router(api_router)