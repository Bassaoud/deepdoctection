from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import deepdoctection as dd
import os
import uuid
from pathlib import Path
import shutil

# Modèles Pydantic
class DocumentClassification(BaseModel):
    document_id: str
    document_name: str
    document_type: str
    confidence: float
    description: str
    text_content: str

# Configuration de l'analyseur avec le fichier de configuration personnalisé
config_path = "/app/configs/custom_config.yaml"
analyzer = dd.get_dd_analyzer(path_config_file=config_path)

class TrainingDocument(BaseModel):
    document_id: str
    document_type: str
    status: str

class TrainingResponse(BaseModel):
    trained_documents: List[TrainingDocument]
    message: str

# Création de l'application FastAPI
app = FastAPI(title="Document Classification API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation de l'analyseur
analyzer = dd.get_dd_analyzer()

# Création des dossiers nécessaires
UPLOAD_DIR = Path("uploads")
TRAINING_DIR = Path("training")
UPLOAD_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)

def classify_document_content(text_content: str) -> tuple[str, float, str]:
    """
    Classifie le document basé sur son contenu
    Retourne (type_document, confiance, description)
    """
    # Règles simples de classification basées sur des mots-clés
    keywords = {
        "facture": (
            "invoice",
            0.9,
            "Document de facturation contenant des informations de paiement"
        ),
        "contrat": (
            "contract", 
            0.85,
            "Document contractuel définissant des termes et conditions"
        ),
        "rapport": (
            "report",
            0.8,
            "Rapport contenant des analyses et informations"
        ),
        "formulaire": (
            "form",
            0.75,
            "Formulaire à remplir avec des champs structurés"
        )
    }
    
    text_lower = text_content.lower()
    max_confidence = 0.0
    doc_type = "unknown"
    description = "Document non classifié"
    
    for keyword, (type_name, confidence, desc) in keywords.items():
        if keyword in text_lower:
            if confidence > max_confidence:
                max_confidence = confidence
                doc_type = type_name
                description = desc
    
    return doc_type, max_confidence if max_confidence > 0 else 0.3, description

@app.post("/analyze/", response_model=DocumentClassification)
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyse et classifie un document PDF
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")
    
    try:
        # Générer un ID unique pour le document
        doc_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{doc_id}.pdf"
        
        # Sauvegarder le fichier
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyser le document avec deepdoctection
        df = analyzer.analyze(path=str(file_path))
        df.reset_state()
        
        # Extraire le texte de toutes les pages
        text_content = ""
        for page in df:
            text_content += page.text + "\n"
        
        # Classifier le document
        doc_type, confidence, description = classify_document_content(text_content)
        
        # Nettoyage du fichier temporaire
        file_path.unlink()
        
        return DocumentClassification(
            document_id=doc_id,
            document_name=file.filename,
            document_type=doc_type,
            confidence=confidence,
            description=description,
            text_content=text_content[:500] + "..." if len(text_content) > 500 else text_content
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse du document: {str(e)}")

@app.post("/train/", response_model=TrainingResponse)
async def train_model(files: List[UploadFile] = File(...), document_type: str = None):
    """
    Endpoint pour entraîner le modèle avec de nouveaux documents
    """
    if not document_type:
        raise HTTPException(status_code=400, detail="Le type de document est requis")
    
    trained_docs = []
    
    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
                
            doc_id = str(uuid.uuid4())
            file_path = TRAINING_DIR / f"{doc_id}.pdf"
            
            # Sauvegarder le fichier
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Ajouter à la liste des documents traités
            trained_docs.append(
                TrainingDocument(
                    document_id=doc_id,
                    document_type=document_type,
                    status="processed"
                )
            )
            
        return TrainingResponse(
            trained_documents=trained_docs,
            message=f"Formation réussie pour {len(trained_docs)} documents"
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'entraînement: {str(e)}"
        )

# Initialize the deepdoctection analyzer
analyzer = dd.get_dd_analyzer()

@app.post("/classify_pdf/")
async def classify_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    
    # Analyze the PDF
    df = analyzer.analyze(path=file.filename)
    df.reset_state()
    
    # Get the first page
    page = next(iter(df))
    
    # Extract document type (assuming it's available)
    doc_type = page.document_type if page.document_type else "Unknown"
    
    return {"document_type": doc_type}


@app.get("/health")
async def health_check():
    """
    Vérification de l'état de l'API
    """
    return {"status": "healthy", "analyzer": "ready"}