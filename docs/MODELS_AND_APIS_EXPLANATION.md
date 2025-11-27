# Models and APIs Explanation - Agentic Brand Classifier V3

## 🔍 SerpAPI Usage

### **Answer: NO, SerpAPI is NOT currently being used**

### Current Status:
- ❌ **Not Active**: SerpAPI is configured but **not actually used** in the codebase
- 📦 **Dependency Present**: `serpapi>=0.1.4` is in `requirements.txt`
- 🔑 **API Key Check**: Code checks for `SERPAPI_API_KEY` environment variable
- 📝 **Future Enhancement**: Marked as TODO for future implementation

### Where SerpAPI is Referenced:

1. **Campaign Agent** (`agent/campaign_agent.py`):
   ```python
   self.api_key = os.getenv("SERPAPI_API_KEY")  # Line 39
   ```
   - API key is retrieved but **never used**
   - Comment says: `# - SerpAPI Google News search (already configured but not used)`

2. **Documentation**:
   - README mentions it's optional
   - Architecture flowchart shows it as a future enhancement
   - TODO comments indicate it could be used for brand validation

### Why Not Used:
- **Cost Efficiency**: System is designed to work **100% locally** with Ollama
- **Speed**: Local processing is faster than external API calls
- **Privacy**: No data leaves the local environment
- **Simplicity**: Current implementation achieves goals without external APIs

### Future Potential Use:
If enabled, SerpAPI could be used for:
- Brand validation via Google search
- Real campaign data from Google News
- Brand mention tracking
- Evidence gathering for confidence scoring

---

## 🤖 Models Used to Refine Prompt Results

The system uses **3 main models/components** to process and refine prompts:

### 1. **DSPy Framework with Ollama/phi3** (Primary LLM)

**What it is:**
- **DSPy**: A framework for building LLM applications with structured prompts
- **Ollama**: Local LLM runtime (runs models locally, no cloud needed)
- **phi3**: Microsoft's small language model (3.8B parameters)

**Configuration:**
```python
model_name = os.getenv("DSPY_MODEL_NAME", "ollama/phi3")
max_tokens = int(os.getenv("DSPY_MAX_TOKENS", "4096"))
temperature = float(os.getenv("DSPY_TEMPERATURE", "0.2"))
```

**Where it's used:**
- ✅ **Brand Agent**: 
  - `BrandValidationSignature` - Validates which entities are brands
  - `BrandExpansionSignature` - Infers related brands (competitors, parent companies)
- ✅ **Category Agent**: 
  - `CategoryExtractionSignature` - Extracts product/service categories
- ✅ **Campaign Agent**: 
  - `CampaignExtractionSignature` - Extracts campaign information
- ✅ **Reach Agent**: 
  - `ReachEstimationSignature` - Estimates campaign reach
- ✅ **Brand Lift Agent**: 
  - `BrandLiftEstimationSignature` - Calculates brand lift percentage

**What it does:**
- Takes structured prompts (signatures) with input/output fields
- Uses the LLM to extract/validate information
- Returns structured outputs (brands, categories, campaigns, etc.)

**Benefits:**
- Local processing (no API costs)
- Fast inference (~2-8 seconds per query)
- Privacy (data stays local)
- Structured outputs (reliable parsing)

---

### 2. **SpaCy Transformer Model (en_core_web_trf)** (NER)

**What it is:**
- **SpaCy**: Natural Language Processing library
- **en_core_web_trf**: Transformer-based English model (RoBERTa-based)
- Used for Named Entity Recognition (NER)

**Where it's used:**
- ✅ **Brand Agent** (`brand_extraction/entity_extractor.py`):
  - Extracts named entities (ORG, PERSON, etc.) from prompts
  - Provides initial candidates for brand extraction

**What it does:**
- Analyzes text to find entities (organizations, people, locations)
- Tags entities with labels (ORG, PERSON, GPE, etc.)
- Provides start/end positions in text
- Part-of-speech tagging

**Example:**
```
Input: "Which tablet is best: iPad Pro or Samsung Galaxy Tab?"
Output: 
  - "iPad Pro" → ORG entity
  - "Samsung Galaxy Tab" → ORG entity
```

**Benefits:**
- Fast entity extraction (~50-200ms)
- High accuracy for common entities
- Provides context for brand extraction

---

### 3. **Fuzzy Matching & Confidence Scoring** (Rule-based)

**What it is:**
- Custom algorithms for matching and scoring
- Not a neural model, but rule-based logic

**Where it's used:**
- ✅ **Brand Agent**: 
  - Fuzzy matching against known brands list
  - Match score calculation (0.0-1.0)
- ✅ **Category Agent**: 
  - Fuzzy matching against known categories
  - Category normalization
- ✅ **All Agents**: 
  - Multi-factor confidence scoring
  - Range validation for numeric predictions

**What it does:**
- Compares extracted entities against known lists
- Calculates similarity scores (fuzzy matching)
- Determines confidence levels based on multiple factors:
  - NER entity count
  - Known brand/category matching
  - Extraction ratio
  - Keyword presence
  - Context signals

**Benefits:**
- Fast (no LLM calls needed)
- Accurate for known entities
- Provides confidence metrics

---

## 🔄 Complete Processing Flow

### Step-by-Step for a Single Prompt:

1. **User Input**: "Which tablet is best: iPad Pro or Samsung Galaxy Tab?"

2. **SpaCy NER Extraction**:
   - Extracts: "iPad Pro" (ORG), "Samsung Galaxy Tab" (ORG)
   - Time: ~50-200ms

3. **DSPy Brand Validation**:
   - Takes NER entities + prompt
   - Uses Ollama/phi3 to validate which are brands
   - Returns: "iPad Pro, Samsung"
   - Time: ~2-4 seconds

4. **DSPy Brand Expansion**:
   - Takes explicit brands + context
   - Uses Ollama/phi3 to infer related brands
   - Returns: "Apple" (parent of iPad)
   - Time: ~2-4 seconds

5. **Fuzzy Matching & Scoring**:
   - Matches against known brands list
   - Calculates match scores
   - Time: ~10-50ms

6. **Confidence Calculation**:
   - Multi-factor confidence scoring
   - Time: ~10-50ms

7. **Category Extraction** (DSPy):
   - Uses brand context + prompt
   - Extracts category: "Electronics, Tablets"
   - Time: ~2-4 seconds

**Total Time**: ~6-12 seconds for Brand + Category extraction

---

## 📊 Model Comparison

| Model/Component | Type | Purpose | Speed | Cost |
|----------------|------|---------|-------|------|
| **Ollama/phi3** | LLM | Brand/Category/Campaign extraction | 2-8s | Free (local) |
| **SpaCy (en_core_web_trf)** | NER | Entity extraction | 50-200ms | Free (local) |
| **Fuzzy Matching** | Algorithm | Brand/Category matching | 10-50ms | Free |
| **SerpAPI** | API | Brand validation (not used) | N/A | Paid (if used) |

---

## 🎯 Key Takeaways

1. **No External APIs**: System works 100% locally
2. **Two Main Models**:
   - **DSPy + Ollama/phi3**: For intelligent extraction/inference
   - **SpaCy**: For fast entity recognition
3. **SerpAPI**: Configured but not used (future enhancement)
4. **All Processing is Local**: 
   - No cloud API costs
   - Fast inference
   - Complete privacy

---

## 🔮 Future Enhancements (TODOs in Code)

If SerpAPI were to be enabled:
- Brand validation via Google search
- Real campaign data from Google News
- Evidence gathering for confidence scoring

**Current Status**: These are marked as TODO comments in the code but not implemented.

---

**Date**: 2025-01-27  
**Status**: ✅ Current implementation uses only local models (DSPy/Ollama + SpaCy)

