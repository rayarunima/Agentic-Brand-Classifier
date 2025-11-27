# Agentic Brand Classifier V3 - System Architecture

## Mermaid Diagram Code

```mermaid
flowchart TD
    Start([START]) --> StreamlitUI[Streamlit UI<br/>User Prompt Input]
    
    StreamlitUI --> Preprocess[Preprocessing<br/>Prompt Cleaning]
    Preprocess --> NER[NER Extraction<br/>SpaCy Transformer Model]
    
    NER --> Orchestrator[Agent Orchestrator<br/>Pipeline Execution]
    
    %% Phase 1: Core Agents
    Orchestrator --> BrandAgent[Brand Agent<br/>DSPy + NER + Confidence]
    BrandAgent --> BrandConf[Confidence Scoring<br/>Multi-factor Analysis]
    BrandConf --> BrandResult[Brand Results<br/>Names + Match Scores]
    
    BrandResult --> CategoryAgent[Category Agent<br/>DSPy + Brand Context]
    CategoryAgent --> CategoryConf[Confidence Scoring<br/>Fuzzy Matching]
    CategoryConf --> CategoryResult[Category Result<br/>+ Normalized Category]
    
    %% Phase 2: Context-Dependent Agents
    BrandResult --> CampaignAgent[Campaign Agent<br/>DSPy + Brand Context]
    CampaignAgent --> CampaignConf[Confidence Scoring]
    CampaignConf --> CampaignResult[Campaign Insights]
    
    BrandResult --> ReachAgent[Reach Agent<br/>DSPy + Multi-Context]
    CategoryResult --> ReachAgent
    CampaignResult --> ReachAgent
    ReachAgent --> ReachConf[Confidence Scoring<br/>Range Validation]
    ReachConf --> ReachResult[Reach Estimation]
    
    BrandResult --> BrandLiftAgent[Brand Lift Agent<br/>DSPy + Campaign Context]
    CampaignResult --> BrandLiftAgent
    BrandLiftAgent --> BrandLiftConf[Confidence Scoring<br/>Category-Aware]
    BrandLiftConf --> BrandLiftResult[Brand Lift %]
    
    %% Results Aggregation
    BrandResult --> ResultsAgg[Results Aggregation<br/>+ Performance Metrics]
    CategoryResult --> ResultsAgg
    CampaignResult --> ResultsAgg
    ReachResult --> ResultsAgg
    BrandLiftResult --> ResultsAgg
    
    ResultsAgg --> StreamlitOutput[Streamlit UI Output<br/>Brands, Category, Campaign,<br/>Reach, Brand Lift, Metrics]
    
    StreamlitOutput --> PerformanceTracker[Performance Tracker<br/>Latency, Throughput, Cost]
    StreamlitOutput --> QueryHistory[Query History<br/>Session-based Analytics]
    
    StreamlitOutput --> Stop([STOP])
    
    %% Configuration & Utils
    Config[Configuration<br/>agent_config.py<br/>Known Brands, Categories]
    Config -.->|Provides| BrandAgent
    Config -.->|Provides| CategoryAgent
    Config -.->|Provides| CampaignAgent
    Config -.->|Provides| ReachAgent
    Config -.->|Provides| BrandLiftAgent
    
    ConfidenceUtils[Confidence Utils<br/>Multi-factor Scoring<br/>Fuzzy Matching]
    ConfidenceUtils -.->|Used by| BrandConf
    ConfidenceUtils -.->|Used by| CategoryConf
    ConfidenceUtils -.->|Used by| CampaignConf
    ConfidenceUtils -.->|Used by| ReachConf
    ConfidenceUtils -.->|Used by| BrandLiftConf
    
    %% Styling
    classDef startEnd fill:#e1f5e1,stroke:#4caf50,stroke-width:3px
    classDef ui fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef agent fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef context fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    classDef confidence fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef config fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5
    classDef output fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    
    class Start,Stop startEnd
    class StreamlitUI,Preprocess,StreamlitOutput,PerformanceTracker,QueryHistory ui
    class Orchestrator agent
    class BrandAgent,CategoryAgent core
    class CampaignAgent,ReachAgent,BrandLiftAgent context
    class BrandConf,CategoryConf,CampaignConf,ReachConf,BrandLiftConf confidence
    class Config,ConfidenceUtils config
    class NER,BrandResult,CategoryResult,CampaignResult,ReachResult,BrandLiftResult,ResultsAgg output
```

## How to Generate/View the Diagram

### Option 1: Online Mermaid Editors (Recommended)
1. **Mermaid Live Editor**: https://mermaid.live/
   - Copy the code above (everything between ```mermaid and ```)
   - Paste into the editor
   - View and export as PNG/SVG

2. **GitHub/GitLab**: 
   - Create a `.md` file with the code
   - Push to GitHub/GitLab
   - View directly in the repository

### Option 2: VS Code Extension
1. Install "Markdown Preview Mermaid Support" extension
2. Open this `.md` file
3. Use Markdown preview (Cmd+Shift+V / Ctrl+Shift+V)

### Option 3: Command Line (Mermaid CLI)
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i system_architecture_v3.md -o diagram.png
```

### Option 4: Documentation Sites
- **Notion**: Supports Mermaid diagrams
- **Confluence**: With Mermaid plugin
- **Obsidian**: Native Mermaid support

## Architecture Overview

### Core Components

1. **Streamlit UI** (`ui/app.py`)
   - User interface for prompt input
   - Results visualization
   - Performance metrics dashboard
   - Query history and analytics

2. **NER Extraction** (`brand_extraction/entity_extractor.py`)
   - Uses SpaCy transformer model (`en_core_web_trf`)
   - Extracts named entities (ORG, PERSON, etc.)
   - Provides context for brand agent

3. **Agent Orchestrator** (`agent/orchestrator.py`)
   - Coordinates all agents
   - Manages execution order
   - Passes context between agents
   - Tracks performance metrics

### Agent Pipeline

#### Phase 1: Core Agents (Always Enabled)
- **Brand Agent**: Extracts brand names using DSPy + NER
- **Category Agent**: Infers category using DSPy + Brand context

#### Phase 2: Context-Dependent Agents (Optional)
- **Campaign Agent**: Requires Brand context
- **Reach Agent**: Requires Brand, Category, Campaign context
- **Brand Lift Agent**: Requires Brand, Campaign context

### Key Features

1. **Intelligent Context Passing**
   - Brand results inform Category extraction
   - Campaign uses Brand context
   - Reach uses Brand + Category + Campaign
   - Brand Lift uses Brand + Campaign

2. **Confidence Scoring** (`utils/confidence_scoring.py`)
   - Multi-factor confidence calculation
   - Fuzzy matching for accuracy
   - Context-aware adjustments
   - Range validation for numeric predictions

3. **Configuration Management** (`config/agent_config.py`)
   - Centralized configuration
   - Known brands, categories, keywords
   - External JSON override support
   - Environment variable support

4. **Performance Tracking** (`utils/performance.py`)
   - Real-time latency tracking
   - Throughput metrics
   - Cost comparison (Ollama vs Cloud APIs)
   - System resource monitoring

## Execution Flow

1. **User Input** → Streamlit UI receives prompt
2. **Preprocessing** → Prompt cleaned and prepared
3. **NER Extraction** → Entities extracted using SpaCy
4. **Orchestrator** → Manages agent pipeline execution
5. **Phase 1 Agents**:
   - Brand Agent extracts brands (uses NER + DSPy)
   - Category Agent infers category (uses Brand context + DSPy)
6. **Phase 2 Agents** (if enabled):
   - Campaign Agent (uses Brand)
   - Reach Agent (uses Brand + Category + Campaign)
   - Brand Lift Agent (uses Brand + Campaign)
7. **Results Aggregation** → All results combined with metrics
8. **UI Output** → Results displayed with visualizations
9. **Tracking** → Performance metrics and query history updated

## Technology Stack

- **UI**: Streamlit
- **NER**: SpaCy Transformer Model
- **LLM**: DSPy with Ollama (phi3)
- **Visualization**: Plotly
- **Performance**: psutil for system metrics
- **Configuration**: Python config module with JSON override

## Benefits

1. **Cost Efficiency**: Local Ollama processing (90%+ savings vs cloud APIs)
2. **Intelligent Orchestration**: Context-aware agent chaining
3. **Confidence Scoring**: Multi-factor reliability metrics
4. **Performance Tracking**: Real-time metrics and analytics
5. **Scalability**: Batch processing with parallel execution
6. **Maintainability**: Centralized configuration and modular design

