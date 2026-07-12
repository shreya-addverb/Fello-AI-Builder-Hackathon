# Fello AI Account Intelligence & Enrichment System

Fello is an AI-powered account intelligence prototype built for the Fello AI Builder Hackathon. It converts minimal company input or website visitor activity into structured company research, buying-intent context, an account summary, and recommended sales actions.

The project focuses on a working end-to-end flow: user input goes through a FastAPI research pipeline, public web evidence is collected with Tavily and Firecrawl, Gemini extracts structured intelligence, and the React frontend presents the final account report.

## Problem

Sales and marketing teams often receive weak signals:

- Anonymous website visitors with only IP, page visits, and session metadata.
- Incomplete company inputs such as only a company name or domain.
- Analytics data that does not explain who the visitor is, what the company does, or what sales should do next.

Fello helps turn those raw signals into sales-ready intelligence.

## Core Features

- Company identification from company name, domain, visitor domain, or public IP ownership.
- Company enrichment from public evidence, including website, industry, company size, headquarters, founding year, description, revenue, and footprint when available.
- Website and public web research using Tavily search and Firecrawl scraping.
- Technology detection from public evidence where tools or stack signals are visible.
- Leadership discovery for possible decision makers such as executives, sales leaders, marketing leaders, or RevOps contacts.
- Business-signal discovery for hiring, funding, expansion, product launches, partnerships, recognition, and growth mentions.
- Persona inference and intent scoring for visitor-analysis requests.
- AI-generated account summary and sales recommendations.
- Stage-by-stage pipeline status with provider usage, timing, errors, and partial results.

## System Architecture

```text
React + TypeScript frontend
        |
        v
FastAPI backend
        |
        v
Request validation and AnalysisContext creation
        |
        v
Account Intelligence Pipeline
        |
        +--> Company identification
        +--> Company enrichment
        +--> Technology detection
        +--> Leadership discovery
        +--> Business signals
        +--> Persona inference       (visitor requests)
        +--> Intent scoring          (visitor requests)
        +--> AI summary
        +--> Sales recommendations

Research providers:
Tavily       -> public web/news search
Firecrawl    -> website/page scraping
Gemini       -> structured reasoning and summary generation
ipapi.co     -> visitor IP ownership lookup
```

## System Design

### 1. Frontend

The frontend is a React + TypeScript operations console. It allows users to run:

- Company analysis with company name and optional domain.
- Visitor analysis with JSON visitor activity.

It displays:

- Pipeline progress.
- Company profile.
- Technology stack.
- Leadership.
- Business signals.
- Persona and intent.
- AI summary.
- Recommended sales actions.
- Optional developer/raw output.

### 2. Backend API

The backend is built with FastAPI. It exposes:

- `POST /api/analyze/company`
- `POST /api/analyze/visitor`
- `GET /api/health`
- `GET /api/system`

Each request creates a single `AnalysisContext`. Every stage reads from and writes to that context, so the final response has one consistent source of truth.

### 3. Research Layer

The research layer wraps external providers behind common interfaces:

- Search provider: Tavily
- Crawl provider: Firecrawl
- Reasoning provider: Gemini

This keeps pipeline stages independent from raw provider APIs and makes failures easier to handle.

### 4. Pipeline Stages

The pipeline is sequential because later stages depend on earlier context:

1. Identify the company.
2. Enrich firmographic profile.
3. Detect visible technologies.
4. Discover leaders.
5. Find business signals.
6. Infer visitor persona, if visitor data is provided.
7. Score buying intent, if visitor behavior exists.
8. Generate account summary.
9. Generate sales recommendations.

Each stage returns `completed`, `no_data`, `failed`, or `skipped`. Optional research sections can return `no_data` without crashing the entire analysis.

### 5. Evidence and AI Reasoning

The system collects live public evidence, then asks Gemini to produce structured JSON. The backend validates citations and schemas before storing results. When evidence is weak or missing, the system returns partial output instead of inventing unsupported details.

## Tech Stack

### Frontend

- React
- TypeScript
- Vite
- React Router
- TanStack Query
- Framer Motion
- Lucide React

### Backend

- Python
- FastAPI
- Pydantic
- HTTPX
- Uvicorn

### AI and Research Providers

- Gemini for structured reasoning and summarization
- Tavily for web/news search
- Firecrawl for website scraping
- ipapi.co for visitor IP organization lookup

## Local Setup

### Requirements

- Python 3.13+
- Node.js 22+
- npm

### 1. Clone the repository

```bash
git clone https://github.com/shreya-addverb/Fello-AI-Builder-Hackathon.git
cd Fello-AI-Builder-Hackathon
```

### 2. Create and activate Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

### 3. Install frontend dependencies

```powershell
cd frontend
npm ci
cd ..
```

### 4. Create environment file

Create `.env` in the repository root:

```dotenv
TAVILY_API_KEY=your_tavily_key
FIRECRAWL_API_KEY=your_firecrawl_key
GEMINI_API_KEY=your_gemini_key

TAVILY_SEARCH_URL=https://api.tavily.com/search
FIRECRAWL_SCRAPE_URL=https://api.firecrawl.dev/v2/scrape
GEMINI_GENERATE_URL=https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
GEMINI_MODEL=gemini-3.1-flash-lite

RESEARCH_TIMEOUT_SECONDS=60
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Do not commit `.env`. It is already ignored by Git.

## Running Locally

### Start backend

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### Start frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

## Example Requests

### Company Analysis

```http
POST /api/analyze/company
Content-Type: application/json

{
  "company_name": "Redfin",
  "domain": "redfin.com"
}
```

Company analysis is best when both company name and domain are provided.

### Visitor Analysis

```http
POST /api/analyze/visitor
Content-Type: application/json

{
  "visitor_id": "visitor-001",
  "ip": "8.8.8.8",
  "domain": "redfin.com",
  "pages_visited": ["/pricing", "/case-studies", "/ai-sales-agent"],
  "time_on_site_seconds": 240,
  "visits_this_week": 3,
  "referral_source": "search",
  "device_type": "desktop",
  "visitor_location": "California, USA"
}
```

Persona and intent are generated only for visitor-analysis requests because they depend on behavior signals.

## Testing

Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

Build frontend:

```powershell
cd frontend
npm run build
```

## Deployment

### Render

The repository includes `render.yaml`. Render builds the frontend, installs backend dependencies, and runs FastAPI as a single web service.

Required Render environment variables:

- `TAVILY_API_KEY`
- `FIRECRAWL_API_KEY`
- `GEMINI_API_KEY`
- `TAVILY_SEARCH_URL`
- `FIRECRAWL_SCRAPE_URL`
- `GEMINI_GENERATE_URL`
- `GEMINI_MODEL`
- `RESEARCH_TIMEOUT_SECONDS`

### Docker

```bash
docker build -t fello .
docker run --env-file .env -p 8000:8000 fello
```

Open:

```text
http://localhost:8000
```

## Reliability Notes

- Results depend on public web evidence and provider availability.
- Some company websites block scraping or expose limited public information.
- Technology stack, leadership, and business signals may return empty when evidence is weak.
- Provider failures are captured at the stage level instead of crashing the whole pipeline.
- The system favors partial grounded output over unsupported fabrication.

## Hackathon Demo Flow

Recommended demo:

1. Open the frontend.
2. Run company analysis with:

```text
Company: Redfin
Domain: redfin.com
```

3. Show the generated company profile, summary, and recommendations.
4. Run visitor analysis with simulated behavior.
5. Explain how persona and intent differ from company-only enrichment.
6. Open developer mode to show raw pipeline stages and provider traces.

## Repository Hygiene

Ignored files include:

- `.env`
- `.venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- Python cache files
- test/build cache files

Never commit API keys or provider credentials.
