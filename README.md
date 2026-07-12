# Fello Account Intelligence

Fello is an evidence-grounded AI account intelligence prototype built for the Fello AI Builder Hackathon. It accepts either a company name or website visitor activity and produces structured company research, inferred buyer intent, an executive briefing, and sales actions.

## Capabilities

- Identifies companies from names, domains, or public visitor IP ownership.
- Enriches company profile, website, industry, size, location, founding year, and description.
- Detects technologies only when direct public evidence is available.
- Discovers current leaders with source links and confidence.
- Searches for hiring, funding, expansion, product, partnership, and growth signals.
- Infers visitor persona and buying intent from behavioral activity.
- Produces a grounded executive summary and recommended sales actions.
- Distinguishes completed, no-evidence, failed, and not-applicable stages.
- Provides a responsive light/dark operations console and raw developer output.

## Architecture

```text
React + TypeScript
        |
        v
FastAPI request validation
        |
        v
Company identification -> enrichment -> technology -> leadership -> signals
        |                                                       |
        +---------------- persona + intent <--------------------+
                                |
                                v
                     summary -> sales actions

Research: Tavily search + Firecrawl extraction + Gemini structured reasoning
Visitor IP ownership: ipapi.co
```

All research stages share one validated `AnalysisContext`. External documents are treated as untrusted evidence. AI outputs use strict schemas and source checks. If synthesis is unavailable, the final briefing and actions degrade conservatively from verified upstream facts rather than inventing company data.

## Local setup

Requirements: Python 3.13+, Node.js 22+, and npm.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
cd frontend
npm ci
cd ..
```

Create `.env` at the repository root:

```dotenv
TAVILY_API_KEY=your_key
FIRECRAWL_API_KEY=your_key
GEMINI_API_KEY=your_key
TAVILY_SEARCH_URL=https://api.tavily.com/search
FIRECRAWL_SCRAPE_URL=https://api.firecrawl.dev/v2/scrape
GEMINI_GENERATE_URL=https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
GEMINI_MODEL=gemini-3.1-flash-lite
RESEARCH_TIMEOUT_SECONDS=45
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload
```

Start the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. API documentation is at `http://127.0.0.1:8000/docs`.

## API

### Company analysis

```http
POST /api/analyze/company
Content-Type: application/json

{"company_name":"Addverb","domain":"addverb.com"}
```

At least one of `company_name` or `domain` is required. Supplying both improves accuracy.

### Visitor analysis

```http
POST /api/analyze/visitor
Content-Type: application/json

{
  "visitor_id":"visitor-001",
  "ip":"8.8.8.8",
  "domain":"example.org",
  "pages_visited":["/pricing","/case-studies"],
  "time_on_site_seconds":240,
  "visits_this_week":3,
  "referral_source":"search",
  "device_type":"desktop",
  "visitor_location":"Bengaluru, India"
}
```

Persona and intent are behavioral outputs and are therefore not generated for company-only requests. A visitor domain is optional; public IP ownership lookup is attempted when it is absent. Residential, VPN, cloud, private, or reserved IPs may not identify an employer.

## Reliability and data integrity

- Missing evidence is returned as empty/unknown, never as fabricated company data.
- Technology and business signals are optional because public evidence may be unavailable.
- Leadership claims require cited evidence; current CEO claims are instructed to prefer official sources.
- Provider failures do not crash the whole pipeline.
- Stage results expose status, duration, and a user-facing explanation.
- Secrets are read only from environment variables and `.env` is ignored by Git.

## Deployment

### Render

The included `render.yaml` creates one Python web service. Its build installs the backend, builds the frontend, and FastAPI serves the resulting static application. Add the three provider secrets when prompted by Render.

### Docker

```bash
docker build -t fello .
docker run --env-file .env -p 8000:8000 fello
```

Open `http://localhost:8000`.

## Submission checklist

- Publish the repository without `.env` or credentials.
- Deploy using `render.yaml` or the Dockerfile.
- Record a 5–10 minute Loom showing one company analysis and one visitor analysis.
- Explain evidence validation, partial results, and the difference between company intelligence and behavioral intent.
- Optional: create two slides covering the problem, architecture, and a sourced report.

## Known external limitations

Results depend on public web evidence and third-party provider availability and quotas. IP ownership identifies the network organization, which is not always the visitor's employer. The prototype intentionally exposes uncertainty instead of filling unsupported fields.
