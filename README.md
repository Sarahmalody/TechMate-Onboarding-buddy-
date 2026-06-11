# TechMate Onboarding — UFO Mission Control

> A gamified browser-based onboarding dashboard for new joiners at TechMate Inc., powered by a self-contained RAG system, a deterministic leave policy rule engine, and a UFO-themed interactive UI. Built as part of a data science internship project at Tech Mahindra

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How the RAG System Works](#how-the-rag-system-works)
- [How the Rule Engine Works](#how-the-rule-engine-works)
- [Policy Documents](#policy-documents)
- [Setup and Running](#setup-and-running)
- [Dashboard Pages](#dashboard-pages)
- [Leave Management](#leave-management)
- [Gamification System](#gamification-system)
- [Challenges and Solutions](#challenges-and-solutions)
- [Planned Improvements](#planned-improvements)
- [Author](#author)

---

## Project Overview

The **TechMate UFO Onboarding Dashboard** is a single-file browser application that helps new employees navigate their first days at TechMate Inc. It combines:

- A **RAG-powered chatbot** (Ask UFO) that answers onboarding questions from 5 separate policy documents
- A **leave management module** with real-time policy validation using a deterministic rule engine
- A **gamified mission system** with XP tracking, level progression, and UFO celebration animations
- An **office slang dictionary** so new joiners can understand TechMate's internal language from Day 1

The entire system runs in the browser with no server, no backend, and no installation required. The only external dependency is a free Groq API key for the chatbot's LLM generation layer.

---

## Live Demo

To run the project:

1. Download or clone this repository
2. Open `ufo_onboarding_dashboard.html` in any browser
3. Navigate to the **Ask UFO** page
4. Enter your free Groq API key when prompted (get one at [console.groq.com](https://console.groq.com))
5. Start asking questions and completing missions

No terminal. No Python. No installation.

---

## Features

### Ask UFO — RAG Chatbot
- Answers natural language questions from 5 separate policy documents
- TF-IDF based retrieval with cosine similarity search built entirely in JavaScript
- LLaMA 3.1 8B via Groq API for natural language generation
- Quick chip buttons and category shortcuts for common questions
- API key stored in localStorage so you only enter it once

### Leave Management
- Leave application form with real-time policy rule validation
- Validates against the full leave policy — Casual, Sick, Earned, Maternity, Paternity, Bereavement, and Comp-Off
- Automatically determines approval chain based on number of days
- Displays warnings, policy references, and override messages
- Leave balance tracker and request history

### Mission Board
- 15 onboarding missions covering HR, IT, social, and training tasks
- Click to complete — triggers UFO celebration overlay with particle animations
- XP system with randomised alien celebration messages

### Progress Tracking
- 5-level progression from Cadet to True TM
- Live XP bar and mission counter across all pages
- UFO mood indicator that changes based on overall progress

### Office Slang Dictionary
- 18 TechMate-specific terms with definitions and usage examples
- Covers standup, sprint, retro, LGTM, OOO, WFH, deep work, and more

### Key Contacts
- HR, IT, canteen, manager, onboarding buddy, and security contacts

---

## Project Structure

```
TechMate-Onboarding/
│
├── ufo_onboarding_dashboard.html   # Complete self-contained application
│
├── leave_policy.txt                # Full HR Leave Policy (Version 2.1)
├── policy_laptop.txt               # Laptop fetching and IT desk policy
├── policy_login.txt                # Login, email, VPN, MFA, and access policy
├── policy_canteen.txt              # Canteen timings and meal card policy
├── policy_slangs.txt               # Office slangs and TechMate culture guide
│
├── .gitignore
└── README.md
```

> The `.txt` policy files serve as the **source of truth** for the RAG system. Their contents are embedded as JavaScript strings inside the HTML dashboard. When policy changes, update the `.txt` file and mirror the change inside the HTML.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Application | Single HTML file | Entire app in one file — no build tools, no framework |
| Fonts | Google Fonts CDN | Space Grotesk and Orbitron typefaces |
| RAG Retrieval | TF-IDF in JavaScript | Converts policy text to vectors, cosine similarity search |
| LLM | LLaMA 3.1 8B Instant | Natural language answer generation |
| LLM Provider | Groq API (Free Tier) | Fast LPU-based inference, free at console.groq.com |
| Rule Engine | Vanilla JavaScript | Deterministic policy validation and override logic |
| State Management | JavaScript variables | In-memory mission and XP tracking |
| Key Storage | localStorage | Groq API key persisted across sessions |

---

## How the RAG System Works

RAG stands for **Retrieval-Augmented Generation**. Instead of asking the LLM to answer from general training knowledge (which leads to hallucination), RAG first retrieves the relevant policy text and gives it to the LLM along with the question.

### The 3 Steps Inside the Dashboard

**Step 1 — Chunking**

Each policy document is split into overlapping chunks of 250 words with a 50-word overlap. The overlap ensures no policy rule is cut off at a chunk boundary.

```
Policy text (e.g. leave_policy.txt content)
        ↓
Sliding window chunker
chunk_size = 250 words, overlap = 50 words
        ↓
List of overlapping text chunks
```

**Step 2 — TF-IDF Retrieval**

When the user asks a question, it is compared against all chunks using TF-IDF scoring and cosine similarity. The top 3 most relevant chunks are returned.

```
User question
        ↓
TF-IDF scoring against all chunks from all 5 policy documents
        ↓
Cosine similarity ranking
        ↓
Top 3 most relevant chunks returned
```

TF-IDF (Term Frequency — Inverse Document Frequency) gives high scores to words that appear frequently in one chunk but rarely across all chunks, making them strong signals for what that chunk is about.

**Step 3 — LLM Generation**

The retrieved chunks and the user question are sent to LLaMA 3.1 via Groq API. The system prompt explicitly instructs the model to answer only from the provided context.

```
[System: You are UFO, answer only from provided context]
[Retrieved chunk 1]
[Retrieved chunk 2]
[Retrieved chunk 3]
[Employee question]
        ↓
Groq API → LLaMA 3.1 8B Instant
        ↓
Natural language answer grounded in actual policy
```

---

## How the Rule Engine Works

The leave management module uses a deterministic rule engine that validates leave requests independently of the LLM. It never guesses — it always applies the exact policy rule.

### Validation Logic

```
User selects leave type and number of days
        ↓
Rule engine detects leave type
        ↓
Applies the relevant policy rule

Casual Leave:   days > 3   → REJECTED (max 3 consecutive days)
Sick Leave:     days > 2   → WARNING (medical certificate required)
Earned Leave:   days < 3   → REJECTED (minimum 3 days per application)
Earned Leave:   probation  → REJECTED (not available during probation)
        ↓
Approval chain determined automatically

1 to 3 days  → Reporting Manager only
4 to 7 days  → Reporting Manager + Department Head
8+ days      → Reporting Manager + Department Head + HR
        ↓
Structured result displayed with answer, reason,
approval chain, warnings, and policy references
```

### Why a Rule Engine Alongside RAG

The LLM is probabilistic — it can occasionally produce incorrect outputs even with good context. For specific high-stakes policy facts like maximum casual leave days or minimum earned leave, a deterministic rule always produces the correct answer. The rule engine acts as a safety net over the LLM layer.

---

## Policy Documents

| File | Content | Sections |
|---|---|---|
| `leave_policy.txt` | Full leave policy Version 2.1 | CL, SL, EL, Maternity, Paternity, Bereavement, Comp-Off, Approval hierarchy, Restrictions |
| `policy_laptop.txt` | IT asset policy | Collection process, IT desk details, accessories, return policy, software rules |
| `policy_login.txt` | Access and credentials policy | Corporate email, password rules, VPN setup, MFA, tool access, login support |
| `policy_canteen.txt` | Food and facilities policy | Timings, meal card, top-up, guest meals, theme lunch, special diets |
| `policy_slangs.txt` | Culture and language guide | 24 office slangs with definitions and usage examples |

---

## Setup and Running

### Requirements

- Any modern browser (Chrome, Firefox, Edge, Safari)
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Steps

```
1. Clone or download this repository

2. Open ufo_onboarding_dashboard.html in your browser
   (double-click the file or drag it into a browser window)

3. Go to the Ask UFO page from the sidebar

4. Enter your Groq API key when prompted
   The key is saved to localStorage and will not be asked again

5. Start using the dashboard
```

### Getting a Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account (no credit card required)
3. Go to API Keys in the left sidebar
4. Click Create API Key
5. Copy the key — it starts with `gsk_`
6. Paste it into the dashboard when prompted

> The key is stored only in your browser's localStorage. It is never sent anywhere except directly to the Groq API.

---

## Dashboard Pages

| Page | Access | Description |
|---|---|---|
| Mission Control | 🏠 Dashboard | Overview with stats, UFO mood, next missions, and progress bar |
| Mission Board | 🎯 Missions | All 15 onboarding tasks — click to complete and earn XP |
| My Progress | 📊 Progress | Step-by-step journey with level indicator and XP tracker |
| Leave Management | 🏖️ Leave | Apply for leave with real-time policy validation |
| Ask UFO | 💬 Ask UFO | RAG chatbot for any onboarding question |
| Office Slangs | 🔤 Slangs | Dictionary of 18 TechMate terms |
| Key Contacts | 📞 Contacts | HR, IT, canteen, and team contacts |

---

## Leave Management

The leave management module covers all leave types defined in `leave_policy.txt`:

| Leave Type | Entitlement | Key Rule |
|---|---|---|
| Casual Leave (CL) | 12 days per year | Max 3 consecutive days, no carry forward |
| Sick Leave (SL) | 12 days per year | Medical certificate required after 2 days |
| Earned Leave (EL) | 18 days per year | Min 3 days, 7 days advance notice, not during probation |
| Maternity Leave | 182 days paid | 4 weeks advance notice, medical documentation |
| Paternity Leave | 5 days paid | Within 30 days of child birth |
| Bereavement Leave | 5 days paid | Immediate family only |
| Comp-Off | As earned | Must use within 30 days, cannot be encashed |

---

## Gamification System

| Element | Description |
|---|---|
| XP System | Each mission awards 50 to 200 XP. Total pool is 1500 XP |
| Level Progression | Cadet (0%) → Rookie (30%) → Field Agent (60%) → Senior Agent (80%) → True TM (100%) |
| UFO Mood | UFO emoji and status message changes as progress increases |
| Celebration Overlay | Completing a mission triggers a full-screen UFO animation with particle explosion and randomised alien message |
| Undo Support | Click a completed mission again to undo it — XP is deducted accordingly |

---

## Challenges and Solutions

| Challenge | Root Cause | Resolution |
|---|---|---|
| LLM API cost | Paid APIs like OpenAI not viable for development | Adopted Groq API with free-tier access to LLaMA 3.1 |
| Model inference speed | Some providers showed 10 to 20 second response times | Switched to Groq LPU hardware reducing latency to under 2 seconds |
| LLM hallucination on policy specifics | LLM generated plausible but incorrect leave limits | Implemented deterministic rule engine that overrides incorrect outputs |
| LLM model deprecation mid-development | llama3-8b-8192 was decommissioned by Groq during development | Migrated to llama-3.1-8b-instant with no changes to prompt logic |
| Semantic retrieval without ML libraries | Browser environment cannot install sentence-transformers | Implemented TF-IDF retrieval natively in JavaScript |
| Secure API key management | Keys stored in environment variables are not persistent | API key stored in browser localStorage, .env file approach planned for Phase 2 backend |

---

## Planned Improvements

### Phase 2 — Backend Integration
- Build a Flask or FastAPI backend so the dashboard calls a Python server instead of Groq directly
- Upgrade retrieval from TF-IDF to `sentence-transformers/all-MiniLM-L6-v2` for true semantic search
- Migrate vector store to ChromaDB for persistent, production-grade embedding storage
- Implement `.env` file based API key management using `python-dotenv`

### Phase 3 — Agentic Capabilities
- Integrate LangChain for multi-step reasoning and tool use
- Connect to HR portal API for real-time leave balance checking and application submission
- Add conversation memory for follow-up questions within a session
- Expand RAG to support additional policy documents dynamically

### Phase 4 — Hosting and Deployment
- Containerise the application using Docker
- Deploy to AWS, Azure, or Google Cloud with auto-scaling
- Integrate with Microsoft Teams or Slack for in-platform querying
- Implement role-based access control for employees, managers, and HR admins
- Set up monitoring dashboards for query volume, response accuracy, and API usage

---

## Author

**Diya Sarah Arun**
Master of Computer Applications — Visvesvaraya Technological University (VTU)
AI Developer

---

## License

This project is developed as part of an academic internship. All company names, policy content, and employee data used in this project are fictional and created for demonstration purposes only.
