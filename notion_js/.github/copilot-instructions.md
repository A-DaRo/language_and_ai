# GitHub Copilot Instructions for Notion Recursive Scraper

## Project Overview
This project is a **Reactive Event-Driven Micro-Kernel** Node.js application designed to create perfect 1:1 offline replicas of Notion sites. It recursively scrapes Notion pages, preserving visual hierarchy, interactivity, and content in a fully browsable offline format.

**Primary Goal:** Generate a local folder structure that mirrors the Notion page hierarchy, with all assets downloaded and links rewritten for offline navigation.

## Documentation Strategy
**CRITICAL:** Do not guess at architecture or implementation details. Always refer to the authoritative documentation in the `Docs/` folder:
- **`Docs/ARCHITECTURE.md`**: The definitive guide on system design, Master-Worker patterns, IPC protocols, and state management.
- **`Docs/QUICKSTART.md`**: Instructions for running and configuring the application.
- **`Docs/JSDocs.md`**: API documentation for core components.

## High-Level Architecture
The system employs a distributed **Master-Worker** architecture to ensure scalability and fault tolerance.
- **Master Process (Orchestration):** Handles state, queuing, and coordination. It does *not* perform heavy scraping.
- **Worker Processes (Execution):** Isolated processes that run Puppeteer to scrape pages, download assets, and parse HTML. They are stateless and communicate only via IPC.

### Core Workflow
The scraping process follows a strict two-phase strategy:
1.  **Discovery Phase:** Lightweight traversal to map the site structure.
2.  **Execution Phase:** Parallel downloading and processing of pages based on the discovered map.

## Repository Structure
- **`src/cluster/`**: Worker lifecycle management (spawning, proxies).
- **`src/core/`**: Shared infrastructure (Config, Logger, EventBus, ProtocolDefinitions).
- **`src/domain/`**: Data models (e.g., `PageContext`).
- **`src/orchestration/`**: Master-side logic (QueueManager, ClusterOrchestrator).
- **`src/worker/`**: Worker-side logic (TaskRunner, WorkerEntrypoint).
- **`src/scraping/`** & **`src/processing/`**: Puppeteer logic and content manipulation.
- **`tests/`**: Jest test suites (Unit and Integration).