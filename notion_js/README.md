# Notion Recursive Scraper

A sophisticated, object-oriented Node.js application designed to create perfect offline replicas of Notion sites. By combining a distributed micro-kernel architecture with advanced scraping techniques, it preserves the exact visual hierarchy, interactivity, and content of your Notion pages in a fully browsable offline format.

## Core Concepts

### 1. Perfect Offline Fidelity
The system goes beyond simple HTML saving. It captures the complete state of a Notion page, including:
- **Visual Identity**: Downloads all CSS, fonts, and images to ensure pixel-perfect replication.
- **Interactivity**: Preserves JavaScript functionality where possible, keeping toggles and layout elements working.
- **Navigation**: Automatically rewrites all internal links to relative file paths, allowing you to click through your offline site just like the live version.

### 2. Deep Hierarchical Structure
Unlike flat scrapers, this tool understands Notion's nested nature. It recursively traverses your page tree to build a matching folder structure on your disk. A sub-page in Notion becomes a sub-folder in your output, maintaining the logical organization of your content.

### 3. Distributed Micro-Kernel Architecture
Built for performance and stability, the system employs a Master-Worker architecture:
- **Master Process**: The "brain" that orchestrates the workflow, manages the global queue, and handles conflict resolution. It remains lightweight and stable.
- **Worker Processes**: The "muscle" that performs the heavy lifting of rendering pages, downloading assets, and processing HTML. These run in isolated processes, meaning a crash in one worker never brings down the entire system.

### 4. Two-Phase Execution Strategy
To ensure efficiency and correctness, the scraping process is divided into two distinct phases:
- **Discovery Phase**: A lightweight, high-speed traversal that maps out the entire site structure without downloading heavy assets. This builds a complete graph of your site, allowing for global optimization and duplicate detection.
- **Execution Phase**: Once the plan is set, workers perform the deep scrape, downloading assets and saving files in parallel. This separation prevents wasted effort and ensures a clean, conflict-free output.

## System Behavior

The scraper operates as an autonomous agent that mirrors human browsing behavior but at scale:
1.  **Bootstrap**: It assesses system resources to spawn an optimal number of worker processes.
2.  **Discovery**: It navigates through your Notion site, identifying every link and building a virtual map of the content.
3.  **Optimization**: It analyzes the map to resolve circular links, handle duplicate references, and prune the graph for the most logical hierarchy.
4.  **Download**: It executes the download plan, fetching thousands of assets and rewriting tens of thousands of lines of code to make them work offline.
5.  **Completion**: It delivers a self-contained folder that you can open in any web browser, with no internet connection required.

## Getting Started

Ready to create your offline backup?

*   **[Quick Start Guide](Docs/QUICKSTART.md)**: Follow this step-by-step guide to install and run the scraper in minutes.

## Technical Documentation

For developers and engineers interested in the internal mechanics, design patterns, and extension points:

*   **[Architecture Documentation](Docs/ARCHITECTURE.md)**: A deep dive into the system's design, including the event bus, IPC protocol, and state management.
