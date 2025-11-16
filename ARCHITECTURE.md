# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           NOTION SCRAPER SYSTEM                          │
│                         Perfect Offline Replication                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR: NotionScraper.js                                     │
│  • Initializes Puppeteer browser                                        │
│  • Coordinates all components                                           │
│  • Generates final statistics                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │                                                   │
        ▼                                                   ▼
┌──────────────────┐                              ┌──────────────────┐
│   Config.js      │◄─────────────────────────────│   Logger.js      │
│  Configuration   │  Used by all components      │  Logging System  │
│  • URLs          │                              │  • [COOKIE]      │
│  • Timeouts      │                              │  • [TOGGLE]      │
│  • Depth limits  │                              │  • [SCRAPE]      │
│  • Selectors     │                              │  • [LINK-REWRITE]│
└──────────────────┘                              └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: RECURSIVE SCRAPING                                             │
└─────────────────────────────────────────────────────────────────────────┘

        RecursiveScraper.js (Orchestrates scraping loop)
                    │
                    │ Maintains scrapeQueue and allContexts[]
                    │
                    ▼
        ┌────────────────────────────────────┐
        │      PageScraper.js                │
        │  • Scrapes individual pages        │
        │  • Registers PageContext           │
        │  • Tracks visited URLs             │
        └────────────────────────────────────┘
                    │
                    │ Uses these components:
                    │
        ┌───────────┴───────────┬─────────────┬─────────────┐
        ▼                       ▼             ▼             ▼
┌───────────────┐    ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐
│CookieHandler  │    │ContentExpander   │  │LinkExtractor │  │AssetDownloader│
│• Reject banner│    │• Load more       │  │• Find links  │  │• Download    │
│• Click OK     │    │• Scroll page     │  │• Extract ctx │  │• Sanitize    │
│• Wait reload  │    │• Expand toggles  │  │• Categorize  │  │• Retry logic │
└───────────────┘    └──────────────────┘  └──────────────┘  └──────────────┘

                ┌──────────────────────────────────────┐
                │     PageContext.js                   │
                │  Hierarchical page representation    │
                │  • URL, title, depth                 │
                │  • Parent/child relationships        │
                │  • Section/subsection                │
                │  • getRelativePath()                 │
                │  • getRelativePathTo() ← KEY!        │
                └──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: LINK REWRITING                                                 │
└─────────────────────────────────────────────────────────────────────────┘

        RecursiveScraper.js
                │
                │ After all pages scraped:
                │ For each PageContext in allContexts[]
                │
                ▼
        PageScraper.rewriteLinksInFile(context)
                │
                ├─ Read HTML file
                ├─ Parse with JSDOM
                ├─ Find all <a href> tags
                ├─ Check if URL is in urlToContextMap
                ├─ Calculate relative path
                ├─ Rewrite href attribute
                └─ Save modified HTML

┌─────────────────────────────────────────────────────────────────────────┐
│  DATA FLOW EXAMPLE                                                       │
└─────────────────────────────────────────────────────────────────────────┘

1. User starts scraper
2. NotionScraper initializes browser
3. RecursiveScraper creates root PageContext
4. RecursiveScraper adds root to queue
5. Loop: While queue not empty
   a. PageScraper.scrapePage(context)
      - CookieHandler.handle() [first page only]
      - ContentExpander.expandAll()
      - LinkExtractor.extractLinks() → links[]
      - AssetDownloader.downloadAndRewriteImages()
      - Save HTML to disk
      - Register context in urlToContextMap
   b. For each link in links[]
      - Create child PageContext (parent = current)
      - Add child to queue
6. Loop ends when queue empty
7. RecursiveScraper starts Phase 2
8. For each context in allContexts[]
   - PageScraper.rewriteLinksInFile(context)
9. Print statistics

┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT STRUCTURE                                                        │
└─────────────────────────────────────────────────────────────────────────┘

downloaded_course_material/
├── Main_Page/
│   ├── index.html ◄──────── All <a> tags rewritten
│   └── images/
│       ├── 1-image.jpg
│       └── 2-logo.png
├── Syllabus/
│   ├── index.html ◄──────── Links: ../Material/Week_1/index.html
│   ├── images/
│   └── Course_Overview/    ◄──────── Nested folder!
│       ├── index.html
│       └── images/
└── Material/
    ├── Week_1/
    │   ├── index.html
    │   ├── images/
    │   ├── Introduction/   ◄──────── Deep nesting!
    │   │   ├── index.html
    │   │   └── images/
    │   └── Lecture_Notes/
    │       └── index.html
    └── Week_2/
        └── ...

┌─────────────────────────────────────────────────────────────────────────┐
│  KEY INNOVATIONS                                                         │
└─────────────────────────────────────────────────────────────────────────┘

1. TWO-PHASE ARCHITECTURE
   Phase 1: Scrape all pages, build context tree
   Phase 2: Rewrite all links using context map
   ✓ Ensures all target pages exist before rewriting

2. PARENT-CHILD PAGE CONTEXTS
   Each PageContext knows its parent
   Path building traverses parent chain
   ✓ Creates true nested folder structure

3. RELATIVE PATH CALCULATION
   getRelativePathTo(target) method
   Finds common ancestor, calculates ../../../
   ✓ Generates correct relative links

4. RESILIENT ASSET DOWNLOADING
   Sanitize filenames with MD5 fallback
   Retry with exponential backoff
   Handle redirects and special characters
   ✓ Downloads succeed even with complex URLs

5. COMPLETE HTML PRESERVATION
   Save full HTML with CSS/JS intact
   Parse with JSDOM for safe manipulation
   Only modify <a href> attributes
   ✓ Perfect visual and functional replication

┌─────────────────────────────────────────────────────────────────────────┐
│  STATISTICS OUTPUT                                                       │
└─────────────────────────────────────────────────────────────────────────┘

[STATS] Total pages scraped: 47
[STATS] Total assets downloaded: 312
[STATS] Total internal links rewritten: 189  ◄── NEW!
[STATS] Total time elapsed: 8m 23s
[STATS]
[STATS] The downloaded site is now fully browsable offline!
[STATS] Open downloaded_course_material/Main_Page/index.html in your browser.
[STATS]
[STATS] Page hierarchy:
  ├─ Main_Page (root)
  ├─ Syllabus (Syllabus)
  │  ├─ Course_Overview (Syllabus/Course_Overview)
  │  └─ Grading_Policy (Syllabus/Grading_Policy)
  └─ Material (Material)
     ├─ Week_1 (Material/Week_1)
     │  ├─ Introduction (Material/Week_1/Introduction)
     │  └─ Lecture_Notes (Material/Week_1/Lecture_Notes)
     └─ Week_2 (Material/Week_2)
        └─ ...
```
