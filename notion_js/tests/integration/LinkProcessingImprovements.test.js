/**
 * @fileoverview Integration tests for link processing improvements
 * @module tests/integration/LinkProcessingImprovements.test.js
 */

const PageGraph = require('../../src/orchestration/PageGraph');
const EdgeClassifier = require('../../src/orchestration/analysis/EdgeClassifier');
const PageContext = require('../../src/domain/PageContext');
const BlockIDExtractor = require('../../src/extraction/BlockIDExtractor');
const BlockIDMapper = require('../../src/processing/BlockIDMapper');
const LinkRewriter = require('../../src/processing/LinkRewriter');
const Config = require('../../src/core/Config');
const Logger = require('../../src/core/Logger');
const { JSDOM } = require('jsdom');
const fs = require('fs/promises');
const path = require('path');
const os = require('os');

describe('Link Processing Improvements - Integration Tests', () => {
  let tempDir;
  let config;
  let logger;

  beforeEach(async () => {
    tempDir = path.join(os.tmpdir(), `link-test-${Date.now()}`);
    await fs.mkdir(tempDir, { recursive: true });

    // Initialize logger (use singleton)
    logger = Logger.getInstance();

    // Mock config
    config = {
      getBaseUrl: () => 'https://notion.so'
    };
  });

  afterEach(async () => {
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (e) {
      // Ignore cleanup errors
    }
  });

  describe('Edge Classification in Page Discovery', () => {
    it('should classify edges correctly during discovery', () => {
      const classifier = new EdgeClassifier();

      // Create a page hierarchy
      const root = new PageContext('https://notion.so/root-29root', 'Root Page', 0);
      const child1 = new PageContext('https://notion.so/child1-29c1', 'Child 1', 1, root, root.id);
      const child2 = new PageContext('https://notion.so/child2-29c2', 'Child 2', 1, root, root.id);
      const grandchild = new PageContext('https://notion.so/grandchild-29gc', 'Grandchild', 2, child1, child1.id);

      const contextMap = new Map([
        [root.id, root],
        [child1.id, child1],
        [child2.id, child2],
        [grandchild.id, grandchild]
      ]);

      classifier.setContextMap(contextMap);

      // Test FORWARD edges (parent -> child)
      const forwardClassification = classifier.classifyEdge(root, child1);
      expect(forwardClassification.type).toBe('FORWARD');

      // Test BACK edges (child -> parent)
      const backClassification = classifier.classifyEdge(child1, root);
      expect(backClassification.type).toBe('BACK');
      expect(backClassification.isAncestor).toBe(true);

      // Test sibling edge (child -> sibling)
      const siblingClassification = classifier.classifyEdge(child1, child2);
      expect(siblingClassification.type).toBe('BACK');
    });

    it('should build page graph during discovery', () => {
      const graph = new PageGraph();
      const classifier = new EdgeClassifier();

      const root = new PageContext('https://notion.so/root-29root', 'Root', 0);
      const parent = new PageContext('https://notion.so/parent-29parent', 'Parent', 1, root, root.id);
      const child = new PageContext('https://notion.so/child-29child', 'Child', 2, parent, parent.id);

      const contextMap = new Map([
        [root.id, root],
        [parent.id, parent],
        [child.id, child]
      ]);

      classifier.setContextMap(contextMap);

      // Simulate discovery process
      graph.addNode(root.id, root);
      const classification1 = classifier.classifyEdge(root, parent);
      graph.addEdge(root.id, parent.id, classification1);
      graph.addNode(parent.id, parent);

      const classification2 = classifier.classifyEdge(parent, child);
      graph.addEdge(parent.id, child.id, classification2);
      graph.addNode(child.id, child);

      // Verify graph structure
      expect(graph.getOutgoingEdges(root.id).size).toBe(1);
      expect(graph.getOutgoingEdges(parent.id).size).toBe(1);
      expect(graph.getEdgeClassification(root.id, parent.id).type).toBe('FORWARD');
      expect(graph.getEdgeClassification(parent.id, child.id).type).toBe('FORWARD');
    });
  });

  describe('Block ID Extraction and Mapping', () => {
    it('should extract and persist block IDs', async () => {
      const html = `
        <html>
          <body>
            <div data-block-id="29d979ee-ca9f-81f7-b82f-e4b983834212">Section 1</div>
            <p data-block-id="12345678-1234-5678-1234-567812345678">Paragraph</p>
          </body>
        </html>
      `;

      const dom = new JSDOM(html);
      const document = dom.window.document;

      // Extract block IDs
      const extractor = new BlockIDExtractor();
      const blockMap = extractor.extractBlockIDs(document);

      expect(blockMap.size).toBe(2);

      // Save and reload block map
      const mapper = new BlockIDMapper();
      await mapper.saveBlockMap('test-page', tempDir, blockMap);

      const loaded = await mapper.loadBlockMap(tempDir);

      expect(loaded.size).toBe(2);
      expect(loaded.get('29d979eeca9f81f7b82fe4b983834212')).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });

    it('should map raw block IDs from URLs to formatted IDs', () => {
      const mapper = new BlockIDMapper();
      const blockMap = new Map([
        ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834212']
      ]);

      // Raw ID from URL
      const rawId = '29d979eeca9f81f7b82fe4b983834212';

      // Get formatted ID
      const formattedId = mapper.getFormattedId(rawId, blockMap);

      expect(formattedId).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });
  });

  describe('Link Rewriting with Block Anchors', () => {
    it('should rewrite links with proper block anchors', async () => {
      const htmlFilePath = path.join(tempDir, 'index.html');
      const html = `
        <html>
          <body>
            <h1>Main Page</h1>
            <a href="https://notion.so/section-29def#29d979eeca9f81f7b82fe4b983834212">Link to Section</a>
          </body>
        </html>
      `;

      await fs.writeFile(htmlFilePath, html);

      // Create page contexts
      const root = new PageContext('https://notion.so/root-29root', 'Root', 0);
      root.htmlFilePath = htmlFilePath;
      root.pathSegments = []; // Root has empty pathSegments

      const targetPage = new PageContext('https://notion.so/section-29def', 'Section', 1, root, root.id);
      targetPage.htmlFilePath = path.join(tempDir, 'Section', 'index.html');
      targetPage.pathSegments = ['Section']; // Set after construction since parentContext is passed but pathSegments computed at construction

      // Save block map for target page
      const blockMap = new Map([
        ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834212']
      ]);

      const sectionDir = path.join(tempDir, 'Section');
      await fs.mkdir(sectionDir, { recursive: true });

      const mapper = new BlockIDMapper();
      await mapper.saveBlockMap(targetPage.id, sectionDir, blockMap);

      // Create URL-to-context map
      const urlToContextMap = new Map([
        ['https://notion.so/root-29root', root],
        ['https://notion.so/section-29def', targetPage]
      ]);

      // Load block maps
      const blockMapCache = new Map([
        [root.id, new Map()],
        [targetPage.id, blockMap]
      ]);

      // Rewrite links
      const rewriter = new LinkRewriter(config, logger);
      await rewriter.rewriteLinksInFile(root, urlToContextMap, null, blockMapCache);

      // Check that HTML was modified
      const modifiedHtml = await fs.readFile(htmlFilePath, 'utf-8');

      // Should contain relative path to Section + formatted block ID
      // Root (pathSegments: []) -> Section (pathSegments: ['Section'])
      // Result: Section/index.html#blockid
      expect(modifiedHtml).toContain('Section/index.html#29d979ee-ca9f-81f7-b82f-e4b983834212');
    });

    it('should handle same-page links with block anchors', async () => {
      const htmlFilePath = path.join(tempDir, 'index.html');
      const html = `
        <html>
          <body>
            <h1 data-block-id="29d979ee-ca9f-81f7-b82f-e4b983834212">Main Title</h1>
            <a href="https://notion.so/page-29page#29d979eeca9f81f7b82fe4b983834212">Jump to Title</a>
          </body>
        </html>
      `;

      await fs.writeFile(htmlFilePath, html);

      const page = new PageContext('https://notion.so/page-29page', 'Page', 0);
      page.htmlFilePath = htmlFilePath;

      const urlToContextMap = new Map([
        ['https://notion.so/page-29page', page]
      ]);

      const blockMapCache = new Map([
        [page.id, new Map([
          ['29d979eeca9f81f7b82fe4b983834212', '29d979ee-ca9f-81f7-b82f-e4b983834212']
        ])]
      ]);

      const rewriter = new LinkRewriter(config, logger);
      await rewriter.rewriteLinksInFile(page, urlToContextMap, null, blockMapCache);

      const modifiedHtml = await fs.readFile(htmlFilePath, 'utf-8');

      // Same-page link should have just the hash
      expect(modifiedHtml).toContain('#29d979ee-ca9f-81f7-b82f-e4b983834212');
    });
  });

  describe('Complete Discovery to Rewriting Flow', () => {
    it('should handle complete flow from discovery to link rewriting', async () => {
      // Setup: Create a simple hierarchy
      const root = new PageContext('https://notion.so/index-29root', 'Index', 0);
      root.pathSegments = []; // Root has empty segments
      
      const about = new PageContext('https://notion.so/about-29about', 'About', 1, root, root.id);
      about.pathSegments = ['About']; // Child has its sanitized title

      // Save HTML files - use capitalized directory names matching pathSegments
      const indexDir = tempDir; // Root is at tempDir itself
      const aboutDir = path.join(tempDir, 'About');

      await fs.mkdir(aboutDir, { recursive: true });

      root.htmlFilePath = path.join(indexDir, 'index.html');
      about.htmlFilePath = path.join(aboutDir, 'index.html');

      const indexHtml = `
        <html>
          <body>
            <h1 data-block-id="11111111-1111-1111-1111-111111111111">Home</h1>
            <a href="https://notion.so/about-29about#22222222222222222222222222222222">About Page</a>
          </body>
        </html>
      `;

      const aboutHtml = `
        <html>
          <body>
            <h1 data-block-id="22222222-2222-2222-2222-222222222222">About</h1>
            <a href="https://notion.so/index-29root#11111111111111111111111111111111">Back to Home</a>
          </body>
        </html>
      `;

      await fs.writeFile(root.htmlFilePath, indexHtml);
      await fs.writeFile(about.htmlFilePath, aboutHtml);

      // Extract and save block maps
      const mapper = new BlockIDMapper();
      const extractor = new BlockIDExtractor();

      const indexDom = new JSDOM(indexHtml);
      const indexBlockMap = extractor.extractBlockIDs(indexDom.window.document);
      await mapper.saveBlockMap(root.id, indexDir, indexBlockMap);

      const aboutDom = new JSDOM(aboutHtml);
      const aboutBlockMap = extractor.extractBlockIDs(aboutDom.window.document);
      await mapper.saveBlockMap(about.id, aboutDir, aboutBlockMap);

      // Create context map
      const urlToContextMap = new Map([
        ['https://notion.so/index-29root', root],
        ['https://notion.so/about-29about', about]
      ]);

      // Load block maps
      const blockMapCache = new Map([
        [root.id, indexBlockMap],
        [about.id, aboutBlockMap]
      ]);

      // Rewrite links
      const rewriter = new LinkRewriter(config, logger);
      await rewriter.rewriteLinksInFile(root, urlToContextMap, null, blockMapCache);
      await rewriter.rewriteLinksInFile(about, urlToContextMap, null, blockMapCache);

      // Verify rewriting
      const rewrittenIndex = await fs.readFile(root.htmlFilePath, 'utf-8');
      const rewrittenAbout = await fs.readFile(about.htmlFilePath, 'utf-8');

      // Index (root, pathSegments: []) -> About (pathSegments: ['About'])
      // Result: About/index.html
      expect(rewrittenIndex).toContain('About/index.html');
      expect(rewrittenIndex).toContain('#22222222-2222-2222-2222-222222222222');

      // About (pathSegments: ['About']) -> Index (root, pathSegments: [])
      // upLevels: 1, downSegments: []
      // Result: ../index.html
      expect(rewrittenAbout).toContain('../index.html');
      expect(rewrittenAbout).toContain('#11111111-1111-1111-1111-111111111111');
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle pages with no block IDs gracefully', async () => {
      const htmlFilePath = path.join(tempDir, 'index.html');
      const html = '<html><body><p>No block IDs here</p></body></html>';
      await fs.writeFile(htmlFilePath, html);

      const page = new PageContext('https://notion.so/page-29page', 'Page', 0);
      page.htmlFilePath = htmlFilePath;

      const urlToContextMap = new Map([[page.url, page]]);
      const blockMapCache = new Map([[page.id, new Map()]]);

      const rewriter = new LinkRewriter(config, logger);
      const linkCount = await rewriter.rewriteLinksInFile(page, urlToContextMap, null, blockMapCache);

      expect(linkCount).toBeGreaterThanOrEqual(0);
    });

    it('should handle missing block maps gracefully', () => {
      const mapper = new BlockIDMapper();

      // Try to get formatted ID without block map
      const formatted = mapper.getFormattedId('29d979eeca9f81f7b82fe4b983834212', null);

      // Should apply fallback formatting
      expect(formatted).toBe('29d979ee-ca9f-81f7-b82f-e4b983834212');
    });
  });
});
