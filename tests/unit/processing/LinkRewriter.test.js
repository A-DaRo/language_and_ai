const LinkRewriter = require('../../../src/processing/LinkRewriter');
const { createMockPageContext } = require('../../helpers/factories');
const path = require('path');

const mockLogger = {
  info: jest.fn(),
  success: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn()
};

const mockConfig = {
  getBaseUrl: () => 'https://notion.so',
  OUTPUT_DIR: '/tmp/output'
};

const mockCssDownloader = {
  downloadAndRewriteCss: jest.fn().mockResolvedValue({
    modified: true,
    stylesheets: 2,
    assets: 5,
    inlineStyles: 1
  })
};

// Mock fs/promises
jest.mock('fs/promises');
const fs = require('fs/promises');

describe('LinkRewriter', () => {
  let linkRewriter;

  beforeEach(() => {
    linkRewriter = new LinkRewriter(mockConfig, mockLogger, mockCssDownloader);
    jest.clearAllMocks();
  });

  test('rewrites internal Notion links to relative paths', async () => {
    const parentContext = createMockPageContext({
      url: 'https://notion.so/Parent-29' + 'a'.repeat(30),
      title: 'Parent',
      depth: 0
    });
    parentContext.htmlFilePath = '/tmp/output/Parent/index.html';
    
    const childContext = createMockPageContext({
      url: 'https://notion.so/Child-29' + 'b'.repeat(30),
      title: 'Child',
      depth: 1
    });
    childContext.htmlFilePath = '/tmp/output/Parent/Child/index.html';
    parentContext.addChild(childContext);
    
    const html = `
      <html>
        <body>
          <a href="https://notion.so/Child-29${'b'.repeat(30)}">Child Link</a>
          <a href="https://external.com">External Link</a>
        </body>
      </html>
    `;
    
    fs.readFile = jest.fn().mockResolvedValue(html);
    fs.writeFile = jest.fn().mockResolvedValue();
    
    const urlMap = new Map([
      [childContext.url, childContext]
    ]);
    
    const count = await linkRewriter.rewriteLinksInFile(parentContext, urlMap);
    
    expect(count).toBeGreaterThan(0);
    expect(fs.writeFile).toHaveBeenCalled();
    
    // Verify the rewritten HTML was saved
    const savedHtml = fs.writeFile.mock.calls[0][1];
    expect(savedHtml).toMatch(/href="(\.\.\/)Child\/index\.html"/); // Relative path with ../ prefix
    expect(savedHtml).toMatch(/href="https:\/\/external\.com"/); // External unchanged
  });

  test('skips pages without htmlFilePath', async () => {
    const context = createMockPageContext({ title: 'Test' });
    context.htmlFilePath = null; // No file path
    
    const count = await linkRewriter.rewriteLinksInFile(context, new Map());
    
    expect(count).toBe(0);
    expect(mockLogger.warn).toHaveBeenCalledWith(
      'LINK-REWRITE',
      expect.stringContaining('No HTML file path')
    );
  });

  test('calculates relative paths between sibling pages', async () => {
    const root = createMockPageContext({
      url: 'https://notion.so/Root-29' + 'r'.repeat(30),
      title: 'Root',
      depth: 0
    });
    root.htmlFilePath = '/tmp/output/Root/index.html';
    
    const childA = createMockPageContext({
      url: 'https://notion.so/ChildA-29' + 'a'.repeat(30),
      title: 'ChildA',
      depth: 1
    });
    childA.htmlFilePath = '/tmp/output/Root/ChildA/index.html';
    childA.parentContext = root;
    
    const childB = createMockPageContext({
      url: 'https://notion.so/ChildB-29' + 'b'.repeat(30),
      title: 'ChildB',
      depth: 1
    });
    childB.htmlFilePath = '/tmp/output/Root/ChildB/index.html';
    childB.parentContext = root;
    
    const html = `
      <html>
        <body>
          <a href="https://notion.so/ChildB-29${'b'.repeat(30)}">Sibling Link</a>
        </body>
      </html>
    `;
    
    fs.readFile = jest.fn().mockResolvedValue(html);
    fs.writeFile = jest.fn().mockResolvedValue();
    
    const urlMap = new Map([[childB.url, childB]]);
    
    await linkRewriter.rewriteLinksInFile(childA, urlMap);
    
    const savedHtml = fs.writeFile.mock.calls[0][1];
    expect(savedHtml).toMatch(/\.\.\/ChildB\/index\.html/); // Up one, down to sibling
  });

  test('invokes CSS downloader when provided', async () => {
    const context = createMockPageContext({ title: 'Test' });
    context.htmlFilePath = '/tmp/output/Test/index.html';
    
    fs.readFile = jest.fn().mockResolvedValue('<html><body></body></html>');
    fs.writeFile = jest.fn().mockResolvedValue();
    
    await linkRewriter.rewriteLinksInFile(context, new Map());
    
    expect(mockCssDownloader.downloadAndRewriteCss).toHaveBeenCalled();
    expect(mockLogger.success).toHaveBeenCalledWith(
      'CSS',
      expect.stringContaining('Localized')
    );
  });
});
