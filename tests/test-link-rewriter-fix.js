
const LinkRewriter = require('../src/processing/LinkRewriter');
const PageContext = require('../src/domain/PageContext');
const { JSDOM } = require('jsdom');
const path = require('path');

// Mock dependencies
const mockConfig = {
    getBaseUrl: () => 'https://www.notion.so',
    OUTPUT_DIR: 'output'
};

const mockLogger = {
    info: () => {},
    debug: () => {},
    warn: () => {},
    error: () => {},
    success: () => {}
};

const mockCssDownloader = {
    downloadAndRewriteCss: async () => ({ modified: false })
};

// Mock fs
const fs = require('fs/promises');
const originalReadFile = fs.readFile;
const originalWriteFile = fs.writeFile;

// Test Data
const pageAUrl = 'https://www.notion.so/Page-A-11111111111111111111111111111111';
const pageBUrl = 'https://www.notion.so/Page-B-22222222222222222222222222222222';

const pageAContext = new PageContext(pageAUrl, 'Page A');
const pageBContext = new PageContext(pageBUrl, 'Page B');

// Mock file system state
const mockFiles = {
    [path.join('output', 'Page_A', 'index.html')]: `
        <html>
            <body>
                <a id="link1" href="${pageBUrl}">Link to Page B</a>
                <a id="link2" href="${pageBUrl}#section1">Link to Page B Section</a>
                <a id="link3" href="${pageAUrl}#top">Link to Self (Full URL)</a>
                <a id="link4" href="#local">Local Anchor</a>
            </body>
        </html>
    `
};

// Setup mocks
fs.readFile = async (filePath) => {
    if (mockFiles[filePath]) return mockFiles[filePath];
    throw new Error(`File not found: ${filePath}`);
};

let writtenFiles = {};
fs.writeFile = async (filePath, content) => {
    writtenFiles[filePath] = content;
};

async function runTest() {
    console.log('Starting LinkRewriter Test...');

    const rewriter = new LinkRewriter(mockConfig, mockLogger, mockCssDownloader);
    
    const urlToContextMap = new Map();
    urlToContextMap.set(pageAUrl, pageAContext);
    urlToContextMap.set(pageBUrl, pageBContext);

    // Setup Page A context path
    pageAContext.htmlFilePath = path.join('output', 'Page_A', 'index.html');
    
    // Mock getRelativePathTo
    // Page A is at output/Page_A/index.html
    // Page B is at output/Page_B/index.html (assuming sibling for simplicity)
    // Relative path from A to B should be ../Page_B/index.html
    
    // We need to ensure PageContext.getRelativePathTo works or mock it.
    // Since we are using real PageContext, we need to set up the structure correctly.
    // Let's manually mock getRelativePathTo for simplicity and isolation.
    pageAContext.getRelativePathTo = (target) => {
        if (target === pageBContext) return '../Page_B/index.html';
        if (target === pageAContext) return 'index.html';
        return 'unknown';
    };

    await rewriter.rewriteLinksInFile(pageAContext, urlToContextMap);

    const outputHtml = writtenFiles[pageAContext.htmlFilePath];
    if (!outputHtml) {
        console.error('FAILED: No file written');
        return;
    }

    const dom = new JSDOM(outputHtml);
    const doc = dom.window.document;

    const link1 = doc.getElementById('link1').getAttribute('href');
    const link2 = doc.getElementById('link2').getAttribute('href');
    const link3 = doc.getElementById('link3').getAttribute('href');
    const link4 = doc.getElementById('link4').getAttribute('href');

    console.log(`Link 1 (Page B): ${link1}`);
    console.log(`Link 2 (Page B + Anchor): ${link2}`);
    console.log(`Link 3 (Self + Anchor): ${link3}`);
    console.log(`Link 4 (Local Anchor): ${link4}`);

    let passed = true;

    if (link1 !== '../Page_B/index.html') {
        console.error('FAILED: Link 1 should be ../Page_B/index.html');
        passed = false;
    }

    if (link2 !== '../Page_B/index.html#section1') {
        console.error('FAILED: Link 2 should be ../Page_B/index.html#section1');
        passed = false;
    }

    if (link3 !== '#top') {
        console.error('FAILED: Link 3 should be #top');
        passed = false;
    }

    if (link4 !== '#local') {
        console.error('FAILED: Link 4 should be #local');
        passed = false;
    }

    if (passed) {
        console.log('SUCCESS: All link rewriting tests passed!');
    } else {
        console.log('FAILURE: Some tests failed.');
    }

    // Restore fs
    fs.readFile = originalReadFile;
    fs.writeFile = originalWriteFile;
}

runTest().catch(console.error);
