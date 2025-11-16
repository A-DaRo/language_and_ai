const puppeteer = require('puppeteer');
const axios = require('axios');
const fs = require('fs/promises');
const path = require('path');
const { URL } = require('url');

// --- CONFIGURATION ---
const NOTION_PAGE_URL = 'https://mctenthij.notion.site/JBC090-Language-AI-29d979eeca9f81469905f51d65beefae';
const OUTPUT_DIR = 'downloaded_course_material';
const MAX_EXPANSION_DEPTH = 2; // Set max depth for expanding toggles.
const MAIN_CONTENT_SELECTOR = 'div.notion-page-content'; // The area where toggles should be expanded.
// --- END CONFIGURATION ---

/**
 * Looks for and clicks the "Reject all" cookie button if it exists.
 * Handles the confirmation dialog that appears after clicking.
 * Waits for the page to reload and stabilize.
 * @param {import('puppeteer').Page} page The Puppeteer page object.
 */
async function handleCookieBanner(page) {
  console.log('[COOKIE] Checking for cookie consent banner...');
  try {
    // Wait for the cookie banner to appear
    const bannerSelector = 'div[aria-live="polite"]';
    await page.waitForSelector(bannerSelector, { timeout: 5000 });
    console.log('[COOKIE] Cookie banner detected.');
    
    // Click the "Reject all" button
    const rejectButtonClicked = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('div[role="button"]'));
      const rejectButton = buttons.find(btn => btn.textContent.includes('Reject all'));
      if (rejectButton) {
        rejectButton.click();
        return true;
      }
      return false;
    });
    
    if (rejectButtonClicked) {
      console.log('[COOKIE] Clicked "Reject all" button.');
      
      // Wait for the confirmation dialog to appear
      await new Promise(resolve => setTimeout(resolve, 500));
      console.log('[COOKIE] Waiting for confirmation dialog...');
      
      // Look for and click the "OK" button in the confirmation dialog
      const dialogSelector = 'div[role="dialog"][aria-modal="true"]';
      await page.waitForSelector(dialogSelector, { timeout: 5000 });
      console.log('[COOKIE] Confirmation dialog detected.');
      
      const okButtonClicked = await page.evaluate(() => {
        const dialog = document.querySelector('div[role="dialog"][aria-modal="true"]');
        if (dialog) {
          const buttons = Array.from(dialog.querySelectorAll('div[role="button"]'));
          const okButton = buttons.find(btn => btn.textContent.trim() === 'OK');
          if (okButton) {
            okButton.click();
            return true;
          }
        }
        return false;
      });
      
      if (okButtonClicked) {
        console.log('[COOKIE] Clicked "OK" button. Waiting for page reload...');
        
        // Wait for navigation after clicking OK (page will reload)
        await page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 30000 });
        console.log('[COOKIE] Page reloaded successfully.');
        
        // Additional wait for any dynamic content to load
        await new Promise(resolve => setTimeout(resolve, 2000));
      } else {
        console.log('[COOKIE] Could not find "OK" button in dialog.');
      }
    } else {
      console.log('[COOKIE] Could not find "Reject all" button.');
    }
  } catch (error) {
    console.log('[COOKIE] No cookie banner found or timeout occurred. Continuing...');
  }
}

/**
 * Finds and clicks all expandable toggle blocks within a specific part of the page.
 * Specifically targets Notion toggle blocks with the correct class structure.
 * @param {import('puppeteer').Page} page - The Puppeteer page object.
 * @param {number} maxDepth - The maximum number of nested toggle levels to expand.
 * @param {string} scopeSelector - The CSS selector for the container to search within.
 */
async function expandTogglesWithDepthLimit(page, maxDepth, scopeSelector) {
  console.log(`[TOGGLE] Starting toggle expansion within '${scopeSelector}' (max depth: ${maxDepth})...`);
  
  let currentDepth = 0;
  let totalTogglesExpanded = 0;
  
  while (currentDepth < maxDepth) {
    currentDepth++;
    console.log(`[TOGGLE] Processing depth level ${currentDepth}/${maxDepth}...`);
    
    // This command runs inside the browser and finds/clicks all collapsed toggles
    const togglesClickedCount = await page.evaluate((scopeSel) => {
      const scopeElement = document.querySelector(scopeSel);
      if (!scopeElement) {
        return { clicked: 0, error: `Scope element "${scopeSel}" not found` };
      }
      
      // Find all toggle blocks that are collapsed (aria-expanded="false")
      // These are the disclosure triangles that expand/collapse content
      const collapsedToggles = Array.from(
        scopeElement.querySelectorAll('[role="button"][aria-expanded="false"]')
      ).filter(toggle => {
        // Filter to only include actual toggle blocks (those with the caret icon)
        const svg = toggle.querySelector('svg.arrowCaretDownFillSmall');
        return svg !== null;
      });
      
      let clicked = 0;
      collapsedToggles.forEach(toggle => {
        try {
          toggle.click();
          clicked++;
        } catch (e) {
          console.error('Failed to click toggle:', e);
        }
      });
      
      return { clicked, error: null };
    }, scopeSelector);

    if (togglesClickedCount.error) {
      console.log(`[TOGGLE] ERROR: ${togglesClickedCount.error}`);
      break;
    }

    if (togglesClickedCount.clicked === 0) {
      console.log('[TOGGLE] No more collapsed toggles found. Expansion complete.');
      break;
    }

    totalTogglesExpanded += togglesClickedCount.clicked;
    console.log(`[TOGGLE] Expanded ${togglesClickedCount.clicked} toggles at this depth. Waiting for content to load...`);
    
    // Wait longer for content to fully load after expanding toggles
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  if (currentDepth >= maxDepth) {
    console.log(`[TOGGLE] WARNING: Reached maximum depth of ${maxDepth}. Further nested toggles will not be expanded.`);
  }
  
  console.log(`[TOGGLE] Toggle expansion finished. Total toggles expanded: ${totalTogglesExpanded}`);
}

async function downloadAsset(url, localPath) {
  try {
    const response = await axios({ method: 'GET', url, responseType: 'arraybuffer' });
    await fs.writeFile(localPath, response.data);
    console.log(`[DOWNLOAD] Success: ${path.basename(localPath)}`);
  } catch (error) {
    console.error(`[DOWNLOAD] ERROR: Failed to download ${url}: ${error.message}`);
  }
}

async function downloadFullPage() {
  console.log('[MAIN] ========================================');
  console.log('[MAIN] Starting Notion page download process');
  console.log('[MAIN] ========================================');
  
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  try {
    const imagesDir = path.join(OUTPUT_DIR, 'images');
    await fs.mkdir(imagesDir, { recursive: true });
    console.log(`[MAIN] Output directory created: ./${OUTPUT_DIR}`);

    console.log(`[MAIN] Navigating to: ${NOTION_PAGE_URL}`);
    await page.goto(NOTION_PAGE_URL, { waitUntil: 'networkidle0', timeout: 60000 });
    const pageTitle = await page.title();
    console.log(`[MAIN] Page loaded successfully. Title: "${pageTitle}"`);

    // Handle the cookie banner and wait for it to be removed
    await handleCookieBanner(page);

    // Expand any database or gallery views
    console.log('[DATABASE] Looking for database/gallery "Load more" buttons...');
    let loadMoreClicked = 0;
    try {
      // Keep clicking "Load more" buttons until there are none left
      while (true) {
        const clicked = await page.evaluate(() => {
          // Look for "Load more" or similar buttons in databases
          const loadMoreButtons = Array.from(document.querySelectorAll('div[role="button"]'))
            .filter(btn => {
              const text = btn.textContent.toLowerCase();
              return text.includes('load more') || 
                     text.includes('show more') || 
                     text.includes('view more');
            });
          
          if (loadMoreButtons.length > 0) {
            loadMoreButtons[0].click();
            return true;
          }
          return false;
        });
        
        if (!clicked) break;
        
        loadMoreClicked++;
        console.log(`[DATABASE] Clicked "Load more" button (${loadMoreClicked}). Waiting for content...`);
        await new Promise(resolve => setTimeout(resolve, 1500));
      }
      
      if (loadMoreClicked > 0) {
        console.log(`[DATABASE] Expanded ${loadMoreClicked} database views.`);
      } else {
        console.log('[DATABASE] No "Load more" buttons found.');
      }
    } catch (error) {
      console.log('[DATABASE] Error while expanding databases:', error.message);
    }

    // Scroll to the bottom of the page to ensure all content loads
    console.log('[MAIN] Scrolling to bottom of page to trigger lazy-loading...');
    await page.evaluate(async () => {
      await new Promise((resolve) => {
        let totalHeight = 0;
        const distance = 100;
        const timer = setInterval(() => {
          const scrollHeight = document.body.scrollHeight;
          window.scrollBy(0, distance);
          totalHeight += distance;
          
          if (totalHeight >= scrollHeight) {
            clearInterval(timer);
            resolve();
          }
        }, 100);
      });
    });
    console.log('[MAIN] Reached bottom of page. Waiting for content to stabilize...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Scroll back to top
    await page.evaluate(() => window.scrollTo(0, 0));
    console.log('[MAIN] Scrolled back to top.');

    // Expand toggles within the main content area
    await expandTogglesWithDepthLimit(page, MAX_EXPANSION_DEPTH, MAIN_CONTENT_SELECTOR);

    console.log('[IMAGE] Identifying and downloading all visible images...');
    const urlMap = {};

    const images = await page.$$eval('img', imgs => 
        imgs.map(img => img.src).filter(src => src.startsWith('http'))
    );
    console.log(`[IMAGE] Found ${images.length} images to process.`);
    
    for (const [index, imageUrl] of images.entries()) {
        const urlObj = new URL(imageUrl);
        const imageName = path.basename(urlObj.pathname).split('?')[0];
        const localImageName = `${index + 1}-${decodeURIComponent(imageName)}`;
        const localImagePath = path.join(imagesDir, localImageName);
        const relativePath = path.join('images', localImageName).replace(/\\/g, '/');
        urlMap[imageUrl] = relativePath;
        await downloadAsset(imageUrl, localImagePath);
    }
    
    console.log('[IMAGE] Rewriting image paths in the HTML to point to local files...');
    await page.evaluate(map => {
      document.querySelectorAll('img').forEach(img => {
        if (map[img.src]) img.src = map[img.src];
      });
    }, urlMap);
    console.log('[IMAGE] All image paths rewritten successfully.');

    const finalHtml = await page.content();
    const htmlFilePath = path.join(OUTPUT_DIR, 'index.html');
    await fs.writeFile(htmlFilePath, finalHtml);
    console.log(`[MAIN] Main page saved to: ./${htmlFilePath}`);

  } catch (error) {
    console.error('[MAIN] ERROR: An unexpected error occurred:', error);
  } finally {
    console.log('[MAIN] Closing browser...');
    await browser.close();
    console.log('[MAIN] ========================================');
    console.log('[MAIN] Download process completed');
    console.log('[MAIN] ========================================');
  }
}

downloadFullPage();