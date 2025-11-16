const puppeteer = require('puppeteer');
const axios = require('axios');
const fs = require('fs/promises');
const path = require('path');
const { URL } = require('url');

// --- CONFIGURATION ---
const NOTION_PAGE_URL = 'https://mctenthij.notion.site/JBC090-Language-AI-29d979eeca9f81469905f51d65beefae';
const OUTPUT_DIR = 'downloaded_course_material';
// --- END CONFIGURATION ---

/**
 * A helper function to download a file from a URL and save it locally.
 * @param {string} url - The URL of the asset to download.
 * @param {string} localPath - The local file path to save the asset to.
 */
async function downloadAsset(url, localPath) {
  try {
    const response = await axios({
      method: 'GET',
      url: url,
      responseType: 'arraybuffer', // Important for binary files like images
    });
    await fs.writeFile(localPath, response.data);
    console.log(`  ✅ Downloaded: ${url}`);
  } catch (error) {
    console.error(`  ❌ Failed to download ${url}: ${error.message}`);
  }
}

/**
 * The main function to scrape and download the Notion page.
 */
async function downloadFullPage() {
  console.log('🚀 Starting Notion page download process...');
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  try {
    // 1. Create directory structure
    const imagesDir = path.join(OUTPUT_DIR, 'images');
    await fs.mkdir(imagesDir, { recursive: true });
    console.log(`📁 Output directory created at: ./${OUTPUT_DIR}`);

    // 2. Navigate to the page and wait for it to fully load
    console.log(`🌐 Navigating to: ${NOTION_PAGE_URL}`);
    await page.goto(NOTION_PAGE_URL, { waitUntil: 'networkidle0', timeout: 60000 });
    const pageTitle = await page.title();
    console.log(`👍 Page loaded successfully. Title: "${pageTitle}"`);

    // 3. Find, download, and map all images
    console.log('🖼️ Identifying and downloading images...');
    const urlMap = {}; // Maps original URL to new local path

    const images = await page.$$eval('img', imgs => 
        imgs.map(img => img.src).filter(src => src.startsWith('http'))
    );
    console.log(`🔎 Found ${images.length} images to process.`);
    
    let imageCounter = 0;
    for (const imageUrl of images) {
        imageCounter++;
        const urlObj = new URL(imageUrl);
        const imageName = path.basename(urlObj.pathname);
        const localImageName = `${imageCounter}-${imageName}`; // Prepend counter to avoid name collisions
        const localImagePath = path.join(imagesDir, localImageName);
        const relativePath = path.join('images', localImageName).replace(/\\/g, '/'); // Use forward slashes for HTML

        urlMap[imageUrl] = relativePath;
        await downloadAsset(imageUrl, localImagePath);
    }
    
    // 4. Rewrite the HTML in the browser's context to use local paths
    console.log('✏️ Rewriting image paths in the HTML...');
    await page.evaluate(map => {
      document.querySelectorAll('img').forEach(img => {
        if (map[img.src]) {
          img.src = map[img.src];
        }
      });
    }, urlMap);
    console.log('✅ All image paths rewritten.');

    // 5. Save the final, modified HTML
    const finalHtml = await page.content();
    const htmlFilePath = path.join(OUTPUT_DIR, 'index.html');
    await fs.writeFile(htmlFilePath, finalHtml);
    console.log(`💾 Main page saved to: ./${htmlFilePath}`);

  } catch (error) {
    console.error('❌ An unexpected error occurred:', error);
  } finally {
    console.log('Closing browser...');
    await browser.close();
    console.log('🎉 Download process finished!');
  }
}

downloadFullPage();