const NotionScraper = require('./src/NotionScraper');

/**
 * Main entry point for the Notion scraper
 */
async function main() {
  const scraper = new NotionScraper();
  
  try {
    await scraper.run();
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  }
}

// Run if this is the main module
if (require.main === module) {
  main();
}

module.exports = { NotionScraper };
