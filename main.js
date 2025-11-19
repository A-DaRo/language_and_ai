const NotionScraper = require('./src/orchestration/NotionScraper');

/**
 * Main entry point for the Notion scraper
 */
function parseArgs(argv) {
  const options = {
    dryRunOnly: false,
    autoConfirm: false
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dry-run') {
      options.dryRunOnly = true;
    } else if (arg === '--yes') {
      options.autoConfirm = true;
    } else if (arg.startsWith('--max-depth')) {
      let value = null;
      if (arg.includes('=')) {
        value = arg.split('=')[1];
      } else if (i + 1 < argv.length) {
        value = argv[++i];
      }
      const parsed = parseInt(value, 10);
      if (!Number.isNaN(parsed)) {
        options.initialMaxDepth = parsed;
      }
    }
  }
  return options;
}

async function main() {
  const cliOptions = parseArgs(process.argv.slice(2));
  const scraper = new NotionScraper();
  
  try {
    await scraper.run(cliOptions);
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
