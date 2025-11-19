/**
 * @fileoverview Main Entry Point for Cluster Mode
 * @description Bootstraps the Micro-Kernel distributed scraping system.
 * This is the new entry point for the cluster-based architecture.
 * 
 * For legacy monolithic mode, use main.js instead.
 */

// Set UTF-8 encoding for console output on Windows
if (process.platform === 'win32') {
  try {
    process.stdout.setDefaultEncoding('utf8');
    process.stderr.setDefaultEncoding('utf8');
  } catch (e) {
    // Ignore if fails
  }
}

const { ClusterOrchestrator } = require('./src/orchestration/ClusterOrchestrator');
const Config = require('./src/core/Config');
const Logger = require('./src/core/Logger');
const SystemEventBus = require('./src/core/SystemEventBus');

/**
 * Parse command-line arguments
 * @param {Array<string>} argv - Command-line arguments
 * @returns {Object} Parsed options
 */
function parseArgs(argv) {
  const options = {
    maxDepth: null,
    dryRun: false
  };
  
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    
    if (arg === '--dry-run') {
      options.dryRun = true;
    } else if (arg.startsWith('--max-depth')) {
      let value = null;
      if (arg.includes('=')) {
        value = arg.split('=')[1];
      } else if (i + 1 < argv.length) {
        value = argv[++i];
      }
      const parsed = parseInt(value, 10);
      if (!Number.isNaN(parsed)) {
        options.maxDepth = parsed;
      }
    } else if (arg === '--help' || arg === '-h') {
      console.log(`
Notion Recursive Scraper - Cluster Mode

Usage: node main-cluster.js [options]

Options:
  --max-depth <n>    Maximum recursion depth (default: from config)
  --dry-run          Discovery only, no downloads (plan phase)
  --help, -h         Show this help message

Examples:
  node main-cluster.js
  node main-cluster.js --max-depth 3
  node main-cluster.js --dry-run --max-depth 2
      `);
      process.exit(0);
    }
  }
  
  return options;
}

/**
 * Setup graceful shutdown handlers
 * @param {ClusterOrchestrator} orchestrator - Orchestrator instance
 */
function setupShutdownHandlers(orchestrator) {
  let shuttingDown = false;
  
  const shutdown = async (signal) => {
    if (shuttingDown) {
      console.log('\nForce exit...');
      process.exit(1);
    }
    
    shuttingDown = true;
    console.log(`\n\nReceived ${signal}. Shutting down gracefully...`);
    
    try {
      await orchestrator.shutdown();
      process.exit(0);
    } catch (error) {
      console.error('Error during shutdown:', error);
      process.exit(1);
    }
  };
  
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  
  process.on('uncaughtException', async (error) => {
    console.error('Uncaught exception:', error);
    await shutdown('uncaughtException');
  });
  
  process.on('unhandledRejection', async (reason, promise) => {
    console.error('Unhandled rejection at:', promise, 'reason:', reason);
    await shutdown('unhandledRejection');
  });
}

/**
 * Main entry point
 */
async function main() {
  const config = new Config();
  const logger = new Logger();
  const cliOptions = parseArgs(process.argv.slice(2));
  
  try {
    logger.separator('Notion Scraper - Cluster Mode');
    logger.info('MAIN', 'Initializing distributed scraping system...');
    
    // Initialize orchestrator
    const orchestrator = new ClusterOrchestrator(config, logger);
    
    // Setup shutdown handlers
    setupShutdownHandlers(orchestrator);
    
    // Emit system initialization event
    const eventBus = SystemEventBus.getInstance();
    eventBus.emit('SYSTEM:INIT', { config });
    
    // Start orchestration
    const maxDepth = cliOptions.maxDepth || config.MAX_RECURSION_DEPTH;
    logger.info('MAIN', `Target URL: ${config.NOTION_PAGE_URL}`);
    logger.info('MAIN', `Output directory: ${config.OUTPUT_DIR}`);
    logger.info('MAIN', `Max depth: ${maxDepth}`);
    
    if (cliOptions.dryRun) {
      logger.info('MAIN', 'DRY RUN MODE: Discovery only, no downloads will occur');
    }
    
    const result = await orchestrator.start(config.NOTION_PAGE_URL, maxDepth, cliOptions.dryRun);
    
    // Display results
    if (cliOptions.dryRun) {
      logger.separator('Discovery Complete (Dry Run)');
      logger.success('MAIN', `Total pages discovered: ${result.stats.discovered}`);
      logger.info('MAIN', 'No files downloaded (dry run mode)');
      logger.info('MAIN', '');
      logger.info('MAIN', 'To proceed with downloading, run without --dry-run flag:');
      logger.info('MAIN', `  node main-cluster.js --max-depth ${maxDepth}`);
    } else {
      logger.separator('Scraping Complete');
      logger.success('MAIN', `Total pages discovered: ${result.stats.discovered}`);
      logger.success('MAIN', `Total pages downloaded: ${result.stats.downloaded}`);
      
      if (result.stats.failed > 0) {
        logger.warn('MAIN', `Failed tasks: ${result.stats.failed}`);
      }
      
      logger.info('MAIN', `Output saved to: ${config.OUTPUT_DIR}`);
    }
    
    // Cleanup
    await orchestrator.shutdown();
    
    process.exit(0);
    
  } catch (error) {
    logger.error('MAIN', 'Fatal error:', error);
    console.error(error.stack);
    process.exit(1);
  }
}

// Run if this is the main module
if (require.main === module) {
  main();
}

module.exports = { main };
