#!/usr/bin/env node
/**
 * @fileoverview Main Entry Point for Cluster Mode
 * @description Initializes and orchestrates the distributed scraping workflow.
 * Implements the new logging and UI dashboard system.
 */

const { ClusterOrchestrator } = require('./src/orchestration/ClusterOrchestrator');
const Config = require('./src/core/Config');
const Logger = require('./src/core/Logger');
const DashboardController = require('./src/ui/DashboardController');
const SystemEventBus = require('./src/core/SystemEventBus');

/**
 * Main application entry point
 * @async
 */
async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run') || args.includes('-d');
  const help = args.includes('--help') || args.includes('-h');
  
  if (help) {
    console.log(`
Usage: node main-cluster.js [options]

Options:
  --dry-run, -d    Run discovery phase only (no downloads)
  --help, -h       Show this help message

Example:
  node main-cluster.js                # Full scrape
  node main-cluster.js --dry-run      # Discovery only
    `);
    process.exit(0);
  }

  const config = new Config();
  const logger = Logger.getInstance();
  const eventBus = SystemEventBus.getInstance();

  // Phase 1: Initialize Logger with ONLY File strategy (dashboard will handle visual output)
  logger.init({
    console: false, // Disable console to prevent terminal pollution
    file: config.LOG_FILE_ENABLED,
    outputDir: config.LOG_DIR,
    logSubdir: 'logs'
  });

  // Temporarily use console for initial startup messages before dashboard
  logger.addStrategy(new (require('./src/core/logger/ConsoleStrategy'))());
  
  logger.separator('JBC090 Language & AI - Notion Scraper');
  logger.info('MAIN', 'Starting application...');
  logger.info('CONFIG', `Target URL: ${config.NOTION_PAGE_URL}`);
  logger.info('CONFIG', `Output directory: ${config.OUTPUT_DIR}`);
  logger.info('CONFIG', `Max recursion depth: ${config.MAX_RECURSION_DEPTH}`);
  
  if (dryRun) {
    logger.info('MODE', 'Dry run mode enabled - discovery phase only');
  }

  try {
    // Create orchestrator (this initializes browserManager)
    const orchestrator = new ClusterOrchestrator(config, logger);

    // Set up dashboard to start after bootstrap completes
    let dashboardController = null;
    eventBus.once('BOOTSTRAP:COMPLETE', ({ workerCount }) => {
      // Phase 2: Create and start Dashboard Controller (now that workers exist)
      dashboardController = new DashboardController(eventBus, orchestrator.browserManager);
      dashboardController.start();

      // Phase 3: Switch logger from console to dashboard mode (removes console strategy)
      logger.switchMode('dashboard', { dashboardInstance: dashboardController.getDashboard() });
      
      // Listen for phase change to switch back to dashboard mode if needed
      // Registered here to ensure it runs AFTER DashboardController has handled the event (and restarted)
      eventBus.on('PHASE:CHANGED', ({ phase }) => {
        if (phase === 'download' && dashboardController) {
          // Switch logger back to dashboard mode
          logger.switchMode('dashboard', { dashboardInstance: dashboardController.getDashboard() });
        }
      });
    });

    // Listen for dashboard stop signal (user confirmation phase)
    eventBus.on('PHASE:STOPPING_DASHBOARD', () => {
      // Switch logger back to console mode for user confirmation
      logger.switchMode('console');
    });

    // Run the full workflow
    // The orchestrator handles all phases internally
    await orchestrator.start(
      config.NOTION_PAGE_URL,
      config.MAX_RECURSION_DEPTH,
      dryRun
    );

    // Stop the dashboard before showing final stats
    if (dashboardController) {
      dashboardController.stop();
    }
    
    // Switch back to console for final output
    logger.switchMode('console');

    // Get final statistics
    const stats = orchestrator.queueManager.getStatistics();
    
    logger.separator('Scraping Complete');
    logger.success('STATS', `Total pages discovered: ${stats.discovered}`);
    logger.success('STATS', `Total pages downloaded: ${stats.downloaded}`);
    logger.info('STATS', `Failed downloads: ${stats.failed}`);
    logger.info('STATS', `Output directory: ${config.OUTPUT_DIR}`);
    logger.separator();

    logger.info('MAIN', 'All operations completed successfully');
    logger.close();
    
    process.exit(0);

  } catch (error) {
    logger.error('MAIN', 'Fatal error during execution', error);
    logger.close();
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\nReceived SIGINT. Shutting down gracefully...');
  const logger = Logger.getInstance();
  logger.close();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n\nReceived SIGTERM. Shutting down gracefully...');
  const logger = Logger.getInstance();
  logger.close();
  process.exit(0);
});

// Start the application
if (require.main === module) {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

module.exports = { main };
