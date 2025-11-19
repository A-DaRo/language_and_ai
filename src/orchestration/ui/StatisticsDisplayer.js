const path = require('path');

/**
 * Formats and displays scraping statistics including audit results and file hierarchy.
 * 
 * @classdesc Responsible for generating the final statistics report after scraping completes.
 * Aggregates metrics from various components (page scraper, asset downloader, CSS downloader)
 * and presents them in a human-readable format. Also displays the integrity audit results
 * and the final page hierarchy.
 * 
 * @see NotionScraper
 * @see IntegrityAuditor
 * @see PageContext
 */
class StatisticsDisplayer {
  /**
   * @param {Logger} logger - Logger instance for formatted output.
   */
  constructor(logger) {
    this.logger = logger;
  }

  /**
   * @summary Print comprehensive scraping statistics and results.
   * 
   * Displays a formatted report including:
   * - Total pages scraped
   * - Assets downloaded (images, files, CSS, CSS assets)
   * - Links rewritten
   * - Time elapsed
   * - Integrity audit results
   * - Page hierarchy
   * 
   * @param {PageContext} rootContext - The root page context.
   * @param {number} totalLinksRewritten - Number of internal links rewritten.
   * @param {Object} auditSummary - Audit results from IntegrityAuditor.
   * @param {number} auditSummary.missingFiles - Count of missing HTML files.
   * @param {number} auditSummary.residualNotionLinks - Count of unrewritten Notion URLs.
   * @param {number} auditSummary.externalStylesheets - Count of external stylesheets.
   * @param {number} auditSummary.issuesFound - Total issues detected.
   * @param {Object} scraperStats - Statistics from PageProcessor.
   * @param {number} scraperStats.pagesScraped - Number of pages processed.
   * @param {Object} assetStats - Statistics from AssetDownloader.
   * @param {number} assetStats.totalAssets - Number of images downloaded.
   * @param {Object} fileStats - Statistics from FileDownloader.
   * @param {number} fileStats.totalFiles - Number of files downloaded.
   * @param {Object} cssStats - Statistics from CssDownloader.
   * @param {number} cssStats.totalCss - Number of CSS files downloaded.
   * @param {number} cssStats.totalCssAssets - Number of CSS assets localized.
   * @param {string} outputDir - Root output directory path.
   * @param {string} elapsedTime - Formatted elapsed time string.
   * @returns {void}
   * 
   * @example
   * displayer.printStatistics(rootContext, 45, auditSummary, 
   *   { pagesScraped: 12 },
   *   { totalAssets: 34 },
   *   { totalFiles: 5 },
   *   { totalCss: 3, totalCssAssets: 8 },
   *   '/output/dir',
   *   '00:02:34'
   * );
   */
  printStatistics(
    rootContext,
    totalLinksRewritten,
    auditSummary,
    scraperStats,
    assetStats,
    fileStats,
    cssStats,
    outputDir,
    elapsedTime
  ) {
    this.logger.separator('Scraping Statistics');
    
    this.logger.info('STATS', `Total pages scraped: ${scraperStats.pagesScraped}`);
    this.logger.info('STATS', `Total images downloaded: ${assetStats.totalAssets}`);
    this.logger.info('STATS', `Total stylesheets downloaded: ${cssStats.totalCss}`);
    this.logger.info('STATS', `Total CSS assets localized: ${cssStats.totalCssAssets}`);
    this.logger.info('STATS', `Total files downloaded: ${fileStats.totalFiles}`);
    this.logger.info('STATS', `Total internal links rewritten: ${totalLinksRewritten}`);
    this.logger.info('STATS', `Total time elapsed: ${elapsedTime}`);
    this.logger.info('STATS', '');
    this.logger.info('STATS', 'SUCCESS: The downloaded site is now fully browsable offline!');
    this.logger.info('STATS', `Open ${path.join(outputDir, 'Main_Page', 'index.html')} in your browser.`);
    this.logger.info('STATS', '');

    if (auditSummary) {
      this.logger.info(
        'STATS',
        `Integrity audit → missing HTML: ${auditSummary.missingFiles}, residual Notion URLs: ${auditSummary.residualNotionLinks}, external stylesheets: ${auditSummary.externalStylesheets}`
      );
      if (auditSummary.issuesFound === 0) {
        this.logger.success('STATS', 'Integrity audit passed with no outstanding issues.');
      } else {
        this.logger.warn('STATS', `Integrity audit flagged ${auditSummary.issuesFound} potential issue(s). See audit log for details.`);
      }
      this.logger.info('STATS', '');
    }
    
    this.logger.info('STATS', 'Page hierarchy:');
    this._printHierarchy(rootContext, 0);
    
    this.logger.separator();
  }

  /**
   * @summary Recursively print page hierarchy with indentation.
   * 
   * Internal helper that displays the page structure as an indented list,
   * showing both the display title and the relative path for each page.
   * 
   * @param {PageContext} context - The page context node to print.
   * @param {number} indent - Current indentation level.
   * @private
   */
  _printHierarchy(context, indent) {
    const prefix = '  '.repeat(indent) + '├─ ';
    const relativePath = context.getRelativePath() || 'root';
    const title = context.displayTitle || context.title;
    console.log(`${prefix}${title} (${relativePath})`);
    
    for (const child of context.children) {
      this._printHierarchy(child, indent + 1);
    }
  }
}

module.exports = StatisticsDisplayer;
