const fs = require('fs/promises');

/**
 * Performs a post-scrape integrity audit to ensure the offline archive is truly self-contained.
 * Checks for missing HTML files, residual Notion-hosted links, and external stylesheet references.
 */
class IntegrityAuditor {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;

    try {
      const url = new URL(this.config.NOTION_PAGE_URL);
      this.baseHostname = url.hostname;
    } catch (error) {
      this.baseHostname = 'notion.site';
    }
  }

  /**
   * Audit all saved page contexts.
   * @param {PageContext[]} contexts
   * @returns {Promise<{missingFiles:number, residualNotionLinks:number, externalStylesheets:number, issuesFound:number}>}
   */
  async audit(contexts) {
    this.logger.separator('Final Integrity Audit');

    let missingFiles = 0;
    let residualNotionLinks = 0;
    let externalStylesheets = 0;

    for (const context of contexts) {
      const filePath = context.getFilePath(this.config.OUTPUT_DIR);

      const pageLabel = `${context.title}`;

      try {
        await fs.access(filePath);
      } catch (error) {
        missingFiles++;
        this.logger.error('AUDIT', `Missing HTML for ${pageLabel} (expected at ${filePath})`);
        continue;
      }

      let html;
      try {
        html = await fs.readFile(filePath, 'utf-8');
      } catch (error) {
        missingFiles++;
        this.logger.error('AUDIT', `Unable to read HTML for ${pageLabel}: ${error.message}`);
        continue;
      }

      // Search for residual Notion-hosted URLs that should have been localized
      const notionRegex = new RegExp(`https?:\/\/[^"']*${this.baseHostname}[^"']*`, 'gi');
      let notionMatch;
      let localResidualCount = 0;
      while ((notionMatch = notionRegex.exec(html)) !== null) {
        residualNotionLinks++;
        localResidualCount++;
        if (localResidualCount <= 3) {
          this.logger.warn('AUDIT', `Residual Notion URL in ${pageLabel}: ${notionMatch[0]}`);
        }
      }

      // Identify external stylesheets that were not localized
      const externalCssRegex = /<link[^>]+rel=["']stylesheet["'][^>]+href=["'](http[^"']+)["']/gi;
      let cssMatch;
      let localCssCount = 0;
      while ((cssMatch = externalCssRegex.exec(html)) !== null) {
        externalStylesheets++;
        localCssCount++;
        if (localCssCount <= 3) {
          this.logger.warn('AUDIT', `External stylesheet reference in ${pageLabel}: ${cssMatch[1]}`);
        }
      }
    }

    const issuesFound = missingFiles + residualNotionLinks + externalStylesheets;

    if (issuesFound === 0) {
      this.logger.success('AUDIT', 'Integrity audit passed. All pages localized successfully.');
    } else {
      this.logger.warn('AUDIT', `Integrity audit detected ${issuesFound} potential issue(s). Review warnings above.`);
    }

    return {
      missingFiles,
      residualNotionLinks,
      externalStylesheets,
      issuesFound
    };
  }
}

module.exports = IntegrityAuditor;
