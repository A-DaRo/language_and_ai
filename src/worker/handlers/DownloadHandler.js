/**
 * @fileoverview Download task handler
 * @module worker/handlers/DownloadHandler
 * @description Handles IPC_DOWNLOAD messages in worker process.
 */

const Logger = require('../../core/Logger');
const WorkerFileSystem = require('../io/WorkerFileSystem');
const ScrapingPipeline = require('../pipeline/ScrapingPipeline');
const NavigationStep = require('../pipeline/steps/NavigationStep');
const CookieConsentStep = require('../pipeline/steps/CookieConsentStep');
const ExpansionStep = require('../pipeline/steps/ExpansionStep');
const AssetDownloadStep = require('../pipeline/steps/AssetDownloadStep');
const LinkRewriterStep = require('../pipeline/steps/LinkRewriterStep');
const HtmlWriteStep = require('../pipeline/steps/HtmlWriteStep');
const CookieHandler = require('../../processing/CookieHandler');
const ContentExpander = require('../../processing/ContentExpander');
const AssetDownloader = require('../../download/AssetDownloader');
const CssDownloader = require('../../download/CssDownloader');
const FileDownloader = require('../../download/FileDownloader');
const BlockIDExtractor = require('../../extraction/BlockIDExtractor');
const BlockIDMapper = require('../../processing/BlockIDMapper');

const path = require('path');
const { JSDOM } = require('jsdom');

/**
 * @class DownloadHandler
 * @classdesc Executes the full scraping pipeline.
 */
class DownloadHandler {
  /**
   * @param {import('puppeteer').Browser} browser - Puppeteer browser
   * @param {Object} context - Worker context
   */
  constructor(browser, context = {}) {
    this.browser = browser;
    this.page = null;
    this.config = context.config;
    this.cookies = context.cookies || [];
    this.titleRegistry = context.titleRegistry || {};
    this.logger = Logger.getInstance();

    // Initialize pipeline components (instantiate once per worker for cache efficiency)
    this.cookieHandler = new CookieHandler(this.config, this.logger);
    this.contentExpander = new ContentExpander(this.config, this.logger);
    this.assetDownloader = new AssetDownloader(this.config, this.logger);
    this.cssDownloader = new CssDownloader(this.config, this.logger);
    this.fileDownloader = new FileDownloader(this.config, this.logger);
    this.fileSystem = new WorkerFileSystem(this.logger);

    // Block ID extraction and mapping
    this.blockIDExtractor = new BlockIDExtractor();
    this.blockIDMapper = new BlockIDMapper();
  }

  /**
   * @async
   * @summary Executes the ScrapingPipeline
   * @description
   * 1. Validates absolute paths in payload
   * 2. Configures pipeline steps
   * 3. Runs the pipeline
   * 4. Returns execution statistics
   * @param {Object} payload - Download payload
   * @returns {Promise<Object>} Download result
   */
  async handle(payload) {
    this._validatePayload(payload);

    const displayTitle = this.titleRegistry[payload.pageId] || 'Unknown';
    this.logger.info('DOWNLOAD', `Starting: ${displayTitle}`);

    // Create or reuse page
    if (!this.page) {
      this.page = await this.browser.newPage();
      await this.page.setViewport({ width: 1920, height: 1080 });
    }

    // Initialize pipeline context
    const pipelineContext = {
      browser: this.browser,
      page: this.page,
      config: this.config,
      logger: this.logger,
      payload: payload,
      fileSystem: this.fileSystem,
      stats: {
        assetsDownloaded: 0,
        linksRewritten: 0
      },
      downloadedAssets: []
    };

    // Construct the scraping pipeline
    const pipeline = new ScrapingPipeline(
      [
        new NavigationStep(),
        new CookieConsentStep(this.cookieHandler),
        new ExpansionStep(this.contentExpander),
        new AssetDownloadStep(
          this.assetDownloader,
          this.cssDownloader,
          this.fileDownloader
        ),
        new LinkRewriterStep(),
        new HtmlWriteStep()
      ],
      this.logger
    );

    try {
      await pipeline.execute(pipelineContext);

      // NEW: Extract block IDs from saved HTML
      await this._extractAndSaveBlockIds(payload);

      this.logger.success('DOWNLOAD', `Completed: ${displayTitle}`);

      return {
        success: true,
        pageId: payload.pageId,
        url: payload.url,
        savedPath: payload.savePath,
        assetsDownloaded: pipelineContext.stats.assetsDownloaded,
        linksRewritten: pipelineContext.stats.linksRewritten,
        blockMapSaved: true
      };

    } catch (error) {
      this.logger.error('DOWNLOAD', `Failed: ${displayTitle}`, error);
      throw error;
    }
  }

  /**
   * Extract block IDs from saved HTML and save mapping
   * @private
   * @param {Object} payload - Download payload
   */
  async _extractAndSaveBlockIds(payload) {
    try {
      const fs = require('fs/promises');
      const htmlContent = await fs.readFile(payload.savePath, 'utf-8');
      const dom = new JSDOM(htmlContent);
      const document = dom.window.document;

      // Extract block IDs from HTML
      const blockMap = this.blockIDExtractor.extractBlockIDs(document);

      // Save mapping to disk
      if (blockMap.size > 0) {
        const saveDir = path.dirname(payload.savePath);
        await this.blockIDMapper.saveBlockMap(payload.pageId, saveDir, blockMap);
        this.logger.debug('BLOCK-ID', `Saved ${blockMap.size} block IDs for ${payload.pageId}`);
      }
    } catch (error) {
      // Log but don't fail - block ID extraction is optional
      this.logger.debug('BLOCK-ID', `Failed to extract block IDs: ${error.message}`);
    }
  }

  /**
   * @private
   * @param {Object} payload - Download payload to validate
   * @throws {Error} If payload is invalid
   */
  _validatePayload(payload) {
    const requiredFields = ['url', 'pageId', 'savePath'];

    for (const field of requiredFields) {
      if (!payload[field]) {
        throw new Error(`Invalid payload: missing required field '${field}'`);
      }
    }

    if (!path.isAbsolute(payload.savePath)) {
      throw new Error(`Invalid payload: savePath must be absolute. Received: ${payload.savePath}`);
    }

    this.logger.debug('VALIDATION', `Payload validated for ${payload.pageId}`);
  }

  /**
   * @async
   * @summary Cleanup resources
   */
  async cleanup() {
    if (this.page) {
      await this.page.close();
      this.page = null;
    }
  }
}

module.exports = DownloadHandler;
