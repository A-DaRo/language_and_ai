const PipelineStep = require('../PipelineStep');
const path = require('path');

/**
 * @fileoverview Asset Download Step
 * @module worker/pipeline/steps/AssetDownloadStep
 */

/**
 * @class AssetDownloadStep
 * @extends PipelineStep
 * @classdesc Downloads all external resources (images, CSS, files) and rewrites references.
 * This is the most time-consuming step in the pipeline, handling potentially dozens
 * of asset downloads per page.
 */
class AssetDownloadStep extends PipelineStep {
  /**
   * @constructor
   * @param {AssetDownloader} assetDownloader - Image/media downloader instance.
   * @param {CssDownloader} cssDownloader - CSS and dependent asset downloader.
   * @param {FileDownloader} fileDownloader - File (PDF, etc.) downloader.
   */
  constructor(assetDownloader, cssDownloader, fileDownloader) {
    super('AssetDownload');
    this.assetDownloader = assetDownloader;
    this.cssDownloader = cssDownloader;
    this.fileDownloader = fileDownloader;
  }
  
  /**
   * @method process
   * @summary Downloads CSS, Images, and embedded Files.
   * @description
   * 1. Calculates output directory from absolute savePath
   * 2. Downloads and rewrites images via AssetDownloader
   * 3. Downloads and rewrites CSS (including fonts/images in CSS) via CssDownloader
   * 4. Downloads and rewrites file attachments via FileDownloader
   * 5. Updates context.stats with download counts
   * 
   * All downloaders modify the DOM in-place, rewriting URLs to local paths.
   * 
   * @param {PipelineContext} context - Pipeline context.
   * @returns {Promise<void>}
   */
  async process(context) {
    const { page, payload, logger, stats } = context;
    
    // Calculate output directory from absolute savePath
    // savePath is like "/abs/path/course_material/Page_Name/index.html"
    // We want "/abs/path/course_material/Page_Name/" for assets
    const outputDir = path.dirname(payload.savePath);
    logger.info('ASSETS', `Output directory: ${outputDir}`);
    
    // Initialize asset statistics
    let totalAssets = 0;
    
    try {
      // Download images and rewrite <img> tags
      logger.info('ASSETS', 'Downloading images...');
      await this.assetDownloader.downloadAndRewriteImages(page, outputDir);
      const imageStats = this.assetDownloader.getStats();
      totalAssets += imageStats.totalAssets;
      logger.success('ASSETS', `Downloaded ${imageStats.totalAssets} image(s)`);
      
      // Download CSS and dependent assets (fonts, background images)
      // USE THE NEW METHOD HERE to avoid JSDOM/Puppeteer interface mismatch
      if (this.cssDownloader) {
        logger.info('ASSETS', 'Downloading stylesheets...');
        await this.cssDownloader.downloadFromPuppeteer(page, outputDir);
        const cssStats = this.cssDownloader.getStats();
        totalAssets += cssStats.stylesheetsDownloaded || 0;
        logger.success('ASSETS', `Downloaded ${cssStats.stylesheetsDownloaded || 0} stylesheet(s)`);
      }
      
      // Download file attachments (PDFs, etc.)
      if (this.fileDownloader) {
        logger.info('ASSETS', 'Downloading file attachments...');
        await this.fileDownloader.downloadAndRewriteFiles(page, outputDir);
        const fileStats = this.fileDownloader.getStats();
        totalAssets += fileStats.totalFiles || 0;
        logger.success('ASSETS', `Downloaded ${fileStats.totalFiles || 0} file(s)`);
      }
      
      // Update stats
      stats.assetsDownloaded = totalAssets;
      logger.success('ASSETS', `Total assets downloaded: ${totalAssets}`);
      
    } catch (error) {
      logger.error('ASSETS', 'Asset download failed', error);
      throw error;
    }
  }
}

module.exports = AssetDownloadStep;
