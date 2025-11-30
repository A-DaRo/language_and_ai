const PipelineStep = require('../PipelineStep');

/**
 * @fileoverview HTML Write Step
 * @module worker/pipeline/steps/HtmlWriteStep
 */

/**
 * @class HtmlWriteStep
 * @extends PipelineStep
 * @classdesc Finalizes the scraping process by saving the modified DOM to disk.
 * This is the critical step that actually persists the downloaded content.
 */
class HtmlWriteStep extends PipelineStep {
  constructor() {
    super('HtmlWrite');
  }
  
  /**
   * @method process
   * @summary Serializes and saves the current page state to disk.
   * @description
   * 1. Extracts final HTML content from page.content()
   * 2. Uses WorkerFileSystem.safeWrite() with the absolute savePath from payload
   * 3. Explicit logging ensures write operation is visible (anti-ghost)
   * 
   * The WorkerFileSystem will:
   * - Verify the path is absolute
   * - Create parent directories
   * - Write the file
   * - Log the operation with byte count
   * 
   * @param {PipelineContext} context - Pipeline context.
   * @returns {Promise<void>}
   * @throws {Error} If file write fails or path is invalid.
   */
  async process(context) {
    const { page, payload, fileSystem, logger } = context;
    
    logger.info('HTML-WRITE', 'Extracting final page content...');
    
    // Get the complete HTML after all modifications
    const html = await page.content();
    
    logger.info('HTML-WRITE', `Writing HTML to: ${payload.savePath}`);
    
    // Use WorkerFileSystem for safe, logged write
    await fileSystem.safeWrite(payload.savePath, html);
    
    logger.success('HTML-WRITE', `Successfully saved page to: ${payload.savePath}`);
  }
}

module.exports = HtmlWriteStep;
