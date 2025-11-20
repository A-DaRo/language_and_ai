/**
 * @fileoverview Linear Execution Controller
 * @module worker/pipeline/ScrapingPipeline
 * @description Orchestrates sequential execution of scraping steps with error isolation.
 */

/**
 * @typedef {Object} PipelineContext
 * @description Shared mutable state passed between pipeline steps.
 * @property {import('puppeteer').Browser} browser - Puppeteer browser instance.
 * @property {import('puppeteer').Page} page - Puppeteer page instance.
 * @property {Config} config - System configuration.
 * @property {Logger} logger - Logger instance.
 * @property {Object} payload - The download task payload from Master.
 * @property {WorkerFileSystem} fileSystem - File I/O abstraction.
 * @property {Object} stats - Accumulator for step statistics.
 * @property {Array<string>} downloadedAssets - List of downloaded asset paths.
 */

/**
 * @class ScrapingPipeline
 * @classdesc Orchestrates the sequential execution of scraping steps.
 * Implements the "Pipe and Filter" pattern to isolate failures and enable testing.
 * 
 * Each step receives the same PipelineContext object, allowing steps to:
 * - Read inputs from previous steps
 * - Write outputs for subsequent steps
 * - Accumulate statistics
 * 
 * If any step throws an error, the pipeline halts immediately and propagates the error.
 */
class ScrapingPipeline {
  /**
   * @constructor
   * @param {Array<PipelineStep>} steps - Ordered list of steps to execute.
   * @param {Logger} logger - Logger instance for pipeline-level logging.
   */
  constructor(steps, logger) {
    this.steps = steps;
    this.logger = logger;
  }
  
  /**
   * @method execute
   * @async
   * @summary Runs the pipeline for a specific task.
   * @description Iterates through steps in order. Passes the shared `PipelineContext` to each.
   * Stops execution immediately if a step throws an error. Logs progress after each step.
   * 
   * @param {PipelineContext} context - Shared state object (Browser, Page, Config, Payload).
   * @returns {Promise<void>}
   * @throws {Error} Propagates error from failing step with step name context.
   */
  async execute(context) {
    this.logger.info('PIPELINE', `Starting pipeline with ${this.steps.length} step(s)`);
    
    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      const stepNumber = i + 1;
      
      try {
        this.logger.info('PIPELINE', `[${stepNumber}/${this.steps.length}] Executing: ${step.name}`);
        
        const startTime = Date.now();
        await step.process(context);
        const duration = Date.now() - startTime;
        
        this.logger.success('PIPELINE', `[${stepNumber}/${this.steps.length}] Completed: ${step.name} (${duration}ms)`);
        
      } catch (error) {
        this.logger.error('PIPELINE', `[${stepNumber}/${this.steps.length}] Failed: ${step.name}`, error);
        
        // Enhance error with step context
        const enhancedError = new Error(`Pipeline failed at step "${step.name}": ${error.message}`);
        enhancedError.originalError = error;
        enhancedError.stepName = step.name;
        enhancedError.stepNumber = stepNumber;
        
        throw enhancedError;
      }
    }
    
    this.logger.success('PIPELINE', 'Pipeline completed successfully');
  }
}

module.exports = ScrapingPipeline;
