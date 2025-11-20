/**
 * @fileoverview Pipeline Step Base Class
 * @module worker/pipeline/PipelineStep
 * @description Defines the interface for individual processing steps in the scraping pipeline.
 */

/**
 * @class PipelineStep
 * @abstract
 * @classdesc Base class for a single unit of work in the scraping process.
 * Each step operates on a shared PipelineContext and performs a specific operation.
 * 
 * Steps should be:
 * - Atomic: Do one thing well
 * - Idempotent: Safe to retry if they fail
 * - Self-contained: No hidden dependencies on other steps
 */
class PipelineStep {
  /**
   * @constructor
   * @param {string} name - Human-readable name for this step (used in logging).
   */
  constructor(name) {
    this.name = name;
  }
  
  /**
   * @method process
   * @abstract
   * @async
   * @summary Execute this step's logic.
   * @description Implementations should modify the context object as needed.
   * Should throw errors rather than returning error objects for cleaner control flow.
   * 
   * @param {PipelineContext} context - Mutable context object shared across pipeline.
   * @returns {Promise<void>}
   * @throws {Error} If step execution fails.
   */
  async process(context) {
    throw new Error(`PipelineStep.process() must be implemented by subclass: ${this.name}`);
  }
}

module.exports = PipelineStep;
