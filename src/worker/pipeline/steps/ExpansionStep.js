const PipelineStep = require('../PipelineStep');

/**
 * @fileoverview Content Expansion Step
 * @module worker/pipeline/steps/ExpansionStep
 */

/**
 * @class ExpansionStep
 * @extends PipelineStep
 * @classdesc Expands all collapsible content on the page.
 * Uses ContentExpander to reveal toggles, databases, and other hidden elements.
 */
class ExpansionStep extends PipelineStep {
  /**
   * @constructor
   * @param {ContentExpander} contentExpander - Content expander instance.
   */
  constructor(contentExpander) {
    super('ContentExpansion');
    this.contentExpander = contentExpander;
  }
  
  /**
   * @method process
   * @summary Expands all toggles and collapsible elements.
   * @description Uses ContentExpander to scroll and click expandable elements.
   * Critical for ensuring complete page capture.
   * 
   * @param {PipelineContext} context - Pipeline context.
   * @returns {Promise<void>}
   */
  async process(context) {
    const { page, logger } = context;
    
    logger.info('EXPANSION', 'Expanding all collapsible content...');
    await this.contentExpander.expandAll(page);
    logger.success('EXPANSION', 'Content expansion complete');
  }
}

module.exports = ExpansionStep;
