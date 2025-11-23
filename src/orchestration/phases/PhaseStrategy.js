/**
 * @fileoverview Abstract Base Class for Orchestration Phases
 * @module orchestration/phases/PhaseStrategy
 * @description Defines the contract for all orchestration phase implementations.
 */

/**
 * @abstract
 * @class PhaseStrategy
 * @classdesc Base class for all orchestration phases in the Master process.
 * Each phase encapsulates a distinct stage of the scraping workflow.
 */
class PhaseStrategy {
  /**
   * @param {ClusterOrchestrator} orchestrator - Reference to the main orchestrator context
   */
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.logger = orchestrator.logger;
    this.queueManager = orchestrator.queueManager;
    this.browserManager = orchestrator.browserManager;
    this.config = orchestrator.config;
    this.eventBus = orchestrator.eventBus;
  }

  /**
   * @abstract
   * @async
   * @method execute
   * @summary Executes the logic for this specific phase
   * @description Must be implemented by concrete strategies.
   * @returns {Promise<void>} Resolves when the phase is complete
   * @throws {Error} If the phase fails critically
   */
  async execute() {
    throw new Error('PhaseStrategy.execute() must be implemented by subclass');
  }
}

module.exports = PhaseStrategy;
