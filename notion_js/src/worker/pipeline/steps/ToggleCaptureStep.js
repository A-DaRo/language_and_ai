/**
 * @file ToggleCaptureStep.js
 * @module worker/pipeline/steps/ToggleCaptureStep
 * @description Pipeline step for capturing dual-state toggle content.
 * 
 * This step integrates ToggleStateCapture into the ScrapingPipeline, capturing
 * both collapsed and expanded states of toggle elements for offline interactivity.
 * 
 * Execution Position: After ExpansionStep (scrolling), before AssetDownloadStep.
 * This ensures lazy-loaded content is available but toggles haven't been altered.
 * 
 * @see {@link ../../../processing/ToggleStateCapture.js}
 * @see {@link ../../../processing/OfflineToggleController.js}
 */

'use strict';

const PipelineStep = require('../PipelineStep');
const ToggleStateCapture = require('../../../processing/ToggleStateCapture');
const OfflineToggleController = require('../../../processing/OfflineToggleController');

/**
 * @class ToggleCaptureStep
 * @extends PipelineStep
 * @classdesc Captures toggle states and injects offline interactivity controller.
 * 
 * This step performs two operations:
 * 1. Captures dual-state HTML for all toggles using ToggleStateCapture
 * 2. Injects OfflineToggleController JavaScript for runtime interactivity
 * 
 * The captured states are stored in context.toggleStates for potential
 * use by subsequent steps or for debugging/logging.
 */
class ToggleCaptureStep extends PipelineStep {
    /**
     * Create a new ToggleCaptureStep.
     * @param {Object} [options={}] - Configuration options
     * @param {boolean} [options.injectController=true] - Whether to inject controller script
     * @param {number} [options.animationWait=300] - Wait time for toggle animations
     * @param {number} [options.maxToggles=100] - Maximum toggles to process
     */
    constructor(options = {}) {
        super('ToggleCapture');
        this.options = {
            injectController: options.injectController !== false,
            animationWait: options.animationWait || 300,
            maxToggles: options.maxToggles || 100
        };
    }

    /**
     * Execute toggle capture and controller injection.
     * @async
     * @param {PipelineContext} context - Pipeline context
     * @returns {Promise<void>}
     */
    async process(context) {
        const { page, logger } = context;

        logger.info('TOGGLE-CAPTURE', 'Starting dual-state toggle capture...');

        // Create ToggleStateCapture instance
        const capture = new ToggleStateCapture(logger, {
            animationWait: this.options.animationWait,
            maxToggles: this.options.maxToggles
        });

        // Capture all toggle states
        const result = await capture.captureAllToggleStates(page);

        // Store result in context for potential downstream use
        context.toggleStates = result.toggleStates;
        context.stats.togglesCaptured = result.capturedCount;
        context.stats.togglesFailed = result.failedCount;

        // Inject offline controller if we captured any toggles
        if (result.capturedCount > 0 && this.options.injectController) {
            await this._injectOfflineController(page, result.toggleStates, logger);
        } else if (result.capturedCount === 0) {
            logger.info('TOGGLE-CAPTURE', 'No toggles found - skipping controller injection');
        }

        logger.success('TOGGLE-CAPTURE', 
            `Toggle capture complete: ${result.capturedCount} captured, ${result.failedCount} failed`);
    }

    /**
     * Inject the offline toggle controller script into the page.
     * @async
     * @private
     * @param {import('puppeteer').Page} page - Puppeteer page
     * @param {Map<string, ToggleState>} toggleStates - Captured toggle states
     * @param {Logger} logger - Logger instance
     */
    async _injectOfflineController(page, toggleStates, logger) {
        logger.info('TOGGLE-CAPTURE', 'Injecting offline toggle controller...');

        // Generate controller script with embedded state data
        const controllerScript = OfflineToggleController.generateScript(toggleStates);

        // Inject into page body (end)
        await page.evaluate((script) => {
            const scriptEl = document.createElement('script');
            scriptEl.id = 'notion-offline-toggle-controller';
            scriptEl.textContent = script;
            document.body.appendChild(scriptEl);
        }, controllerScript);

        // Also inject the CSS for toggle styling
        const controllerStyles = OfflineToggleController.generateStyles();
        await page.evaluate((css) => {
            const styleEl = document.createElement('style');
            styleEl.id = 'notion-offline-toggle-styles';
            styleEl.textContent = css;
            document.head.appendChild(styleEl);
        }, controllerStyles);

        logger.success('TOGGLE-CAPTURE', 'Offline controller injected successfully');
    }
}

module.exports = ToggleCaptureStep;
