/**
 * @file ToggleStateCapture.js
 * @module processing/ToggleStateCapture
 * @description Captures dual-state toggle content for offline interactivity.
 * 
 * This class solves the fundamental tradeoff in toggle handling:
 * - "Expand All" approach: Captures complete content but destroys interactivity
 * - "Expand None" approach: Preserves interactivity but loses hidden content
 * - **Dual-State Capture**: Captures BOTH states, enabling complete content WITH interactivity
 * 
 * Algorithm:
 * 1. Find all toggle elements on page (role="button" with aria-expanded)
 * 2. For each toggle, record collapsed state HTML
 * 3. Click to expand, record expanded state HTML
 * 4. Click to restore original collapsed state
 * 5. Return state map for injection by OfflineToggleController
 * 
 * Uses HtmlFacade abstraction for context-agnostic DOM operations.
 * 
 * @see {@link ../Docs/29112025_Codebase_Review.md} Section 5.5
 * @see {@link ./OfflineToggleController.js}
 */

'use strict';

const { HtmlFacadeFactory } = require('../html');

/**
 * @typedef {Object} ToggleState
 * @property {string} toggleId - Unique identifier (data-block-id or generated)
 * @property {string} collapsedHtml - HTML when aria-expanded="false"
 * @property {string} expandedHtml - HTML when aria-expanded="true"
 * @property {string} triggerSelector - CSS selector for click target
 * @property {boolean} initiallyExpanded - Whether toggle was expanded when captured
 */

/**
 * @typedef {Object} CaptureResult
 * @property {Map<string, ToggleState>} toggleStates - Map of toggle ID to state
 * @property {number} capturedCount - Number of toggles successfully captured
 * @property {number} failedCount - Number of toggles that failed to capture
 * @property {Array<string>} errors - Error messages from failed captures
 */

/**
 * @class ToggleStateCapture
 * @classdesc Captures dual-state toggle content for offline interactivity.
 * 
 * Implements a non-destructive capture strategy that:
 * - Records both collapsed and expanded HTML states
 * - Restores original toggle state after capture
 * - Handles nested toggles gracefully
 * - Skips destructive/dangerous interactive elements
 * 
 * @example
 * const capture = new ToggleStateCapture(logger);
 * const result = await capture.captureAllToggleStates(page);
 * console.log(`Captured ${result.capturedCount} toggles`);
 */
class ToggleStateCapture {
    /**
     * Create a new ToggleStateCapture instance.
     * @param {Logger} logger - Logger instance for progress tracking
     * @param {Object} [options={}] - Configuration options
     * @param {number} [options.animationWait=300] - Wait time for toggle animations (ms)
     * @param {number} [options.maxToggles=100] - Maximum toggles to capture (prevents runaway)
     * @param {Array<string>} [options.skipPatterns=[]] - Text patterns to skip (destructive actions)
     */
    constructor(logger, options = {}) {
        this.logger = logger;
        this.options = {
            animationWait: options.animationWait || 300,
            maxToggles: options.maxToggles || 100,
            skipPatterns: options.skipPatterns || ['delete', 'remove', 'share', 'export', 'duplicate']
        };
        
        /** @type {Map<string, ToggleState>} */
        this.capturedToggles = new Map();
    }

    /**
     * Capture all toggle states on a page.
     * @async
     * @param {import('puppeteer').Page} page - Puppeteer page instance
     * @returns {Promise<CaptureResult>} Capture result with toggle states
     */
    async captureAllToggleStates(page) {
        this.capturedToggles.clear();
        const errors = [];
        let failedCount = 0;

        // Create HtmlFacade for page operations
        const facade = HtmlFacadeFactory.forPage(page);

        // Find all toggle elements
        const toggles = await this._findToggles(facade);
        this.logger.info('TOGGLE-CAPTURE', `Found ${toggles.length} toggle elements`);

        // Limit to max toggles
        const togglesToProcess = toggles.slice(0, this.options.maxToggles);
        if (toggles.length > this.options.maxToggles) {
            this.logger.warn('TOGGLE-CAPTURE', 
                `Limiting capture to ${this.options.maxToggles} toggles (found ${toggles.length})`);
        }

        for (const toggle of togglesToProcess) {
            try {
                // Check if toggle should be skipped
                if (await this._shouldSkipToggle(facade, toggle)) {
                    this.logger.debug('TOGGLE-CAPTURE', 'Skipping potentially destructive toggle');
                    continue;
                }

                const state = await this._captureToggleState(facade, page, toggle);
                if (state) {
                    this.capturedToggles.set(state.toggleId, state);
                }
            } catch (error) {
                failedCount++;
                errors.push(error.message);
                this.logger.debug('TOGGLE-CAPTURE', `Failed to capture toggle: ${error.message}`);
            }
        }

        this.logger.success('TOGGLE-CAPTURE', 
            `Captured ${this.capturedToggles.size} toggles (${failedCount} failed)`);

        return {
            toggleStates: this.capturedToggles,
            capturedCount: this.capturedToggles.size,
            failedCount,
            errors
        };
    }

    /**
     * Find all toggle elements on the page.
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @returns {Promise<Array<Object>>} Array of toggle element wrappers
     */
    async _findToggles(facade) {
        // Primary selector: Notion toggle elements with aria-expanded
        const toggleSelector = '[role="button"][aria-expanded]';
        return await facade.query(toggleSelector);
    }

    /**
     * Check if a toggle should be skipped (potentially destructive).
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {Object} toggle - Toggle element wrapper
     * @returns {Promise<boolean>} True if toggle should be skipped
     */
    async _shouldSkipToggle(facade, toggle) {
        try {
            const textContent = await facade.getTextContent(toggle);
            const text = (textContent || '').toLowerCase();
            
            // Skip if text contains any dangerous patterns
            return this.options.skipPatterns.some(pattern => text.includes(pattern));
        } catch {
            return false;
        }
    }

    /**
     * Capture collapsed and expanded states of a single toggle.
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {import('puppeteer').Page} page - Puppeteer page (for waitForTimeout)
     * @param {Object} toggleElement - Toggle element wrapper
     * @returns {Promise<ToggleState|null>} Toggle state or null if capture failed
     */
    async _captureToggleState(facade, page, toggleElement) {
        // Generate unique toggle ID
        const toggleId = await this._getToggleId(facade, page, toggleElement);
        
        // Get current state (expanded or collapsed)
        const isExpanded = await this._isToggleExpanded(facade, toggleElement);
        
        // Capture current state HTML (content area)
        const currentHtml = await this._getToggleContentHtml(facade, page, toggleElement);
        
        // Click to toggle state
        await toggleElement.click();
        await this._waitForAnimation(page);
        
        // Capture opposite state HTML
        const oppositeHtml = await this._getToggleContentHtml(facade, page, toggleElement);
        
        // Restore original state (click again)
        await toggleElement.click();
        await this._waitForAnimation(page);
        
        // Build trigger selector
        const triggerSelector = `[data-block-id="${toggleId}"] [role="button"][aria-expanded]`;
        
        return {
            toggleId,
            collapsedHtml: isExpanded ? oppositeHtml : currentHtml,
            expandedHtml: isExpanded ? currentHtml : oppositeHtml,
            triggerSelector,
            initiallyExpanded: isExpanded
        };
    }

    /**
     * Get or generate a unique ID for a toggle element.
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {import('puppeteer').Page} page - Puppeteer page
     * @param {Object} toggleElement - Toggle element wrapper
     * @returns {Promise<string>} Unique toggle ID
     */
    async _getToggleId(facade, page, toggleElement) {
        // Try to find existing block ID from parent
        const blockId = await page.evaluate(el => {
            const block = el.closest('[data-block-id]');
            return block ? block.getAttribute('data-block-id') : null;
        }, toggleElement.handle);
        
        if (blockId) {
            return blockId;
        }
        
        // Generate a unique ID
        return `toggle-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Check if a toggle is currently expanded.
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {Object} toggleElement - Toggle element wrapper
     * @returns {Promise<boolean>} True if expanded
     */
    async _isToggleExpanded(facade, toggleElement) {
        const ariaExpanded = await facade.getAttribute(toggleElement, 'aria-expanded');
        return ariaExpanded === 'true';
    }

    /**
     * Get HTML content associated with a toggle (the hidden content area).
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {import('puppeteer').Page} page - Puppeteer page
     * @param {Object} toggleElement - Toggle element wrapper
     * @returns {Promise<string>} Toggle content HTML
     */
    async _getToggleContentHtml(facade, page, toggleElement) {
        return await page.evaluate(el => {
            // Navigate from toggle button to content container
            const block = el.closest('[data-block-id]');
            if (!block) return '';
            
            // Notion toggle content is typically in:
            // 1. A sibling div with notion-toggle-content class
            // 2. A child container that's hidden/shown
            // 3. The last child element of the block
            const contentArea = 
                block.querySelector('.notion-toggle-content') ||
                block.querySelector('[style*="display: none"]') ||
                block.querySelector('[style*="display: block"]') ||
                block.querySelector('[data-content-editable-leaf]')?.parentElement ||
                block.lastElementChild;
            
            if (!contentArea || contentArea === el) {
                // If content area is the button itself, return empty
                return '';
            }
            
            return contentArea.outerHTML;
        }, toggleElement.handle);
    }

    /**
     * Wait for toggle animation to complete.
     * @async
     * @private
     * @param {import('puppeteer').Page} page - Puppeteer page
     */
    async _waitForAnimation(page) {
        await new Promise(resolve => setTimeout(resolve, this.options.animationWait));
    }

    /**
     * Get the captured toggle states as a plain object (for serialization).
     * @returns {Object<string, ToggleState>} Plain object of toggle states
     */
    getToggleStatesObject() {
        return Object.fromEntries(this.capturedToggles);
    }

    /**
     * Clear captured states.
     */
    clear() {
        this.capturedToggles.clear();
    }
}

module.exports = ToggleStateCapture;
