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
 * 1. Find all toggle BLOCKS on page (.notion-toggle-block elements)
 * 2. For each block, locate the clickable button ([role="button"][aria-expanded])
 * 3. Record collapsed state HTML (content div, if any)
 * 4. Click to expand, wait for React to insert content, record expanded state
 * 5. Click to restore original collapsed state
 * 6. Return state map for injection by OfflineToggleController
 * 
 * @design NOTION DOM STRUCTURE
 * Notion toggle blocks have this structure:
 * ```html
 * <div data-block-id="..." class="notion-selectable notion-toggle-block">
 *   <!-- Header row with arrow + title -->
 *   <div style="display: flex">
 *     <div class="notion-list-item-box-left">
 *       <div role="button" aria-expanded="false|true">...</div>
 *     </div>
 *     <div>Toggle Title</div>
 *   </div>
 *   <!-- Content: Dynamically inserted when expanded -->
 *   <div style="padding-left: 26px">
 *     <!-- Child blocks appear here -->
 *   </div>
 * </div>
 * ```
 * 
 * Key insight: Content is a CHILD div with padding-left, NOT a sibling.
 * Content is conditionally rendered by React (not just hidden with CSS).
 * 
 * @fixes Issue #3 - Toggle content empty capture
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
     * @param {number} [options.contentWait=2000] - Max wait time for content to appear (ms)
     * @param {number} [options.maxToggles=100] - Maximum toggles to capture (prevents runaway)
     * @param {Array<string>} [options.skipPatterns=[]] - Text patterns to skip (destructive actions)
     */
    constructor(logger, options = {}) {
        this.logger = logger;
        this.options = {
            animationWait: options.animationWait || 300,
            contentWait: options.contentWait || 2000,
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
     * @returns {Promise<Array<Object>>} Array of toggle info objects
     * 
     * @description Targets .notion-toggle-block containers (not the button inside).
     * For each block, locates the clickable button for interaction.
     * 
     * @fixes Issue #3 - Correct toggle discovery using .notion-toggle-block
     */
    async _findToggles(facade) {
        // Primary selector: Notion toggle BLOCKS (the container, not the button)
        const toggleBlockSelector = '.notion-toggle-block';
        const blocks = await facade.query(toggleBlockSelector);
        
        // Build toggle info objects with both block and button references
        const toggles = [];
        for (const block of blocks) {
            // Find the clickable button inside each block
            const buttons = await facade.query('[role="button"][aria-expanded]', block);
            if (buttons.length > 0) {
                toggles.push({
                    block: block,           // The container for content extraction
                    button: buttons[0],     // The clickable element
                    blockId: await facade.getAttribute(block, 'data-block-id')
                });
            }
        }
        
        // Fallback: If no .notion-toggle-block found, try the old selector
        // This handles non-standard toggle implementations
        if (toggles.length === 0) {
            const legacyToggles = await facade.query('[role="button"][aria-expanded]');
            for (const toggle of legacyToggles) {
                toggles.push({
                    block: null,            // No block container found
                    button: toggle,         // Use button directly
                    blockId: null           // Will be resolved in _getToggleId
                });
            }
        }
        
        return toggles;
    }

    /**
     * Check if a toggle should be skipped (potentially destructive).
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {Object} toggleInfo - Toggle info object with block and button references
     * @returns {Promise<boolean>} True if toggle should be skipped
     */
    async _shouldSkipToggle(facade, toggleInfo) {
        try {
            // Check the button's text content for dangerous patterns
            const button = toggleInfo.button || toggleInfo;
            const textContent = await facade.getTextContent(button);
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
     * @param {Object} toggleInfo - Toggle info object with block and button references
     * @returns {Promise<ToggleState|null>} Toggle state or null if capture failed
     * 
     * @description Captures both collapsed and expanded states:
     * 1. Get toggle ID from block's data-block-id
     * 2. Check current state (expanded/collapsed)
     * 3. Capture current content HTML
     * 4. Click to toggle state
     * 5. Wait for React to render new content
     * 6. Capture opposite state HTML
     * 7. Restore original state
     * 
     * @fixes Issue #3 - Uses toggleInfo structure with block and button references
     */
    async _captureToggleState(facade, page, toggleInfo) {
        const { block, button, blockId: precomputedBlockId } = toggleInfo;
        
        // Get toggle ID (prefer precomputed, fallback to extraction)
        const toggleId = precomputedBlockId || await this._getToggleId(facade, page, toggleInfo);
        
        // Get current state (expanded or collapsed)
        const isExpanded = await this._isToggleExpanded(facade, button);
        
        // Capture current state HTML (content area)
        const currentHtml = await this._getToggleContentHtml(facade, page, toggleInfo);
        
        // Click to toggle state
        await button.click();
        
        // Wait for content to appear (not just animation)
        await this._waitForContentChange(facade, page, toggleInfo, currentHtml);
        
        // Capture opposite state HTML
        const oppositeHtml = await this._getToggleContentHtml(facade, page, toggleInfo);
        
        // Restore original state (click again)
        await button.click();
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
     * @param {Object} toggleInfo - Toggle info object with block and button
     * @returns {Promise<string>} Unique toggle ID
     */
    async _getToggleId(facade, page, toggleInfo) {
        // If we have a block with data-block-id, use it
        if (toggleInfo.block) {
            const blockId = await facade.getAttribute(toggleInfo.block, 'data-block-id');
            if (blockId) {
                return blockId;
            }
        }
        
        // Fallback: Try to find block ID from button's parent chain
        const button = toggleInfo.button || toggleInfo;
        const blockId = await page.evaluate(el => {
            const block = el.closest('[data-block-id]');
            return block ? block.getAttribute('data-block-id') : null;
        }, button.handle);
        
        if (blockId) {
            return blockId;
        }
        
        // Generate a unique ID as last resort
        return `toggle-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Check if a toggle is currently expanded.
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {Object} buttonElement - Button element wrapper
     * @returns {Promise<boolean>} True if expanded
     */
    async _isToggleExpanded(facade, buttonElement) {
        const ariaExpanded = await facade.getAttribute(buttonElement, 'aria-expanded');
        return ariaExpanded === 'true';
    }

    /**
     * Get HTML content associated with a toggle (the hidden content area).
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {import('puppeteer').Page} page - Puppeteer page
     * @param {Object} toggleInfo - Toggle info object with block and button
     * @returns {Promise<string>} Toggle content HTML
     * 
     * @design NOTION TOGGLE STRUCTURE
     * Notion toggles have content as CHILD div, not sibling:
     * ```html
     * <div class="notion-toggle-block">
     *   <div style="display: flex">Header with arrow + title</div>
     *   <div style="padding-left: 26px">Content (when expanded)</div>
     * </div>
     * ```
     * 
     * Content has:
     * - padding-left for indentation
     * - Child blocks with data-block-id
     * - Only present when expanded (React conditional rendering)
     * 
     * @fixes Issue #3 - Correct content extraction from second child div
     */
    async _getToggleContentHtml(facade, page, toggleInfo) {
        const { block, button } = toggleInfo;
        
        // Use the block if available, otherwise find it from button
        const blockHandle = block ? block.handle : await page.evaluate(el => {
            return el.closest('.notion-toggle-block') || el.closest('[data-block-id]');
        }, button.handle);
        
        if (!blockHandle) {
            // Fallback to legacy extraction
            return this._getLegacyToggleContentHtml(facade, page, button || toggleInfo);
        }
        
        return await page.evaluate(blockEl => {
            if (!blockEl) return '';
            
            // Get all direct children of the toggle block
            const children = Array.from(blockEl.children);
            
            if (children.length < 2) {
                // Only header present - no content (collapsed state)
                return '';
            }
            
            // Strategy 1: Find content div by padding-left (most reliable)
            // Content container has left padding for indentation
            for (let i = 1; i < children.length; i++) {
                const child = children[i];
                const style = window.getComputedStyle(child);
                const paddingLeft = parseFloat(style.paddingLeft) || 0;
                
                // Content typically has 20-30px padding-left
                if (paddingLeft >= 18 && child.offsetHeight > 10) {
                    return child.outerHTML;
                }
            }
            
            // Strategy 2: Second child is content (first is header with flex display)
            for (let i = 0; i < children.length; i++) {
                const child = children[i];
                const style = window.getComputedStyle(child);
                
                // Header has display: flex
                if (style.display === 'flex') {
                    // Next sibling should be content
                    if (i + 1 < children.length) {
                        const nextChild = children[i + 1];
                        if (nextChild.offsetHeight > 10) {
                            return nextChild.outerHTML;
                        }
                    }
                    break;
                }
            }
            
            // Strategy 3: Collect all children except the first (header)
            // This handles cases where content spans multiple divs
            if (children.length > 1) {
                const contentChildren = children.slice(1).filter(c => {
                    // Skip empty or tiny elements
                    return c.offsetHeight > 5 || c.innerHTML.trim().length > 0;
                });
                if (contentChildren.length > 0) {
                    return contentChildren.map(c => c.outerHTML).join('');
                }
            }
            
            // Strategy 4: Look for nested blocks inside the toggle
            const nestedBlocks = blockEl.querySelectorAll(':scope > div > [data-block-id]');
            if (nestedBlocks.length > 0) {
                // Find the parent container of these nested blocks
                const contentContainer = nestedBlocks[0].parentElement;
                if (contentContainer && contentContainer !== blockEl) {
                    return contentContainer.outerHTML;
                }
            }
            
            return '';
        }, block ? block.handle : blockHandle);
    }
    
    /**
     * Legacy content extraction for non-standard toggle implementations.
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {import('puppeteer').Page} page - Puppeteer page
     * @param {Object} toggleElement - Toggle element wrapper (button)
     * @returns {Promise<string>} Toggle content HTML
     */
    async _getLegacyToggleContentHtml(facade, page, toggleElement) {
        return await page.evaluate(el => {
            // Navigate from toggle button to content container
            const block = el.closest('[data-block-id]');
            if (!block) return '';
            
            const blockId = block.getAttribute('data-block-id');
            
            // Strategy 1: Find sibling content container with matching ID reference
            const siblingContent = document.querySelector(
                `[data-content-for="${blockId}"], [data-toggle-content="${blockId}"]`
            );
            if (siblingContent) {
                return siblingContent.outerHTML;
            }
            
            // Strategy 2: Check aria-controls attribute for controlled element
            const ariaControls = el.getAttribute('aria-controls');
            if (ariaControls) {
                const controlledElement = document.getElementById(ariaControls);
                if (controlledElement) {
                    return controlledElement.outerHTML;
                }
            }
            
            // Strategy 3: Find next sibling with toggle content characteristics
            let sibling = block.nextElementSibling;
            while (sibling) {
                const classes = sibling.className || '';
                if ((classes.includes('toggle-content') || 
                     classes.includes('notion-column-list') ||
                     !sibling.hasAttribute('data-block-id')) &&
                    sibling.children.length > 0) {
                    return sibling.outerHTML;
                }
                if (sibling.hasAttribute('data-block-id')) break;
                sibling = sibling.nextElementSibling;
            }
            
            // Strategy 4: Look for indented child content blocks
            const indentedContent = block.querySelector(
                ':scope > div:not([role="button"]):not(:first-child)'
            );
            if (indentedContent && indentedContent !== el.closest('div')) {
                if (indentedContent.offsetHeight > 30) {
                    return indentedContent.outerHTML;
                }
            }
            
            return '';
        }, toggleElement.handle);
    }

    /**
     * Wait for content to appear after toggle click.
     * Polls for content change with timeout.
     * @async
     * @private
     * @param {HtmlFacade} facade - HtmlFacade instance
     * @param {import('puppeteer').Page} page - Puppeteer page
     * @param {Object} toggleInfo - Toggle info object with block and button
     * @param {string} previousHtml - HTML content before click
     * @returns {Promise<string>} Current content HTML after change (or timeout)
     * 
     * @description Waits for React to re-render the toggle content after clicking.
     * Returns immediately if content changes (appears/disappears).
     * Falls back to animation wait on timeout.
     * 
     * @fixes Issue #3 - Returns the new content HTML, handles empty → content transitions
     */
    async _waitForContentChange(facade, page, toggleInfo, previousHtml) {
        const startTime = Date.now();
        const checkInterval = 100;
        const timeout = this.options.contentWait;
        
        while (Date.now() - startTime < timeout) {
            const currentHtml = await this._getToggleContentHtml(facade, page, toggleInfo);
            
            // Success conditions:
            // 1. Content changed from previous state
            // 2. For expand: content appeared (was empty, now has content)
            // 3. For collapse: content disappeared (had content, now empty)
            if (currentHtml !== previousHtml) {
                const prevLen = (previousHtml || '').length;
                const currLen = (currentHtml || '').length;
                
                // Content appeared/disappeared or changed significantly
                if (prevLen === 0 || currLen === 0 || Math.abs(currLen - prevLen) > 10) {
                    return currentHtml;
                }
            }
            
            // Wait before next check
            await new Promise(resolve => setTimeout(resolve, checkInterval));
        }
        
        // Timeout - log warning and return current state
        this.logger.warn('TOGGLE-CAPTURE', 
            `Content wait timeout (previous: ${(previousHtml || '').length} bytes)`);
        
        return await this._getToggleContentHtml(facade, page, toggleInfo);
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
