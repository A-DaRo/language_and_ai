/**
 * @file PuppeteerHtmlElement.js
 * @description Wrapper class for Puppeteer ElementHandle.
 * 
 * Provides a consistent element interface for browser-context DOM operations.
 * Wraps Puppeteer's ElementHandle to expose a uniform API that matches
 * the JsdomHtmlElement interface.
 * 
 * @module html/elements/PuppeteerHtmlElement
 * @see {@link ../PuppeteerHtmlFacade.js}
 */

'use strict';

/**
 * @class PuppeteerHtmlElement
 * @description Wrapper for Puppeteer ElementHandle with unified element interface.
 * 
 * This wrapper allows consistent element handling across facade implementations.
 * All methods are async to match Puppeteer's async nature.
 * 
 * @example
 * const element = new PuppeteerHtmlElement(elementHandle);
 * await element.click();
 * await element.focus();
 */
class PuppeteerHtmlElement {
    /**
     * Create a new PuppeteerHtmlElement wrapper.
     * @param {import('puppeteer').ElementHandle} handle - Puppeteer ElementHandle
     */
    constructor(handle) {
        /**
         * @type {import('puppeteer').ElementHandle}
         * @description The underlying Puppeteer ElementHandle
         */
        this.handle = handle;
    }

    /**
     * Click the element.
     * @async
     * @param {Object} [options={}] - Click options
     * @param {number} [options.delay=0] - Time to wait between mousedown and mouseup in ms
     * @param {string} [options.button='left'] - Mouse button ('left', 'right', 'middle')
     * @param {number} [options.clickCount=1] - Number of clicks
     * @returns {Promise<void>}
     */
    async click(options = {}) {
        await this.handle.click(options);
    }

    /**
     * Focus the element.
     * @async
     * @returns {Promise<void>}
     */
    async focus() {
        await this.handle.focus();
    }

    /**
     * Type text into the element (for input/textarea elements).
     * @async
     * @param {string} text - Text to type
     * @param {Object} [options={}] - Type options
     * @param {number} [options.delay=0] - Time to wait between key presses in ms
     * @returns {Promise<void>}
     */
    async type(text, options = {}) {
        await this.handle.type(text, options);
    }

    /**
     * Hover over the element.
     * @async
     * @returns {Promise<void>}
     */
    async hover() {
        await this.handle.hover();
    }

    /**
     * Take a screenshot of the element.
     * @async
     * @param {Object} [options={}] - Screenshot options
     * @returns {Promise<Buffer|string>} Screenshot data
     */
    async screenshot(options = {}) {
        return await this.handle.screenshot(options);
    }

    /**
     * Press a key while focused on this element.
     * @async
     * @param {string} key - Key to press (e.g., 'Enter', 'Tab')
     * @param {Object} [options={}] - Press options
     * @returns {Promise<void>}
     */
    async press(key, options = {}) {
        await this.handle.press(key, options);
    }

    /**
     * Select option(s) in a select element.
     * @async
     * @param {...string} values - Values to select
     * @returns {Promise<string[]>} Array of selected values
     */
    async select(...values) {
        return await this.handle.select(...values);
    }

    /**
     * Scroll element into view.
     * @async
     * @returns {Promise<void>}
     */
    async scrollIntoView() {
        await this.handle.scrollIntoView();
    }

    /**
     * Get the bounding box of the element.
     * @async
     * @returns {Promise<Object|null>} Bounding box with x, y, width, height or null
     */
    async boundingBox() {
        return await this.handle.boundingBox();
    }

    /**
     * Check if element is visible.
     * @async
     * @returns {Promise<boolean>} True if visible
     */
    async isVisible() {
        return await this.handle.isVisible();
    }

    /**
     * Check if element is hidden.
     * @async
     * @returns {Promise<boolean>} True if hidden
     */
    async isHidden() {
        return await this.handle.isHidden();
    }

    /**
     * Dispose of the element handle.
     * @async
     * @returns {Promise<void>}
     */
    async dispose() {
        await this.handle.dispose();
    }
}

module.exports = { PuppeteerHtmlElement };
