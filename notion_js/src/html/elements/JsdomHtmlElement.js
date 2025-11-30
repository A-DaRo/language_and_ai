/**
 * @file JsdomHtmlElement.js
 * @description Wrapper class for JSDOM Element.
 * 
 * Provides a consistent element interface for server-context DOM operations.
 * Wraps JSDOM's Element to expose a uniform API that matches the
 * PuppeteerHtmlElement interface.
 * 
 * Note: Browser interaction methods (click, focus, type) have limited
 * functionality in JSDOM as there's no real browser rendering engine.
 * 
 * @module html/elements/JsdomHtmlElement
 * @see {@link ../JsdomHtmlFacade.js}
 */

'use strict';

/**
 * @class JsdomHtmlElement
 * @description Wrapper for JSDOM Element with unified element interface.
 * 
 * This wrapper allows consistent element handling across facade implementations.
 * Methods are async to match the PuppeteerHtmlElement interface, even though
 * JSDOM operations are synchronous.
 * 
 * @example
 * const element = new JsdomHtmlElement(domElement);
 * await element.click(); // Dispatches click event
 */
class JsdomHtmlElement {
    /**
     * Create a new JsdomHtmlElement wrapper.
     * @param {Element} element - JSDOM Element
     */
    constructor(element) {
        /**
         * @type {Element}
         * @description The underlying JSDOM Element
         */
        this.element = element;
    }

    /**
     * Simulate a click on the element.
     * Dispatches a MouseEvent for event listeners, but does not
     * perform actual browser navigation or interaction.
     * @async
     * @returns {Promise<void>}
     */
    async click() {
        const window = this.element.ownerDocument.defaultView;
        if (window && window.MouseEvent) {
            const event = new window.MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window
            });
            this.element.dispatchEvent(event);
        }
    }

    /**
     * Simulate focusing the element.
     * Dispatches a focus event, but JSDOM has limited focus support.
     * @async
     * @returns {Promise<void>}
     */
    async focus() {
        if (typeof this.element.focus === 'function') {
            this.element.focus();
        }
        
        const window = this.element.ownerDocument.defaultView;
        if (window && window.FocusEvent) {
            const event = new window.FocusEvent('focus', {
                bubbles: false,
                cancelable: false,
                view: window
            });
            this.element.dispatchEvent(event);
        }
    }

    /**
     * Simulate typing text into the element.
     * Sets the value property for input/textarea elements.
     * Note: Does not simulate individual keystrokes like Puppeteer.
     * @async
     * @param {string} text - Text to type
     * @returns {Promise<void>}
     */
    async type(text) {
        if ('value' in this.element) {
            this.element.value = text;
            
            // Dispatch input event
            const window = this.element.ownerDocument.defaultView;
            if (window && window.InputEvent) {
                const event = new window.InputEvent('input', {
                    bubbles: true,
                    cancelable: true,
                    data: text
                });
                this.element.dispatchEvent(event);
            }
        }
    }

    /**
     * Simulate hovering over the element.
     * Dispatches mouseenter event.
     * @async
     * @returns {Promise<void>}
     */
    async hover() {
        const window = this.element.ownerDocument.defaultView;
        if (window && window.MouseEvent) {
            const event = new window.MouseEvent('mouseenter', {
                bubbles: false,
                cancelable: false,
                view: window
            });
            this.element.dispatchEvent(event);
        }
    }

    /**
     * Get the bounding client rect.
     * Note: JSDOM does not perform layout, so this returns a default rect.
     * @async
     * @returns {Promise<Object>} Bounding box with zeros (no layout in JSDOM)
     */
    async boundingBox() {
        // JSDOM doesn't perform layout, return null-like box
        if (typeof this.element.getBoundingClientRect === 'function') {
            const rect = this.element.getBoundingClientRect();
            return {
                x: rect.x || 0,
                y: rect.y || 0,
                width: rect.width || 0,
                height: rect.height || 0
            };
        }
        return { x: 0, y: 0, width: 0, height: 0 };
    }

    /**
     * Check if element is visible.
     * Note: JSDOM has limited visibility support (no CSS/layout).
     * @async
     * @returns {Promise<boolean>} True if element exists and isn't hidden
     */
    async isVisible() {
        // Basic check - JSDOM doesn't compute styles
        const style = this.element.style;
        if (style.display === 'none' || style.visibility === 'hidden') {
            return false;
        }
        return this.element.offsetParent !== undefined;
    }

    /**
     * Check if element is hidden.
     * @async
     * @returns {Promise<boolean>} True if hidden
     */
    async isHidden() {
        return !(await this.isVisible());
    }

    /**
     * Scroll element into view (no-op in JSDOM).
     * @async
     * @returns {Promise<void>}
     */
    async scrollIntoView() {
        // No-op: JSDOM doesn't have viewport/scrolling
        if (typeof this.element.scrollIntoView === 'function') {
            this.element.scrollIntoView();
        }
    }

    /**
     * Simulate pressing a key.
     * Dispatches keydown and keyup events.
     * @async
     * @param {string} key - Key to press
     * @returns {Promise<void>}
     */
    async press(key) {
        const window = this.element.ownerDocument.defaultView;
        if (window && window.KeyboardEvent) {
            const keydownEvent = new window.KeyboardEvent('keydown', {
                key,
                bubbles: true,
                cancelable: true
            });
            const keyupEvent = new window.KeyboardEvent('keyup', {
                key,
                bubbles: true,
                cancelable: true
            });
            this.element.dispatchEvent(keydownEvent);
            this.element.dispatchEvent(keyupEvent);
        }
    }

    /**
     * Select option(s) in a select element.
     * @async
     * @param {...string} values - Values to select
     * @returns {Promise<string[]>} Array of selected values
     */
    async select(...values) {
        if (this.element.tagName === 'SELECT') {
            const selectedValues = [];
            for (const option of this.element.options) {
                if (values.includes(option.value)) {
                    option.selected = true;
                    selectedValues.push(option.value);
                } else if (!this.element.multiple) {
                    option.selected = false;
                }
            }
            
            // Dispatch change event
            const window = this.element.ownerDocument.defaultView;
            if (window && window.Event) {
                const event = new window.Event('change', {
                    bubbles: true,
                    cancelable: true
                });
                this.element.dispatchEvent(event);
            }
            
            return selectedValues;
        }
        return [];
    }

    /**
     * Dispose of the element reference (no-op in JSDOM).
     * @async
     * @returns {Promise<void>}
     */
    async dispose() {
        // No disposal needed for JSDOM elements
        // Could set this.element = null if memory cleanup is needed
    }
}

module.exports = { JsdomHtmlElement };
