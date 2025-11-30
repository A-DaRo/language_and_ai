/**
 * @file HtmlFacade.js
 * @description Abstract interface for HTML DOM manipulation.
 * 
 * This facade provides a unified interface for HTML manipulation across different
 * execution contexts (browser via Puppeteer, server via JSDOM). It abstracts away
 * the implementation differences, allowing components like LinkRewriter, AssetDownloader,
 * and ToggleController to work uniformly regardless of context.
 * 
 * Design Goals:
 * - Unified Interface: Single API for DOM manipulation
 * - Context Agnostic: Same operations work in browser and server
 * - Chainable: Methods return promises for fluent async patterns
 * - Testable: Easy to mock for unit testing
 * - Extensible: New implementations can support additional contexts
 * 
 * @module html/HtmlFacade
 */

'use strict';

/**
 * @readonly
 * @enum {string}
 * @description Execution context enum for HTML operations.
 */
const Context = Object.freeze({
    /** Browser context - operations execute in Puppeteer page */
    BROWSER: 'browser',
    /** Server context - operations execute in JSDOM */
    SERVER: 'server'
});

/**
 * @abstract
 * @class HtmlFacade
 * @description Abstract base class for HTML DOM manipulation facades.
 * 
 * Provides a unified interface for DOM operations that can be implemented
 * for different execution contexts (Puppeteer for browser, JSDOM for server).
 * 
 * @example
 * // Usage pattern - works with both implementations
 * const links = await facade.query('a[href]');
 * for (const link of links) {
 *     const href = await facade.getAttribute(link, 'href');
 *     if (href.startsWith('/')) {
 *         await facade.setAttribute(link, 'href', './pages' + href);
 *     }
 * }
 */
class HtmlFacade {
    /**
     * @static
     * @type {Object}
     * @description Context enum for identifying execution environment.
     */
    static Context = Context;

    /**
     * Create a new HtmlFacade instance.
     * @throws {Error} If instantiated directly
     */
    constructor() {
        if (new.target === HtmlFacade) {
            throw new Error('HtmlFacade is abstract and cannot be instantiated directly');
        }
    }

    /**
     * Get the execution context of this facade.
     * @abstract
     * @returns {string} The context (Context.BROWSER or Context.SERVER)
     */
    getContext() {
        throw new Error('Abstract method getContext() must be implemented');
    }

    /**
     * Query all elements matching a CSS selector.
     * @abstract
     * @async
     * @param {string} selector - CSS selector string
     * @returns {Promise<Array<Object>>} Array of wrapped elements
     */
    async query(selector) {
        throw new Error('Abstract method query() must be implemented');
    }

    /**
     * Query first element matching a CSS selector.
     * @abstract
     * @async
     * @param {string} selector - CSS selector string
     * @returns {Promise<Object|null>} Wrapped element or null if not found
     */
    async queryOne(selector) {
        throw new Error('Abstract method queryOne() must be implemented');
    }

    /**
     * Get an attribute value from an element.
     * @abstract
     * @async
     * @param {Object} element - Wrapped element
     * @param {string} name - Attribute name
     * @returns {Promise<string|null>} Attribute value or null
     */
    async getAttribute(element, name) {
        throw new Error('Abstract method getAttribute() must be implemented');
    }

    /**
     * Set an attribute value on an element.
     * @abstract
     * @async
     * @param {Object} element - Wrapped element
     * @param {string} name - Attribute name
     * @param {string} value - Attribute value
     * @returns {Promise<void>}
     */
    async setAttribute(element, name, value) {
        throw new Error('Abstract method setAttribute() must be implemented');
    }

    /**
     * Get the inner HTML of an element.
     * @abstract
     * @async
     * @param {Object} element - Wrapped element
     * @returns {Promise<string>} Inner HTML content
     */
    async getInnerHtml(element) {
        throw new Error('Abstract method getInnerHtml() must be implemented');
    }

    /**
     * Set the inner HTML of an element.
     * @abstract
     * @async
     * @param {Object} element - Wrapped element
     * @param {string} html - HTML content to set
     * @returns {Promise<void>}
     */
    async setInnerHtml(element, html) {
        throw new Error('Abstract method setInnerHtml() must be implemented');
    }

    /**
     * Create a new element with the specified tag name.
     * @abstract
     * @async
     * @param {string} tagName - HTML tag name (e.g., 'div', 'script', 'style')
     * @returns {Promise<Object>} Wrapped newly created element
     */
    async createElement(tagName) {
        throw new Error('Abstract method createElement() must be implemented');
    }

    /**
     * Append a child element to a parent element.
     * @abstract
     * @async
     * @param {Object} parent - Wrapped parent element
     * @param {Object} child - Wrapped child element to append
     * @returns {Promise<void>}
     */
    async appendChild(parent, child) {
        throw new Error('Abstract method appendChild() must be implemented');
    }

    /**
     * Insert an element before a reference element.
     * @abstract
     * @async
     * @param {Object} newElement - Wrapped element to insert
     * @param {Object} reference - Wrapped reference element
     * @returns {Promise<void>}
     */
    async insertBefore(newElement, reference) {
        throw new Error('Abstract method insertBefore() must be implemented');
    }

    /**
     * Serialize the document to an HTML string.
     * @abstract
     * @async
     * @returns {Promise<string>} Serialized HTML content
     */
    async serialize() {
        throw new Error('Abstract method serialize() must be implemented');
    }

    /**
     * Inject a script into the document.
     * @abstract
     * @async
     * @param {string} scriptContent - JavaScript code to inject
     * @param {Object} [options={}] - Injection options
     * @param {string} [options.placement='body-end'] - Where to inject ('head', 'body-end')
     * @param {string|null} [options.id=null] - Script element ID
     * @returns {Promise<void>}
     */
    async injectScript(scriptContent, options = {}) {
        throw new Error('Abstract method injectScript() must be implemented');
    }

    /**
     * Inject a style block into the document head.
     * @abstract
     * @async
     * @param {string} cssContent - CSS rules to inject
     * @param {Object} [options={}] - Injection options
     * @param {string|null} [options.id=null] - Style element ID
     * @returns {Promise<void>}
     */
    async injectStyle(cssContent, options = {}) {
        throw new Error('Abstract method injectStyle() must be implemented');
    }

    /**
     * Get the text content of an element.
     * @async
     * @param {Object} element - Wrapped element
     * @returns {Promise<string>} Text content
     */
    async getTextContent(element) {
        throw new Error('Abstract method getTextContent() must be implemented');
    }

    /**
     * Set the text content of an element.
     * @async
     * @param {Object} element - Wrapped element
     * @param {string} text - Text content to set
     * @returns {Promise<void>}
     */
    async setTextContent(element, text) {
        throw new Error('Abstract method setTextContent() must be implemented');
    }

    /**
     * Remove an element from the document.
     * @async
     * @param {Object} element - Wrapped element to remove
     * @returns {Promise<void>}
     */
    async removeElement(element) {
        throw new Error('Abstract method removeElement() must be implemented');
    }

    /**
     * Check if an element has a specific class.
     * @async
     * @param {Object} element - Wrapped element
     * @param {string} className - Class name to check
     * @returns {Promise<boolean>} True if element has the class
     */
    async hasClass(element, className) {
        throw new Error('Abstract method hasClass() must be implemented');
    }

    /**
     * Add a class to an element.
     * @async
     * @param {Object} element - Wrapped element
     * @param {string} className - Class name to add
     * @returns {Promise<void>}
     */
    async addClass(element, className) {
        throw new Error('Abstract method addClass() must be implemented');
    }

    /**
     * Remove a class from an element.
     * @async
     * @param {Object} element - Wrapped element
     * @param {string} className - Class name to remove
     * @returns {Promise<void>}
     */
    async removeClass(element, className) {
        throw new Error('Abstract method removeClass() must be implemented');
    }
}

module.exports = { HtmlFacade, Context };
