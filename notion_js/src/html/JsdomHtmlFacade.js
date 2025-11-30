/**
 * @file JsdomHtmlFacade.js
 * @description Server-context HtmlFacade implementation using JSDOM.
 * 
 * Provides DOM manipulation capabilities for server-side HTML processing.
 * Uses JSDOM to parse and manipulate HTML without a browser.
 * 
 * Use this implementation when:
 * - Processing saved HTML files after scraping
 * - Performing post-processing transformations
 * - Working with static HTML content
 * - Link rewriting, asset path updates
 * 
 * @module html/JsdomHtmlFacade
 * @see {@link ./HtmlFacade.js}
 * @see {@link ../Docs/29112025_Codebase_Review.md} Section 6.6
 */

'use strict';

const { HtmlFacade } = require('./HtmlFacade');
const { JsdomHtmlElement } = require('./elements/JsdomHtmlElement');

/**
 * @class JsdomHtmlFacade
 * @extends HtmlFacade
 * @description Server-context implementation of HtmlFacade using JSDOM.
 * 
 * Executes DOM operations synchronously using JSDOM. Suitable for
 * post-processing saved HTML files where JavaScript execution is
 * not needed.
 * 
 * @example
 * // Create from HTML string
 * const facade = JsdomHtmlFacade.fromHtml('<html><body>...</body></html>');
 * 
 * // Or from file
 * const facade = await JsdomHtmlFacade.fromFile('./page.html');
 * 
 * // Manipulate and save
 * const links = await facade.query('a[href]');
 * for (const link of links) {
 *     await facade.setAttribute(link, 'href', './local-path.html');
 * }
 * await facade.saveToFile('./modified.html');
 */
class JsdomHtmlFacade extends HtmlFacade {
    /**
     * Create a new JsdomHtmlFacade.
     * @param {import('jsdom').JSDOM} dom - JSDOM instance
     */
    constructor(dom) {
        super();
        
        /**
         * @type {import('jsdom').JSDOM}
         * @description The JSDOM instance
         */
        this.dom = dom;
        
        /**
         * @type {Document}
         * @description The JSDOM document for convenience
         */
        this.document = dom.window.document;
    }

    /**
     * Create facade from HTML string.
     * @static
     * @param {string} html - HTML content
     * @returns {JsdomHtmlFacade} New facade instance
     */
    static fromHtml(html) {
        const { JSDOM } = require('jsdom');
        const dom = new JSDOM(html);
        return new JsdomHtmlFacade(dom);
    }

    /**
     * Create facade from file path.
     * @static
     * @async
     * @param {string} filePath - Path to HTML file
     * @returns {Promise<JsdomHtmlFacade>} New facade instance
     */
    static async fromFile(filePath) {
        const fs = require('fs/promises');
        const html = await fs.readFile(filePath, 'utf-8');
        return JsdomHtmlFacade.fromHtml(html);
    }

    /**
     * Get the execution context.
     * @returns {string} Always returns Context.SERVER
     */
    getContext() {
        return HtmlFacade.Context.SERVER;
    }

    /**
     * Query all elements matching a CSS selector.
     * @async
     * @param {string} selector - CSS selector string
     * @returns {Promise<JsdomHtmlElement[]>} Array of wrapped elements
     */
    async query(selector) {
        const elements = this.document.querySelectorAll(selector);
        return Array.from(elements).map(el => new JsdomHtmlElement(el));
    }

    /**
     * Query first element matching a CSS selector.
     * @async
     * @param {string} selector - CSS selector string
     * @returns {Promise<JsdomHtmlElement|null>} Wrapped element or null
     */
    async queryOne(selector) {
        const element = this.document.querySelector(selector);
        return element ? new JsdomHtmlElement(element) : null;
    }

    /**
     * Get an attribute value from an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @param {string} name - Attribute name
     * @returns {Promise<string|null>} Attribute value or null
     */
    async getAttribute(element, name) {
        return element.element.getAttribute(name);
    }

    /**
     * Set an attribute value on an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @param {string} name - Attribute name
     * @param {string} value - Attribute value
     * @returns {Promise<void>}
     */
    async setAttribute(element, name, value) {
        element.element.setAttribute(name, value);
    }

    /**
     * Get the inner HTML of an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @returns {Promise<string>} Inner HTML content
     */
    async getInnerHtml(element) {
        return element.element.innerHTML;
    }

    /**
     * Set the inner HTML of an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @param {string} html - HTML content to set
     * @returns {Promise<void>}
     */
    async setInnerHtml(element, html) {
        element.element.innerHTML = html;
    }

    /**
     * Create a new element with the specified tag name.
     * @async
     * @param {string} tagName - HTML tag name
     * @returns {Promise<JsdomHtmlElement>} Wrapped newly created element
     */
    async createElement(tagName) {
        const element = this.document.createElement(tagName);
        return new JsdomHtmlElement(element);
    }

    /**
     * Append a child element to a parent element.
     * @async
     * @param {JsdomHtmlElement} parent - Wrapped parent element
     * @param {JsdomHtmlElement} child - Wrapped child element
     * @returns {Promise<void>}
     */
    async appendChild(parent, child) {
        parent.element.appendChild(child.element);
    }

    /**
     * Insert an element before a reference element.
     * @async
     * @param {JsdomHtmlElement} newElement - Element to insert
     * @param {JsdomHtmlElement} reference - Reference element
     * @returns {Promise<void>}
     */
    async insertBefore(newElement, reference) {
        reference.element.parentNode.insertBefore(
            newElement.element,
            reference.element
        );
    }

    /**
     * Serialize the document to an HTML string.
     * @async
     * @returns {Promise<string>} Serialized HTML content
     */
    async serialize() {
        return this.dom.serialize();
    }

    /**
     * Inject a script into the document.
     * @async
     * @param {string} scriptContent - JavaScript code to inject
     * @param {Object} [options={}] - Injection options
     * @param {string} [options.placement='body-end'] - Where to inject
     * @param {string|null} [options.id=null] - Script element ID
     * @returns {Promise<void>}
     */
    async injectScript(scriptContent, options = {}) {
        const { placement = 'body-end', id = null } = options;
        
        const script = this.document.createElement('script');
        script.textContent = scriptContent;
        if (id) script.id = id;
        
        const target = placement === 'head'
            ? this.document.head
            : this.document.body;
        target.appendChild(script);
    }

    /**
     * Inject a style block into the document head.
     * @async
     * @param {string} cssContent - CSS rules to inject
     * @param {Object} [options={}] - Injection options
     * @param {string|null} [options.id=null] - Style element ID
     * @returns {Promise<void>}
     */
    async injectStyle(cssContent, options = {}) {
        const { id = null } = options;
        
        const style = this.document.createElement('style');
        style.textContent = cssContent;
        if (id) style.id = id;
        this.document.head.appendChild(style);
    }

    /**
     * Get the text content of an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @returns {Promise<string>} Text content
     */
    async getTextContent(element) {
        return element.element.textContent;
    }

    /**
     * Set the text content of an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @param {string} text - Text content to set
     * @returns {Promise<void>}
     */
    async setTextContent(element, text) {
        element.element.textContent = text;
    }

    /**
     * Remove an element from the document.
     * @async
     * @param {JsdomHtmlElement} element - Element to remove
     * @returns {Promise<void>}
     */
    async removeElement(element) {
        element.element.remove();
    }

    /**
     * Check if an element has a specific class.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @param {string} className - Class name to check
     * @returns {Promise<boolean>} True if element has the class
     */
    async hasClass(element, className) {
        return element.element.classList.contains(className);
    }

    /**
     * Add a class to an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @param {string} className - Class name to add
     * @returns {Promise<void>}
     */
    async addClass(element, className) {
        element.element.classList.add(className);
    }

    /**
     * Remove a class from an element.
     * @async
     * @param {JsdomHtmlElement} element - Wrapped element
     * @param {string} className - Class name to remove
     * @returns {Promise<void>}
     */
    async removeClass(element, className) {
        element.element.classList.remove(className);
    }

    /**
     * Save serialized HTML to file.
     * @async
     * @param {string} filePath - Output file path
     * @returns {Promise<void>}
     */
    async saveToFile(filePath) {
        const fs = require('fs/promises');
        const html = await this.serialize();
        await fs.writeFile(filePath, html, 'utf-8');
    }

    /**
     * Get the underlying JSDOM instance.
     * @returns {import('jsdom').JSDOM} The JSDOM instance
     */
    getDom() {
        return this.dom;
    }

    /**
     * Get the underlying Document.
     * @returns {Document} The JSDOM document
     */
    getDocument() {
        return this.document;
    }

    /**
     * Get the body element.
     * @async
     * @returns {Promise<JsdomHtmlElement>} Wrapped body element
     */
    async getBody() {
        return new JsdomHtmlElement(this.document.body);
    }

    /**
     * Get the head element.
     * @async
     * @returns {Promise<JsdomHtmlElement>} Wrapped head element
     */
    async getHead() {
        return new JsdomHtmlElement(this.document.head);
    }

    /**
     * Get the document element (html).
     * @async
     * @returns {Promise<JsdomHtmlElement>} Wrapped html element
     */
    async getDocumentElement() {
        return new JsdomHtmlElement(this.document.documentElement);
    }

    /**
     * Get or set the document title.
     * @async
     * @param {string} [newTitle] - New title (optional)
     * @returns {Promise<string>} Current or new title
     */
    async title(newTitle) {
        if (newTitle !== undefined) {
            this.document.title = newTitle;
        }
        return this.document.title;
    }
}

module.exports = { JsdomHtmlFacade };
