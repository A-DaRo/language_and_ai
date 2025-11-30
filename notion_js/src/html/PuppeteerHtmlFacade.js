/**
 * @file PuppeteerHtmlFacade.js
 * @description Browser-context HtmlFacade implementation using Puppeteer.
 * 
 * Provides DOM manipulation capabilities for live browser pages via Puppeteer.
 * All operations execute in the browser context using page.evaluate().
 * 
 * Use this implementation when:
 * - Working with live Puppeteer pages during scraping
 * - Need to interact with JavaScript-rendered content
 * - Require real browser behavior (events, styles, layout)
 * 
 * @module html/PuppeteerHtmlFacade
 * @see {@link ./HtmlFacade.js}
 * @see {@link ../Docs/29112025_Codebase_Review.md} Section 6.5
 */

'use strict';

const { HtmlFacade } = require('./HtmlFacade');
const { PuppeteerHtmlElement } = require('./elements/PuppeteerHtmlElement');

/**
 * @class PuppeteerHtmlFacade
 * @extends HtmlFacade
 * @description Browser-context implementation of HtmlFacade using Puppeteer.
 * 
 * Executes DOM operations in the browser via page.evaluate(). Suitable for
 * manipulating live pages during the scraping process where JavaScript
 * execution and real browser behavior is needed.
 * 
 * @example
 * const facade = new PuppeteerHtmlFacade(page);
 * const links = await facade.query('a[href]');
 * for (const link of links) {
 *     const href = await facade.getAttribute(link, 'href');
 *     console.log('Link:', href);
 * }
 */
class PuppeteerHtmlFacade extends HtmlFacade {
    /**
     * Create a new PuppeteerHtmlFacade.
     * @param {import('puppeteer').Page} page - Puppeteer Page instance
     */
    constructor(page) {
        super();
        
        /**
         * @type {import('puppeteer').Page}
         * @description The Puppeteer page instance
         */
        this.page = page;
    }

    /**
     * Get the execution context.
     * @returns {string} Always returns Context.BROWSER
     */
    getContext() {
        return HtmlFacade.Context.BROWSER;
    }

    /**
     * Query all elements matching a CSS selector.
     * @async
     * @param {string} selector - CSS selector string
     * @returns {Promise<PuppeteerHtmlElement[]>} Array of wrapped elements
     */
    async query(selector) {
        const handles = await this.page.$$(selector);
        return handles.map(h => new PuppeteerHtmlElement(h));
    }

    /**
     * Query first element matching a CSS selector.
     * @async
     * @param {string} selector - CSS selector string
     * @returns {Promise<PuppeteerHtmlElement|null>} Wrapped element or null
     */
    async queryOne(selector) {
        const handle = await this.page.$(selector);
        return handle ? new PuppeteerHtmlElement(handle) : null;
    }

    /**
     * Get an attribute value from an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @param {string} name - Attribute name
     * @returns {Promise<string|null>} Attribute value or null
     */
    async getAttribute(element, name) {
        return await this.page.evaluate(
            (el, attrName) => el.getAttribute(attrName),
            element.handle,
            name
        );
    }

    /**
     * Set an attribute value on an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @param {string} name - Attribute name
     * @param {string} value - Attribute value
     * @returns {Promise<void>}
     */
    async setAttribute(element, name, value) {
        await this.page.evaluate(
            (el, attrName, attrValue) => el.setAttribute(attrName, attrValue),
            element.handle,
            name,
            value
        );
    }

    /**
     * Get the inner HTML of an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @returns {Promise<string>} Inner HTML content
     */
    async getInnerHtml(element) {
        return await this.page.evaluate(
            el => el.innerHTML,
            element.handle
        );
    }

    /**
     * Set the inner HTML of an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @param {string} html - HTML content to set
     * @returns {Promise<void>}
     */
    async setInnerHtml(element, html) {
        await this.page.evaluate(
            (el, htmlContent) => { el.innerHTML = htmlContent; },
            element.handle,
            html
        );
    }

    /**
     * Create a new element with the specified tag name.
     * @async
     * @param {string} tagName - HTML tag name
     * @returns {Promise<PuppeteerHtmlElement>} Wrapped newly created element
     */
    async createElement(tagName) {
        const handle = await this.page.evaluateHandle(
            tag => document.createElement(tag),
            tagName
        );
        return new PuppeteerHtmlElement(handle.asElement());
    }

    /**
     * Append a child element to a parent element.
     * @async
     * @param {PuppeteerHtmlElement} parent - Wrapped parent element
     * @param {PuppeteerHtmlElement} child - Wrapped child element
     * @returns {Promise<void>}
     */
    async appendChild(parent, child) {
        await this.page.evaluate(
            (p, c) => p.appendChild(c),
            parent.handle,
            child.handle
        );
    }

    /**
     * Insert an element before a reference element.
     * @async
     * @param {PuppeteerHtmlElement} newElement - Element to insert
     * @param {PuppeteerHtmlElement} reference - Reference element
     * @returns {Promise<void>}
     */
    async insertBefore(newElement, reference) {
        await this.page.evaluate(
            (newEl, refEl) => refEl.parentNode.insertBefore(newEl, refEl),
            newElement.handle,
            reference.handle
        );
    }

    /**
     * Serialize the document to an HTML string.
     * @async
     * @returns {Promise<string>} Serialized HTML content
     */
    async serialize() {
        return await this.page.content();
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
        
        await this.page.evaluate(
            (content, placement, scriptId) => {
                const script = document.createElement('script');
                script.textContent = content;
                if (scriptId) script.id = scriptId;
                
                const target = placement === 'head'
                    ? document.head
                    : document.body;
                target.appendChild(script);
            },
            scriptContent,
            placement,
            id
        );
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
        
        await this.page.evaluate(
            (content, styleId) => {
                const style = document.createElement('style');
                style.textContent = content;
                if (styleId) style.id = styleId;
                document.head.appendChild(style);
            },
            cssContent,
            id
        );
    }

    /**
     * Get the text content of an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @returns {Promise<string>} Text content
     */
    async getTextContent(element) {
        return await this.page.evaluate(
            el => el.textContent,
            element.handle
        );
    }

    /**
     * Set the text content of an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @param {string} text - Text content to set
     * @returns {Promise<void>}
     */
    async setTextContent(element, text) {
        await this.page.evaluate(
            (el, textContent) => { el.textContent = textContent; },
            element.handle,
            text
        );
    }

    /**
     * Remove an element from the document.
     * @async
     * @param {PuppeteerHtmlElement} element - Element to remove
     * @returns {Promise<void>}
     */
    async removeElement(element) {
        await this.page.evaluate(
            el => el.remove(),
            element.handle
        );
    }

    /**
     * Check if an element has a specific class.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @param {string} className - Class name to check
     * @returns {Promise<boolean>} True if element has the class
     */
    async hasClass(element, className) {
        return await this.page.evaluate(
            (el, cls) => el.classList.contains(cls),
            element.handle,
            className
        );
    }

    /**
     * Add a class to an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @param {string} className - Class name to add
     * @returns {Promise<void>}
     */
    async addClass(element, className) {
        await this.page.evaluate(
            (el, cls) => el.classList.add(cls),
            element.handle,
            className
        );
    }

    /**
     * Remove a class from an element.
     * @async
     * @param {PuppeteerHtmlElement} element - Wrapped element
     * @param {string} className - Class name to remove
     * @returns {Promise<void>}
     */
    async removeClass(element, className) {
        await this.page.evaluate(
            (el, cls) => el.classList.remove(cls),
            element.handle,
            className
        );
    }

    /**
     * Wait for a selector to appear in the document.
     * @async
     * @param {string} selector - CSS selector to wait for
     * @param {Object} [options={}] - Wait options
     * @param {number} [options.timeout=30000] - Timeout in milliseconds
     * @returns {Promise<PuppeteerHtmlElement>} The appeared element
     */
    async waitForSelector(selector, options = {}) {
        const handle = await this.page.waitForSelector(selector, options);
        return handle ? new PuppeteerHtmlElement(handle) : null;
    }

    /**
     * Evaluate JavaScript in the page context.
     * @async
     * @param {Function|string} pageFunction - Function to evaluate
     * @param {...*} args - Arguments to pass to the function
     * @returns {Promise<*>} Result of evaluation
     */
    async evaluate(pageFunction, ...args) {
        return await this.page.evaluate(pageFunction, ...args);
    }

    /**
     * Get the underlying Puppeteer page.
     * @returns {import('puppeteer').Page} The Puppeteer page instance
     */
    getPage() {
        return this.page;
    }
}

module.exports = { PuppeteerHtmlFacade };
