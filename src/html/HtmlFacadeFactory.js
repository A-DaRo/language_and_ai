/**
 * @file HtmlFacadeFactory.js
 * @description Factory for creating appropriate HtmlFacade instances.
 * 
 * Provides static factory methods for creating HtmlFacade instances
 * based on the available context (Puppeteer page, HTML string, or file).
 * 
 * This factory simplifies facade creation by hiding implementation details
 * and selecting the appropriate concrete class automatically.
 * 
 * @module html/HtmlFacadeFactory
 * @see {@link ./HtmlFacade.js}
 * @see {@link ../Docs/29112025_Codebase_Review.md} Section 6.7
 */

'use strict';

const { PuppeteerHtmlFacade } = require('./PuppeteerHtmlFacade');
const { JsdomHtmlFacade } = require('./JsdomHtmlFacade');
const { HtmlFacade, Context } = require('./HtmlFacade');

/**
 * @class HtmlFacadeFactory
 * @description Factory for creating HtmlFacade instances based on context.
 * 
 * Use this factory to create the appropriate facade implementation without
 * needing to know which concrete class to instantiate.
 * 
 * @example
 * // For browser context (Puppeteer)
 * const browserFacade = HtmlFacadeFactory.forPage(page);
 * 
 * // For server context (JSDOM)
 * const serverFacade = HtmlFacadeFactory.fromHtml('<html>...</html>');
 * const fileFacade = await HtmlFacadeFactory.fromFile('./page.html');
 */
class HtmlFacadeFactory {
    /**
     * Create facade for a Puppeteer page.
     * @static
     * @param {import('puppeteer').Page} page - Puppeteer Page instance
     * @returns {PuppeteerHtmlFacade} Browser-context facade
     * @throws {Error} If page is null or undefined
     */
    static forPage(page) {
        if (!page) {
            throw new Error('HtmlFacadeFactory.forPage requires a valid Puppeteer page');
        }
        return new PuppeteerHtmlFacade(page);
    }

    /**
     * Create facade from HTML string.
     * @static
     * @param {string} html - HTML content
     * @returns {JsdomHtmlFacade} Server-context facade
     * @throws {Error} If html is empty or not a string
     */
    static fromHtml(html) {
        if (!html || typeof html !== 'string') {
            throw new Error('HtmlFacadeFactory.fromHtml requires a non-empty HTML string');
        }
        return JsdomHtmlFacade.fromHtml(html);
    }

    /**
     * Create facade from file.
     * @static
     * @async
     * @param {string} filePath - Path to HTML file
     * @returns {Promise<JsdomHtmlFacade>} Server-context facade
     * @throws {Error} If file cannot be read
     */
    static async fromFile(filePath) {
        if (!filePath || typeof filePath !== 'string') {
            throw new Error('HtmlFacadeFactory.fromFile requires a valid file path');
        }
        return await JsdomHtmlFacade.fromFile(filePath);
    }

    /**
     * Create an empty JSDOM facade for building HTML from scratch.
     * @static
     * @param {Object} [options={}] - Creation options
     * @param {string} [options.title=''] - Document title
     * @param {string} [options.lang='en'] - Document language
     * @returns {JsdomHtmlFacade} Empty server-context facade
     */
    static createEmpty(options = {}) {
        const { title = '', lang = 'en' } = options;
        const html = `<!DOCTYPE html><html lang="${lang}"><head><meta charset="utf-8"><title>${title}</title></head><body></body></html>`;
        return JsdomHtmlFacade.fromHtml(html);
    }

    /**
     * Check if a value is a Puppeteer page.
     * @static
     * @param {*} value - Value to check
     * @returns {boolean} True if value appears to be a Puppeteer page
     */
    static isPuppeteerPage(value) {
        return value && 
               typeof value === 'object' && 
               typeof value.evaluate === 'function' &&
               typeof value.$ === 'function' &&
               typeof value.$$ === 'function';
    }

    /**
     * Auto-detect context and create appropriate facade.
     * @static
     * @async
     * @param {import('puppeteer').Page|string} source - Puppeteer page, HTML string, or file path
     * @returns {Promise<HtmlFacade>} Appropriate facade instance
     * @throws {Error} If source type cannot be determined
     */
    static async create(source) {
        // Check if it's a Puppeteer page
        if (HtmlFacadeFactory.isPuppeteerPage(source)) {
            return HtmlFacadeFactory.forPage(source);
        }
        
        // Check if it's a string
        if (typeof source === 'string') {
            // Check if it looks like HTML (starts with < or DOCTYPE)
            const trimmed = source.trim();
            if (trimmed.startsWith('<') || trimmed.toLowerCase().startsWith('<!doctype')) {
                return HtmlFacadeFactory.fromHtml(source);
            }
            
            // Assume it's a file path
            return await HtmlFacadeFactory.fromFile(source);
        }
        
        throw new Error('HtmlFacadeFactory.create: unable to determine source type');
    }

    /**
     * Get the Context enum for checking facade context.
     * @static
     * @type {Object}
     */
    static get Context() {
        return Context;
    }

    /**
     * Check if a facade is in browser context.
     * @static
     * @param {HtmlFacade} facade - Facade to check
     * @returns {boolean} True if browser context
     */
    static isBrowserContext(facade) {
        return facade.getContext() === Context.BROWSER;
    }

    /**
     * Check if a facade is in server context.
     * @static
     * @param {HtmlFacade} facade - Facade to check
     * @returns {boolean} True if server context
     */
    static isServerContext(facade) {
        return facade.getContext() === Context.SERVER;
    }
}

module.exports = { HtmlFacadeFactory };
