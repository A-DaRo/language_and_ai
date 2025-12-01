/**
 * @file index.js
 * @description Module exports for the html package.
 * 
 * This module provides a unified interface for HTML DOM manipulation
 * across different execution contexts (browser via Puppeteer, server via JSDOM).
 * 
 * @module html
 * @see {@link ../Docs/29112025_Codebase_Review.md} Section 6
 * 
 * @example
 * const {
 *     HtmlFacade,
 *     HtmlFacadeFactory,
 *     PuppeteerHtmlFacade,
 *     JsdomHtmlFacade,
 *     Context
 * } = require('./src/html');
 * 
 * // Create facade for Puppeteer page
 * const browserFacade = HtmlFacadeFactory.forPage(page);
 * 
 * // Create facade from HTML string
 * const serverFacade = HtmlFacadeFactory.fromHtml('<html>...</html>');
 * 
 * // Create facade from file
 * const fileFacade = await HtmlFacadeFactory.fromFile('./page.html');
 */

'use strict';

// Core interface and types
const { HtmlFacade, Context } = require('./HtmlFacade');

// Concrete implementations
const { PuppeteerHtmlFacade } = require('./PuppeteerHtmlFacade');
const { JsdomHtmlFacade } = require('./JsdomHtmlFacade');

// Factory
const { HtmlFacadeFactory } = require('./HtmlFacadeFactory');

// Element wrappers
const { PuppeteerHtmlElement } = require('./elements/PuppeteerHtmlElement');
const { JsdomHtmlElement } = require('./elements/JsdomHtmlElement');

module.exports = {
    // Core interface
    HtmlFacade,
    Context,
    
    // Implementations
    PuppeteerHtmlFacade,
    JsdomHtmlFacade,
    
    // Factory (primary entry point)
    HtmlFacadeFactory,
    
    // Element wrappers (for advanced use)
    PuppeteerHtmlElement,
    JsdomHtmlElement
};
