const PipelineStep = require('../PipelineStep');

/**
 * @fileoverview Cookie Consent Step
 * @module worker/pipeline/steps/CookieConsentStep
 */

/**
 * @class CookieConsentStep
 * @extends PipelineStep
 * @classdesc Handles cookie consent banner dismissal as a precautionary measure.
 * 
 * Even though cookies are shared across workers after bootstrap, this step
 * provides an additional safety check to handle any cookie banners that might
 * appear before interacting with the page.
 * 
 * This step:
 * - Detects cookie consent banners
 * - Clicks "Reject all" button if present
 * - Handles confirmation dialogs
 * - Fails gracefully if no banner is present
 */
class CookieConsentStep extends PipelineStep {
  /**
   * @param {CookieHandler} cookieHandler - Cookie handler instance
   */
  constructor(cookieHandler) {
    super('CookieConsent');
    this.cookieHandler = cookieHandler;
  }
  
  /**
   * @method process
   * @summary Attempts to dismiss cookie consent banner if present.
   * @description This is a precautionary step that runs in the download phase
   * even though cookies are shared after bootstrap. It ensures no cookie
   * banners interfere with page interaction.
   * 
   * @param {PipelineContext} context - Pipeline context.
   * @returns {Promise<void>}
   */
  async process(context) {
    const { page, logger } = context;
    
    logger.debug('COOKIE-CONSENT', 'Checking for cookie consent banner...');
    
    try {
      // Use the cookie handler to check and dismiss banner
      const bannerHandled = await this.cookieHandler.ensureConsent(page, 'download-phase');
      
      if (bannerHandled) {
        logger.info('COOKIE-CONSENT', 'Cookie banner dismissed successfully');
      } else {
        logger.debug('COOKIE-CONSENT', 'No cookie banner present');
      }
    } catch (error) {
      // Don't fail the entire pipeline if cookie handling fails
      logger.warn('COOKIE-CONSENT', `Cookie handling failed: ${error.message}`);
    }
  }
}

module.exports = CookieConsentStep;
