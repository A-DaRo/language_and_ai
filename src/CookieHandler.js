/**
 * Handles cookie consent banners on Notion pages
 */
class CookieHandler {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
  }
  
  /**
   * Handle the cookie banner and confirmation dialog
   */
  async handle(page) {
    this.logger.info('COOKIE', 'Checking for cookie consent banner...');
    
    try {
      // Wait for the cookie banner to appear
      await page.waitForSelector(this.config.SELECTORS.cookieBanner, { 
        timeout: this.config.TIMEOUT_COOKIE_BANNER 
      });
      this.logger.info('COOKIE', 'Cookie banner detected.');
      
      // Click the "Reject all" button
      const rejectButtonClicked = await this._clickRejectButton(page);
      
      if (rejectButtonClicked) {
        this.logger.info('COOKIE', 'Clicked "Reject all" button.');
        
        // Handle the confirmation dialog
        await this._handleConfirmationDialog(page);
      } else {
        this.logger.warn('COOKIE', 'Could not find "Reject all" button.');
      }
    } catch (error) {
      this.logger.info('COOKIE', 'No cookie banner found or timeout occurred. Continuing...');
    }
  }
  
  /**
   * Click the "Reject all" button
   */
  async _clickRejectButton(page) {
    return await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('div[role="button"]'));
      const rejectButton = buttons.find(btn => btn.textContent.includes('Reject all'));
      if (rejectButton) {
        rejectButton.click();
        return true;
      }
      return false;
    });
  }
  
  /**
   * Handle the confirmation dialog that appears after rejecting cookies
   */
  async _handleConfirmationDialog(page) {
    try {
      // Wait a moment for the dialog to appear
      await new Promise(resolve => setTimeout(resolve, 500));
      this.logger.info('COOKIE', 'Waiting for confirmation dialog...');
      
      // Wait for the dialog
      await page.waitForSelector(this.config.SELECTORS.confirmDialog, { 
        timeout: this.config.TIMEOUT_DIALOG 
      });
      this.logger.info('COOKIE', 'Confirmation dialog detected.');
      
      // Click the OK button
      const okButtonClicked = await page.evaluate(() => {
        const dialog = document.querySelector('div[role="dialog"][aria-modal="true"]');
        if (dialog) {
          const buttons = Array.from(dialog.querySelectorAll('div[role="button"]'));
          const okButton = buttons.find(btn => btn.textContent.trim() === 'OK');
          if (okButton) {
            okButton.click();
            return true;
          }
        }
        return false;
      });
      
      if (okButtonClicked) {
        this.logger.info('COOKIE', 'Clicked "OK" button. Waiting for page reload...');
        
        // Wait for navigation
        await page.waitForNavigation({ 
          waitUntil: 'networkidle0', 
          timeout: this.config.TIMEOUT_NAVIGATION 
        });
        this.logger.success('COOKIE', 'Page reloaded successfully.');
        
        // Wait for content to stabilize
        await new Promise(resolve => setTimeout(resolve, this.config.WAIT_AFTER_COOKIE));
      } else {
        this.logger.warn('COOKIE', 'Could not find "OK" button in dialog.');
      }
    } catch (error) {
      this.logger.error('COOKIE', 'Error handling confirmation dialog', error);
    }
  }
}

module.exports = CookieHandler;
