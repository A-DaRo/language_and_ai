/**
 * @fileoverview Path Module Index
 * @module domain/path
 * @description Exports all path-related components for the Notion scraper.
 * 
 * This module implements the Path Strategy Pattern for resolving links
 * between pages in the scraped hierarchy.
 * 
 * @see PathStrategy - Abstract base class for path resolution
 * @see IntraPathStrategy - Same-page anchor links
 * @see InterPathStrategy - Cross-page navigation links
 * @see ExternalPathStrategy - External URL pass-through
 * @see PathStrategyFactory - Strategy selection and invocation
 * @see PathCalculator - Legacy path calculator (to be deprecated)
 */

const PathStrategy = require('./PathStrategy');
const IntraPathStrategy = require('./IntraPathStrategy');
const InterPathStrategy = require('./InterPathStrategy');
const ExternalPathStrategy = require('./ExternalPathStrategy');
const PathStrategyFactory = require('./PathStrategyFactory');
const PathCalculator = require('./PathCalculator');

module.exports = {
  PathStrategy,
  IntraPathStrategy,
  InterPathStrategy,
  ExternalPathStrategy,
  PathStrategyFactory,
  PathCalculator
};
