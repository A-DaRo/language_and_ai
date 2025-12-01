/**
 * @fileoverview Resolvers Index
 * @module domain/path/resolvers
 * @description Exports all path resolver implementations.
 * 
 * @see PathResolver - Base interface
 * @see PathResolverFactory - Factory for resolver selection
 */

'use strict';

const IntraPageResolver = require('./IntraPageResolver');
const InterPageResolver = require('./InterPageResolver');
const ExternalUrlResolver = require('./ExternalUrlResolver');
const FilesystemResolver = require('./FilesystemResolver');

module.exports = {
  IntraPageResolver,
  InterPageResolver,
  ExternalUrlResolver,
  FilesystemResolver
};
