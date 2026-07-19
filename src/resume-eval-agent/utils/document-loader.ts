/**
 * @fileoverview Document loading utilities
 * Handles loading and extracting text from PDF, DOCX, and TXT files
 */

import * as fs from 'fs';
import * as path from 'path';

/**
 * Load and extract text from a PDF file
 * Note: Requires pdfjs-dist to be installed
 * @param filePath - Path to the PDF file
 * @returns Extracted text from PDF
 * @throws Error if file not found or PDF parsing fails
 */
export async function loadPDF(filePath: string): Promise<string> {
  try {
    const pdfParse = require('pdf-parse');
    const fileBuffer = fs.readFileSync(filePath);
    const pdfData = await pdfParse(fileBuffer);
    return pdfData.text;
  } catch (error) {
    throw new Error(`Failed to parse PDF: ${(error as Error).message}`);
  }
}

/**
 * Load and extract text from a DOCX file
 * Note: Requires mammoth to be installed
 * @param filePath - Path to the DOCX file
 * @returns Extracted text from DOCX
 * @throws Error if file not found or DOCX parsing fails
 */
export async function loadDOCX(filePath: string): Promise<string> {
  try {
    const mammoth = require('mammoth');
    const result = await mammoth.extractRawText({ path: filePath });
    return result.value;
  } catch (error) {
    throw new Error(`Failed to parse DOCX: ${(error as Error).message}`);
  }
}

/**
 * Load text from a plain text file
 * @param filePath - Path to the TXT file
 * @returns File contents
 * @throws Error if file not found or cannot be read
 */
export async function loadTextFile(filePath: string): Promise<string> {
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch (error) {
    throw new Error(`Failed to read text file: ${(error as Error).message}`);
  }
}

/**
 * Load document from file, auto-detecting format by extension
 * Supports .pdf, .docx, and .txt files
 * @param filePath - Path to the document
 * @returns Extracted text content
 * @throws Error if format not supported or file cannot be read
 */
export async function loadDocument(filePath: string): Promise<string> {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === '.pdf') {
    return loadPDF(filePath);
  } else if (ext === '.docx') {
    return loadDOCX(filePath);
  } else if (ext === '.txt') {
    return loadTextFile(filePath);
  } else {
    throw new Error(`Unsupported file format: ${ext}. Supported formats: .pdf, .docx, .txt`);
  }
}

/**
 * Validate that a file exists and is readable
 * @param filePath - Path to check
 * @throws Error if file doesn't exist or isn't readable
 */
export function validateFilePath(filePath: string): void {
  try {
    fs.accessSync(filePath, fs.constants.R_OK);
  } catch (error) {
    throw new Error(`File not accessible: ${filePath}`);
  }
}
