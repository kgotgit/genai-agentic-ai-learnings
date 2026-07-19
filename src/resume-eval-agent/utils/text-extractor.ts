/**
 * @fileoverview Text extraction and normalization utilities
 * Helper functions for parsing and cleaning resume/JD text
 */

/**
 * Extract email address from text using regex pattern
 * Finds first valid email format
 * @param text - Text to search
 * @returns Email address or null if not found
 */
export function extractEmail(text: string): string | null {
  const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
  const match = text.match(emailRegex);
  return match ? match[0] : null;
}

/**
 * Extract phone number from text using regex pattern
 * Supports multiple formats: +1-800-555-1234, (800) 555-1234, 800.555.1234, etc.
 * @param text - Text to search
 * @returns Phone number or null if not found
 */
export function extractPhoneNumber(text: string): string | null {
  const phoneRegex = /(?:\+1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})/;
  const match = text.match(phoneRegex);
  return match ? match[0] : null;
}

/**
 * Extract total years of experience from text
 * Searches for patterns like "5 years", "3.5 years", "10+ years"
 * @param text - Text to search (typically experience section)
 * @returns Experience string (e.g., "5 years") or "Unknown"
 */
export function extractTotalExperience(text: string): string {
  const expRegex = /(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)/i;
  const match = text.match(expRegex);
  if (match) {
    return `${match[1]} years`;
  }
  return 'Unknown';
}

/**
 * Normalize and clean text
 * - Removes extra whitespace
 * - Fixes line breaks
 * - Removes special characters (optional)
 * @param text - Raw text to normalize
 * @returns Cleaned text
 */
export function normalizeText(text: string): string {
  return text
    .replace(/\r\n/g, '\n') // Normalize line breaks
    .replace(/\n\s*\n/g, '\n') // Remove multiple blank lines
    .replace(/\s+/g, ' ') // Replace multiple spaces with single space
    .trim();
}

/**
 * Split text into sections based on common headers
 * Useful for separating experience, education, skills, etc.
 * @param text - Full resume text
 * @returns Map of section name to section content
 */
export function splitSectionsByHeaders(text: string): Record<string, string> {
  const sections: Record<string, string> = {};
  const headerPatterns = [
    /^(EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE)$/im,
    /^(EDUCATION|ACADEMIC BACKGROUND)$/im,
    /^(SKILLS|TECHNICAL SKILLS|COMPETENCIES)$/im,
    /^(PROJECTS|PORTFOLIO)$/im,
    /^(CERTIFICATIONS|LICENSES|CERTIFICATIONS & LICENSES)$/im,
    /^(CONTACT|CONTACT INFORMATION)$/im,
  ];

  let currentSection = 'header';
  const lines = text.split('\n');
  let currentContent = '';

  for (const line of lines) {
    let foundHeader = false;
    for (const pattern of headerPatterns) {
      if (pattern.test(line)) {
        if (currentContent.trim()) {
          sections[currentSection] = normalizeText(currentContent);
        }
        currentSection = line.trim().toLowerCase();
        currentContent = '';
        foundHeader = true;
        break;
      }
    }
    if (!foundHeader) {
      currentContent += line + '\n';
    }
  }

  if (currentContent.trim()) {
    sections[currentSection] = normalizeText(currentContent);
  }

  return sections;
}

/**
 * Truncate text to maximum character length
 * Preserves word boundaries
 * @param text - Text to truncate
 * @param maxChars - Maximum character limit
 * @returns Truncated text
 */
export function truncateText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const truncated = text.substring(0, maxChars);
  const lastSpace = truncated.lastIndexOf(' ');
  return lastSpace > 0 ? truncated.substring(0, lastSpace) + '...' : truncated + '...';
}

/**
 * Extract unique skills from text by matching common skill patterns
 * @param text - Text to search (typically skills section)
 * @returns Array of extracted skills
 */
export function extractSkillsFromText(text: string): string[] {
  const skillKeywords = [
    'python', 'java', 'typescript', 'javascript', 'c\\+\\+', 'csharp', 'go', 'rust', 'kotlin',
    'sql', 'mongodb', 'postgres', 'mysql', 'redis', 'dynamodb',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
    'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
    'git', 'linux', 'machine learning', 'ai', 'nlp', 'llm', 'deep learning',
    'agile', 'scrum', 'rest api', 'graphql', 'microservices'
  ];

  const skills: string[] = [];
  const lowerText = text.toLowerCase();

  for (const skill of skillKeywords) {
    const regex = new RegExp(`\\b${skill}\\b`, 'gi');
    if (regex.test(lowerText)) {
      // Capitalize first letter
      skills.push(skill.split(/\s+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '));
    }
  }

  return [...new Set(skills)]; // Remove duplicates
}
