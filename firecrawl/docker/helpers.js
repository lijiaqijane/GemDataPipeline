/**
 * DeepSeek helper functions for FireCrawl.
 * This file is injected into the patched JS files via require().
 */

function isDeepSeekModel(modelId) {
  var lower = (modelId || '').toLowerCase();
  return (
    lower.indexOf('deepseek') >= 0 ||
    lower.indexOf('deep-seek') >= 0 ||
    lower.indexOf('ds-') >= 0 ||
    lower.indexOf('dv-') >= 0 ||
    lower === 'deepseek-chat' ||
    lower === 'deepseek-reasoner' ||
    lower.indexOf('deepseek-v') === 0 ||
    lower.indexOf('deepseek-r') === 0
  );
}

function safeParseJSON(text) {
  if (!text || typeof text !== 'string') {
    return { success: false, error: 'Empty or non-string input' };
  }
  var cleaned = text.trim()
    .replace(/^```(?:json)?\s*\n?/i, '')
    .replace(/\n?```\s*$/g, '')
    .replace(/[\u200B-\u200D\uFEFF\u00AD\u2060\u200E\u200F]/g, '');
  try { return { success: true, data: JSON.parse(cleaned) }; } catch (_) {}
  var firstBrace = cleaned.indexOf('{');
  var firstBracket = cleaned.indexOf('[');
  var startIdx = firstBrace !== -1 && (firstBracket === -1 || firstBrace < firstBracket) ? firstBrace : firstBracket;
  if (startIdx === -1) return { success: false, error: 'No JSON found' };
  var lastBrace = cleaned.lastIndexOf('}');
  var lastBracket = cleaned.lastIndexOf(']');
  var endIdx = lastBrace !== -1 && (lastBracket === -1 || lastBrace > lastBracket) ? lastBrace : lastBracket;
  if (endIdx === -1 || endIdx <= startIdx) return { success: false, error: 'No matching bracket' };
  cleaned = cleaned.slice(startIdx, endIdx + 1);
  try { return { success: true, data: JSON.parse(cleaned) }; } catch (_) {}
  try { return { success: true, data: JSON.parse(cleaned.replace(/'/g, '"')) }; } catch (_) {}
  try { return { success: true, data: JSON.parse(cleaned.replace(/,\s*([}\]])/g, '$1')) }; } catch (_) {}
  try { return { success: true, data: JSON.parse(cleaned.replace(/'/g, '"').replace(/,\s*([}\]])/g, '$1')) }; } catch (_) {}
  return { success: false, error: 'Failed to parse JSON after all attempts' };
}

module.exports = { isDeepSeekModel: isDeepSeekModel, safeParseJSON: safeParseJSON };
