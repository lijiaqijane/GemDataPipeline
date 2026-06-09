import { encoding_for_model } from "@dqbd/tiktoken";
import { TiktokenModel } from "@dqbd/tiktoken";
import {
  Document,
  ExtractOptions,
  ExtractOptionsInput,
  TokenUsage,
} from "../../../controllers/v1/types";
import { Logger } from "winston";
import { logger } from "../../../lib/logger";
import { modelPrices } from "../../../lib/extract/usage/model-prices";
import { generateObject, generateText, LanguageModel, NoObjectGeneratedError } from "ai";
import { jsonSchema } from "ai";
import { getModel } from "../../../lib/generic-ai";
import { z } from "zod";

// Get max tokens from model prices
const getModelLimits_F0 = (model: string) => {
  const modelConfig = modelPrices[model];
  if (!modelConfig) {
    // Default fallback values
    return {
      maxInputTokens: 8192,
      maxOutputTokens: 4096,
      maxTokens: 12288,
    };
  }
  return {
    maxInputTokens: modelConfig.max_input_tokens || modelConfig.max_tokens,
    maxOutputTokens: modelConfig.max_output_tokens || modelConfig.max_tokens,
    maxTokens: modelConfig.max_tokens,
  };
};

class LLMRefusalError extends Error {
  public refusal: string;

  constructor(refusal: string) {
    super("LLM refused to extract the website's content");
    this.refusal = refusal;
  }
}

/**
 * Check if the model is a DeepSeek model (or accessed via DeepSeek-compatible API).
 * Used to apply DeepSeek-specific JSON parsing workarounds.
 */
function isDeepSeekModel(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  return (
    lower.includes("deepseek") ||
    lower.includes("deep-seek") ||
    lower.includes("ds-") ||
    // Volcengine DeepSeek models from ark.cn-beijing.volces.com
    lower.includes("dv-") ||
    // Common DeepSeek model naming patterns
    lower === "deepseek-chat" ||
    lower === "deepseek-reasoner" ||
    lower.startsWith("deepseek-v") ||
    lower.startsWith("deepseek-r")
  );
}

/**
 * Robust JSON parser that handles common quirks when LLMs (especially DeepSeek)
 * return malformed JSON. Handles:
 * - Markdown code block wrapping (```json ... ```)
 * - Extra text before/after the JSON
 * - Zero-width and invisible characters
 * - Single quotes instead of double quotes
 * - Trailing commas
 * - Multiple JSON objects (takes the first valid one)
 */
function safeParseJSON(text: string): { success: boolean; data?: any; error?: string } {
  if (!text || typeof text !== "string") {
    return { success: false, error: "Empty or non-string input" };
  }

  let cleaned = text.trim();

  // Remove markdown code block markers (```json ... ``` or ``` ... ```)
  cleaned = cleaned.replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/g, "");

  // Remove zero-width and invisible Unicode characters
  cleaned = cleaned.replace(/[\u200B-\u200D\uFEFF\u00AD\u2060\u200E\u200F]/g, "");

  // Try direct parse first
  try {
    return { success: true, data: JSON.parse(cleaned) };
  } catch (_) {
    // Continue to advanced parsing
  }

  // Find first { or [ and matching last } or ]
  const firstBrace = cleaned.indexOf("{");
  const firstBracket = cleaned.indexOf("[");
  const startIdx =
    firstBrace !== -1 && (firstBracket === -1 || firstBrace < firstBracket)
      ? firstBrace
      : firstBracket;

  if (startIdx === -1) {
    return { success: false, error: "No JSON object or array found in the response" };
  }

  const lastBrace = cleaned.lastIndexOf("}");
  const lastBracket = cleaned.lastIndexOf("]");
  const endIdx =
    lastBrace !== -1 && (lastBracket === -1 || lastBrace > lastBracket)
      ? lastBrace
      : lastBracket;

  if (endIdx === -1 || endIdx <= startIdx) {
    return { success: false, error: "Could not find matching closing bracket" };
  }

  cleaned = cleaned.slice(startIdx, endIdx + 1);

  // Try parsing after extracting JSON region
  try {
    return { success: true, data: JSON.parse(cleaned) };
  } catch (_) {}

  // Replace single quotes with double quotes (common DeepSeek quirk)
  // But be careful not to break already-valid JSON strings containing apostrophes
  try {
    const singleQuoteReplaced = cleaned.replace(/'/g, '"');
    return { success: true, data: JSON.parse(singleQuoteReplaced) };
  } catch (_) {}

  // Remove trailing commas before closing brackets
  try {
    const noTrailingCommas = cleaned.replace(/,\s*([}\]])/g, "$1");
    return { success: true, data: JSON.parse(noTrailingCommas) };
  } catch (_) {}

  // Try parsing with both single quote replacement AND trailing comma fix
  try {
    const combined = cleaned.replace(/'/g, '"').replace(/,\s*([}\]])/g, "$1");
    return { success: true, data: JSON.parse(combined) };
  } catch (_) {}

  return { success: false, error: "Failed to parse JSON after all attempts" };
}

function normalizeSchema(x: any): any {
  if (typeof x !== "object" || x === null) return x;

  if (x["$defs"] !== null && typeof x["$defs"] === "object") {
    x["$defs"] = Object.fromEntries(
      Object.entries(x["$defs"]).map(([name, schema]) => [
        name,
        normalizeSchema(schema),
      ]),
    );
  }

  if (x && x.anyOf) {
    x.anyOf = x.anyOf.map(x => normalizeSchema(x));
  }

  if (x && x.oneOf) {
    x.oneOf = x.oneOf.map(x => normalizeSchema(x));
  }

  if (x && x.allOf) {
    x.allOf = x.allOf.map(x => normalizeSchema(x));
  }

  if (x && x.not) {
    x.not = normalizeSchema(x.not);
  }

  if (x && x.type === "object") {
    return {
      ...x,
      properties: Object.fromEntries(
        Object.entries(x.properties || {}).map(([k, v]) => [
          k,
          normalizeSchema(v),
        ]),
      ),
      required: Object.keys(x.properties || {}),
      additionalProperties: false,
    };
  } else if (x && x.type === "array") {
    return {
      ...x,
      items: normalizeSchema(x.items),
    };
  } else {
    return x;
  }
}

interface TrimResult {
  text: string;
  numTokens: number;
  warning?: string;
}

function trimToTokenLimit_F0(
  text: string,
  maxTokens: number,
  modelId: string = "gpt-4o-mini",
  previousWarning?: string,
): TrimResult {
  try {
    const encoder = encoding_for_model(modelId as TiktokenModel);
    try {
      const tokens = encoder.encode(text);
      const numTokens = tokens.length;

      if (numTokens <= maxTokens) {
        return { text, numTokens };
      }

      const modifier = 3;
      // Start with 3 chars per token estimation
      let currentText = text.slice(0, Math.floor(maxTokens * modifier) - 1);

      // Keep trimming until we're under the token limit
      while (true) {
        const currentTokens = encoder.encode(currentText);
        if (currentTokens.length <= maxTokens) {
          const warning = `The extraction content would have used more tokens (${numTokens}) than the maximum we allow (${maxTokens}). -- the input has been automatically trimmed.`;
          return {
            text: currentText,
            numTokens: currentTokens.length,
            warning: previousWarning
              ? `${warning} ${previousWarning}`
              : warning,
          };
        }
        const overflow = currentTokens.length * modifier - maxTokens - 1;
        // If still over limit, remove another chunk
        currentText = currentText.slice(
          0,
          Math.floor(currentText.length - overflow),
        );
      }
    } catch (e) {
      throw e;
    } finally {
      encoder.free();
    }
  } catch (error) {
    // Fallback to a more conservative character-based approach
    const estimatedCharsPerToken = 2.8;
    const safeLength = maxTokens * estimatedCharsPerToken;
    const trimmedText = text.slice(0, Math.floor(safeLength));

    const warning = `Failed to derive number of LLM tokens the extraction might use -- the input has been automatically trimmed to the maximum number of tokens (${maxTokens}) we support.`;

    return {
      text: trimmedText,
      numTokens: maxTokens, // We assume we hit the max in this fallback case
      warning: previousWarning ? `${warning} ${previousWarning}` : warning,
    };
  }
}

export async function generateCompletions_F0({
  logger,
  options,
  markdown,
  previousWarning,
  isExtractEndpoint,
  model = getModel("gpt-4o-mini"),
  mode = "object",
  metadata,
}: {
  model?: LanguageModel;
  modelName?: string;
  logger: Logger;
  options: ExtractOptionsInput;
  markdown?: string;
  previousWarning?: string;
  isExtractEndpoint?: boolean;
  mode?: "object" | "no-object";
  metadata: {
    teamId: string;
    functionId?: string;
    extractId?: string;
    scrapeId?: string;
  };
}): Promise<{
  extract: any;
  numTokens: number;
  warning: string | undefined;
  totalUsage: TokenUsage;
  model: string;
}> {
  let extract: any;
  let warning: string | undefined;

  if (markdown === undefined) {
    throw new Error("document.markdown is undefined -- this is unexpected");
  }

  const modelId = typeof model === "string" ? model : model.modelId;

  const { maxInputTokens } = getModelLimits_F0(modelId);
  // Calculate 80% of max input tokens (for content)
  const maxTokensSafe = Math.floor(maxInputTokens * 0.8);

  // Use the new trimming function
  const {
    text: trimmedMarkdown,
    numTokens,
    warning: trimWarning,
  } = trimToTokenLimit_F0(markdown, maxTokensSafe, modelId, previousWarning);

  markdown = trimmedMarkdown;
  warning = trimWarning;

  try {
    const prompt =
      options.prompt !== undefined
        ? `Transform the following content into structured JSON output based on the provided schema and this user request: ${options.prompt}. If schema is provided, strictly follow it.\n\n${markdown}`
        : `Transform the following content into structured JSON output based on the provided schema if any.\n\n${markdown}`;

    if (mode === "no-object") {
      const result = await generateText({
        model: model,
        prompt: options.prompt + (markdown ? `\n\nData:${markdown}` : ""),
        temperature: options.temperature ?? 0,
        system: options.systemPrompt,
        providerOptions: {
          google: {
            labels: {
              functionId: metadata.functionId ?? "unspecified",
              extractId: metadata.extractId ?? "unspecified",
              scrapeId: metadata.scrapeId ?? "unspecified",
              teamId: metadata.teamId,
            },
          },
        },
        experimental_telemetry: {
          isEnabled: true,
          functionId: metadata.functionId,
          metadata: {
            teamId: metadata.teamId,
            ...(metadata.extractId
              ? {
                  langfuseTraceId: "extract:" + metadata.extractId,
                  extractId: metadata.extractId,
                }
              : {}),
            ...(metadata.scrapeId
              ? {
                  langfuseTraceId: "scrape:" + metadata.scrapeId,
                  scrapeId: metadata.scrapeId,
                }
              : {}),
          },
        },
      });

      extract = result.text;

      return {
        extract,
        warning,
        numTokens,
        totalUsage: {
          promptTokens: numTokens,
          completionTokens: result.usage?.outputTokens ?? 0,
          totalTokens: numTokens + (result.usage?.outputTokens ?? 0),
        },
        model: modelId,
      };
    }

    let schema = options.schema;
    // Normalize the bad json schema users write (mogery)
    if (schema && !(schema instanceof z.ZodType)) {
      // let schema = options.schema;
      if (schema) {
        schema = removeDefaultProperty_F0(schema);
      }

      if (schema && schema.type === "array") {
        schema = {
          type: "object",
          properties: {
            items: options.schema,
          },
          required: ["items"],
          additionalProperties: false,
        };
      } else if (schema && typeof schema === "object" && !schema.type) {
        schema = {
          type: "object",
          properties: Object.fromEntries(
            Object.entries(schema).map(([key, value]) => {
              return [key, removeDefaultProperty_F0(value)];
            }),
          ),
          required: Object.keys(schema),
          additionalProperties: false,
        };
      }

      schema = normalizeSchema(schema);
    }

    // For DeepSeek models (and similar OpenAI-compatible models that may not fully
    // support the response_format parameter), use generateText with explicit JSON
    // instructions instead of generateObject. This avoids issues with structured
    // output support in non-OpenAI models.
    if (isDeepSeekModel(modelId)) {
      logger.info("Using generateText fallback for DeepSeek model", { modelId });

      const jsonPrompt = schema
        ? `You must respond with valid JSON only, no markdown formatting, no explanation.
The JSON must conform to this schema: ${JSON.stringify(schema)}

${options.prompt !== undefined
  ? `User request: ${options.prompt}`
  : ""}

Content to extract from:
${markdown}`
        : `You must respond with valid JSON only, no markdown formatting, no explanation.
${options.prompt !== undefined
  ? `User request: ${options.prompt}`
  : ""}

Content to extract from:
${markdown}`;

      const result = await generateText({
        model: model,
        prompt: jsonPrompt,
        temperature: options.temperature ?? 0,
        system: (options.systemPrompt || "") + "\nYou are a JSON extraction assistant. Always respond with valid JSON only. Do not wrap JSON in markdown code blocks. Do not include any text before or after the JSON.",
        providerOptions: {
          google: {
            labels: {
              functionId: metadata.functionId ?? "unspecified",
              extractId: metadata.extractId ?? "unspecified",
              scrapeId: metadata.scrapeId ?? "unspecified",
              teamId: metadata.teamId,
            },
          },
        },
        experimental_telemetry: {
          isEnabled: true,
          functionId: metadata.functionId,
          metadata: {
            teamId: metadata.teamId,
            ...(metadata.extractId
              ? {
                  langfuseTraceId: "extract:" + metadata.extractId,
                  extractId: metadata.extractId,
                }
              : {}),
            ...(metadata.scrapeId
              ? {
                  langfuseTraceId: "scrape:" + metadata.scrapeId,
                  scrapeId: metadata.scrapeId,
                }
              : {}),
          },
        },
      });

      const parsed = safeParseJSON(result.text);
      if (parsed.success) {
        extract = parsed.data;

        // If the users actually wants the items object, they can specify it as 'required' in the schema
        // otherwise, we just return the items array
        if (
          options.schema &&
          options.schema.type === "array" &&
          !schema?.required?.includes("items")
        ) {
          extract = extract?.items;
        }

        return {
          extract,
          warning,
          numTokens: result.usage?.inputTokens ?? 0,
          totalUsage: {
            promptTokens: result.usage?.inputTokens ?? 0,
            completionTokens: result.usage?.outputTokens ?? 0,
            totalTokens: (result.usage?.inputTokens ?? 0) + (result.usage?.outputTokens ?? 0),
          },
          model: modelId,
        };
      } else {
        logger.error("Failed to parse DeepSeek JSON response", {
          parseError: parsed.error,
          textPeek: result.text.substring(0, 500),
        });
        // Fall through to try generateObject as fallback
      }
    }

    const repairConfig = {
      experimental_repairText: async ({ text, error }) => {
        // First, try our robust JSON parser before anything else
        const robustParsed = safeParseJSON(text);
        if (robustParsed.success) {
          logger.debug("Repaired text using safeParseJSON");
          return JSON.stringify(robustParsed.data);
        }

        // Try LLM-based repair as fallback
        const { text: fixedText } = await generateText({
          model: model,
          prompt: `Fix this JSON that had the following error: ${error}\n\nOriginal text:\n${text}\n\nReturn only the fixed JSON, no explanation.`,
          system:
            "You are a JSON repair expert. Your only job is to fix malformed JSON and return valid JSON that matches the original structure and intent as closely as possible. Do not include any explanation or commentary - only return the fixed JSON. Do not return it in a Markdown code block, just plain JSON.",
          providerOptions: {
            google: {
              labels: {
                functionId: metadata.functionId ?? "unspecified",
                extractId: metadata.extractId ?? "unspecified",
                scrapeId: metadata.scrapeId ?? "unspecified",
                teamId: metadata.teamId,
              },
            },
          },
          experimental_telemetry: {
            isEnabled: true,
            functionId: metadata.functionId,
            metadata: {
              teamId: metadata.teamId,
              ...(metadata.extractId
                ? {
                    langfuseTraceId: "extract:" + metadata.extractId,
                    extractId: metadata.extractId,
                  }
                : {}),
              ...(metadata.scrapeId
                ? {
                    langfuseTraceId: "scrape:" + metadata.scrapeId,
                    scrapeId: metadata.scrapeId,
                  }
                : {}),
            },
          },
        });
        return fixedText;
      },
    };

    const generateObjectConfig = {
      model: model,
      prompt: prompt,
      temperature: options.temperature ?? 0,
      system: options.systemPrompt,
      ...(schema && {
        schema: schema instanceof z.ZodType ? schema : jsonSchema(schema),
      }),
      ...(!schema && { output: "no-schema" as const }),
      ...repairConfig,
      ...(!schema && {
        onError: (error: Error) => {
          console.error(error);
        },
      }),
      providerOptions: {
        google: {
          labels: {
            functionId: metadata.functionId ?? "unspecified",
            extractId: metadata.extractId ?? "unspecified",
            scrapeId: metadata.scrapeId ?? "unspecified",
            teamId: metadata.teamId,
          },
        },
      },
      experimental_telemetry: {
        isEnabled: true,
        functionId: metadata.functionId,
        metadata: {
          teamId: metadata.teamId,
          ...(metadata.extractId
            ? {
                langfuseTraceId: "extract:" + metadata.extractId,
                extractId: metadata.extractId,
              }
            : {}),
          ...(metadata.scrapeId
            ? {
                langfuseTraceId: "scrape:" + metadata.scrapeId,
                scrapeId: metadata.scrapeId,
              }
            : {}),
        },
      },
    } satisfies Parameters<typeof generateObject>[0];

    let result: {
      object: any;
      usage?: {
        inputTokens?: number;
        outputTokens?: number;
        totalTokens?: number;
      };
    } | undefined;

    let generateObjectFailed = false;
    try {
      result = await generateObject(generateObjectConfig);
    } catch (error) {
      generateObjectFailed = true;
      // Handle NoObjectGeneratedError - the model returned text that wasn't valid JSON.
      // This is common with DeepSeek and other OpenAI-compatible models that may
      // wrap JSON in markdown code blocks, add extra text, or use non-standard formatting.
      if (NoObjectGeneratedError.isInstance(error)) {
        logger.warn("NoObjectGeneratedError caught, attempting to parse text response", {
          error: error.message,
          modelId,
          textPeek: error.text
            ? JSON.stringify(error.text.substring(0, 300))
            : "N/A",
        });

        if (error.text) {
          // Try our robust JSON parser first
          const parsed = safeParseJSON(error.text);
          if (parsed.success) {
            logger.info("Successfully parsed JSON from error text using safeParseJSON");
            result = {
              object: parsed.data,
              usage: {
                inputTokens: error.usage?.inputTokens ?? 0,
                outputTokens: error.usage?.outputTokens ?? 0,
                totalTokens: error.usage?.totalTokens ?? 0,
              },
            };
          } else {
            // If safeParseJSON failed, try using the LLM to repair the text
            logger.warn("safeParseJSON failed, attempting LLM-based repair", {
              parseError: parsed.error,
            });
            try {
              const { text: fixedText } = await generateText({
                model: model,
                prompt: `Fix this JSON that had the following error: ${error.message}\n\nOriginal text:\n${error.text}\n\nReturn only the fixed JSON, no explanation.`,
                system:
                  "You are a JSON repair expert. Your only job is to fix malformed JSON and return valid JSON that matches the original structure and intent as closely as possible. Do not include any explanation or commentary - only return the fixed JSON. Do not return it in a Markdown code block, just plain JSON.",
                providerOptions: {
                  google: {
                    labels: {
                      functionId: metadata.functionId ?? "unspecified",
                      extractId: metadata.extractId ?? "unspecified",
                      scrapeId: metadata.scrapeId ?? "unspecified",
                      teamId: metadata.teamId,
                    },
                  },
                },
                experimental_telemetry: {
                  isEnabled: true,
                  functionId: metadata.functionId,
                  metadata: {
                    teamId: metadata.teamId,
                    ...(metadata.extractId
                      ? {
                          langfuseTraceId: "extract:" + metadata.extractId,
                          extractId: metadata.extractId,
                        }
                      : {}),
                    ...(metadata.scrapeId
                      ? {
                          langfuseTraceId: "scrape:" + metadata.scrapeId,
                          scrapeId: metadata.scrapeId,
                        }
                      : {}),
                  },
                },
              });
              const repaired = safeParseJSON(fixedText);
              if (repaired.success) {
                logger.info("Successfully repaired JSON using LLM repair");
                result = {
                  object: repaired.data,
                  usage: {
                    inputTokens: error.usage?.inputTokens ?? 0,
                    outputTokens: error.usage?.outputTokens ?? 0,
                    totalTokens: error.usage?.totalTokens ?? 0,
                  },
                };
              } else {
                throw new Error(
                  `Failed to parse JSON even after LLM repair: ${repaired.error}`,
                );
              }
            } catch (repairError) {
              logger.error("Failed to repair JSON via LLM", {
                error: repairError.message,
              });
              throw repairError;
            }
          }
        } else {
          throw error;
        }
      } else if (error.message?.includes("refused")) {
        throw new LLMRefusalError(error.message);
      } else {
        throw error;
      }
    }

    if (!result) {
      throw new Error("generateObject returned undefined result");
    }

    extract = result.object;

    // If the users actually wants the items object, they can specify it as 'required' in the schema
    // otherwise, we just return the items array
    if (
      options.schema &&
      options.schema.type === "array" &&
      !schema?.required?.includes("items")
    ) {
      extract = extract?.items;
    }

    // Token usage
    const promptTokens = result.usage?.inputTokens ?? numTokens;
    const completionTokens = result.usage?.outputTokens ?? 0;

    return {
      extract,
      warning,
      numTokens: promptTokens,
      totalUsage: {
        promptTokens,
        completionTokens,
        totalTokens: promptTokens + completionTokens,
      },
      model: modelId,
    };
  } catch (error) {
    if (error.message?.includes("refused")) {
      throw new LLMRefusalError(error.message);
    }
    logger.error("LLM extraction failed", {
      error: error.message,
      modelId,
      mode,
    });
    throw error;
  }
}

function removeDefaultProperty_F0(schema: any): any {
  if (typeof schema !== "object" || schema === null) return schema;

  const rest = { ...schema };

  // unsupported global keys
  delete rest.default;

  // unsupported object keys
  delete rest.patternProperties;
  delete rest.unevaluatedProperties;
  delete rest.propertyNames;
  delete rest.minProperties;
  delete rest.maxProperties;

  // unsupported string keys
  delete rest.minLength;
  delete rest.maxLength;
  delete rest.pattern;
  delete rest.format;

  // unsupported number keys
  delete rest.minimum;
  delete rest.maximum;
  delete rest.multipleOf;

  // unsupported array keys
  delete rest.unevaluatedItems;
  delete rest.contains;
  delete rest.minContains;
  delete rest.maxContains;
  delete rest.minItems;
  delete rest.maxItems;
  delete rest.uniqueItems;

  for (const key in rest) {
    if (Array.isArray(rest[key])) {
      rest[key] = rest[key].map((item: any) => removeDefaultProperty_F0(item));
    } else if (typeof rest[key] === "object" && rest[key] !== null) {
      rest[key] = removeDefaultProperty_F0(rest[key]);
    }
  }

  return rest;
}

export async function generateSchemaFromPrompt_F0(
  prompt: string,
  metadata: { teamId: string; functionId?: string; extractId?: string },
): Promise<any> {
  const model = getModel("gpt-4o-mini");
  const temperatures = [0, 0.1, 0.3]; // Different temperatures to try
  let lastError: Error | null = null;

  for (const temp of temperatures) {
    try {
      const { extract } = await generateCompletions_F0({
        logger: logger.child({
          method: "generateSchemaFromPrompt/generateCompletions",
        }),
        model: model,
        options: {
          mode: "llm",
          systemPrompt: `You are a schema generator for a web scraping system. Generate a JSON schema based on the user's prompt.
Consider:
1. The type of data being requested
2. Required fields vs optional fields
3. Appropriate data types for each field
4. Nested objects and arrays where appropriate

Valid JSON schema, has to be simple. No crazy properties. OpenAI has to support it.
Supported types
The following types are supported for Structured Outputs:

String
Number
Boolean
Integer
Object
Array
Enum
anyOf

Formats are not supported. Min/max are not supported. Anything beyond the above is not supported. Keep it simple with types and descriptions.
Optionals are not supported.
DO NOT USE FORMATS.
Keep it simple. Don't create too many properties, just the ones that are needed. Don't invent properties.
Return a valid JSON schema object with properties that would capture the information requested in the prompt.`,
          prompt: `Generate a JSON schema for extracting the following information: ${prompt}`,
          temperature: temp,
        },
        markdown: prompt,
        metadata: {
          ...metadata,
          functionId: metadata.functionId
            ? metadata.functionId + "/generateSchemaFromPrompt_F0"
            : "generateSchemaFromPrompt_F0",
        },
      });

      return extract;
    } catch (error) {
      lastError = error as Error;
      logger.warn(`Failed attempt with temperature ${temp}: ${error.message}`);
      continue;
    }
  }

  // If we get here, all attempts failed
  throw new Error(
    `Failed to generate schema after all attempts. Last error: ${lastError?.message}`,
  );
}
