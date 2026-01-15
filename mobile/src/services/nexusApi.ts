/**
 * NEXUS API Service
 *
 * Handles communication with the NEXUS backend for HAI-DEF model inference.
 *
 * HAI-DEF Models Used:
 * - MedSigLIP: Anemia and Jaundice detection from images
 * - HeAR: Cry analysis for asphyxia detection
 * - MedGemma: Clinical synthesis and recommendations
 */

import * as FileSystem from 'expo-file-system';

// API Configuration
const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

interface AnemiaResult {
  is_anemic: boolean;
  confidence: number;
  risk_level: string;
  estimated_hemoglobin: number;
  recommendation: string;
  anemia_score: number;
  healthy_score: number;
}

interface JaundiceResult {
  has_jaundice: boolean;
  confidence: number;
  severity: string;
  estimated_bilirubin: number;
  needs_phototherapy: boolean;
  recommendation: string;
  kramer_zone: number;
}

interface CryResult {
  is_abnormal: boolean;
  asphyxia_risk: number;
  cry_type: string;
  risk_level: string;
  recommendation: string;
  features: {
    f0_mean: number;
    f0_std: number;
    duration: number;
    voiced_ratio: number;
  };
}

interface CombinedResult {
  summary: string;
  severity_level: string;
  severity_description: string;
  immediate_actions: string[];
  referral_needed: boolean;
  referral_urgency: string;
  follow_up: string;
  urgent_conditions: string[];
  model: string;
}

export interface SynthesisResult {
  synthesis: string;
  recommendation: string;
  immediate_actions: string[];
  severity: string;
  referral_needed: boolean;
  confidence: number;
}

interface SynthesisInput {
  patient_type: 'pregnant' | 'newborn';
  danger_signs: string[];
  anemia_result?: Partial<AnemiaResult>;
  jaundice_result?: Partial<JaundiceResult>;
  cry_result?: Partial<CryResult>;
}

interface AssessmentData {
  conjunctivaImage: string | null;
  skinImage: string | null;
  cryAudio: string | null;
}

/**
 * Upload file and get base64 encoding
 */
async function fileToBase64(uri: string): Promise<string> {
  const base64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  return base64;
}

/**
 * Error thrown when API is unavailable
 */
export class ApiUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiUnavailableError';
  }
}

/**
 * Analyze conjunctiva image for anemia using MedSigLIP
 */
export async function analyzeAnemia(imageUri: string): Promise<AnemiaResult> {
  const base64Image = await fileToBase64(imageUri);

  const response = await fetch(`${API_BASE_URL}/api/anemia/detect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: base64Image,
      model: 'medsiglip',
    }),
  });

  if (!response.ok) {
    const errorMsg = `Anemia analysis failed: HTTP ${response.status}`;
    console.error(errorMsg);
    throw new ApiUnavailableError(errorMsg);
  }

  return await response.json();
}

/**
 * Analyze skin/sclera image for jaundice using MedSigLIP
 */
export async function analyzeJaundice(imageUri: string): Promise<JaundiceResult> {
  const base64Image = await fileToBase64(imageUri);

  const response = await fetch(`${API_BASE_URL}/api/jaundice/detect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: base64Image,
      model: 'medsiglip',
    }),
  });

  if (!response.ok) {
    const errorMsg = `Jaundice analysis failed: HTTP ${response.status}`;
    console.error(errorMsg);
    throw new ApiUnavailableError(errorMsg);
  }

  return await response.json();
}

/**
 * Analyze infant cry audio using HeAR
 */
export async function analyzeCry(audioUri: string): Promise<CryResult> {
  const base64Audio = await fileToBase64(audioUri);

  const response = await fetch(`${API_BASE_URL}/api/cry/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      audio: base64Audio,
      model: 'hear',
    }),
  });

  if (!response.ok) {
    const errorMsg = `Cry analysis failed: HTTP ${response.status}`;
    console.error(errorMsg);
    throw new ApiUnavailableError(errorMsg);
  }

  return await response.json();
}

/**
 * Run combined assessment using all HAI-DEF models and MedGemma synthesis
 */
export async function runCombinedAssessment(data: AssessmentData): Promise<CombinedResult> {
  const payload: Record<string, string> = {};

  if (data.conjunctivaImage) {
    payload.conjunctiva_image = await fileToBase64(data.conjunctivaImage);
  }
  if (data.skinImage) {
    payload.skin_image = await fileToBase64(data.skinImage);
  }
  if (data.cryAudio) {
    payload.cry_audio = await fileToBase64(data.cryAudio);
  }

  const response = await fetch(`${API_BASE_URL}/api/combined/assess`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ...payload,
      synthesizer: 'medgemma',
    }),
  });

  if (!response.ok) {
    const errorMsg = `Combined assessment failed: HTTP ${response.status}`;
    console.error(errorMsg);
    throw new ApiUnavailableError(errorMsg);
  }

  return await response.json();
}

/**
 * Synthesize clinical findings using MedGemma
 *
 * When useLocalFallback is true and API fails, returns deterministic
 * rule-based synthesis. Otherwise throws ApiUnavailableError.
 */
export async function synthesize(
  input: SynthesisInput,
  useLocalFallback = false
): Promise<SynthesisResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/synthesize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...input,
        model: 'medgemma',
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Synthesis error:', error);

    // Only use local fallback if explicitly requested (for offline mode)
    if (!useLocalFallback) {
      throw new ApiUnavailableError(
        `MedGemma synthesis unavailable: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }

    // Deterministic rule-based synthesis (WHO IMNCI protocols)
    return synthesizeLocally(input);
  }
}

/**
 * Local rule-based synthesis following WHO IMNCI protocols.
 * Used only when explicitly requested for offline mode.
 */
function synthesizeLocally(input: SynthesisInput): SynthesisResult {
  const hasDangerSigns = input.danger_signs.length > 0;
  const hasAnemia = input.anemia_result?.is_anemic;
  const hasJaundice = input.jaundice_result?.has_jaundice;
  const hasAbnormalCry = input.cry_result?.is_abnormal;

  const findings: string[] = [];
  const actions: string[] = [];

  if (hasDangerSigns) {
    findings.push(`Danger signs present: ${input.danger_signs.join(', ')}`);
    actions.push('Immediate clinical evaluation required');
  }

  if (hasAnemia) {
    findings.push('Anemia detected based on conjunctival pallor assessment');
    actions.push('Consider hemoglobin testing and iron supplementation');
  }

  if (hasJaundice) {
    const needsPhoto = input.jaundice_result?.needs_phototherapy;
    findings.push(`Neonatal jaundice detected${needsPhoto ? ' requiring phototherapy' : ''}`);
    if (needsPhoto) {
      actions.push('Initiate phototherapy');
    }
    actions.push('Monitor bilirubin levels');
  }

  if (hasAbnormalCry) {
    const risk = input.cry_result?.asphyxia_risk ?? 0;
    findings.push(`Abnormal cry pattern detected with ${(risk * 100).toFixed(0)}% asphyxia risk`);
    actions.push('Neurological assessment recommended');
  }

  if (findings.length === 0) {
    findings.push('No significant abnormalities detected in screening');
  }

  if (actions.length === 0) {
    actions.push('Continue routine care and monitoring');
  }

  const severity = hasDangerSigns || (hasAbnormalCry && (input.cry_result?.asphyxia_risk ?? 0) > 0.5)
    ? 'RED'
    : (hasAnemia || hasJaundice || hasAbnormalCry)
      ? 'YELLOW'
      : 'GREEN';

  return {
    synthesis: findings.join('. ') + '. [Rule-based synthesis - MedGemma unavailable]',
    recommendation: actions[0],
    immediate_actions: actions,
    severity,
    referral_needed: severity === 'RED' || (severity === 'YELLOW' && input.jaundice_result?.needs_phototherapy === true),
    confidence: 0.65, // Lower confidence for rule-based
  };
}

/**
 * Check API health status
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      timeout: 5000,
    } as RequestInit);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * WHO IMNCI Protocol response type
 */
export interface ProtocolResult {
  name: string;
  source: string;
  condition: string;
  steps: string[];
  referral_criteria: string;
  warning_signs: string[];
}

/**
 * Get WHO IMNCI protocol for a specific condition
 *
 * Available conditions:
 * - anemia: Maternal anemia management
 * - jaundice: Neonatal jaundice management
 * - asphyxia: Birth asphyxia management
 * - danger_signs_newborn: Newborn danger signs
 * - danger_signs_pregnant: Pregnancy danger signs
 */
export async function getProtocol(condition: string): Promise<ProtocolResult> {
  const response = await fetch(`${API_BASE_URL}/api/protocol/${encodeURIComponent(condition)}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorMsg = `Protocol fetch failed: HTTP ${response.status}`;
    console.error(errorMsg);
    throw new ApiUnavailableError(errorMsg);
  }

  return await response.json();
}

/**
 * List all available WHO IMNCI protocols
 */
export async function listProtocols(): Promise<{ protocols: Array<{ condition: string; name: string; source: string }>; count: number }> {
  const response = await fetch(`${API_BASE_URL}/api/protocols`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorMsg = `Protocols list fetch failed: HTTP ${response.status}`;
    console.error(errorMsg);
    throw new ApiUnavailableError(errorMsg);
  }

  return await response.json();
}

/**
 * Get information about available HAI-DEF models
 */
export async function getModelsInfo(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE_URL}/api/models`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorMsg = `Models info fetch failed: HTTP ${response.status}`;
    console.error(errorMsg);
    throw new ApiUnavailableError(errorMsg);
  }

  return await response.json();
}

// Named export for easy importing
export const nexusApi = {
  analyzeAnemia,
  analyzeJaundice,
  analyzeCry,
  runCombinedAssessment,
  synthesize,
  checkApiHealth,
  getProtocol,
  listProtocols,
  getModelsInfo,
};
