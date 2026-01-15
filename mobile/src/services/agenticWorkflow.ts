/**
 * Agentic Workflow Engine
 *
 * Orchestrates multi-agent system for comprehensive maternal-neonatal assessments
 * Based on WHO IMNCI protocols with HAI-DEF model integration
 *
 * Agents:
 * - Triage Agent: Initial risk stratification
 * - Image Agent: MedSigLIP-powered visual analysis
 * - Audio Agent: HeAR-powered acoustic analysis
 * - Protocol Agent: WHO IMNCI guideline application
 * - Referral Agent: Decision synthesis and routing
 */

import { edgeAI, AnemiaResult, JaundiceResult, CryResult } from './edgeAI';
import { nexusApi, SynthesisResult } from './nexusApi';

// ============================================================================
// Types
// ============================================================================

export type PatientType = 'pregnant' | 'newborn';

export type WorkflowState =
  | 'idle'
  | 'triaging'
  | 'analyzing_image'
  | 'analyzing_audio'
  | 'applying_protocol'
  | 'synthesizing'
  | 'complete'
  | 'error';

export type SeverityLevel = 'RED' | 'YELLOW' | 'GREEN';

export interface DangerSign {
  id: string;
  label: string;
  severity: 'critical' | 'high' | 'medium';
  present: boolean;
}

export interface PatientInfo {
  // Common
  patientId?: string;

  // Maternal
  gestationalWeeks?: number;
  gravida?: number;
  para?: number;
  lastMenstrualPeriod?: string;

  // Newborn
  ageHours?: number;
  birthWeight?: number;
  deliveryType?: 'normal' | 'cesarean' | 'assisted';
  apgarScore?: number;
  gestationalAgeAtBirth?: number;
}

export interface WorkflowInput {
  patientType: PatientType;
  patientInfo: PatientInfo;
  dangerSigns: DangerSign[];
  conjunctivaImageUri?: string;  // For anemia screening
  skinImageUri?: string;          // For jaundice detection
  cryAudioUri?: string;           // For cry analysis
  additionalNotes?: string;
}

export interface AgentResult {
  agentName: string;
  success: boolean;
  data?: Record<string, unknown>;
  error?: string;
  processingTimeMs: number;
}

export interface WorkflowResult {
  success: boolean;
  patientType: PatientType;
  whoClassification: SeverityLevel;

  // Agent results
  triageResult: TriageResult;
  imageResults: ImageAnalysisResult;
  audioResults?: AudioAnalysisResult;
  protocolResult: ProtocolResult;
  referralResult: ReferralResult;

  // Synthesis
  clinicalSynthesis: string;
  recommendation: string;
  immediateActions: string[];

  // Metadata
  processingTimeMs: number;
  agentResults: AgentResult[];
  timestamp: string;
}

export interface TriageResult {
  riskLevel: SeverityLevel;
  criticalSignsDetected: boolean;
  criticalSigns: string[];
  immediateReferralNeeded: boolean;
  score: number;
}

export interface ImageAnalysisResult {
  anemia?: AnemiaResult;
  jaundice?: JaundiceResult;
}

export interface AudioAnalysisResult {
  cry?: CryResult;
}

export interface ProtocolResult {
  classification: SeverityLevel;
  applicableProtocols: string[];
  treatmentRecommendations: string[];
  followUpSchedule: string;
}

export interface ReferralResult {
  referralNeeded: boolean;
  urgency: 'immediate' | 'urgent' | 'routine' | 'none';
  facilityLevel: 'primary' | 'secondary' | 'tertiary';
  reason: string;
  timeframe: string;
}

// ============================================================================
// Workflow State Machine
// ============================================================================

type WorkflowCallback = (state: WorkflowState, progress: number) => void;

class WorkflowStateMachine {
  private state: WorkflowState = 'idle';
  private onStateChange?: WorkflowCallback;

  constructor(callback?: WorkflowCallback) {
    this.onStateChange = callback;
  }

  transition(newState: WorkflowState, progress: number): void {
    this.state = newState;
    this.onStateChange?.(this.state, progress);
  }

  getState(): WorkflowState {
    return this.state;
  }
}

// ============================================================================
// Individual Agents
// ============================================================================

/**
 * Triage Agent
 * Performs initial risk stratification based on danger signs
 */
class TriageAgent {
  async process(
    patientType: PatientType,
    dangerSigns: DangerSign[],
    patientInfo: PatientInfo
  ): Promise<TriageResult> {
    // Timestamp for future metrics tracking
    void Date.now();

    // Calculate risk score
    let score = 0;
    const criticalSigns: string[] = [];

    for (const sign of dangerSigns) {
      if (sign.present) {
        switch (sign.severity) {
          case 'critical':
            score += 30;
            criticalSigns.push(sign.label);
            break;
          case 'high':
            score += 15;
            criticalSigns.push(sign.label);
            break;
          case 'medium':
            score += 5;
            break;
        }
      }
    }

    // Additional risk factors from patient info
    if (patientType === 'pregnant') {
      if (patientInfo.gestationalWeeks && patientInfo.gestationalWeeks < 28) {
        score += 10; // Preterm risk
      }
      if (patientInfo.gestationalWeeks && patientInfo.gestationalWeeks > 42) {
        score += 15; // Post-term risk
      }
    } else if (patientType === 'newborn') {
      if (patientInfo.birthWeight && patientInfo.birthWeight < 2500) {
        score += 10; // Low birth weight
      }
      if (patientInfo.apgarScore !== undefined && patientInfo.apgarScore < 7) {
        score += 15; // Low APGAR
      }
      if (patientInfo.ageHours !== undefined && patientInfo.ageHours < 24) {
        score += 5; // First day of life
      }
    }

    // Determine risk level
    let riskLevel: SeverityLevel;
    if (score >= 30 || criticalSigns.length > 0) {
      riskLevel = 'RED';
    } else if (score >= 15) {
      riskLevel = 'YELLOW';
    } else {
      riskLevel = 'GREEN';
    }

    const criticalSignsDetected = criticalSigns.length > 0;
    const immediateReferralNeeded = riskLevel === 'RED' && criticalSignsDetected;

    return {
      riskLevel,
      criticalSignsDetected,
      criticalSigns,
      immediateReferralNeeded,
      score,
    };
  }
}

/**
 * Image Analysis Agent
 * Uses MedSigLIP for visual analysis (anemia, jaundice)
 */
class ImageAnalysisAgent {
  async process(
    patientType: PatientType,
    conjunctivaImageUri?: string,
    skinImageUri?: string
  ): Promise<ImageAnalysisResult> {
    const results: ImageAnalysisResult = {};

    // Anemia screening (both maternal and newborn)
    if (conjunctivaImageUri) {
      try {
        results.anemia = await edgeAI.analyzeAnemia(conjunctivaImageUri);
      } catch (error) {
        console.error('Anemia analysis error:', error);
        results.anemia = {
          is_anemic: false,
          confidence: 0,
          risk_level: 'low',
          estimated_hemoglobin: 0,
          recommendation: 'Analysis failed - please retry',
          anemia_score: 0,
          healthy_score: 0,
          inference_mode: 'edge',
        };
      }
    }

    // Jaundice detection (newborn only)
    if (patientType === 'newborn' && skinImageUri) {
      try {
        results.jaundice = await edgeAI.analyzeJaundice(skinImageUri);
      } catch (error) {
        console.error('Jaundice analysis error:', error);
        results.jaundice = {
          has_jaundice: false,
          confidence: 0,
          severity: 'none',
          estimated_bilirubin: 0,
          needs_phototherapy: false,
          recommendation: 'Analysis failed - please retry',
          kramer_zone: 0,
          inference_mode: 'edge',
        };
      }
    }

    return results;
  }
}

/**
 * Audio Analysis Agent
 * Uses HeAR for acoustic analysis (cry patterns)
 */
class AudioAnalysisAgent {
  async process(cryAudioUri?: string): Promise<AudioAnalysisResult> {
    const results: AudioAnalysisResult = {};

    if (cryAudioUri) {
      try {
        results.cry = await edgeAI.analyzeCry(cryAudioUri);
      } catch (error) {
        console.error('Cry analysis error:', error);
        results.cry = {
          is_abnormal: false,
          asphyxia_risk: 0,
          cry_type: 'unknown',
          risk_level: 'low',
          recommendation: 'Analysis failed - please retry',
          inference_mode: 'edge',
        };
      }
    }

    return results;
  }
}

/**
 * Protocol Agent
 * Applies WHO IMNCI guidelines
 */
class ProtocolAgent {
  async process(
    patientType: PatientType,
    triageResult: TriageResult,
    imageResults: ImageAnalysisResult,
    audioResults?: AudioAnalysisResult
  ): Promise<ProtocolResult> {
    const applicableProtocols: string[] = [];
    const treatmentRecommendations: string[] = [];
    let classification: SeverityLevel = triageResult.riskLevel;

    // Maternal protocols
    if (patientType === 'pregnant') {
      applicableProtocols.push('WHO IMNCI Maternal Care');

      if (imageResults.anemia?.is_anemic) {
        applicableProtocols.push('Anemia Management Protocol');
        treatmentRecommendations.push('Initiate iron supplementation (60mg elemental iron + 400mcg folic acid daily)');
        treatmentRecommendations.push('Dietary counseling for iron-rich foods');

        if (imageResults.anemia.estimated_hemoglobin &&
            imageResults.anemia.estimated_hemoglobin < 7) {
          classification = 'RED';
          treatmentRecommendations.push('URGENT: Severe anemia - consider blood transfusion');
        } else if (classification !== 'RED') {
          classification = 'YELLOW';
        }
      }

      // Add danger sign protocols
      if (triageResult.criticalSignsDetected) {
        applicableProtocols.push('Emergency Obstetric Care Protocol');
        treatmentRecommendations.push('Immediate assessment for emergency obstetric conditions');
      }
    }

    // Newborn protocols
    if (patientType === 'newborn') {
      applicableProtocols.push('WHO IMNCI Newborn Care');

      // Jaundice protocol
      if (imageResults.jaundice?.has_jaundice) {
        applicableProtocols.push('Neonatal Jaundice Protocol');

        if (imageResults.jaundice.needs_phototherapy) {
          classification = classification === 'RED' ? 'RED' : 'YELLOW';
          treatmentRecommendations.push('Initiate phototherapy');
          treatmentRecommendations.push('Monitor bilirubin levels every 6-12 hours');
        } else {
          treatmentRecommendations.push('Continue breastfeeding');
          treatmentRecommendations.push('Monitor for increasing jaundice');
        }

        // Check for severe jaundice
        if (imageResults.jaundice.estimated_bilirubin &&
            imageResults.jaundice.estimated_bilirubin > 20) {
          classification = 'RED';
          treatmentRecommendations.push('URGENT: Severe hyperbilirubinemia - consider exchange transfusion');
        }
      }

      // Cry analysis protocol
      if (audioResults?.cry?.is_abnormal) {
        applicableProtocols.push('Birth Asphyxia Assessment Protocol');

        if (audioResults.cry.asphyxia_risk > 0.7) {
          classification = 'RED';
          treatmentRecommendations.push('URGENT: High asphyxia risk - immediate neonatal assessment');
        } else if (audioResults.cry.asphyxia_risk > 0.4) {
          classification = classification === 'RED' ? 'RED' : 'YELLOW';
          treatmentRecommendations.push('Monitor neurological status');
          treatmentRecommendations.push('Consider head ultrasound');
        }
      }

      // Anemia in newborn
      if (imageResults.anemia?.is_anemic) {
        applicableProtocols.push('Neonatal Anemia Protocol');
        treatmentRecommendations.push('Check hematocrit and reticulocyte count');
        classification = classification === 'RED' ? 'RED' : 'YELLOW';
      }
    }

    // Determine follow-up schedule based on classification
    let followUpSchedule: string;
    switch (classification) {
      case 'RED':
        followUpSchedule = 'Immediate referral - no follow-up at this level';
        break;
      case 'YELLOW':
        followUpSchedule = 'Follow-up in 2-3 days or sooner if condition worsens';
        break;
      default:
        followUpSchedule = patientType === 'newborn'
          ? 'Routine follow-up in 1 week'
          : 'Routine antenatal follow-up as scheduled';
    }

    return {
      classification,
      applicableProtocols,
      treatmentRecommendations,
      followUpSchedule,
    };
  }
}

/**
 * Referral Agent
 * Synthesizes all results to determine referral decision
 */
class ReferralAgent {
  async process(
    patientType: PatientType,
    triageResult: TriageResult,
    protocolResult: ProtocolResult,
    imageResults: ImageAnalysisResult,
    audioResults?: AudioAnalysisResult
  ): Promise<ReferralResult> {
    let referralNeeded = false;
    let urgency: ReferralResult['urgency'] = 'none';
    let facilityLevel: ReferralResult['facilityLevel'] = 'primary';
    const reasons: string[] = [];

    // Check critical danger signs
    if (triageResult.immediateReferralNeeded) {
      referralNeeded = true;
      urgency = 'immediate';
      facilityLevel = 'tertiary';
      reasons.push(`Critical danger signs: ${triageResult.criticalSigns.join(', ')}`);
    }

    // Check protocol classification
    if (protocolResult.classification === 'RED') {
      referralNeeded = true;
      if (urgency !== 'immediate') {
        urgency = 'urgent';
      }
      if (facilityLevel === 'primary') {
        facilityLevel = 'secondary';
      }
    } else if (protocolResult.classification === 'YELLOW') {
      if (!referralNeeded) {
        urgency = 'routine';
      }
    }

    // Specific condition checks
    if (patientType === 'pregnant') {
      if (imageResults.anemia?.is_anemic &&
          imageResults.anemia.estimated_hemoglobin &&
          imageResults.anemia.estimated_hemoglobin < 7) {
        referralNeeded = true;
        urgency = urgency === 'immediate' ? 'immediate' : 'urgent';
        facilityLevel = 'secondary';
        reasons.push('Severe anemia requiring blood transfusion');
      }
    }

    if (patientType === 'newborn') {
      // Severe jaundice
      if (imageResults.jaundice?.needs_phototherapy) {
        referralNeeded = true;
        urgency = urgency === 'immediate' ? 'immediate' : 'urgent';
        facilityLevel = facilityLevel === 'tertiary' ? 'tertiary' : 'secondary';
        reasons.push('Jaundice requiring phototherapy');
      }

      // High asphyxia risk
      if (audioResults?.cry?.asphyxia_risk &&
          audioResults.cry.asphyxia_risk > 0.7) {
        referralNeeded = true;
        urgency = 'immediate';
        facilityLevel = 'tertiary';
        reasons.push('High birth asphyxia risk');
      }
    }

    // Determine timeframe
    let timeframe: string;
    switch (urgency) {
      case 'immediate':
        timeframe = 'Within 1 hour';
        break;
      case 'urgent':
        timeframe = 'Within 4-6 hours';
        break;
      case 'routine':
        timeframe = 'Within 24-48 hours';
        break;
      default:
        timeframe = 'Not applicable';
    }

    return {
      referralNeeded,
      urgency,
      facilityLevel,
      reason: reasons.length > 0 ? reasons.join('; ') : 'No referral required',
      timeframe,
    };
  }
}

// ============================================================================
// Main Workflow Engine
// ============================================================================

export class AgenticWorkflowEngine {
  private triageAgent: TriageAgent;
  private imageAgent: ImageAnalysisAgent;
  private audioAgent: AudioAnalysisAgent;
  private protocolAgent: ProtocolAgent;
  private referralAgent: ReferralAgent;
  private stateMachine: WorkflowStateMachine;

  constructor(onStateChange?: WorkflowCallback) {
    this.triageAgent = new TriageAgent();
    this.imageAgent = new ImageAnalysisAgent();
    this.audioAgent = new AudioAnalysisAgent();
    this.protocolAgent = new ProtocolAgent();
    this.referralAgent = new ReferralAgent();
    this.stateMachine = new WorkflowStateMachine(onStateChange);
  }

  async execute(input: WorkflowInput): Promise<WorkflowResult> {
    const startTime = Date.now();
    const agentResults: AgentResult[] = [];

    try {
      // Step 1: Triage
      this.stateMachine.transition('triaging', 10);
      const triageStart = Date.now();
      const triageResult = await this.triageAgent.process(
        input.patientType,
        input.dangerSigns,
        input.patientInfo
      );
      agentResults.push({
        agentName: 'TriageAgent',
        success: true,
        data: triageResult as unknown as Record<string, unknown>,
        processingTimeMs: Date.now() - triageStart,
      });

      // Early exit for critical cases
      if (triageResult.immediateReferralNeeded) {
        this.stateMachine.transition('complete', 100);
        return this.buildEarlyReferralResult(input, triageResult, startTime, agentResults);
      }

      // Step 2: Image Analysis
      this.stateMachine.transition('analyzing_image', 30);
      const imageStart = Date.now();
      const imageResults = await this.imageAgent.process(
        input.patientType,
        input.conjunctivaImageUri,
        input.skinImageUri
      );
      agentResults.push({
        agentName: 'ImageAnalysisAgent',
        success: true,
        data: imageResults as unknown as Record<string, unknown>,
        processingTimeMs: Date.now() - imageStart,
      });

      // Step 3: Audio Analysis (newborn only)
      let audioResults: AudioAnalysisResult | undefined;
      if (input.patientType === 'newborn' && input.cryAudioUri) {
        this.stateMachine.transition('analyzing_audio', 50);
        const audioStart = Date.now();
        audioResults = await this.audioAgent.process(input.cryAudioUri);
        agentResults.push({
          agentName: 'AudioAnalysisAgent',
          success: true,
          data: audioResults as unknown as Record<string, unknown>,
          processingTimeMs: Date.now() - audioStart,
        });
      }

      // Step 4: Protocol Application
      this.stateMachine.transition('applying_protocol', 70);
      const protocolStart = Date.now();
      const protocolResult = await this.protocolAgent.process(
        input.patientType,
        triageResult,
        imageResults,
        audioResults
      );
      agentResults.push({
        agentName: 'ProtocolAgent',
        success: true,
        data: protocolResult as unknown as Record<string, unknown>,
        processingTimeMs: Date.now() - protocolStart,
      });

      // Step 5: Referral Decision
      const referralStart = Date.now();
      const referralResult = await this.referralAgent.process(
        input.patientType,
        triageResult,
        protocolResult,
        imageResults,
        audioResults
      );
      agentResults.push({
        agentName: 'ReferralAgent',
        success: true,
        data: referralResult as unknown as Record<string, unknown>,
        processingTimeMs: Date.now() - referralStart,
      });

      // Step 6: Clinical Synthesis with MedGemma
      this.stateMachine.transition('synthesizing', 90);
      const synthesisStart = Date.now();
      let synthesisResult: SynthesisResult | undefined;
      try {
        synthesisResult = await nexusApi.synthesize({
          patient_type: input.patientType,
          danger_signs: input.dangerSigns.filter(s => s.present).map(s => s.label),
          anemia_result: imageResults.anemia,
          jaundice_result: imageResults.jaundice,
          cry_result: audioResults?.cry,
        });
      } catch (error) {
        console.error('Synthesis error:', error);
      }
      agentResults.push({
        agentName: 'MedGemmaSynthesis',
        success: !!synthesisResult,
        data: synthesisResult as unknown as Record<string, unknown>,
        processingTimeMs: Date.now() - synthesisStart,
      });

      // Build final result
      this.stateMachine.transition('complete', 100);

      return {
        success: true,
        patientType: input.patientType,
        whoClassification: protocolResult.classification,
        triageResult,
        imageResults,
        audioResults,
        protocolResult,
        referralResult,
        clinicalSynthesis: synthesisResult?.synthesis || this.generateFallbackSynthesis(
          input.patientType,
          triageResult,
          imageResults,
          audioResults,
          protocolResult
        ),
        recommendation: synthesisResult?.recommendation || protocolResult.treatmentRecommendations[0] || 'Continue routine care',
        immediateActions: synthesisResult?.immediate_actions || protocolResult.treatmentRecommendations,
        processingTimeMs: Date.now() - startTime,
        agentResults,
        timestamp: new Date().toISOString(),
      };

    } catch (error) {
      this.stateMachine.transition('error', 0);
      throw error;
    }
  }

  private buildEarlyReferralResult(
    input: WorkflowInput,
    triageResult: TriageResult,
    startTime: number,
    agentResults: AgentResult[]
  ): WorkflowResult {
    return {
      success: true,
      patientType: input.patientType,
      whoClassification: 'RED',
      triageResult,
      imageResults: {},
      protocolResult: {
        classification: 'RED',
        applicableProtocols: ['Emergency Referral Protocol'],
        treatmentRecommendations: ['IMMEDIATE REFERRAL REQUIRED'],
        followUpSchedule: 'After emergency care',
      },
      referralResult: {
        referralNeeded: true,
        urgency: 'immediate',
        facilityLevel: 'tertiary',
        reason: `Critical danger signs detected: ${triageResult.criticalSigns.join(', ')}`,
        timeframe: 'Immediately - within 1 hour',
      },
      clinicalSynthesis: `URGENT: Critical danger signs detected (${triageResult.criticalSigns.join(', ')}). Immediate referral to higher-level facility is required. This patient requires emergency care that cannot be provided at the current level.`,
      recommendation: 'IMMEDIATE REFERRAL to tertiary care facility',
      immediateActions: [
        'Arrange emergency transport',
        'Call receiving facility',
        'Provide pre-referral treatment as per protocol',
        'Accompany patient with referral note',
      ],
      processingTimeMs: Date.now() - startTime,
      agentResults,
      timestamp: new Date().toISOString(),
    };
  }

  private generateFallbackSynthesis(
    patientType: PatientType,
    triageResult: TriageResult,
    imageResults: ImageAnalysisResult,
    audioResults: AudioAnalysisResult | undefined,
    protocolResult: ProtocolResult
  ): string {
    const parts: string[] = [];

    // Classification
    parts.push(`Classification: ${protocolResult.classification}`);

    // Triage summary
    if (triageResult.criticalSignsDetected) {
      parts.push(`Critical signs: ${triageResult.criticalSigns.join(', ')}`);
    }

    // Image analysis summary
    if (patientType === 'pregnant' && imageResults.anemia) {
      parts.push(
        imageResults.anemia.is_anemic
          ? `Anemia detected with ${(imageResults.anemia.confidence * 100).toFixed(0)}% confidence`
          : 'No anemia detected'
      );
    }

    if (patientType === 'newborn') {
      if (imageResults.jaundice) {
        parts.push(
          imageResults.jaundice.has_jaundice
            ? `Jaundice detected${imageResults.jaundice.needs_phototherapy ? ' - phototherapy recommended' : ''}`
            : 'No jaundice detected'
        );
      }
      if (audioResults?.cry) {
        parts.push(
          audioResults.cry.is_abnormal
            ? `Abnormal cry pattern detected - ${(audioResults.cry.asphyxia_risk * 100).toFixed(0)}% asphyxia risk`
            : 'Normal cry pattern'
        );
      }
    }

    // Recommendations
    if (protocolResult.treatmentRecommendations.length > 0) {
      parts.push(`Recommendations: ${protocolResult.treatmentRecommendations.slice(0, 2).join('; ')}`);
    }

    return parts.join('. ') + '.';
  }

  getState(): WorkflowState {
    return this.stateMachine.getState();
  }
}

// Export singleton instance
export const agenticWorkflow = new AgenticWorkflowEngine();

// Export factory for custom instances with state callbacks
export const createWorkflowEngine = (onStateChange?: WorkflowCallback): AgenticWorkflowEngine => {
  return new AgenticWorkflowEngine(onStateChange);
};
