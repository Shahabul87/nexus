/**
 * Results Screen
 *
 * Displays analysis results from HAI-DEF models
 * Shows severity, recommendations, and referral decisions
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';

type ResultsScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Results'>;
  route: RouteProp<RootStackParamList, 'Results'>;
};

interface DangerSign {
  id: string;
  label: string;
  severity: 'critical' | 'high' | 'medium';
  present: boolean;
}

interface PatientInfo {
  gestationalWeeks?: number;
  gravida?: number;
  para?: number;
  ageHours?: number;
  birthWeight?: number;
  deliveryType?: string;
  apgarScore?: number;
}

interface ResultData {
  is_anemic?: boolean;
  has_jaundice?: boolean;
  is_abnormal?: boolean;
  confidence?: number;
  risk_level?: string;
  severity?: string;
  severity_level?: string;
  estimated_hemoglobin?: number;
  estimated_bilirubin?: number;
  asphyxia_risk?: number;
  cry_type?: string;
  needs_phototherapy?: boolean;
  recommendation?: string;
  summary?: string;
  immediate_actions?: string[];
  referral_needed?: boolean;
  referral_urgency?: string;
  follow_up?: string;
  // Comprehensive assessment fields
  patientInfo?: PatientInfo;
  dangerSigns?: DangerSign[];
  whoClassification?: 'RED' | 'YELLOW' | 'GREEN';
  clinicalSynthesis?: string;
  medgemmaAnalysis?: string;
  anemiaScreening?: {
    isAnemic: boolean;
    confidence: number;
    estimatedHemoglobin?: number;
  };
  jaundiceScreening?: {
    hasJaundice: boolean;
    confidence: number;
    kramerZone?: number;
    estimatedBilirubin?: number;
    needsPhototherapy: boolean;
  };
  cryAnalysis?: {
    isAbnormal: boolean;
    asphyxiaRisk: number;
    cryType: string;
  };
  riskFactors?: string[];
  protectiveFactors?: string[];
}

export default function ResultsScreen({ navigation, route }: ResultsScreenProps): React.JSX.Element {
  const { type, results } = route.params;
  const data = results as ResultData;

  const getSeverityColor = (): { bg: string; text: string; border: string } => {
    const severity = data.severity_level || data.risk_level || data.severity || 'low';
    switch (severity.toLowerCase()) {
      case 'red':
      case 'high':
      case 'critical':
      case 'severe':
        return { bg: '#f8d7da', text: '#721c24', border: '#f5c6cb' };
      case 'yellow':
      case 'medium':
      case 'moderate':
        return { bg: '#fff3cd', text: '#856404', border: '#ffeeba' };
      default:
        return { bg: '#d4edda', text: '#155724', border: '#c3e6cb' };
    }
  };

  const getSeverityIcon = (): keyof typeof Ionicons.glyphMap => {
    const severity = data.severity_level || data.risk_level || data.severity || 'low';
    switch (severity.toLowerCase()) {
      case 'red':
      case 'high':
      case 'critical':
      case 'severe':
        return 'alert-circle';
      case 'yellow':
      case 'medium':
      case 'moderate':
        return 'warning';
      default:
        return 'checkmark-circle';
    }
  };

  const colors = getSeverityColor();

  const renderAnemiaResults = (): React.JSX.Element => (
    <View style={styles.metricsContainer}>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Anemia Detected</Text>
        <Text style={[styles.metricValue, { color: data.is_anemic ? '#e74c3c' : '#28a745' }]}>
          {data.is_anemic ? 'Yes' : 'No'}
        </Text>
      </View>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Confidence</Text>
        <Text style={styles.metricValue}>
          {((data.confidence || 0) * 100).toFixed(0)}%
        </Text>
      </View>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Est. Hemoglobin</Text>
        <Text style={styles.metricValue}>
          {data.estimated_hemoglobin || 'N/A'} g/dL
        </Text>
      </View>
    </View>
  );

  const renderJaundiceResults = (): React.JSX.Element => (
    <View style={styles.metricsContainer}>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Jaundice Detected</Text>
        <Text style={[styles.metricValue, { color: data.has_jaundice ? '#f39c12' : '#28a745' }]}>
          {data.has_jaundice ? 'Yes' : 'No'}
        </Text>
      </View>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Est. Bilirubin</Text>
        <Text style={styles.metricValue}>
          {data.estimated_bilirubin || 'N/A'} mg/dL
        </Text>
      </View>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Phototherapy</Text>
        <Text style={[styles.metricValue, { color: data.needs_phototherapy ? '#e74c3c' : '#28a745' }]}>
          {data.needs_phototherapy ? 'Needed' : 'Not needed'}
        </Text>
      </View>
    </View>
  );

  const renderCryResults = (): React.JSX.Element => (
    <View style={styles.metricsContainer}>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Abnormal Cry</Text>
        <Text style={[styles.metricValue, { color: data.is_abnormal ? '#e74c3c' : '#28a745' }]}>
          {data.is_abnormal ? 'Yes' : 'No'}
        </Text>
      </View>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Asphyxia Risk</Text>
        <Text style={styles.metricValue}>
          {((data.asphyxia_risk || 0) * 100).toFixed(0)}%
        </Text>
      </View>
      <View style={styles.metricCard}>
        <Text style={styles.metricLabel}>Cry Type</Text>
        <Text style={styles.metricValue}>
          {data.cry_type || 'Unknown'}
        </Text>
      </View>
    </View>
  );

  const renderPatientInfo = (): React.JSX.Element | null => {
    if (!data.patientInfo) return null;
    const info = data.patientInfo;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Patient Information</Text>
        <View style={styles.infoGrid}>
          {info.gestationalWeeks && (
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Gestational Age</Text>
              <Text style={styles.infoValue}>{info.gestationalWeeks} weeks</Text>
            </View>
          )}
          {info.gravida !== undefined && (
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Gravida</Text>
              <Text style={styles.infoValue}>{info.gravida}</Text>
            </View>
          )}
          {info.para !== undefined && (
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Para</Text>
              <Text style={styles.infoValue}>{info.para}</Text>
            </View>
          )}
          {info.ageHours !== undefined && (
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Age</Text>
              <Text style={styles.infoValue}>{info.ageHours} hours</Text>
            </View>
          )}
          {info.birthWeight && (
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Birth Weight</Text>
              <Text style={styles.infoValue}>{info.birthWeight} g</Text>
            </View>
          )}
          {info.deliveryType && (
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Delivery</Text>
              <Text style={styles.infoValue}>{info.deliveryType}</Text>
            </View>
          )}
          {info.apgarScore !== undefined && (
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>APGAR Score</Text>
              <Text style={styles.infoValue}>{info.apgarScore}/10</Text>
            </View>
          )}
        </View>
      </View>
    );
  };

  const renderDangerSigns = (): React.JSX.Element | null => {
    if (!data.dangerSigns || data.dangerSigns.length === 0) return null;
    const presentSigns = data.dangerSigns.filter(sign => sign.present);

    if (presentSigns.length === 0) {
      return (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>WHO Danger Signs</Text>
          <View style={styles.noDangerCard}>
            <Ionicons name="checkmark-circle" size={24} color="#28a745" />
            <Text style={styles.noDangerText}>No danger signs detected</Text>
          </View>
        </View>
      );
    }

    const getSeverityColor = (severity: string): string => {
      switch (severity) {
        case 'critical': return '#dc3545';
        case 'high': return '#e74c3c';
        case 'medium': return '#f39c12';
        default: return '#666';
      }
    };

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>WHO Danger Signs Detected</Text>
        {presentSigns.map((sign) => (
          <View key={sign.id} style={styles.dangerSignItem}>
            <Ionicons
              name={sign.severity === 'critical' ? 'alert-circle' : 'warning'}
              size={20}
              color={getSeverityColor(sign.severity)}
            />
            <Text style={[styles.dangerSignText, { color: getSeverityColor(sign.severity) }]}>
              {sign.label}
            </Text>
            <View style={[styles.severityTag, { backgroundColor: getSeverityColor(sign.severity) }]}>
              <Text style={styles.severityTagText}>{sign.severity.toUpperCase()}</Text>
            </View>
          </View>
        ))}
      </View>
    );
  };

  const renderPregnantWomanResults = (): React.JSX.Element => (
    <View>
      {/* Anemia Screening Results */}
      {data.anemiaScreening && (
        <View style={styles.metricsContainer}>
          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>Anemia Status</Text>
            <Text style={[styles.metricValue, { color: data.anemiaScreening.isAnemic ? '#e74c3c' : '#28a745' }]}>
              {data.anemiaScreening.isAnemic ? 'Detected' : 'Normal'}
            </Text>
          </View>
          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>Confidence</Text>
            <Text style={styles.metricValue}>
              {(data.anemiaScreening.confidence * 100).toFixed(0)}%
            </Text>
          </View>
          {data.anemiaScreening.estimatedHemoglobin && (
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Est. Hemoglobin</Text>
              <Text style={styles.metricValue}>
                {data.anemiaScreening.estimatedHemoglobin} g/dL
              </Text>
            </View>
          )}
        </View>
      )}
    </View>
  );

  const renderNewbornResults = (): React.JSX.Element => (
    <View>
      {/* Jaundice Screening Results */}
      {data.jaundiceScreening && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Jaundice Screening</Text>
          <View style={styles.metricsContainer}>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Jaundice</Text>
              <Text style={[styles.metricValue, { color: data.jaundiceScreening.hasJaundice ? '#f39c12' : '#28a745' }]}>
                {data.jaundiceScreening.hasJaundice ? 'Detected' : 'Normal'}
              </Text>
            </View>
            {data.jaundiceScreening.kramerZone && (
              <View style={styles.metricCard}>
                <Text style={styles.metricLabel}>Kramer Zone</Text>
                <Text style={styles.metricValue}>
                  Zone {data.jaundiceScreening.kramerZone}
                </Text>
              </View>
            )}
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Phototherapy</Text>
              <Text style={[styles.metricValue, { color: data.jaundiceScreening.needsPhototherapy ? '#e74c3c' : '#28a745' }]}>
                {data.jaundiceScreening.needsPhototherapy ? 'Needed' : 'Not needed'}
              </Text>
            </View>
          </View>
        </View>
      )}

      {/* Cry Analysis Results */}
      {data.cryAnalysis && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Cry Analysis</Text>
          <View style={styles.metricsContainer}>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Cry Pattern</Text>
              <Text style={[styles.metricValue, { color: data.cryAnalysis.isAbnormal ? '#e74c3c' : '#28a745' }]}>
                {data.cryAnalysis.isAbnormal ? 'Abnormal' : 'Normal'}
              </Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Asphyxia Risk</Text>
              <Text style={styles.metricValue}>
                {(data.cryAnalysis.asphyxiaRisk * 100).toFixed(0)}%
              </Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>Cry Type</Text>
              <Text style={styles.metricValue}>
                {data.cryAnalysis.cryType}
              </Text>
            </View>
          </View>
        </View>
      )}
    </View>
  );

  const renderClinicalSynthesis = (): React.JSX.Element | null => {
    if (!data.clinicalSynthesis && !data.medgemmaAnalysis) return null;

    return (
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>AI Clinical Synthesis</Text>
        <View style={styles.synthesisCard}>
          <View style={styles.synthesisHeader}>
            <Ionicons name="sparkles" size={20} color="#1a73e8" />
            <Text style={styles.synthesisLabel}>MedGemma Analysis</Text>
          </View>
          <Text style={styles.synthesisText}>
            {data.clinicalSynthesis || data.medgemmaAnalysis}
          </Text>
        </View>
      </View>
    );
  };

  const renderRiskFactors = (): React.JSX.Element | null => {
    if ((!data.riskFactors || data.riskFactors.length === 0) &&
        (!data.protectiveFactors || data.protectiveFactors.length === 0)) return null;

    return (
      <View style={styles.section}>
        {data.riskFactors && data.riskFactors.length > 0 && (
          <View style={styles.factorsSection}>
            <Text style={styles.factorsTitle}>Risk Factors</Text>
            {data.riskFactors.map((factor, index) => (
              <View key={index} style={styles.riskFactorItem}>
                <Ionicons name="alert" size={16} color="#e74c3c" />
                <Text style={styles.riskFactorText}>{factor}</Text>
              </View>
            ))}
          </View>
        )}

        {data.protectiveFactors && data.protectiveFactors.length > 0 && (
          <View style={[styles.factorsSection, { marginTop: 12 }]}>
            <Text style={styles.factorsTitle}>Protective Factors</Text>
            {data.protectiveFactors.map((factor, index) => (
              <View key={index} style={styles.protectiveFactorItem}>
                <Ionicons name="shield-checkmark" size={16} color="#28a745" />
                <Text style={styles.protectiveFactorText}>{factor}</Text>
              </View>
            ))}
          </View>
        )}
      </View>
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Severity Banner */}
      <View style={[styles.severityBanner, { backgroundColor: colors.bg, borderColor: colors.border }]}>
        <Ionicons name={getSeverityIcon()} size={32} color={colors.text} />
        <View style={styles.severityContent}>
          <Text style={[styles.severityTitle, { color: colors.text }]}>
            {(data.severity_level || data.risk_level || data.severity || 'Low').toUpperCase()}
          </Text>
          <Text style={[styles.severitySubtitle, { color: colors.text }]}>
            {type === 'anemia' && 'Anemia Screening Result'}
            {type === 'jaundice' && 'Jaundice Detection Result'}
            {type === 'cry' && 'Cry Analysis Result'}
            {type === 'combined' && 'Combined Assessment Result'}
            {type === 'pregnant' && 'Maternal Assessment Result'}
            {type === 'newborn' && 'Newborn Assessment Result'}
          </Text>
        </View>
      </View>

      {/* Patient Info for comprehensive assessments */}
      {(type === 'pregnant' || type === 'newborn') && renderPatientInfo()}

      {/* WHO Danger Signs for comprehensive assessments */}
      {(type === 'pregnant' || type === 'newborn') && renderDangerSigns()}

      {/* Type-specific metrics */}
      {type === 'anemia' && renderAnemiaResults()}
      {type === 'jaundice' && renderJaundiceResults()}
      {type === 'cry' && renderCryResults()}
      {type === 'pregnant' && renderPregnantWomanResults()}
      {type === 'newborn' && renderNewbornResults()}

      {/* Clinical Synthesis for comprehensive assessments */}
      {(type === 'pregnant' || type === 'newborn') && renderClinicalSynthesis()}

      {/* Risk Factors */}
      {(type === 'pregnant' || type === 'newborn') && renderRiskFactors()}

      {/* Recommendation */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Recommendation</Text>
        <View style={styles.recommendationCard}>
          <Ionicons name="medical" size={24} color="#1a73e8" />
          <Text style={styles.recommendationText}>
            {data.recommendation || data.summary || 'No recommendation available'}
          </Text>
        </View>
      </View>

      {/* Immediate Actions */}
      {data.immediate_actions && data.immediate_actions.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Immediate Actions</Text>
          {data.immediate_actions.map((action, index) => (
            <View key={index} style={styles.actionItem}>
              <Ionicons name="arrow-forward-circle" size={20} color="#1a73e8" />
              <Text style={styles.actionText}>{action}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Referral Decision */}
      {data.referral_needed !== undefined && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Referral Status</Text>
          <View style={[
            styles.referralCard,
            { backgroundColor: data.referral_needed ? '#f8d7da' : '#d4edda' }
          ]}>
            <Ionicons
              name={data.referral_needed ? 'alert-circle' : 'checkmark-circle'}
              size={24}
              color={data.referral_needed ? '#721c24' : '#155724'}
            />
            <View style={styles.referralContent}>
              <Text style={[
                styles.referralTitle,
                { color: data.referral_needed ? '#721c24' : '#155724' }
              ]}>
                {data.referral_needed ? 'REFERRAL NEEDED' : 'No Referral Needed'}
              </Text>
              {data.referral_urgency && (
                <Text style={[
                  styles.referralUrgency,
                  { color: data.referral_needed ? '#721c24' : '#155724' }
                ]}>
                  Urgency: {data.referral_urgency.toUpperCase()}
                </Text>
              )}
            </View>
          </View>
        </View>
      )}

      {/* Follow-up */}
      {data.follow_up && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Follow-up Plan</Text>
          <View style={styles.followUpCard}>
            <Ionicons name="calendar" size={24} color="#666" />
            <Text style={styles.followUpText}>{data.follow_up}</Text>
          </View>
        </View>
      )}

      {/* Action Buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => navigation.navigate('Home')}
        >
          <Ionicons name="home" size={24} color="#fff" />
          <Text style={styles.primaryButtonText}>New Assessment</Text>
        </TouchableOpacity>
      </View>

      {/* Disclaimer */}
      <View style={styles.disclaimer}>
        <Ionicons name="information-circle-outline" size={16} color="#999" />
        <Text style={styles.disclaimerText}>
          This is a screening result only. Always confirm with laboratory tests.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    padding: 16,
  },
  severityBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    marginBottom: 20,
  },
  severityContent: {
    marginLeft: 12,
    flex: 1,
  },
  severityTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  severitySubtitle: {
    fontSize: 14,
    marginTop: 2,
  },
  metricsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 20,
    gap: 8,
  },
  metricCard: {
    flex: 1,
    minWidth: '30%',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
    textAlign: 'center',
  },
  metricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  recommendationCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#e8f4fd',
    padding: 16,
    borderRadius: 12,
  },
  recommendationText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  actionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  actionText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 14,
    color: '#333',
  },
  referralCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
  },
  referralContent: {
    marginLeft: 12,
    flex: 1,
  },
  referralTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  referralUrgency: {
    fontSize: 14,
    marginTop: 4,
  },
  followUpCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
  },
  followUpText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 14,
    color: '#333',
  },
  buttonContainer: {
    marginTop: 20,
    marginBottom: 16,
  },
  primaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a73e8',
    paddingVertical: 16,
    borderRadius: 12,
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  disclaimer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
  },
  disclaimerText: {
    marginLeft: 8,
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
    flex: 1,
  },
  // Patient Info Styles
  infoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
  },
  infoItem: {
    width: '50%',
    paddingVertical: 8,
    paddingHorizontal: 4,
  },
  infoLabel: {
    fontSize: 11,
    color: '#666',
    marginBottom: 2,
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  // Danger Signs Styles
  noDangerCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#d4edda',
    padding: 16,
    borderRadius: 12,
  },
  noDangerText: {
    marginLeft: 12,
    fontSize: 14,
    color: '#155724',
    fontWeight: '500',
  },
  dangerSignItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  dangerSignText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 14,
    fontWeight: '500',
  },
  severityTag: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  severityTagText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#fff',
  },
  // Clinical Synthesis Styles
  synthesisCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#1a73e8',
  },
  synthesisHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  synthesisLabel: {
    marginLeft: 8,
    fontSize: 14,
    fontWeight: '600',
    color: '#1a73e8',
  },
  synthesisText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 22,
  },
  // Risk Factors Styles
  factorsSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  factorsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  riskFactorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
  },
  riskFactorText: {
    marginLeft: 8,
    fontSize: 13,
    color: '#e74c3c',
  },
  protectiveFactorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
  },
  protectiveFactorText: {
    marginLeft: 8,
    fontSize: 13,
    color: '#28a745',
  },
});
