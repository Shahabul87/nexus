import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

type IconName = keyof typeof Ionicons.glyphMap;

export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';
export type WHOClassification = 'GREEN' | 'YELLOW' | 'RED';

interface AnalysisCardProps {
  title: string;
  icon: IconName;
  probability?: number;
  riskLevel?: RiskLevel;
  classification?: WHOClassification;
  metrics?: Array<{ label: string; value: string | number; unit?: string }>;
  children?: React.ReactNode;
}

const riskColors: Record<RiskLevel, { bg: string; text: string; label: string }> = {
  low: { bg: '#d1e7dd', text: '#198754', label: 'Low Risk' },
  moderate: { bg: '#fff3cd', text: '#856404', label: 'Moderate Risk' },
  high: { bg: '#f8d7da', text: '#dc3545', label: 'High Risk' },
  critical: { bg: '#842029', text: '#fff', label: 'Critical' },
};

const classificationColors: Record<WHOClassification, { bg: string; text: string }> = {
  GREEN: { bg: '#198754', text: '#fff' },
  YELLOW: { bg: '#ffc107', text: '#000' },
  RED: { bg: '#dc3545', text: '#fff' },
};

export const AnalysisCard: React.FC<AnalysisCardProps> = ({
  title,
  icon,
  probability,
  riskLevel,
  classification,
  metrics,
  children,
}) => {
  const riskConfig = riskLevel ? riskColors[riskLevel] : null;
  const classConfig = classification ? classificationColors[classification] : null;

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name={icon} size={24} color="#4285f4" />
          <Text style={styles.title}>{title}</Text>
        </View>

        {classification && classConfig && (
          <View style={[styles.classification, { backgroundColor: classConfig.bg }]}>
            <Text style={[styles.classificationText, { color: classConfig.text }]}>
              {classification}
            </Text>
          </View>
        )}
      </View>

      {probability !== undefined && (
        <View style={styles.probabilityContainer}>
          <View style={styles.probabilityBar}>
            <View
              style={[
                styles.probabilityFill,
                {
                  width: `${probability * 100}%`,
                  backgroundColor:
                    probability < 0.3
                      ? '#198754'
                      : probability < 0.7
                      ? '#ffc107'
                      : '#dc3545',
                },
              ]}
            />
          </View>
          <Text style={styles.probabilityText}>
            {(probability * 100).toFixed(1)}%
          </Text>
        </View>
      )}

      {riskLevel && riskConfig && (
        <View style={[styles.riskBadge, { backgroundColor: riskConfig.bg }]}>
          <Ionicons
            name={
              riskLevel === 'critical'
                ? 'alert-circle'
                : riskLevel === 'high'
                ? 'warning'
                : 'shield-checkmark'
            }
            size={16}
            color={riskConfig.text}
          />
          <Text style={[styles.riskText, { color: riskConfig.text }]}>
            {riskConfig.label}
          </Text>
        </View>
      )}

      {metrics && metrics.length > 0 && (
        <View style={styles.metricsGrid}>
          {metrics.map((metric, index) => (
            <View key={index} style={styles.metricItem}>
              <Text style={styles.metricLabel}>{metric.label}</Text>
              <Text style={styles.metricValue}>
                {metric.value}
                {metric.unit && (
                  <Text style={styles.metricUnit}> {metric.unit}</Text>
                )}
              </Text>
            </View>
          ))}
        </View>
      )}

      {children && <View style={styles.content}>{children}</View>}
    </View>
  );
};

// Risk Score Circle Component
interface RiskScoreProps {
  score: number;
  maxScore?: number;
  size?: 'small' | 'medium' | 'large';
}

export const RiskScore: React.FC<RiskScoreProps> = ({
  score,
  maxScore = 100,
  size = 'medium',
}) => {
  const percentage = (score / maxScore) * 100;
  const sizeConfig = {
    small: { diameter: 48, fontSize: 14 },
    medium: { diameter: 72, fontSize: 20 },
    large: { diameter: 96, fontSize: 28 },
  };
  const config = sizeConfig[size];

  const getColor = () => {
    if (percentage < 30) return '#198754';
    if (percentage < 60) return '#ffc107';
    if (percentage < 80) return '#fd7e14';
    return '#dc3545';
  };

  return (
    <View
      style={[
        styles.riskCircle,
        {
          width: config.diameter,
          height: config.diameter,
          borderRadius: config.diameter / 2,
          borderColor: getColor(),
        },
      ]}
    >
      <Text style={[styles.riskScoreText, { fontSize: config.fontSize, color: getColor() }]}>
        {Math.round(score)}
      </Text>
    </View>
  );
};

// Confidence Indicator Component
interface ConfidenceIndicatorProps {
  confidence: number;
  showLabel?: boolean;
}

export const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  showLabel = true,
}) => {
  const getLevel = () => {
    if (confidence >= 0.9) return { label: 'Very High', color: '#198754' };
    if (confidence >= 0.75) return { label: 'High', color: '#28a745' };
    if (confidence >= 0.6) return { label: 'Moderate', color: '#ffc107' };
    if (confidence >= 0.4) return { label: 'Low', color: '#fd7e14' };
    return { label: 'Very Low', color: '#dc3545' };
  };

  const level = getLevel();

  return (
    <View style={styles.confidenceContainer}>
      <View style={styles.confidenceBar}>
        <View
          style={[
            styles.confidenceFill,
            { width: `${confidence * 100}%`, backgroundColor: level.color },
          ]}
        />
      </View>
      {showLabel && (
        <Text style={[styles.confidenceLabel, { color: level.color }]}>
          {level.label} ({(confidence * 100).toFixed(0)}%)
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#212529',
  },
  classification: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  classificationText: {
    fontSize: 12,
    fontWeight: '700',
  },
  probabilityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  probabilityBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    overflow: 'hidden',
  },
  probabilityFill: {
    height: '100%',
    borderRadius: 4,
  },
  probabilityText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#212529',
    minWidth: 48,
    textAlign: 'right',
  },
  riskBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  riskText: {
    fontSize: 13,
    fontWeight: '600',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 8,
  },
  metricItem: {
    minWidth: '45%',
  },
  metricLabel: {
    fontSize: 12,
    color: '#6c757d',
    marginBottom: 2,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#212529',
  },
  metricUnit: {
    fontSize: 12,
    fontWeight: '400',
    color: '#6c757d',
  },
  content: {
    marginTop: 12,
  },

  // Risk Score styles
  riskCircle: {
    borderWidth: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  riskScoreText: {
    fontWeight: '700',
  },

  // Confidence styles
  confidenceContainer: {
    gap: 4,
  },
  confidenceBar: {
    height: 4,
    backgroundColor: '#e9ecef',
    borderRadius: 2,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 2,
  },
  confidenceLabel: {
    fontSize: 11,
    fontWeight: '500',
  },
});

export default AnalysisCard;
