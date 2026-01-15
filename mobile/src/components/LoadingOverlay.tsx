import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  Modal,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface LoadingOverlayProps {
  visible: boolean;
  message?: string;
  submessage?: string;
  progress?: number;
  type?: 'spinner' | 'progress' | 'ai';
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  visible,
  message = 'Loading...',
  submessage,
  progress,
  type = 'spinner',
}) => {
  const pulseAnim = React.useRef(new Animated.Value(1)).current;

  React.useEffect(() => {
    if (visible && type === 'ai') {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();
      return () => pulse.stop();
    }
  }, [visible, type, pulseAnim]);

  const renderLoader = () => {
    switch (type) {
      case 'ai':
        return (
          <Animated.View
            style={[
              styles.aiIcon,
              { transform: [{ scale: pulseAnim }] },
            ]}
          >
            <Ionicons name="hardware-chip-outline" size={48} color="#4285f4" />
          </Animated.View>
        );

      case 'progress':
        return (
          <View style={styles.progressContainer}>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${(progress ?? 0) * 100}%` },
                ]}
              />
            </View>
            <Text style={styles.progressText}>
              {Math.round((progress ?? 0) * 100)}%
            </Text>
          </View>
        );

      default:
        return <ActivityIndicator size="large" color="#4285f4" />;
    }
  };

  return (
    <Modal
      transparent
      animationType="fade"
      visible={visible}
      onRequestClose={() => {}}
    >
      <View style={styles.overlay}>
        <View style={styles.container}>
          {renderLoader()}
          <Text style={styles.message}>{message}</Text>
          {submessage && (
            <Text style={styles.submessage}>{submessage}</Text>
          )}
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  container: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    minWidth: 200,
    maxWidth: 300,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 8,
  },
  aiIcon: {
    marginBottom: 16,
  },
  message: {
    fontSize: 16,
    fontWeight: '600',
    color: '#212529',
    textAlign: 'center',
    marginTop: 16,
  },
  submessage: {
    fontSize: 14,
    color: '#6c757d',
    textAlign: 'center',
    marginTop: 8,
  },
  progressContainer: {
    width: '100%',
    alignItems: 'center',
  },
  progressBar: {
    width: '100%',
    height: 8,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#4285f4',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: '#6c757d',
    marginTop: 8,
  },
});

export default LoadingOverlay;
