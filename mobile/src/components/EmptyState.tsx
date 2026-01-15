import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

type IconName = keyof typeof Ionicons.glyphMap;

interface EmptyStateProps {
  icon?: IconName;
  title: string;
  message: string;
  action?: {
    label: string;
    onPress: () => void;
  };
  variant?: 'default' | 'error' | 'search' | 'offline';
}

const variantConfig: Record<string, { icon: IconName; color: string }> = {
  default: { icon: 'folder-open-outline', color: '#6c757d' },
  error: { icon: 'alert-circle-outline', color: '#dc3545' },
  search: { icon: 'search-outline', color: '#6c757d' },
  offline: { icon: 'cloud-offline-outline', color: '#ffc107' },
};

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  message,
  action,
  variant = 'default',
}) => {
  const config = variantConfig[variant];
  const displayIcon = icon ?? config.icon;

  return (
    <View style={styles.container}>
      <View style={[styles.iconContainer, { backgroundColor: `${config.color}15` }]}>
        <Ionicons name={displayIcon} size={48} color={config.color} />
      </View>

      <Text style={styles.title}>{title}</Text>
      <Text style={styles.message}>{message}</Text>

      {action && (
        <TouchableOpacity style={styles.actionButton} onPress={action.onPress}>
          <Text style={styles.actionText}>{action.label}</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  iconContainer: {
    width: 96,
    height: 96,
    borderRadius: 48,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#212529',
    textAlign: 'center',
    marginBottom: 8,
  },
  message: {
    fontSize: 14,
    color: '#6c757d',
    textAlign: 'center',
    lineHeight: 20,
    maxWidth: 280,
  },
  actionButton: {
    marginTop: 24,
    backgroundColor: '#4285f4',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  actionText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});

export default EmptyState;
