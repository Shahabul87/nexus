import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useOffline } from '../hooks/useOffline';

interface NetworkStatusProps {
  showSyncButton?: boolean;
  compact?: boolean;
}

export const NetworkStatus: React.FC<NetworkStatusProps> = ({
  showSyncButton = true,
  compact = false,
}) => {
  const { isOnline, syncState, pendingCount, sync } = useOffline();
  const slideAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(slideAnim, {
      toValue: isOnline ? 0 : 1,
      duration: 300,
      useNativeDriver: true,
    }).start();
  }, [isOnline, slideAnim]);

  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <View
          style={[
            styles.statusDot,
            { backgroundColor: isOnline ? '#198754' : '#dc3545' },
          ]}
        />
        {!isOnline && (
          <Text style={styles.compactText}>Offline</Text>
        )}
        {pendingCount > 0 && (
          <View style={styles.pendingBadge}>
            <Text style={styles.pendingText}>{pendingCount}</Text>
          </View>
        )}
      </View>
    );
  }

  return (
    <Animated.View
      style={[
        styles.container,
        {
          backgroundColor: isOnline ? '#d1e7dd' : '#f8d7da',
          transform: [
            {
              translateY: slideAnim.interpolate({
                inputRange: [0, 1],
                outputRange: [-60, 0],
              }),
            },
          ],
        },
      ]}
    >
      <View style={styles.content}>
        <Ionicons
          name={isOnline ? 'wifi' : 'cloud-offline'}
          size={20}
          color={isOnline ? '#198754' : '#dc3545'}
        />
        <View style={styles.textContainer}>
          <Text
            style={[
              styles.statusText,
              { color: isOnline ? '#198754' : '#dc3545' },
            ]}
          >
            {isOnline ? 'Online' : 'Offline Mode'}
          </Text>
          {!isOnline && (
            <Text style={styles.subText}>
              Data will sync when connected
            </Text>
          )}
          {pendingCount > 0 && (
            <Text style={styles.pendingCountText}>
              {pendingCount} item{pendingCount !== 1 ? 's' : ''} pending sync
            </Text>
          )}
        </View>
      </View>

      {showSyncButton && isOnline && pendingCount > 0 && (
        <TouchableOpacity
          style={[
            styles.syncButton,
            syncState === 'syncing' && styles.syncButtonDisabled,
          ]}
          onPress={sync}
          disabled={syncState === 'syncing'}
        >
          {syncState === 'syncing' ? (
            <Ionicons name="sync" size={16} color="#fff" />
          ) : (
            <>
              <Ionicons name="cloud-upload" size={16} color="#fff" />
              <Text style={styles.syncButtonText}>Sync Now</Text>
            </>
          )}
        </TouchableOpacity>
      )}
    </Animated.View>
  );
};

// Offline Banner Component for screen headers
export const OfflineBanner: React.FC = () => {
  const { isOnline, pendingCount } = useOffline();

  if (isOnline) return null;

  return (
    <View style={styles.banner}>
      <Ionicons name="cloud-offline" size={16} color="#856404" />
      <Text style={styles.bannerText}>
        You&apos;re offline. Changes will sync when connected.
      </Text>
      {pendingCount > 0 && (
        <View style={styles.bannerBadge}>
          <Text style={styles.bannerBadgeText}>{pendingCount}</Text>
        </View>
      )}
    </View>
  );
};

// Sync Status Indicator for showing sync progress
interface SyncStatusProps {
  onSyncComplete?: () => void;
}

export const SyncStatus: React.FC<SyncStatusProps> = ({ onSyncComplete }) => {
  const { syncState, syncProgress, pendingCount } = useOffline();

  React.useEffect(() => {
    if (syncState === 'idle' && pendingCount === 0) {
      onSyncComplete?.();
    }
  }, [syncState, pendingCount, onSyncComplete]);

  if (syncState !== 'syncing') return null;

  return (
    <View style={styles.syncStatus}>
      <View style={styles.syncStatusHeader}>
        <Ionicons name="sync" size={16} color="#4285f4" />
        <Text style={styles.syncStatusText}>Syncing data...</Text>
      </View>
      <View style={styles.syncProgressBar}>
        <View
          style={[
            styles.syncProgressFill,
            { width: `${syncProgress && syncProgress.total > 0 ? (syncProgress.completed / syncProgress.total) * 100 : 0}%` },
          ]}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 0, 0, 0.1)',
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  textContainer: {
    gap: 2,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '600',
  },
  subText: {
    fontSize: 12,
    color: '#6c757d',
  },
  pendingCountText: {
    fontSize: 12,
    color: '#856404',
    fontWeight: '500',
  },
  syncButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#4285f4',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  syncButtonDisabled: {
    backgroundColor: '#6c757d',
  },
  syncButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },

  // Compact styles
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  compactText: {
    fontSize: 12,
    color: '#dc3545',
    fontWeight: '500',
  },
  pendingBadge: {
    backgroundColor: '#ffc107',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
    minWidth: 20,
    alignItems: 'center',
  },
  pendingText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#000',
  },

  // Banner styles
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fff3cd',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#ffc107',
  },
  bannerText: {
    flex: 1,
    fontSize: 12,
    color: '#856404',
  },
  bannerBadge: {
    backgroundColor: '#856404',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
  },
  bannerBadgeText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#fff',
  },

  // Sync status styles
  syncStatus: {
    backgroundColor: '#e7f1ff',
    padding: 12,
    gap: 8,
  },
  syncStatusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  syncStatusText: {
    fontSize: 12,
    color: '#4285f4',
    fontWeight: '500',
  },
  syncProgressBar: {
    height: 4,
    backgroundColor: '#c7d9f7',
    borderRadius: 2,
    overflow: 'hidden',
  },
  syncProgressFill: {
    height: '100%',
    backgroundColor: '#4285f4',
  },
});

export default NetworkStatus;
