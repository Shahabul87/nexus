/**
 * Home Screen
 *
 * Main navigation hub for NEXUS assessments
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';

type HomeScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Home'>;
};

interface AssessmentCard {
  id: string;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  screen: keyof RootStackParamList;
  haiDefModel: string;
}

// Comprehensive assessment flows (WHO IMNCI based)
const comprehensiveAssessments: AssessmentCard[] = [
  {
    id: 'pregnant',
    title: 'Maternal Assessment',
    description: 'Complete prenatal checkup with danger signs, anemia screening, and AI synthesis',
    icon: 'woman-outline',
    color: '#e91e63',
    screen: 'PregnantWoman',
    haiDefModel: 'MedGemma + MedSigLIP',
  },
  {
    id: 'newborn',
    title: 'Newborn Assessment',
    description: 'Complete neonatal evaluation with danger signs, jaundice, and cry analysis',
    icon: 'happy-outline',
    color: '#00bcd4',
    screen: 'Newborn',
    haiDefModel: 'MedGemma + HeAR + MedSigLIP',
  },
];

// Quick individual assessments
const quickAssessments: AssessmentCard[] = [
  {
    id: 'anemia',
    title: 'Anemia Quick Screen',
    description: 'Screen for anemia using conjunctiva (eye) photo',
    icon: 'eye-outline',
    color: '#e74c3c',
    screen: 'Anemia',
    haiDefModel: 'MedSigLIP',
  },
  {
    id: 'jaundice',
    title: 'Jaundice Quick Check',
    description: 'Detect jaundice from newborn skin photo',
    icon: 'sunny-outline',
    color: '#f39c12',
    screen: 'Jaundice',
    haiDefModel: 'MedSigLIP',
  },
  {
    id: 'cry',
    title: 'Cry Analysis',
    description: 'Analyze infant cry for birth asphyxia signs',
    icon: 'volume-high-outline',
    color: '#9b59b6',
    screen: 'CryAnalysis',
    haiDefModel: 'HeAR',
  },
  {
    id: 'combined',
    title: 'Combined Analysis',
    description: 'Multi-modal evaluation with clinical synthesis',
    icon: 'medical-outline',
    color: '#1a73e8',
    screen: 'CombinedAssessment',
    haiDefModel: 'MedGemma',
  },
];

export default function HomeScreen({ navigation }: HomeScreenProps): React.JSX.Element {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>NEXUS</Text>
          <Text style={styles.subtitle}>
            AI-Powered Maternal-Neonatal Care
          </Text>
          <View style={styles.badge}>
            <Ionicons name="shield-checkmark" size={16} color="#1a73e8" />
            <Text style={styles.badgeText}>Powered by Google HAI-DEF</Text>
          </View>
        </View>

        {/* Comprehensive Assessments - Featured */}
        <View style={styles.cardsContainer}>
          <Text style={styles.sectionTitle}>Comprehensive Assessments</Text>
          <Text style={styles.sectionSubtitle}>WHO IMNCI-based clinical workflows</Text>

          {comprehensiveAssessments.map((card) => (
            <TouchableOpacity
              key={card.id}
              style={[styles.featuredCard, { borderLeftColor: card.color }]}
              onPress={() => navigation.navigate(card.screen as Exclude<keyof RootStackParamList, 'Results'>)}
              activeOpacity={0.7}
            >
              <View style={[styles.iconContainer, { backgroundColor: card.color }]}>
                <Ionicons name={card.icon} size={32} color="#fff" />
              </View>
              <View style={styles.cardContent}>
                <Text style={styles.cardTitle}>{card.title}</Text>
                <Text style={styles.cardDescription}>{card.description}</Text>
                <View style={styles.modelBadge}>
                  <Text style={styles.modelText}>{card.haiDefModel}</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={24} color="#ccc" />
            </TouchableOpacity>
          ))}
        </View>

        {/* Quick Assessments */}
        <View style={styles.cardsContainer}>
          <Text style={styles.sectionTitle}>Quick Assessments</Text>
          <Text style={styles.sectionSubtitle}>Individual screenings</Text>

          {quickAssessments.map((card) => (
            <TouchableOpacity
              key={card.id}
              style={[styles.card, { borderLeftColor: card.color }]}
              onPress={() => navigation.navigate(card.screen as Exclude<keyof RootStackParamList, 'Results'>)}
              activeOpacity={0.7}
            >
              <View style={[styles.iconContainerSmall, { backgroundColor: card.color }]}>
                <Ionicons name={card.icon} size={24} color="#fff" />
              </View>
              <View style={styles.cardContent}>
                <Text style={styles.cardTitleSmall}>{card.title}</Text>
                <Text style={styles.cardDescriptionSmall}>{card.description}</Text>
                <View style={styles.modelBadge}>
                  <Text style={styles.modelText}>{card.haiDefModel}</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#ccc" />
            </TouchableOpacity>
          ))}
        </View>

        {/* Footer Info */}
        <View style={styles.footer}>
          <View style={styles.warningBox}>
            <Ionicons name="information-circle-outline" size={20} color="#856404" />
            <Text style={styles.warningText}>
              This is a screening tool only. Always confirm findings with laboratory tests.
            </Text>
          </View>
          <Text style={styles.footerText}>
            MedGemma Impact Challenge 2026
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    padding: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
    paddingVertical: 16,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#1a73e8',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 12,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f0fe',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  badgeText: {
    marginLeft: 6,
    color: '#1a73e8',
    fontSize: 12,
    fontWeight: '600',
  },
  cardsContainer: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 13,
    color: '#666',
    marginBottom: 16,
  },
  featuredCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 4,
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  iconContainer: {
    width: 56,
    height: 56,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  iconContainerSmall: {
    width: 44,
    height: 44,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  cardContent: {
    flex: 1,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  cardTitleSmall: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 2,
  },
  cardDescription: {
    fontSize: 13,
    color: '#666',
    marginBottom: 8,
  },
  cardDescriptionSmall: {
    fontSize: 12,
    color: '#666',
    marginBottom: 6,
  },
  modelBadge: {
    alignSelf: 'flex-start',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  modelText: {
    fontSize: 11,
    color: '#666',
    fontWeight: '500',
  },
  footer: {
    alignItems: 'center',
  },
  warningBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#fff3cd',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  warningText: {
    flex: 1,
    marginLeft: 8,
    fontSize: 12,
    color: '#856404',
  },
  footerText: {
    fontSize: 12,
    color: '#999',
  },
});
